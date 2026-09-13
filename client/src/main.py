import json
import math
import random
import datetime
import grpc
from concurrent import futures
import time
import torch
import os
import urllib
import copy

# import generated gRPC code
from rpc_calls import RegistryClient, send_weights_to_peer
from rpc_calls import FederatedNodeServicer
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

from model import SentimentPyTorch
from aggregator import apply_fedavg
from utils import calculate_k, get_weights_as_bytes, load_weights_from_bytes, get_model_hash

def start_grpc_server(port: int, my_id: str) -> tuple:
    """setup the client gRPC server."""
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedNodeServicer(my_id=my_id)
    federated_pb2_grpc.add_FederatedNodeServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"Background gRPC server listening on port {port}...")
    return server, servicer

def get_ecs_container_ip():
    metadata_url = os.getenv("ECS_CONTAINER_METADATA_URI_V4")
    
    if not metadata_url:
        return "Variable ECS_CONTAINER_METADATA_URI_V4 not found. Not running on ECS?"

    try:
        with urllib.request.urlopen(metadata_url) as response:
            body = response.read().decode('utf-8')
            metadata = json.loads(body)
            
            networks = metadata.get('Networks', [])
            if networks and 'IPv4Addresses' in networks[0]:
                return networks[0]['IPv4Addresses'][0]
                
    except Exception as e:
        return f"Error reading metadata: {e}"

    return "IP address not found in metadata"

def run_training_loop(config, global_model, servicer, registry_client, MY_ID, device, RESPAWNED):
    """handle federated learning and client failure"""
    
    # extract all variables from config
    training_nodes = config['training_nodes']
    total_rounds = config['total_rounds']
    start_round = config['start_round']
    peers = config['peers']
    weight_wait_timeout = config['weight_wait_timeout']
    training_set_percentage = config['training_set_percentage']
    num_epochs = config['num_epochs']
    
    # preparazione dataset
    bucket_name = "sdcc-dataset-264452429750-us-east-1-an"
    s3_key = "all_data_niid_05_keep_3_train_9.json"
    X_train, Mask_train, Y_train, X_val, Mask_val, Y_val, X_test, Mask_test, Y_test = SentimentPyTorch.prepare_dataset(
        bucket_name=bucket_name, 
        s3_key=s3_key, 
        num_training_nodes=training_nodes, 
        training_set_percentage=training_set_percentage,
        max_samples_per_client=1000,
        seed=42 
    )
    my_samples = len(Y_train)
    
    # setup servicer for gossiping
    servicer.num_samples = my_samples
    servicer.peers = peers
    k = calculate_k(peers)
    servicer.fanout = k
    servicer.received_weights.clear()
    servicer.seen_messages.clear()
    servicer.latest_local_weights = None
    servicer.round_num = start_round
    
    # if a node is the backup from a failure, RESPAWNED = True
    if RESPAWNED:
        print("\nNodo identified as RESPAWNED. Starting weights recovery from peers...")
        
        # ask weights to all the other peers
        for peer in peers: 
            servicer.get_weights_from_peer(peer['ip'], peer['port'], peer['id'], MY_ID, start_round)

        print("Waiting for local weights from peers to complete recovery...")
        start_wait_time = time.time()
        
        # recieve weights from other peers
        while True:
            with servicer.lock:
                rec_weights = servicer.received_weights.get(start_round, [])
                if len(rec_weights) > 0:
                    break
            if time.time() - start_wait_time > weight_wait_timeout:
                break
            time.sleep(0.5)

        with servicer.lock:
            round_payloads = servicer.received_weights.get(start_round, [])

        # need at least one model recieved from another peer to rebuild the model, this is not the first round
        if not round_payloads:
            print("No weights received from peers during recovery. Recovery failed.")
            return

        # fedavg application with the models recieved
        print(f"Recovery: Execution of FedAvg on {len(round_payloads)} models received from peers...")
        round_payloads.sort(key=lambda x: x.sender_id)
        deserialized_models = []

        for request in round_payloads:
            state_dict = load_weights_from_bytes(request.model_weights)
            deserialized_models.append({
                'sender_id': request.sender_id,
                'weights': state_dict,
                'num_samples': request.num_samples
            })

        global_model = apply_fedavg(global_model, deserialized_models)
        servicer.received_weights.clear()
        
        # skip current round since we just recovered it
        start_round += 1
        print(f"Model succesfully restored. The training will resume from round {start_round + 1}.")

    print(f"\nStarting training session. Peers: {len(peers)}, Fanout (k): {k}")
    start_training_time = time.time()

    # actual training cycle
    # variables for time statistics
    total_compute_time_acc = 0.0
    total_comm_time_acc = 0.0
    rounds_executed = 0

    try:
        for round_num in range(start_round, total_rounds):
            print(f"\n{'='*10} ROUND {round_num + 1}/{total_rounds} {'='*10}")
            
            # local Training
            start_compute = time.time()
            global_model, my_samples = SentimentPyTorch.train_local(
                model=global_model, 
                X_train=X_train, Mask_train=Mask_train, Y_train=Y_train, 
                X_val=X_val, Mask_val=Mask_val, Y_val=Y_val, 
                device=device,
                num_epochs=num_epochs
            )
            compute_time = time.time() - start_compute

            # serialization
            start_comm = time.time()
            payload_bytes = get_weights_as_bytes(global_model)
            servicer.latest_local_weights = payload_bytes
            servicer.round_num = round_num
            servicer.seen_messages.add((MY_ID, round_num))
            
            # send weights to other peer with gossiping
            actual_k = min(k, len(peers))
            initial_gossip_peers = random.sample(peers, actual_k)
            
            for peer in initial_gossip_peers:
                payload = federated_pb2.WeightPayload(
                    sender_id=MY_ID,
                    round_number=round_num,
                    model_weights=payload_bytes,
                    num_samples=my_samples
                )
                if send_weights_to_peer(peer['ip'], peer['port'], peer['id'], payload) == 1:
                    print(f"Warning: Peer {peer['id']} unresponsive. Cleanup...")
                    servicer.remove_peer_by_id(peer['id'])
                    peers = servicer.peers
                    registry_client.signal_unresponsive_node(
                        peer['ip'], peer['port'], peer['id'], 
                        training_nodes, total_rounds, start_round, 5, weight_wait_timeout, training_set_percentage, num_epochs
                    )

            # wait weights from other peer
            start_wait_time = time.time()
            print(f"Waiting for weights from peers for round {round_num + 1} (Timeout: {weight_wait_timeout}s)...")
            enough_weights_received = False
            while True:
                with servicer.lock:
                    current_round_weights = servicer.received_weights.get(round_num, [])
                    if len(current_round_weights) >= len(peers):
                        print(f"All weights received from peers for round {round_num + 1}.")
                        break
                    if len(current_round_weights) >= math.ceil(actual_k/2):
                        enough_weights_received = True

                if time.time() - start_wait_time > weight_wait_timeout:
                    if not enough_weights_received:
                        print(f"Warning: Timeout! Proceeding with {len(current_round_weights)} models received.")
                    else:
                        print(f"Received enough weights for round {round_num + 1}.")
                    break
                time.sleep(0.5)
                
            # aggregate peers weights
            with servicer.lock:
                round_payloads = servicer.received_weights.get(round_num, [])

            round_payloads.sort(key=lambda x: x.sender_id)
            deserialized_models = []
            
            local_fc_only = {k: v.cpu() for k, v in global_model.state_dict().items() if k.startswith('fc.')}
            deserialized_models.append({
                'sender_id': MY_ID,
                'weights': copy.deepcopy(local_fc_only),
                'num_samples': my_samples
            })
            
            for request in round_payloads:
                state_dict = load_weights_from_bytes(request.model_weights)
                deserialized_models.append({
                    'sender_id': request.sender_id,
                    'weights': state_dict,
                    'num_samples': request.num_samples
                })
            
            # hash before FedAvg
            fc_hash_before = get_model_hash(global_model, only_trainable=True)
            print(f"VERIFICATION: BEFORE FEDAVG round {round_num + 1} Model Classifier SHA-256: {fc_hash_before}")

            # apply FedAvg
            global_model = apply_fedavg(global_model, deserialized_models)
            
            # clear servicer buffer
            with servicer.lock:
                servicer.received_weights.pop(round_num, None)

            # hash after FedAvg
            fc_hash_after = get_model_hash(global_model, only_trainable=True)
            print(f"VERIFICATION: AFTER FEDAVG round {round_num + 1} Model Classifier SHA-256: {fc_hash_after}")
            
            # execution time for a single round
            comm_time = time.time() - start_comm
            print(f"Round {round_num}: Computing time = {compute_time:.2f}s, Network/Waiting time = {comm_time:.2f}s")

            # for final times print
            total_compute_time_acc += compute_time
            total_comm_time_acc += comm_time
            rounds_executed += 1

        print("\nTRAINING HAS BEEN COMPLETED.")
        total_training_time = time.time() - start_training_time
        print(f"Total training time: {str(datetime.timedelta(seconds = total_training_time))}\n")

        # display time averages for round
        if rounds_executed > 0:
            avg_compute = total_compute_time_acc / rounds_executed
            avg_comm = total_comm_time_acc / rounds_executed
            print(f"MEAN Computing time per round: {avg_compute:.2f}s")
            print(f"MEAN Network/Waiting time per round: {avg_comm:.2f}s")

        # check relative model signature
        fc_hash_final = get_model_hash(global_model, only_trainable=True)
        full_hash_final = get_model_hash(global_model, only_trainable=False)

        print("\n" + "="*10)
        print(f"VERIFICATION: Final Model Classifier SHA-256: {fc_hash_final}")
        print(f"VERIFICATION: Final Model Full SHA-256:       {full_hash_final}")
        print("="*10 + "\n")

        # evaluation
        print("Final evaluation of globlal model on local test set...")
        try:
            SentimentPyTorch.evaluate_global(global_model, X_test, Mask_test, Y_test, device)
        except Exception as e:
            print(f"Error: Final evaluation failed: {e}")

    except Exception as e:
        print(f"Error: Exception occurred during training loop: {e}")

def main():
    # read environment variables
    MY_ID = str(os.getenv("CLIENT_ID", f"node_{random.randint(1000,9999)}"))
    MY_IP = get_ecs_container_ip()
    MY_PORT = int(os.getenv("PORT", 50051))
    REGISTRY_ADDR = os.getenv("REGISTRY_ADRESS", "registry-nlb-99c42fd51de63d80.elb.us-east-1.amazonaws.com:8080")
    STARTER = os.getenv("STARTER", "false").lower() == "true"
    RESPAWNED = os.getenv("RESPAWNED", "false").lower() == "true"
    
    # start gRPC server and registry client
    registry_client = RegistryClient(REGISTRY_ADDR, MY_ID)
    server, servicer = start_grpc_server(MY_PORT, MY_ID)
    registry_client.servicer = servicer

    # important to check if we can use the gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    global_model = SentimentPyTorch(num_class=2).to(device)

    print(f"Init: Node ID: {MY_ID}, IP: {MY_IP}, PORT: {MY_PORT}, STARTER: {STARTER}, RESPAWNED: {RESPAWNED}")

    # check if it is a normal or a respawned node with RESPAWNED
    initial_status = "working" if STARTER else "idle"
    
    is_starter_execution = STARTER

    # in any case register the node to the peers list
    if not RESPAWNED:
        if not registry_client.register_node(MY_IP, MY_PORT, initial_status):
            print("Error connecting to Registry. Exiting.")
            server.stop(grace=0)
            return
    else:
        print("Node started in RESPAWNED mode.")
        if not registry_client.register_respawned_node(MY_IP, MY_PORT):
            print("Error connecting to Registry for respawned node. Exiting.")
            server.stop(grace=0)
            return


    try:
        while True:
            config = None
            
            if is_starter_execution:
                print("\nSTARTER Node. Discovery phase...")
                
                # get environment variables
                training_nodes = int(os.getenv("TRAINING_NODES", 5))
                total_rounds = int(os.getenv("TOTAL_ROUNDS", 5))
                start_round = int(os.getenv("START_ROUND", 0))
                num_peers_required = int(os.getenv("NUM_PEERS_REQUIRED", training_nodes - 1))
                max_retries = int(os.getenv("MAX_DISCOVERY_RETRIES", 5))
                timeout_sec = int(os.getenv("WEIGHT_WAIT_TIMEOUT_SECONDS", 30))
                training_set_percentage = float(os.getenv("TRAINING_SET_PERCENTAGE", 0.7))
                num_epochs = int(os.getenv("NUM_EPOCHS", 1))

                # get peers list
                peers = []
                retries = 0
                while len(peers) < num_peers_required:
                    if retries >= max_retries:
                        print("Error: Discovery timeout.")
                        break
                    print(f"Discovery: Fetching peers ({retries+1}/{max_retries})...")
                    peers = registry_client.get_peer_list(node_request_count=num_peers_required)
                    if len(peers) < num_peers_required:
                        time.sleep(5)
                        retries += 1
                
                # if we have all the peers, give them the config and make them start
                if len(peers) < num_peers_required:
                    print("Error: Unable to start training due to missing peers.")
                else:
                    starter_peer = {"id": MY_ID, "ip": MY_IP, "port": MY_PORT}
                    config = {
                        'training_nodes': training_nodes,
                        'total_rounds': total_rounds,
                        'start_round': start_round,
                        'num_peers_required': num_peers_required,
                        'max_discovery_retries': max_retries,
                        'weight_wait_timeout': timeout_sec,
                        'training_set_percentage': training_set_percentage,
                        'num_epochs': num_epochs,
                        'peers': peers
                    }
                    peers_config = {
                        'training_nodes': training_nodes,
                        'total_rounds': total_rounds,
                        'start_round': start_round,
                        'num_peers_required': num_peers_required,
                        'max_discovery_retries': max_retries,
                        'weight_wait_timeout': timeout_sec,
                        'training_set_percentage': training_set_percentage,
                        'num_epochs': num_epochs,
                        'peers': [*peers, starter_peer]  # Include the starter node itself in the peers list
                    }
                    
                    print("STARTER Node. Sending RPC StartTraining to peers...")
                    for peer in peers:
                        servicer.send_start_training_signal(peer, peers_config)

                is_starter_execution = False

            else:
                # for idle and support node, wait the timeout after a training
                print("\nIDLE: Waiting for training requests (Timeout: 5 minutes)...")
                servicer.start_training_event.clear()
                
                received_signal = servicer.start_training_event.wait(timeout=300)

                if not received_signal:
                    print("\nNo requests in 5 minutes of idle. Shutting down...")
                    break

                config = servicer.pending_training_config
                registry_client.update_status("working")

            # start training
            if config:
                try:
                    run_training_loop(
                        config, global_model, servicer, registry_client, MY_ID, device, RESPAWNED)
                except Exception as e:
                    print(f"Error during training execution: {e}")
                RESPAWNED = False

            # reset global model for new executions
            print("\nRestarting global model for future executions...")
            torch.manual_seed(42)
            global_model = SentimentPyTorch(num_class=2).to(device)

            # set up IDLE state
            print("\nRestoring status to IDLE for 5 minutes...")
            registry_client.update_status("idle")

    except KeyboardInterrupt:
        print("\nShutdown (KeyboardInterrupt).")
    finally:
        # shutdown gRPC server
        print("Shutdown. Unregistering and stopping gRPC...")
        registry_client.unregister_node(MY_IP, MY_PORT)
        server.stop(grace=5)
        print("Done.")
    
    
if __name__ == "__main__":
    main()