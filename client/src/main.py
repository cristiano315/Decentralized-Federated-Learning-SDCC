import json
import random

import grpc
from concurrent import futures
import time
import torch
import os

import urllib
import copy

# Import generated gRPC code
from rpc_calls import RegistryClient, send_weights_to_peer
from rpc_calls import FederatedNodeServicer
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

from model import SentimentPyTorch
from aggregator import apply_fedavg
from utils import calculate_k, get_weights_as_bytes, load_weights_from_bytes, get_model_hash

def start_grpc_server(port: int, my_id: str) -> tuple:
    """
    Initializes and starts the background gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedNodeServicer(my_id=my_id)
    federated_pb2_grpc.add_FederatedNodeServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"[Server] Background gRPC server listening on port {port}...")
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

def main():
    # Configuration
    MY_ID = str(os.getenv("CLIENT_ID"))
    MY_IP = get_ecs_container_ip()
    MY_PORT = int(os.getenv("PORT", 50051))
    REGISTRY_ADDR = os.getenv("REGISTRY_ADRESS")
    TRAINING_NODES = int(os.getenv("TRAINING_NODES", 5))
    TOTAL_ROUNDS = int(os.getenv("TOTAL_ROUNDS", 5))
    START_ROUND = int(os.getenv("START_ROUND", 0))
    NUM_PEERS_REQUIRED = int(os.getenv("NUM_PEERS_REQUIRED", TRAINING_NODES - 1)) # to exclude self node
    MAX_DISCOVERY_RETRIES = int(os.getenv("MAX_DISCOVERY_RETRIES", 5))
    WEIGHT_WAIT_TIMEOUT_SECONDS = int(os.getenv("WEIGHT_WAIT_TIMEOUT_SECONDS", 30))
    
    print("Client is running.")

    # ==========================================
    # 1. Prepare data (Runs ONCE)
    # ==========================================
    bucket_name = "sdcc-dataset-771379920513-us-east-1-an"
    s3_key = "all_data_niid_05_keep_3_train_9.json"
    X_train, Mask_train, Y_train, X_val, Mask_val, Y_val = SentimentPyTorch.prepare_dataset(bucket_name, s3_key)

    my_samples = len(Y_train)
    # ==========================================
    # 2. Prepare global model
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Initializing global model on {device}...")
    
    seed = 42
    torch.manual_seed(seed)
    global_model = SentimentPyTorch(num_class=2)
    global_model.to(device)

    # ==========================================
    # 3. Register to service registry
    # ==========================================
    registry_client = RegistryClient(REGISTRY_ADDR, MY_ID)
    print(f"Client initialized with ID: {MY_ID}, IP: {MY_IP}, Port: {MY_PORT}")
    
    if START_ROUND == 0:
        if not registry_client.register_node(MY_IP, MY_PORT):
            print("Fatal error: Could not connect to Registry. Exiting.")
            return
    else:
        if not registry_client.register_respawned_node(MY_IP, MY_PORT, START_ROUND):
            print("Fatal error: Could not connect to Registry for respawned node. Exiting.")
            return
    
    # ==========================================
    # 4. Start gRPC server
    # ==========================================
    server, servicer = start_grpc_server(MY_PORT, MY_ID)
    # add num_samples to servicer
    servicer.num_samples = my_samples
    registry_client.servicer = servicer

    # ==========================================
    # 5. Discovery
    # ==========================================
    peers = []
    retries = 0

    while len(peers) < (NUM_PEERS_REQUIRED - 1):
        if retries >= MAX_DISCOVERY_RETRIES:
            print("[Error] Discovery timeout reached. Not enough peers.")
            server.stop(grace=0)
            break 
        print(f"[Discovery] Fetching peers from Registry... (Attempt {retries + 1}/{MAX_DISCOVERY_RETRIES})")
        peers = registry_client.get_peer_list(node_request_count=NUM_PEERS_REQUIRED)
        if len(peers) < (NUM_PEERS_REQUIRED - 1):
            time.sleep(5)
            retries += 1
    
    print(f"Found {len(peers)} peers ready for gossip.")
    
    # Calculate dynamic gossip fanout based on the number of peers
    k = calculate_k(peers)
    print(f"[Info] Network of {len(peers) + 1} nodes. Gossip fanout (k) dynamically set to {k}.")

    servicer.peers = peers
    servicer.fanout = k

    start_round = START_ROUND

    # Respawn recovery: if the node is respawned, it should request the local weights from peers to recover the model state
    if start_round != 0:
        print("[Recovery] Nodo respawnato. Richiedo i pesi locali ai peer...")
        # prendo i pesi
        # 1: chiedo a ogni peer trovato i loro pesi
        for peer in peers: 
            # NOTA: get_weights_from_peer deve dire al peer di inviare i suoi pesi al nostro SendWeights
            # Questa funziona popola automaticamente servicer.received_weights con i pesi ricevuti dai peer
            servicer.get_weights_from_peer(peer['ip'], peer['port'], peer['id'], MY_ID, start_round)

        # 2: attendo la risposta
        # D. Wait for Incoming Weights for the current round_num
        print("[Wait] Waiting to receive local weights from peers...")
        start_wait_time = time.time()

        if servicer.received_weights[start_round] is None or len(servicer.received_weights[start_round]) == 0:
            print("[Error] Nessun peso ricevuto. Fallimento del recovery.")
            #FARE RETURN O GESTIONE ERRORE
            return
        # E. Aggregation
        with servicer.lock:
            round_payloads = servicer.received_weights.get(start_round, [])

        round_payloads.sort(key=lambda x: x.sender_id)
        # faccio fedavg
        # evitare di usare il modello locale perché non è stato addestrato
        print(f"[Aggregate] Running FedAvg on {len(round_payloads)} peer models...")

        deserialized_models = []
        # Deserialize received weights and prepare for aggregation
        for request in round_payloads:
            state_dict = load_weights_from_bytes(request.model_weights)
            deserialized_models.append({
                'sender_id': request.sender_id,
                'weights': state_dict,
                'num_samples': request.num_samples
            })

        global_model = apply_fedavg(global_model, deserialized_models)
            
        # F. Clear Buffer for the next round
        servicer.received_weights.clear()
        # setto start_round al round successivo a quello appena completato
        start_round += 1
    
    # ==========================================
    # 6. Training loop
    # ==========================================
    try:
        for round_num in range(start_round, TOTAL_ROUNDS):
            print(f"\n{'='*10} ROUND {round_num + 1} {'='*10}")
            
            # A. Local Training (Updating the global_model)
            print("[Train] Training model on local dataset...")
            global_model, my_samples = SentimentPyTorch.train_local(
                model=global_model, 
                X_train=X_train, 
                Mask_train=Mask_train, 
                Y_train=Y_train, 
                X_val=X_val, 
                Mask_val=Mask_val, 
                Y_val=Y_val, 
                device=device
            )

            # B. Serialize Weights for Gossip
            print("[Serialize] Converting model weights to bytes...")
            payload_bytes = get_weights_as_bytes(global_model)
            servicer.latest_local_weights = payload_bytes # byte di payload per i client che vanno in failure durante il training
            servicer.round_num = round_num

            # Add self to seen messages to avoid processing our own gossip
            servicer.seen_messages.add((MY_ID, round_num))
            
            # C. Gossip: Send weights to a random subset of peers
            actual_k = min(k, len(peers))
            initial_gossip_peers = random.sample(peers, actual_k)
            
            for peer in initial_gossip_peers:
                payload = federated_pb2.WeightPayload(
                    sender_id=MY_ID,
                    round_number=round_num,
                    model_weights=payload_bytes,
                    num_samples=my_samples
                )
                if send_weights_to_peer(peer['ip'], peer['port'], peer['id'], payload) == 1: # If 1 is returned, the peer is unresponsive
                    print(f"[Warning] Peer {peer['id']} is unresponsive. Removing from peer list and signaling to registry.")
                    # 1. Rimuovi dal Servicer in modo thread-safe
                    servicer.remove_peer_by_id(peer['id'])
                    
                    # 2. Aggiorna la reference locale per il ciclo corrente
                    peers = servicer.peers 
                    
                    # 3. Segnala al Go Registry
                    registry_client.signal_unresponsive_node(
                        peer['ip'], peer['port'], peer['id'], 
                        NUM_PEERS_REQUIRED, TOTAL_ROUNDS, 
                        START_ROUND, MAX_DISCOVERY_RETRIES, 
                        WEIGHT_WAIT_TIMEOUT_SECONDS
                    )

            # D. Wait for Incoming Weights for the current round_num
            print(f"[Wait] Waiting to receive weights for round {round_num + 1}...")
            start_wait_time = time.time()

            while True:
                with servicer.lock:
                    current_round_weights = servicer.received_weights.get(round_num, [])
                    if len(current_round_weights) >= len(peers):
                        break

                if time.time() - start_wait_time > WEIGHT_WAIT_TIMEOUT_SECONDS:
                    print(f"[Warning] Timeout! Proceeding with {len(current_round_weights)} received models.")
                    break
                time.sleep(0.5)
                
            # E. Aggregation
            with servicer.lock:
                round_payloads = servicer.received_weights.get(round_num, [])

            round_payloads.sort(key=lambda x: x.sender_id)
            
            print(f"[Aggregate] Running FedAvg on {len(round_payloads)} peer models...")

            deserialized_models = []
            # add local model in decentralized models
            local_fc_only = {k: v.cpu() for k, v in global_model.state_dict().items() if k.startswith('fc.')}
            
            deserialized_models.append({
                    'sender_id':MY_ID,
                    'weights': copy.deepcopy(local_fc_only),
                    'num_samples': my_samples
                })
            # Deserialize received weights and prepare for aggregation
            for request in round_payloads:
                state_dict = load_weights_from_bytes(request.model_weights)
                deserialized_models.append({
                    'sender_id': request.sender_id,
                    'weights': state_dict,
                    'num_samples': request.num_samples
                })
            
            fc_hash = get_model_hash(global_model, only_trainable=True)
            print(f"[VERIFICATION] PRIMA DI FEDAVG round {round_num} Model Classifier SHA-256: {fc_hash}")

            global_model = apply_fedavg(global_model, deserialized_models)
            
            # F. Clear Buffer for the next round
            servicer.received_weights.clear()

            fc_hash = get_model_hash(global_model, only_trainable=True)
            print(f"[VERIFICATION] DOPO FEDAVG round {round_num} Model Classifier SHA-256: {fc_hash}")

    except KeyboardInterrupt:
        print("\n[Shutdown] Training interrupted by user.")
    finally:
        # Keep server running briefly so remaining nodes can still fetch weights
        print("[Shutdown] Waiting for network to complete rounds...")
        time.sleep(60) # Keep gRPC server alive for lagging peers
        
        print("[Shutdown] Stopping background gRPC server...")
        server.stop(grace=0)
        registry_client.unregister_node(MY_IP, MY_PORT)
        print("Done.")

    print("TRAINING HAS BEEN COMPLETED.")

    # ==========================================
    # 7. Verification / Print Model Hash
    # ==========================================
    # Hash dei soli pesi addestrati (es. il classificatore)
    fc_hash = get_model_hash(global_model, only_trainable=True)

    # Hash di tutti i parametri del modello
    full_hash = get_model_hash(global_model, only_trainable=False)

    print("\n" + "="*10)
    print(f"[VERIFICATION] Final Model Classifier SHA-256: {fc_hash}")
    print(f"[VERIFICATION] Final Model Full SHA-256:       {full_hash}")
    print("="*10 + "\n")

    # 8. Evaluation
    # check ability to generalize with the data untouched by the users
    print("Valutazione globale su modello finale")

    try:
        #tokenizza dataset
        X_full, Mask_full, Y_full = SentimentPyTorch.prepare_eval_dataset(bucket_name, s3_key)

        #valuta modello aggregato
        SentimentPyTorch.evaluate_global(global_model, X_full, Mask_full, Y_full, device)
    except Exception as e:
        print(f"[Error] Valutazione globale fallita: {e}")
    
    
if __name__ == "__main__":
    main()