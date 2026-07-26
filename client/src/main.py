import json
import random

import grpc
from concurrent import futures
import time
import torch
import os

import urllib

# Import generated gRPC code
from rpc_calls import RegistryClient, send_weights_to_peer
from rpc_calls import FederatedNodeServicer
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

from model import SentimentPyTorch
from aggregator import apply_fedavg
from utils import calculate_k, get_weights_as_bytes, load_weights_from_bytes

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

    # ==========================================
    # 2. Prepare global model
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Initializing global model on {device}...")
    
    global_model = SentimentPyTorch(num_class=2)
    global_model.to(device)

    # ==========================================
    # 3. Register to service registry
    # ==========================================
    registry_client = RegistryClient(REGISTRY_ADDR, MY_ID)
    print(f"Client initialized with ID: {MY_ID}, IP: {MY_IP}, Port: {MY_PORT}")
    
    if not registry_client.register_node(MY_IP, MY_PORT):
        print("Fatal error: Could not connect to Registry. Exiting.")
        # return UNCOMMENT WHEN NOT TESTING
    
    # ==========================================
    # 4. Start gRPC server
    # ==========================================
    server, servicer = start_grpc_server(MY_PORT, MY_ID)

    # ==========================================
    # 5. Discovery
    # ==========================================
    peers = []
    retries = 0

    while len(peers) < NUM_PEERS_REQUIRED:
        if retries >= MAX_DISCOVERY_RETRIES:
            print("[Error] Discovery timeout reached. Not enough peers.")
            server.stop(grace=0)
            break 
        print(f"[Discovery] Fetching peers from Registry... (Attempt {retries + 1}/{MAX_DISCOVERY_RETRIES})")
        peers = registry_client.get_peer_list(node_request_count=NUM_PEERS_REQUIRED)
        if len(peers) < NUM_PEERS_REQUIRED:
            time.sleep(5)
            retries += 1
    
    print(f"Found {len(peers)} peers ready for gossip.")
    
    # Calculate dynamic gossip fanout based on the number of peers
    k = calculate_k(peers)
    print(f"[Info] Network of {len(peers) + 1} nodes. Gossip fanout (k) dynamically set to {k}.")

    servicer.peers = peers
    servicer.fanout = k
    
    # ==========================================
    # 6. Training loop
    # ==========================================
    try:
        for round_num in range(TOTAL_ROUNDS):
            print(f"\n{'='*30}\n ROUND {round_num + 1}\n{'='*30}")
            
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

            # Add self to seen messages to avoid processing our own gossip
            servicer.seen_messages.add((MY_ID, round_num))
            
            # C. Gossip: Send weights to a random subset of peers
            actual_k = min(k, len(peers))
            initial_gossip_peers = random.sample(peers, actual_k)
            
            for peer in initial_gossip_peers:
                print(f"[Gossip] Sending weights to {peer['id']}...")
                payload = federated_pb2.WeightPayload(
                    sender_id=MY_ID,
                    round_number=round_num,
                    model_weights=payload_bytes,
                    num_samples=my_samples
                )
                send_weights_to_peer(peer['ip'], peer['port'], payload)
                print(f"sent {len(payload_bytes)} bytes") 

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
                
            # print(f"[Info] Received {len(servicer.received_weights)} models.")
            
            # E. Aggregation
            print("[Aggregate] Running FedAvg...")

            with servicer.lock:
                round_payloads = servicer.received_weights.get(round_num, [])

            deserialized_models = []
            # Deserialize received weights and prepare for aggregation
            for request in round_payloads:
                state_dict = load_weights_from_bytes(request.model_weights)
                deserialized_models.append({
                    'weights': state_dict,
                    'num_samples': request.num_samples
                })

            global_model = apply_fedavg(global_model, deserialized_models, my_samples)
            
            # F. Clear Buffer for the next round
            servicer.received_weights.clear()

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
    
if __name__ == "__main__":
    main()