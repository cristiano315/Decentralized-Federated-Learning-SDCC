from random import random

import grpc
from concurrent import futures
import time
import torch

# Import generated gRPC code
from rpc_calls import RegistryClient, send_weights_to_peer
from rpc_calls import FederatedNodeServicer
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

from model import SentimentPyTorch
from aggregator import apply_fedavg
from utils import get_weights_as_bytes, load_weights_from_bytes

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

def main():
    # Configuration
    MY_ID = "client-1"
    MY_IP = "127.0.0.1"
    MY_PORT = 50051
    REGISTRY_ADDR = "127.0.0.1:8080"
    TRAINING_NODES = 5
    TOTAL_ROUNDS = 3
    NUM_PEERS_REQUIRED = TRAINING_NODES - 1 # Exclude self
    MAX_DISCOVERY_RETRIES = 5
    WEIGHT_WAIT_TIMEOUT_SECONDS = 30
    
    print("Client is running.")

    # ==========================================
    # 1. Prepare data (Runs ONCE)
    # ==========================================
    file_path = "./all_data_niid_05_keep_3_train_9.json"
    X_train, Mask_train, Y_train, X_val, Mask_val, Y_val = SentimentPyTorch.prepare_dataset(file_path)
    
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

    servicer.peers = peers
    
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
            gossip_fanout = 2
            k = min(gossip_fanout, len(peers))
            initial_gossip_peers = random.sample(peers, k)
            
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

            # D. Wait for Incoming Weights
            print("[Wait] Waiting to receive weights from peers...")
            start_wait_time = time.time()
            while len(servicer.received_weights) < len(peers):
                elapsed_time = time.time() - start_wait_time
                if elapsed_time > WEIGHT_WAIT_TIMEOUT_SECONDS:
                    print(f"[Warning] Timeout! Proceeding with {len(servicer.received_weights)} received models.")
                    break               
                time.sleep(0.5)
                
            print(f"[Info] Received {len(servicer.received_weights)} models.")
            
            # E. Aggregation
            print("[Aggregate] Running FedAvg...")

            deserialized_models = []
            # Deserialize received weights and prepare for aggregation
            for request in servicer.received_weights:
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
        # Graceful Shutdown
        print("[Shutdown] Stopping background gRPC server...")
        server.stop(grace=0)
        registry_client.unregister_node(MY_IP, MY_PORT)
        print("Done.")

    print("TRAINING HAS BEEN COMPLETED.")
    
if __name__ == "__main__":
    main()