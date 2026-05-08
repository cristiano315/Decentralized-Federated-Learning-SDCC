#NEED TO DECIDE IF WE WANT TO KEEP THE CONTAINER ACTIVE AFTER FINISHING OR NOT.

import grpc
from concurrent import futures
import time

# Import generated gRPC code
from rpc_calls import RegistryClient, send_weights_to_peer
from rpc_calls import FederatedNodeServicer
import federated_pb2
import federated_pb2_grpc

def start_grpc_server(port: int) -> tuple:
    """
    Initializes and starts the background gRPC server.
    Returns the server instance (to stop it safely later) 
    and the servicer instance (to access the received weights).
    """
    # 1. Create a thread pool for handling incoming gRPC requests concurrently
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # 2. Instantiate the servicer class from rpc_calls.py
    servicer = FederatedNodeServicer()

    # 3. Bind the servicer to the gRPC server using the generated stub
    federated_pb2_grpc.add_FederatedNodeServicer_to_server(servicer, server)

    # 4. Bind the server to the designated port and start it
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"[Server] Background gRPC server listening on port {port}...")
    return server, servicer

def main():
    # Dummy main, to implement
    # Configuration: get from AWS Parameter Store or environment variables
    # Set client ID, port, number of rounds and max number of peers to gossip with
    # Dummy valuse TO CHANGE:

    # Inject whit ECS as ambient variables OR use AWS parameter store (boto3)

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

    
    #1 Prepare model

    # TO IMPLEMENT
    
    #2 Register to service registry and get list of peers:
    
    # Initialize the RPC Client wrapper
    registry_client = RegistryClient(REGISTRY_ADDR, MY_ID)
    print(f"Client initialized with ID: {MY_ID}, IP: {MY_IP}, Port: {MY_PORT}")
    
    # Register
    if not registry_client.register_node(MY_IP, MY_PORT):
        print("Fatal error: Could not connect to Registry. Exiting.")
        #return UNCOMMENT WHEN NOT TESTING
    
    
    #3 Start gRPC server
    server, servicer = start_grpc_server(MY_PORT)

    
    #4 Discovery
    # Get list of peers
    peers = []
    retries = 0

    while len(peers) < NUM_PEERS_REQUIRED:
        if retries >= MAX_DISCOVERY_RETRIES:
            print("[Error] Discovery timeout reached. Not enough peers. Exiting.")
            server.stop(grace=0)
            #return # Exit the program UNCOMMENT WHEN NOT TESTING
            break # For testing, we break to proceed with dummy peers TO REMOVE

        print(f"[Discovery] Fetching peers from Registry... (Attempt {retries + 1}/{MAX_DISCOVERY_RETRIES})")
        peers = registry_client.get_peer_list(node_request_count=NUM_PEERS_REQUIRED)
        if len(peers) < NUM_PEERS_REQUIRED:
            time.sleep(5)
            retries += 1
    
    print(f"Found {len(peers)} peers ready for gossip.")
    
    #5 Prepare data

    # TO IMPLEMENT
    
    #6 Training loop

    try:
        for round_num in range(TOTAL_ROUNDS):
            print(f"\n{'='*30}\n ROUND {round_num + 1}\n{'='*30}")
            
            # A. Local Training
            print("[Train] Training model on local dataset...")
            # model.train_local(...) TO IMPLEMENT
            # my_samples = ...
            time.sleep(2) # Simulating training time
            
            # B. Serialize Weights
            # payload_bytes = serialize_weights(model)
            payload_bytes = b"dummy_bytes_for_testing" # Placeholder
            my_samples = 100
            
            # C. Gossip: Send weights to peers
            for peer in peers:
                print(f"[Gossip] Sending weights to {peer['id']}...")
                payload = federated_pb2.WeightPayload(
                    sender_id=MY_ID,
                    round_number=round_num,
                    model_weights=payload_bytes,
                    num_samples=my_samples
                )
                send_weights_to_peer(peer['ip'], peer['port'], payload)
                print(f"sent {len(payload_bytes)} bytes") #fix

            # D. Wait for Incoming Weights
            print("[Wait] Waiting to receive weights from peers...")
            start_wait_time = time.time()
            # The background server is populating 'servicer.received_weights'
            while len(servicer.received_weights) < len(peers):
            # Check if we exceeded the maximum allowed waiting time
                elapsed_time = time.time() - start_wait_time
                if elapsed_time > WEIGHT_WAIT_TIMEOUT_SECONDS:
                    print(f"[Warning] Timeout! Proceeding with only {len(servicer.received_weights)} received models.")
                    break # Proceed to aggregation                
                time.sleep(0.5)
                
            print(f"[Info] Received {len(servicer.received_weights)} models.")
            
            # E. Aggregation
            print("[Aggregate] Running FedAvg...")
            # apply_fedavg(model, servicer.received_weights, my_samples) TO IMPLEMENT
            
            # F. Clear Buffer for the next round
            servicer.received_weights.clear()

    except KeyboardInterrupt:
        print("\n[Shutdown] Training interrupted by user.")
    finally:
        # --- 5. Graceful Shutdown ---
        print("[Shutdown] Stopping background gRPC server...")
        server.stop(grace=0)
        # Unregister from the registry
        registry_client.unregister_node(MY_IP, MY_PORT)
        print("Done.")

    print("TRAINING HAS BEEN COMPLETED.")
    
if __name__ == "__main__":
    main()
