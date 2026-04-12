import grpc

# Import generated gRPC code
from rpc_calls import RegistryClient
import federated_pb2
import federated_pb2_grpc

def main():
    # Dummy main, to implement
    # Configuration: get from AWS Parameter Store or environment variables
    # Set client ID, port, number of rounds and max number of peers to gossip with
    # Dummy valuse TO CHANGE:
    MY_ID = "client-1"
    MY_IP = "127.0.0.1"
    MY_PORT = 50051
    REGISTRY_ADDR = "127.0.0.1:8080"
    TRAINING_NODES = 5
    
    #1 Prepare model
    
    #2 Register to service registry and get list of peers:
    
    # Initialize the RPC Client wrapper
    network_client = RegistryClient(REGISTRY_ADDR, MY_ID)
    print(f"Client initialized with ID: {MY_ID}, IP: {MY_IP}, Port: {MY_PORT}")
    
    # Register
    if not network_client.register_node(MY_IP, MY_PORT):
        print("Fatal error: Could not connect to Registry. Exiting.")
        return
    
    # Get list of peers
    peers = network_client.get_peer_list(node_request_count=TRAINING_NODES)
    print(f"Found {len(peers)} peers ready for gossip.")
    
    #3 Start gRPC server
    
    #4 Prepare data
    
    #5 Discovery
    
    #6 Training loop
    
    print("Client is running.")

if __name__ == "__main__":
    main()