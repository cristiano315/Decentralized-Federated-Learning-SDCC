#File for grpc calls implementation

import grpc
import federated_pb2
import federated_pb2_grpc
from utils import load_weights_from_bytes

# =====================================================================
# REGISTRY CALLS
# =====================================================================

class RegistryClient:
    """
    Client class to interact with the central Go Service Registry for node registration and discovery.
    """
    def __init__(self, registry_address: str, my_id: str):
        self.registry_address = registry_address
        self.my_id = my_id

    def register_node(self, my_ip: str, my_port: int) -> bool:
        """
        Registers this node with the central Go Service Registry.
        """
        try:
            with grpc.insecure_channel(self.registry_address) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                payload = federated_pb2.NodeInfo(
                    node_id=self.my_id,
                    ip_address=my_ip,
                    port=my_port
                )
                response = stub.RegisterNode(payload)
                print(f"[RPC] Registration success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"[RPC Error] Failed to register: {e.details()}")
            return False
            
    def get_peer_list(self, node_request_count: int) -> list:
        """
        Fetches the list of active peers from the Service Registry.
        """
        try:
            with grpc.insecure_channel(self.registry_address) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                request = federated_pb2.DiscoverRequest(
                    node_id=self.my_id,
                    request_count=node_request_count
                )
                response = stub.DiscoverNodes(request)
                
                # Convert gRPC repeated field to a standard Python list
                peers = [{"id": p.node_id, "ip": p.ip_address, "port": p.port} for p in response.peers]
                return peers
        except grpc.RpcError as e:
            print(f"[RPC Error] Discovery failed: {e.details()}")
            return []

# =====================================================================
# P2P GOSSIP CALLS
# =====================================================================

class FederatedNodeServicer(federated_pb2_grpc.FederatedNodeServicer):
    """
    gRPC Server class that runs in the background of each node.
    Listens for incoming weights from peers in the decentralized network.
    """
    def __init__(self):
        # Buffer to store incoming weights during the current training round
        self.received_weights = []

    def SendWeights(self, request, context):
        """
        Triggered when another node calls this RPC.
        """
        # Convert incoming bytes to PyTorch state_dict immediately
        state_dict = load_weights_from_bytes(request.model_weights)
        
        # Store for the aggregation phase
        self.received_weights.append({
            "sender_id": request.sender_id,
            "round_number": request.round_number,
            "state_dict": state_dict,
            "samples": request.num_samples,
        })
        
        return federated_pb2.Ack(success=True, message="Weights received successfully")

def send_weights_to_peer(peer_ip: str, peer_port: int, payload: federated_pb2.WeightPayload):
    """
    Client-side gossip function: Sends local weights to a specific peer.
    """
    print(f"[RPC] Sending weights to peer at {peer_ip}:{peer_port}...")
    peer_address = f"{peer_ip}:{peer_port}"
    try:
        with grpc.insecure_channel(peer_address) as channel:
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            # Send the RPC call
            stub.SendWeights(payload)
    except grpc.RpcError as e:
        # To add logging for CloudWatch
        print(f"Failed to send weights to {peer_address}: {e.code()}")