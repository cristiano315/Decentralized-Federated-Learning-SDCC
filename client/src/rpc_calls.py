#File for grpc calls implementation

import random
import threading

import grpc
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

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
        print(f"DEBUG ip:{my_ip}, with type: {type(my_ip)}, port: {my_port}, with type: {type(my_port)}. Registry address: {self.registry_address}, with type: {type(self.registry_address)}")
        try:
            with grpc.insecure_channel(self.registry_address) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                payload = federated_pb2.NodeInfo(
                    node_id=self.my_id,
                    ip_address=str(my_ip),
                    port=int(my_port)
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
                    request_count=int(node_request_count)
                )
                response = stub.DiscoverNodes(request)
                
                # Convert gRPC repeated field to a standard Python list
                # ATTENZIONE: Il proto definisce il campo come 'nodes', non 'peers'
                peers = [{"id": p.node_id, "ip": p.ip_address, "port": p.port} for p in response.nodes]
                return peers
        except grpc.RpcError as e:
            print(f"[RPC Error] Discovery failed: {e.details()}")
            return []
        
    def ping(self, request, context):
        """
        RPC method to respond to ping requests from the registry.
        """
        if request.node_id == self.my_id:
            return federated_pb2.Ack(success=True, message="Node is alive")
        
    def unregister_node(self, my_ip: str, my_port: int) -> bool:
        """
        Unregisters this node with the central Go Service Registry.
        """
        try:
            with grpc.insecure_channel(self.registry_address) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                payload = federated_pb2.NodeInfo(
                    node_id=self.my_id,
                    ip_address=str(my_ip),
                    port=int(my_port)
                )
                response = stub.UnregisterNode(payload)
                print(f"[RPC] Unregistration success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"[RPC Error] Failed to unregister: {e.details()}")
            return False

# =====================================================================
# P2P GOSSIP CALLS
# =====================================================================

class FederatedNodeServicer(federated_pb2_grpc.FederatedNodeServicer):
    """
    gRPC Server class that runs in the background of each node.
    Listens for incoming weights from peers in the decentralized network.
    """
    def __init__(self, my_id):
        self.my_id = my_id
        self.received_weights = {}
        self.seen_messages = set()
        self.lock = threading.Lock()
        self.peers = []
        self.fanout = 2  # Number of peers to forward the message to for every gossip hop

    def SendWeights(self, request, context):
        """
        Triggered when another node calls this RPC.
        """
        msg_id = (request.sender_id, request.round_number)
        
        with self.lock:
            if msg_id in self.seen_messages:
                # Ignore msg to avoid loops
                return federated_pb2.Ack(success=True, message="Message already seen")
            
            self.seen_messages.add(msg_id)

            rnd = request.round_number
            if rnd not in self.received_weights:
                self.received_weights[rnd] = []
            self.received_weights[rnd].append(request)
            print(f"[Gossip] Received weights from {request.sender_id} with size {len(request.model_weights)} bytes for round {rnd}. Total received for this round: {len(self.received_weights[rnd])}")
            
        # Forward the gossip to a subset of peers in a separate thread to avoid blocking
        threading.Thread(target=self._forward_gossip, args=(request,)).start()
        
        return federated_pb2.Ack(success=True, message="Weights successfully received")

    def _forward_gossip(self, request):
        # Exclude self and sender
        available_peers = [
            p for p in self.peers 
            if p['id'] != request.sender_id and p['id'] != self.my_id
        ]
        
        k = min(self.fanout, len(available_peers))
        if k == 0:
            return
        
        selected_peers = random.sample(available_peers, k)
        
        for peer in selected_peers:
            try:
                # Forward the same identical payload (maintains the original sender_id)
                print(f"[Gossip] Forwarding weights to {peer['id']}...")
                send_weights_to_peer(peer['ip'], peer['port'], peer['id'], request)
            except Exception as e:
                print(f"[Gossip] Error forwarding to {peer['id']}: {e}")



def send_weights_to_peer(peer_ip: str, peer_port: int, peer_id: str, payload: federated_pb2.WeightPayload):
    """
    Client-side gossip function: Sends local weights to a specific peer.
    """
    print(f"[RPC] Sending weights with size: {len(payload.model_weights)} bytes to peer with id {peer_id} at {peer_ip}:{peer_port}...")
    peer_address = f"{peer_ip}:{peer_port}"
    try:
        with grpc.insecure_channel(peer_address) as channel:
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            # Send the RPC call
            stub.SendWeights(payload)
            print(f"[RPC] Sent {len(payload.model_weights)} bytes to peer {peer_id}")
    except grpc.RpcError as e:
        # To add logging for CloudWatch
        print(f"Failed to send weights to {peer_address}: {e.code()}")
        # avvisa che e morto, cosi che il registry lo rimuove e ne crea un altro