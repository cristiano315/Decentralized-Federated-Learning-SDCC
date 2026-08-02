#File for grpc calls implementation

import random
import threading

import grpc
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

# =====================================================================
# P2P GOSSIP CALLS
# =====================================================================

class FederatedNodeServicer(federated_pb2_grpc.FederatedNodeServicer):
    """
    gRPC Server class that runs in the background of each node.
    Listens for incoming weights from peers in the decentralized network.
    """
    def __init__(self, my_id):
        self.my_id = my_id #set up at start of main.py
        self.received_weights = {}
        self.seen_messages = set()
        self.lock = threading.Lock()
        self.peers = []
        self.fanout = 2  # Number of peers to forward the message to for every gossip hop
        self.latest_local_weights = None # set up after every local_train in main.py
        self.num_samples = 0 # set up after defining servicer in main.py
        self.round_num = 0 # set up every after every local_train in main.py

    def remove_peer_by_id(self, peer_id: str):
        """Rimuove un nodo non responsivo dalla lista dei peer locali in modo thread-safe."""
        with self.lock:
            initial_count = len(self.peers)
            self.peers = [p for p in self.peers if p['id'] != peer_id]
            if len(self.peers) < initial_count:
                print(f"[Servicer] Peer {peer_id} rimosso con successo dalla lista locale. Peer rimanenti: {len(self.peers)}")

    def add_or_update_peer(self, peer_info: dict):
        """Aggiunge o aggiorna un peer nella lista locale."""
        with self.lock:
            # Aggiorna se esiste già, altrimenti aggiungi
            self.peers = [p for p in self.peers if p['id'] != peer_info['id']]
            self.peers.append(peer_info)
            print(f"[Servicer] Nuovo peer {peer_info['id']} ({peer_info['ip']}:{peer_info['port']}) aggiunto.")

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
                status = send_weights_to_peer(peer['ip'], peer['port'], peer['id'], request)
                # Se il nodo non risponde, lo rimuoviamo semplicemente dalla topologia locale
                if status == 1:
                    print(f"[Warning] Peer {peer['id']} unreacheable during gossip forward. Removing locally.")
                    self.remove_peer_by_id(peer['id'])
            except Exception as e:
                print(f"[Gossip] Error forwarding to {peer['id']}: {e}")

    def RequestWeights(self, request, context):
        """
        Triggered when another node requests weights from this node.
        """
        rnd = request.round_number
        if self.latest_local_weights is None or self.round_num < rnd:
            print(f"[Gossip] No weights available for round {rnd}. Current round: {self.round_num}")
            return federated_pb2.WeightPayload(
                sender_id=self.my_id,
                round_number=rnd,
                model_weights=b'',
                num_samples=0
            )
        
        print(f"[Gossip] Sending weights for round {rnd} to {request.requester_id} with size {len(self.latest_local_weights)} bytes.")
        return federated_pb2.WeightPayload(
            sender_id=self.my_id,
            round_number=rnd,
            model_weights=self.latest_local_weights,
            num_samples=self.num_samples
        )

    def get_weights_from_peer(self, peer_ip: str, peer_port: int, peer_id: str, my_id: str, round_number_requested: int):
        """
        Client-side function to request weights from a specific peer.
        """
        print(f"[RPC] Requesting weights for round {round_number_requested} from peer with id {peer_id} at {peer_ip}:{peer_port}...")
        peer_address = f"{peer_ip}:{peer_port}"
        try:
            with grpc.insecure_channel(peer_address) as channel:
                stub = federated_pb2_grpc.FederatedNodeStub(channel)
                # Create a WeightRequest payload to request weights
                request_payload = federated_pb2.WeightRequest(
                    requester_id=my_id,
                    round_number=round_number_requested
                )
                # Send the RPC call
                response = stub.RequestWeights(request_payload)
                if response.model_weights == b'':
                    print(f"[Gossip] Peer {peer_id} has no weights for round {round_number_requested}.")
                    return None

                msg_id = (response.sender_id, response.round_number)
                with self.lock:
                    if msg_id in self.seen_messages:
                        print(f"[Gossip] Already received weights from {peer_id} for round {round_number_requested}. Ignoring.")
                        return None
                    
                    self.seen_messages.add(msg_id)

                    rnd = response.round_number
                    if rnd not in self.received_weights:
                        self.received_weights[rnd] = []
                    self.received_weights[rnd].append(response)
                    print(f"[Gossip] Received weights from {response.sender_id} with size {len(response.model_weights)} bytes for round {rnd}. Total received for this round: {len(self.received_weights[rnd])}")
                        
        except grpc.RpcError as e:
            print(f"Failed to request weights from {peer_address}: {e.code()}")
            return None


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
            return 0  # Success
    except grpc.RpcError as e:
        # To add logging for CloudWatch
        print(f"Failed to send weights to {peer_address}: {e.code()}")
        # avvisa che e morto, cosi che il registry lo rimuove e ne crea un altro
        return 1 # Failure

# =====================================================================
# REGISTRY CALLS
# =====================================================================

class RegistryClient:
    """
    Client class to interact with the central Go Service Registry for node registration and discovery.
    """
    def __init__(self, registry_address: str, my_id: str, servicer: FederatedNodeServicer = None):
        self.registry_address = registry_address
        self.my_id = my_id
        self.servicer = servicer

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

    def register_respawned_node(self, my_ip: str, my_port: int) -> bool:
        """
        Registers this node with the central Go Service Registry after a crash.
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
                response = stub.RegisterRespawnedNode(payload)
                print(f"[RPC] Respawned registration success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"[RPC Error] Failed to register respawned node: {e.details()}")
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

    def NotifyUnresponsiveNode(self, request, context):
        """
        RPC method to handle notifications about unresponsive nodes from the registry.
        """
        # Se il servicer è collegato, aggiorna la lista dei peer
        if self.servicer is not None:
            self.servicer.add_or_update_peer({"id": request.node_id, "ip": request.ip_address, "port": request.port})
            print(f"[Registry Notification] Peer {request.node_id} marked as unresponsive. Updated local peer list.")
        else:
            print("[Warning] Servicer non collegato a RegistryClient, impossibile aggiornare la lista dei peer.")

        return federated_pb2.Ack(success=True, message=f"Peer {request.node_id} rimosso dalla topologia locale.")
        
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

    def signal_unresponsive_node(self, peer_ip: str, peer_port: int, peer_id: str, requiredNodes: int, totalRounds: int, startRound: int, maxDiscoveryRetries: int, weightWaitTimeoutSeconds: int):
        """
        Signals to the registry that a peer node is unresponsive.
        """
        try:
            with grpc.insecure_channel(self.registry_address) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                payload = federated_pb2.FullNodeInfo(
                    node_id=peer_id,
                    ip_address=str(peer_ip),
                    port=int(peer_port),
                    required_nodes=requiredNodes,
                    total_rounds=totalRounds,
                    current_round=startRound,
                    max_discovery_retries=maxDiscoveryRetries,
                    weight_wait_timeout_seconds=weightWaitTimeoutSeconds
                )
                response = stub.SignalUnresponsiveNode(payload)
                print(f"[RPC] Signal unresponsive node success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"[RPC Error] Failed to signal unresponsive node: {e.details()}")
            return False