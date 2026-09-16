#File for grpc calls implementation

import threading
import copy

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
        self.latest_local_weights = None # set up after every local_train in main.py
        self.num_samples = 0 # set up after defining servicer in main.py
        self.round_num = 0 # set up every after every local_train in main.py
        self.start_training_event = threading.Event()
        self.pending_training_config = None
        self.is_starter = False
        self.active_config = None

    def StartTraining(self, request, context):
        """
        Called by starter node to send ENV and PEERS list to a support node.
        """
        print(f"Received StartTraining request from {request.starter_id}")

        peers_list = [
                {
                    'id': p.node_id,
                    'ip': p.ip_address,
                    'port': p.port
                } 
                for p in request.peers if p.node_id != self.my_id  # Exclude self
        ]
        
        self.pending_training_config = {
            'training_nodes': request.training_nodes,
            'total_rounds': request.total_rounds,
            'start_round': request.start_round,
            'num_peers_required': request.num_peers_required,
            'max_discovery_retries': request.max_discovery_retries,
            'weight_wait_timeout': request.weight_wait_timeout_seconds,
            'training_set_percentage': request.training_set_percentage,
            'num_epochs': request.num_epochs,
            'peers': peers_list
        }
        
        # Unlock main thread to start training
        self.start_training_event.set()
        return federated_pb2.Ack(success=True, message="Training queued")

    def remove_peer_by_id(self, peer_id: str):
        """Remove unresponsive peer from the local list (thread-safe)."""
        with self.lock:
            initial_count = len(self.peers)
            self.peers = [p for p in self.peers if p['id'] != peer_id]
            if len(self.peers) < initial_count:
                print(f"[Servicer] Peer {peer_id} removed successfully from the local list. Remaining peers: {len(self.peers)}")

    def add_or_update_peer(self, peer_info: dict):
        """Add or update a peer in the local list."""
        with self.lock:
            self.peers = [p for p in self.peers if p['id'] != peer_info['id']]
            self.peers.append(peer_info)
            print(f"New peer {peer_info['id']} ({peer_info['ip']}:{peer_info['port']}) added.")

    def SendWeights(self, request, context):
        """
        Triggered when another node calls this RPC.
        """
        msg_id = (request.sender_id, request.round_number)
        rnd = request.round_number
        MAX_STALENESS = 2
        
        with self.lock:
            if msg_id in self.seen_messages:
                # Ignore msg to avoid loops
                return federated_pb2.Ack(success=True, message="Message already seen")

            if rnd < (self.round_num - MAX_STALENESS):
                print(f"Received message from {request.sender_id} for round {rnd}, which is too old (Current round: {self.current_round}). Dropped the message.")
                return federated_pb2.Ack(success=False, message="Message dropped: too old")
            
            self.seen_messages.add(msg_id)

            if rnd < self.round_num:
                # Insert in current round to use immediately
                target_round = self.round_num
                status_str = f"PAST ROUND {rnd} -> REMAPPED TO CURRENT {self.round_num}"
            else:
                # Save in actual round or future round buffer
                target_round = rnd
                status_str = "CURRENT ROUND" if rnd == self.round_num else "FUTURE ROUND (Buffered)"

            if target_round not in self.received_weights:
                self.received_weights[target_round] = []
            self.received_weights[target_round].append(request)
            print(
            f"RECEIVED - {status_str}] From {request.sender_id} | "
            f"Original Msg Round: {rnd} | Node Round: {self.round_num} | "
            f"Total in buffer for round {target_round}: {len(self.received_weights[target_round])}"
            )
        
        return federated_pb2.Ack(success=True, message="Weights successfully received")

    def RequestWeights(self, request, context):
        """
        Triggered when another node requests weights from this node.
        """
        rnd = request.round_number
        if self.latest_local_weights is None or self.round_num < rnd:
            print(f"No weights available for round {rnd}. Current round: {self.round_num}")
            return federated_pb2.WeightPayload(
                sender_id=self.my_id,
                round_number=rnd,
                model_weights=b'',
                num_samples=0
            )
        
        print(f"Sending weights for round {rnd} to {request.requester_id} with size {len(self.latest_local_weights)} bytes.")
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
        print(f"Requesting weights for round {round_number_requested} from peer with id {peer_id} at {peer_ip}:{peer_port}...")
        peer_address = f"{peer_ip}:{peer_port}"
        try:
            with grpc.insecure_channel(peer_address) as channel:
                stub = federated_pb2_grpc.FederatedNodeStub(channel)
                request_payload = federated_pb2.WeightRequest(
                    requester_id=my_id,
                    round_number=round_number_requested
                )
                response = stub.RequestWeights(request_payload)
                if response.model_weights == b'':
                    print(f"Peer {peer_id} has no weights for round {round_number_requested}.")
                    return None

                msg_id = (response.sender_id, response.round_number)
                with self.lock:
                    if msg_id in self.seen_messages:
                        print(f"Already received weights from {peer_id} for round {round_number_requested}. Ignoring.")
                        return None
                    
                    self.seen_messages.add(msg_id)

                    rnd = response.round_number
                    if rnd not in self.received_weights:
                        self.received_weights[rnd] = []
                    self.received_weights[rnd].append(response)
                    print(f"Received weights from {response.sender_id} with size {len(response.model_weights)} bytes for round {rnd}. Total received for this round: {len(self.received_weights[rnd])}")
                        
        except grpc.RpcError as e:
            print(f"Failed to request weights from {peer_address}: {e.code()}")
            return None

    def Ping(self, request, context):
        """
        RPC method to respond to ping requests from the registry.
        """
        if request.node_id == self.my_id:
            return federated_pb2.Ack(success=True, message="Node is alive")

    def NotifyUnresponsiveNode(self, request, context):
        """
        RPC method to handle notifications about unresponsive nodes from the registry.
        """

        new_peer = {
            "id": request.node_id,
            "ip": request.ip_address,
            "port": request.port
        }

        self.add_or_update_peer(new_peer)
        print(f"Peer {request.node_id} marked as unresponsive. Updated local peer list.")

        # If STARTER node, send config
        if self.is_starter and self.active_config is not None:
            print(f"Sending configuration to respawned node {request.node_id}...")

            with self.lock:
                peers_snapshot = list(self.peers)

            # Send config in a separate thread to avoid blocking the gRPC server
            def push_config_to_respawned():
                config_to_send = copy.deepcopy(self.active_config)
                # Exclude receiver
                config_to_send['peers'] = [p for p in peers_snapshot if p['id'] != request.node_id]

                # Send start training signal
                self.send_start_training_signal(new_peer, config_to_send)

            threading.Thread(target=push_config_to_respawned, daemon=True).start()


        return federated_pb2.Ack(success=True, message=f"Peer {request.node_id} removed from local topology.")

    def send_start_training_signal(self, peer, config):
        try:
            channel = grpc.insecure_channel(f"{peer['ip']}:{peer['port']}")
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            
            peer_proto_list = [
                federated_pb2.NodeInfo(
                    node_id=p['id'], 
                    ip_address=p['ip'], 
                    port=p['port'],
                    status="idle"
                )
                for p in config['peers']
            ]
            
            req = federated_pb2.StartTrainingRequest(
                starter_id=self.my_id,
                training_nodes=config['training_nodes'],
                total_rounds=config['total_rounds'],
                start_round=config['start_round'],
                num_peers_required=config['num_peers_required'],
                max_discovery_retries=config['max_discovery_retries'],
                weight_wait_timeout_seconds=config['weight_wait_timeout'],
                training_set_percentage=config['training_set_percentage'],
                num_epochs=config['num_epochs'],
                peers=peer_proto_list
            )
            
            response = stub.StartTraining(req, timeout=10)
            return response.success
        except Exception as e:
            print(f"Error: Unable to send StartTraining signal to {peer['id']}: {e}")


def send_weights_to_peer(peer_ip: str, peer_port: int, peer_id: str, payload: federated_pb2.WeightPayload):
    """
    Client-side gossip function: Sends local weights to a specific peer.
    """
    print(f"Sending weights with size: {len(payload.model_weights)} bytes to peer with id {peer_id} at {peer_ip}:{peer_port}...")
    peer_address = f"{peer_ip}:{peer_port}"
    try:
        with grpc.insecure_channel(peer_address) as channel:
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            # Send the RPC call
            stub.SendWeights(payload)
            print(f"Sent {len(payload.model_weights)} bytes to peer {peer_id}")
            return 0  # Success
    except grpc.RpcError as e:
        print(f"Failed to send weights to {peer_address}: {e.code()}")
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

    def register_node(self, my_ip: str, my_port: int, status: str) -> bool:
        """
        Registers this node with the central Go Service Registry.
        """
        #print(f"DEBUG ip:{my_ip}, with type: {type(my_ip)}, port: {my_port}, with type: {type(my_port)}. Registry address: {self.registry_address}, with type: {type(self.registry_address)}") UNCOMMENT FOR DEBUGGING
        target = self.registry_address
        if not target.startswith("dns:///") and not target.startswith("ipv4:"):
            target = f"dns:///{target}"
        try:
            with grpc.insecure_channel(target) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                payload = federated_pb2.NodeInfo(
                    node_id=self.my_id,
                    ip_address=str(my_ip),
                    port=int(my_port),
                    status=status
                )
                response = stub.RegisterNode(payload)
                print(f"Registration success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"Error: Failed to register: {e.details()}")
            return False

    def register_respawned_node(self, my_ip: str, my_port: int) -> bool:
        """
        Registers this node with the central Go Service Registry after a crash.
        """
        #rint(f"DEBUG ip:{my_ip}, with type: {type(my_ip)}, port: {my_port}, with type: {type(my_port)}. Registry address: {self.registry_address}, with type: {type(self.registry_address)}") UNCOMMENT FOR DEBUGGING
        target = self.registry_address
        if not target.startswith("dns:///") and not target.startswith("ipv4:"):
            target = f"dns:///{target}"
        try:
            with grpc.insecure_channel(target) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                payload = federated_pb2.NodeInfo(
                    node_id=self.my_id,
                    ip_address=str(my_ip),
                    port=int(my_port),
                    status="working"
                )
                response = stub.RegisterRespawnedNode(payload)
                print(f"Respawned node registration success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"Error: Failed to register respawned node: {e.details()}")
            return False
            
    def get_peer_list(self, node_request_count: int) -> list:
        """
        Fetches the list of active peers from the Service Registry.
        """
        try:
            target = self.registry_address
            if not target.startswith("dns:///") and not target.startswith("ipv4:"):
                target = f"dns:///{target}"
            with grpc.insecure_channel(target) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                request = federated_pb2.DiscoverRequest(
                    node_id=self.my_id,
                    request_count=int(node_request_count)
                )
                response = stub.DiscoverNodes(request)
                
                # Convert gRPC repeated field to a standard Python list
                peers = [{"id": p.node_id, "ip": p.ip_address, "port": p.port} for p in response.nodes]
                return peers
        except grpc.RpcError as e:
            print(f"Error: Discovery failed: {e.details()}")
            return []
        
    def unregister_node(self, my_ip: str, my_port: int) -> bool:
        """
        Unregisters this node with the central Go Service Registry.
        """
        try:
            target = self.registry_address
            if not target.startswith("dns:///") and not target.startswith("ipv4:"):
                target = f"dns:///{target}"
            with grpc.insecure_channel(target) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                payload = federated_pb2.NodeInfo(
                    node_id=self.my_id,
                    ip_address=str(my_ip),
                    port=int(my_port),
                    status="idle"
                )
                response = stub.UnregisterNode(payload)
                print(f"Unregistration success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"Error: Failed to unregister: {e.details()}")
            return False

    def signal_unresponsive_node(self, peer_ip: str, peer_port: int, peer_id: str, requiredNodes: int, totalRounds: int, startRound: int, maxDiscoveryRetries: int, weightWaitTimeoutSeconds: int, trainingSetPercentage: float, numEpochs: int):
        """
        Signals to the registry that a peer node is unresponsive.
        """
        try:
            target = self.registry_address
            if not target.startswith("dns:///") and not target.startswith("ipv4:"):
                target = f"dns:///{target}"
            with grpc.insecure_channel(target) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                payload = federated_pb2.FullNodeInfo(
                    node_id=peer_id,
                    ip_address=str(peer_ip),
                    port=int(peer_port),
                    required_nodes=requiredNodes,
                    total_rounds=totalRounds,
                    current_round=startRound,
                    max_discovery_retries=maxDiscoveryRetries,
                    weight_wait_timeout_seconds=weightWaitTimeoutSeconds,
                    training_set_percentage=trainingSetPercentage,
                    num_epochs=numEpochs
                )
                response = stub.SignalUnresponsiveNode(payload)
                print(f"Signal unresponsive node success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"Error: Failed to signal unresponsive node: {e.details()}")
            return False

    def update_status(self, new_status: str) -> bool:
        """
        Sends an RPC request to the Go Registry to update the node's status ("idle" or "working").
        """
        try:
            req = federated_pb2.ChangeStatusRequest(
                node_id=self.my_id,
                new_status=new_status
            )
            
            target = self.registry_address
            if not target.startswith("dns:///") and not target.startswith("ipv4:"):
                target = f"dns:///{target}"
            with grpc.insecure_channel(target) as channel:
                stub = federated_pb2_grpc.RegistryServiceStub(channel)
                response = stub.ChangeNodeStatus(req, timeout=5)
                
                if response.success:
                    print(f"Status updated successfully to '{new_status}'.")
                    return True
                else:
                    print(f"Error updating status: {response.message}")
                    return False

        except grpc.RpcError as e:
            print(f"Error: Unable to update status on Registry: {e.details()}")
            return False
        except Exception as e:
            print(f"Error: Generic exception during update_status: {e}")
            return False