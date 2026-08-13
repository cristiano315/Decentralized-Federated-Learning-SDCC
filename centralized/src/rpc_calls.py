#File for grpc calls implementation

import random
import threading

import grpc
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

# =====================================================================
# CLIENT CALLS
# =====================================================================

class FederatedNodeServicer(federated_pb2_grpc.FederatedNodeServicer):
    """
    gRPC Server class that runs in the background of each node.
    Listens for incoming messages from the central coordinator.
    """
    def __init__(self, my_id):
        self.my_id = my_id #set up at start of main.py
        self.received_model = None
        self.lock = threading.Lock()
        self.num_samples = 0 # set up after defining servicer in main.py
        self.round_num = 0 # set up every after every local_train in main.py
        self.start_training_event = threading.Event()
        self.pending_training_config = None

    def StartTrainingSession(self, request, context):
        """
        RPC chiamata dal coordinator per inviare le ENV e la lista dei PEERS
        """
        print(f"[RPC] Ricevuta richiesta StartTraining per sessione avviata dal coordinatore")
        
        self.pending_training_config = {
            'aggregator_address': request.aggregator_address,
            'total_rounds': request.total_rounds,
            'start_round': request.start_round,
            'start_index': request.start_index,
            'training_set_percentage': request.training_set_percentage,
            'weight_wait_timeout': request.weight_wait_timeout_seconds,
            'num_epochs': request.num_epochs,
            'num_samples': request.num_samples,
            'truncated_training_length': request.truncated_training_length,
        }
        
        # Sblocchiamo il thread principale che attende in stato IDLE
        self.start_training_event.set()
        return federated_pb2.Ack(success=True, message="Training queued")

    def SendUpdatedModel(self, request, context):
        """
        Triggered when the coordinator sends the updated model after aggregation.
        """
        
        with self.lock:
            self.received_model = request.model_weights
            print(f"[RPC] Ricevuto modello aggiornato dal coordinatore con dimensione {len(request.model_weights)} bytes per il round {self.round_num}")
        
        return federated_pb2.Ack(success=True, message="Model successfully received")

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

    def ping(self, request, context):
        """
        RPC method to respond to ping requests from the registry.
        """
        if request.node_id == self.my_id:
            return federated_pb2.Ack(success=True, message="Node is alive")



def send_weights_to_coordinator(coordinator_address, payload: federated_pb2.WeightPayload):
    """
    Client-side gossip function: Sends local weights to a specific peer.
    """
    print(f"[RPC] Sending weights with size: {len(payload.model_weights)} bytes to coordinator...")
    try:
        with grpc.insecure_channel(coordinator_address) as channel:
            stub = federated_pb2_grpc.FederatedServerStub(channel)
            # Send the RPC call
            stub.SendLocalModel(payload)
            print(f"[RPC] Sent {len(payload.model_weights)} bytes to coordinator")
            return 0  # Success
    except grpc.RpcError as e:
        # To add logging for CloudWatch
        print(f"Failed to send weights to {coordinator_address}: {e.code()}")
        # avvisa che e morto, cosi che il registry lo rimuove e ne crea un altro
        return 1 # Failure


# =====================================================================
# COORDINATOR CALLS
# =====================================================================

class FederatedServerServicer(federated_pb2_grpc.FederatedServerServicer):
    """
    gRPC Server class that runs in the background of the aggregator.
    Listens for incoming messages from the worker nodes.
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

    def send_start_training_signal(self, peer, config):
        try:
            channel = grpc.insecure_channel(f"{peer['ip']}:{peer['port']}")
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            
            req = federated_pb2.StartTrainingCentralizedRequest(
                aggregator_address=config['aggregator_address'],
                total_rounds=config['total_rounds'],
                start_round=config['start_round'],
                start_index=config['start_index'],
                weight_wait_timeout_seconds=config['weight_wait_timeout_seconds'],
                training_set_percentage=config['training_set_percentage'],
                num_epochs = config['num_epochs'],
                num_samples = config['num_samples'],
                truncated_training_length = config['truncated_training_length']
            )
            
            response = stub.StartTrainingSession(req, timeout=10)
            return response.success
        except Exception as e:
            print(f"[Error] Impossibile inviare segnale StartTraining a {peer['id']}: {e}")

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

    def SendLocalModel(self, request, context):
        """
        Triggered when a node sends its local model to the aggregator.
        """
        with self.lock:
            rnd = request.round_number
            if rnd not in self.received_weights:
                self.received_weights[rnd] = []
            self.received_weights[rnd].append(request)
            print(f"[RPC] Ricevuto modello locale da {request.sender_id} con dimensione {len(request.model_weights)} bytes per il round {rnd}. Totale ricevuti per questo round: {len(self.received_weights[rnd])}")
        
        return federated_pb2.Ack(success=True, message="Local model successfully received")

    def GetGlobalModel(self, request, context):
        """
        Triggered when a node requests the global model from the aggregator.
        """
        with self.lock:
            rnd = request.round_number
            if self.latest_local_weights is None or self.round_num < rnd:
                print(f"[RPC] Nessun modello globale disponibile per il round {rnd}. Round corrente: {self.round_num}")
                return federated_pb2.WeightPayload(
                    sender_id=self.my_id,
                    round_number=rnd,
                    model_weights=b'',
                    num_samples=0
                )
            
            print(f"[RPC] Inviato modello globale per il round {rnd} a {request.requester_id} con dimensione {len(self.latest_local_weights)} bytes.")
            return federated_pb2.WeightPayload(
                sender_id=self.my_id,
                round_number=rnd,
                model_weights=self.latest_local_weights,
                num_samples=self.num_samples
            )

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
        print(f"DEBUG ip:{my_ip}, with type: {type(my_ip)}, port: {my_port}, with type: {type(my_port)}. Registry address: {self.registry_address}, with type: {type(self.registry_address)}")
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
                # ATTENZIONE: Il proto definisce il campo come 'nodes', non 'peers'
                peers = [{"id": p.node_id, "ip": p.ip_address, "port": p.port} for p in response.nodes]
                return peers
        except grpc.RpcError as e:
            print(f"[RPC Error] Discovery failed: {e.details()}")
            return []

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
                print(f"[RPC] Unregistration success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"[RPC Error] Failed to unregister: {e.details()}")
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
                print(f"[RPC] Signal unresponsive node success: {response.message}")
                return response.success
        except grpc.RpcError as e:
            print(f"[RPC Error] Failed to signal unresponsive node: {e.details()}")
            return False

    def update_status(self, new_status: str) -> bool:
        """
        Invia una richiesta RPC al Go Registry per aggiornare lo stato del nodo ("idle" o "working").
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
                    print(f"[RegistryClient] Stato aggiornato con successo a '{new_status}'.")
                    return True
                else:
                    print(f"[RegistryClient] Errore aggiornamento stato: {response.message}")
                    return False

        except grpc.RpcError as e:
            print(f"[RegistryClient Error] Impossibile aggiornare lo stato su Registry: {e.details()}")
            return False
        except Exception as e:
            print(f"[RegistryClient Error] Eccezione generica durante update_status: {e}")
            return False