import os
import random
import time
import threading
import torch
import grpc
from concurrent import futures

from utils import load_weights_from_bytes, get_weights_as_bytes, get_model_hash, get_ecs_container_ip, get_training_index_list
from rpc_calls import FederatedServerServicer, RegistryClient, ping_worker_node
from model import SentimentPyTorch
from aggregator import apply_fedavg

import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

class FederatedCoordinator:
    def __init__(self, servicer, registry_addr, config, device):
        self.servicer = servicer
        self.registry_addr = registry_addr
        self.config = config
        self.device = device
        self.registry_client = None
        self.active_nodes = []
        self.indexes = {}
        self.global_model = None

    def send_start_to_single_node(self, node, current_round):
        """Send the training configuration and indices to a single node."""
        current_index = self.indexes[node['id']]
        current_config = {
            'aggregator_address': self.config['aggregator_address'],
            'total_rounds': self.config['total_rounds'],
            'start_round': current_round,
            'start_index': current_index['start_idx'],
            'weight_wait_timeout_seconds': self.config['timeout_sec'],
            'training_set_percentage': self.config['training_set_percentage'],
            'num_epochs': self.config['num_epochs'],
            'num_samples': current_index['num_samples'],
            'truncated_training_length': current_index['truncated_training_length']
        }
        return self.servicer.send_start_training_signal(node, current_config)

    def broadcast_start_signal(self, nodes, current_round = 0):
        """Send start signal to all nodes in parallel."""
        threads = []
        for node in nodes:
            t = threading.Thread(target=self.send_start_to_single_node, args=(node, current_round))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def send_global_model_to_node(self, node, model_bytes):
        """Send the updated global model to a single worker."""
        node_address = f"{node['ip']}:{node['port']}"
        try:
            channel = grpc.insecure_channel(node_address)
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            
            request = federated_pb2.ModelPayload(model_weights=model_bytes)
            stub.SendUpdatedModel(request, timeout=10)
            channel.close()
            return True
        except Exception as e:
            print(f"Error: Failed to send global model to {node_address}: {e}")
            return False

    def broadcast_global_model(self, node_addresses, model_bytes, round_num):
        """Send the updated global model to all nodes in parallel."""
        threads = []
        for node in node_addresses:
            t = threading.Thread(
                target=self.send_global_model_to_node, 
                args=(node, model_bytes)
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def run_coordination_loop(self):
        print("\n" + "="*40)
        print("    STARTING FEDERATED LEARNING COORDINATOR   ")
        print("="*40)

        # Nodes discovery from registry
        active_nodes = []
        retries = 0
        num_peers_required = self.config['training_nodes']
        max_retries = self.config['max_discovery_retries']
        bucket_name = "sdcc-dataset-264452429750-us-east-1-an"
        s3_key = "all_data_niid_05_keep_3_train_9.json"
        while len(active_nodes) < num_peers_required:
            if retries >= max_retries:
                print("Error: Discovery timeout.")
                break
            print(f"Discovery: Fetching peers ({retries+1}/{max_retries})...")
            active_nodes = self.registry_client.get_peer_list(node_request_count=num_peers_required)
            if len(active_nodes) < num_peers_required:
                time.sleep(5)
                retries += 1
        
        if len(active_nodes) < num_peers_required:
            print("Error: Unable to start training due to insufficient peers.")
            return

        self.active_nodes = active_nodes
        print(f"Nodes recruited for training: {self.active_nodes}")

        # Send Start Training Signal to models and initialize the global model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        self.global_model = SentimentPyTorch(num_class=2).to(device)
        print("Sending StartTraining signal to all nodes...")
        self.indexes = get_training_index_list(self.active_nodes, bucket_name, s3_key, self.config['training_set_percentage'], max_samples_per_client=1000)
        self.servicer.coordinator = self
        self.broadcast_start_signal(self.active_nodes, current_round = 0)

        total_rounds = self.config['total_rounds']
        weight_timeout = self.config['timeout_sec']
        min_clients = self.config.get('min_clients', 1)
        expected_node_count = len(self.active_nodes)

        # Training loop
        for round_num in range(total_rounds):
            print(f"\n-------------------- ROUND {round_num + 1}/{total_rounds} --------------------")
            
            with self.servicer.lock:
                self.servicer.current_round = round_num
                self.servicer.received_weights.clear()

            # Wait for weights from all nodes or until timeout
            start_wait_time = time.time()
            while True:
                with self.servicer.lock:
                    received_count = len(self.servicer.received_weights.get(round_num, []))
                
                # IF all nodes have sent their weights, break the loop
                if received_count >= len(self.active_nodes):
                    print(f"Weights received from all nodes ({received_count}/{len(self.active_nodes)}).")
                    break
                
                # Handling Timeout
                if time.time() - start_wait_time > weight_timeout:
                    print(f"Timeout reached for Round {round_num + 1}. Weights received from {received_count}/{len(self.active_nodes)} nodes.")
                    break
                
                time.sleep(1)

            # Fault tolerance
            with self.servicer.lock:
                round_payloads = self.servicer.received_weights.get(round_num, [])
                received_ids = {req.sender_id for req in round_payloads}

            missing_nodes = [node for node in self.active_nodes if node['id'] not in received_ids]
            coord_ip, coord_port = self.config['aggregator_address'].split(':')

            dead_count = 0

            for dead_candidate in missing_nodes:
                print(f"Node {dead_candidate['id']} did not respond. Executing direct ping...")
                is_alive = ping_worker_node(dead_candidate)
                if not is_alive:
                    print(f"Node {dead_candidate['id']} confirmed DEAD. Reporting to Registry...")
                    # Remove from active nodes
                    self.active_nodes = [n for n in self.active_nodes if n['id'] != dead_candidate['id']]
                    dead_count += 1

                    # Signal to the registry to start a replacement container        
                    self.registry_client.signal_unresponsive_node(
                        peer_ip=dead_candidate['ip'],
                        peer_port=dead_candidate['port'],
                        peer_id=dead_candidate['id'],
                        requiredNodes=self.config['training_nodes'],
                        totalRounds=total_rounds,
                        startRound=round_num,
                        maxDiscoveryRetries=5,
                        weightWaitTimeoutSeconds=weight_timeout,
                        trainingSetPercentage=self.config['training_set_percentage'],
                        numEpochs=self.config['num_epochs'],
                        coordinator_ip=coord_ip,
                        coordinator_port=int(coord_port)
                    )
                else:
                    print(f"Node {dead_candidate['id']} is still alive (likely local latency).")

            # Deserialization and FedAvg
            with self.servicer.lock:
                round_payloads = self.servicer.received_weights.get(round_num, [])

            print(f"Executing FedAvg on {len(round_payloads)} contributions...")

            if not round_payloads:
                print(f"Warning: No contributions received for Round {round_num + 1}. Skipping FedAvg and keeping current global model.")
            else:
                round_payloads.sort(key=lambda x: x.sender_id)
                deserialized_models = []
                
                for request in round_payloads:
                    state_dict = load_weights_from_bytes(request.model_weights)
                    deserialized_models.append({
                        'sender_id': request.sender_id,
                        'weights': state_dict,
                        'num_samples': request.num_samples
                    })
            
                # HASH BEFORE FEDAVG
                fc_hash_before = get_model_hash(self.global_model, only_trainable=True)
                print(f"VERIFICATION: BEFORE FEDAVG round {round_num + 1} Model Classifier SHA-256: {fc_hash_before}")

                self.global_model = apply_fedavg(self.global_model, deserialized_models)
                
                # Clear servicer buffer
                with self.servicer.lock:
                    self.servicer.received_weights.pop(round_num, None)

                # HASH AFTER FEDAVG
                fc_hash_after = get_model_hash(self.global_model, only_trainable=True)
                print(f"VERIFICATION: AFTER FEDAVG round {round_num + 1} Model Classifier SHA-256: {fc_hash_after}")

            # Clear servicer buffer
            with self.servicer.lock:
                self.servicer.received_weights.pop(round_num, None)

            # Load local model for verification
            global_bytes = get_weights_as_bytes(self.global_model)
            
            with self.servicer.lock:
                self.servicer.latest_global_bytes = global_bytes
                self.servicer.received_weights.pop(round_num, None)

            print(f"Round {round_num + 1} completed.")

            # Wait for nodes to respawn
            if len(self.active_nodes) < expected_node_count:
                print(f"\nWaiting for respawned nodes ({len(self.active_nodes)}/{expected_node_count} ready)...")
                while len(self.active_nodes) < expected_node_count:
                    self.servicer.node_respawned_event.wait(timeout=120)
                    self.servicer.node_respawned_event.clear()
                    print(f"Actual nodes state: {len(self.active_nodes)}/{expected_node_count}")

                print(f"All the {expected_node_count} nodes are ready and configured.\n")

            # Broadcast new model to workers
            print(f"Sending updated global model to all nodes for Round {round_num + 1}...")
            self.broadcast_global_model(self.active_nodes, global_bytes, round_num)

        print("\n" + "="*40)
        print("  FEDERATED TRAINING COMPLETED SUCCESSFULLY  ")
        print("="*40)


def start_coordinator_server(port: int):
    """Initialize the background gRPC server for the coordinator."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedServerServicer(my_id='COORDINATOR')
    federated_pb2_grpc.add_FederatedServerServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server listening on port: {port}...")
    return server, servicer


def main():
    PORT = int(os.getenv("PORT", 50053))
    REGISTRY_ADDR = os.getenv("REGISTRY_ADDRESS", "centralized-registry-nlb-d298b040b4ebce81.elb.us-east-1.amazonaws.com:8080")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    my_ip = get_ecs_container_ip()
    coordinator_grpc_address = f"{my_ip}:{PORT}"
    training_nodes = int(os.getenv("TRAINING_NODES", 5))

    # Configuration for the federated training process to send to the nodes
    training_config = {
        'training_nodes': training_nodes,
        'total_rounds': int(os.getenv("TOTAL_ROUNDS", 8)),
        'start_round': int(os.getenv("START_ROUND", 0)),
        'num_peers_required': int(os.getenv("NUM_PEERS_REQUIRED", training_nodes - 1)),
        'timeout_sec': int(os.getenv("WEIGHT_WAIT_TIMEOUT_SECONDS", 120)),
        'training_set_percentage': float(os.getenv("TRAINING_SET_PERCENTAGE", 0.7)),
        'aggregator_address': coordinator_grpc_address,
        'num_epochs': int(os.getenv("NUM_EPOCHS", 2)),
        'max_discovery_retries': int(os.getenv("MAX_DISCOVERY_RETRIES", 5))
    }

    # Start coordinator gRPC server
    server, servicer = start_coordinator_server(PORT)

    # Start coordinator logic
    coordinator = FederatedCoordinator(
        servicer=servicer,
        registry_addr=REGISTRY_ADDR,
        config=training_config,
        device=device
    )
    registry_client = RegistryClient(REGISTRY_ADDR, f"{random.randint(1000, 9999)}")
    coordinator.registry_client = registry_client

    try:
        coordinator.run_coordination_loop()
    except KeyboardInterrupt:
        print("\nShutdown (KeyboardInterrupt).")
    finally:
        server.stop(grace=5)
        print("Shutdown. gRPC server stopped.")


if __name__ == "__main__":
    main()