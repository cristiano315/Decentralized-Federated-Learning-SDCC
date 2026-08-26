import os
import time
import json
import random
import threading
import torch
import grpc
from concurrent import futures

from utils import load_weights_from_bytes, get_weights_as_bytes, get_model_hash, get_ecs_container_ip, get_training_index_list
from rpc_calls import FederatedServerServicer, RegistryClient
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

    def broadcast_start_signal(self, nodes, indexes):
        """Invia in parallelo il comando di avvio a tutti i worker."""
        threads = []
        for node in nodes:
            current_index = indexes[node['id']]
            current_config = {
                'aggregator_address': self.config['aggregator_address'],
                'total_rounds': self.config['total_rounds'],
                'start_round': self.config['start_round'],
                'start_index': current_index['start_idx'],
                'weight_wait_timeout_seconds': self.config['timeout_sec'],
                'training_set_percentage': self.config['training_set_percentage'],
                'num_epochs': self.config['num_epochs'],
                'num_samples' : current_index['num_samples'],
                'truncated_training_length' : current_index['truncated_training_length']
            }
            t = threading.Thread(
                target=self.servicer.send_start_training_signal, 
                args=(node, current_config)
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def send_global_model_to_node(self, node, model_bytes):
        """Invia il nuovo modello globale aggregato ad un singolo worker."""
        node_address = f"{node['ip']}:{node['port']}"
        try:
            channel = grpc.insecure_channel(node_address)
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            
            request = federated_pb2.ModelPayload(
                model_weights=model_bytes
            )
            stub.SendUpdatedModel(request, timeout=10)
            channel.close()
            return True
        except Exception as e:
            print(f"[Coordinator Error] Fallito l'invio del modello globale a {node_address}: {e}")
            return False

    def broadcast_global_model(self, node_addresses, model_bytes, round_num):
        """Invia in parallelo il modello aggregato a tutti i worker."""
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
        print("    AVVIO COORDINATORE FEDERATED LEARNING    ")
        print("="*40)

        # 1. Discovery dei Nodi dal Registry
        active_nodes = []
        retries = 0
        num_peers_required = self.config['training_nodes']
        max_retries = self.config['max_discovery_retries']
        bucket_name = "sdcc-dataset-264452429750-us-east-1-an"
        s3_key = "all_data_niid_05_keep_3_train_9.json"
        while len(active_nodes) < num_peers_required:
            if retries >= max_retries:
                print("[Error] Discovery timeout.")
                break
            print(f"[Discovery] Fetching peers ({retries+1}/{max_retries})...")
            active_nodes = self.registry_client.get_peer_list(node_request_count=num_peers_required)
            if len(active_nodes) < num_peers_required:
                time.sleep(5)
                retries += 1
        
        if len(active_nodes) < num_peers_required:
            print("[Aborting] Impossibile avviare il training per assenza peer.")
            return

        print(f"[Coordinator] Nodi reclutati per l'addestramento: {active_nodes}")

        # 2. Avvio dell'addestramento sui nodi (Start Training Signal) e inizializzazione modello
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        global_model = SentimentPyTorch(num_class=2).to(device)
        print("[Coordinator] Invio segnale di avvio (StartTraining) a tutti i nodi...")
        indexes = get_training_index_list(active_nodes, bucket_name, s3_key, self.config['training_set_percentage'], max_samples_per_client=1000)
        self.broadcast_start_signal(active_nodes, indexes)

        total_rounds = self.config['total_rounds']
        weight_timeout = self.config['timeout_sec']
        min_clients = self.config.get('min_clients', 1)

        # 3. Ciclo dei Round FL
        for round_num in range(total_rounds):
            print(f"\n-------------------- ROUND {round_num + 1}/{total_rounds} --------------------")
            
            with self.servicer.lock:
                self.servicer.current_round = round_num
                self.servicer.received_weights.clear()

            # Attesa dei pesi dai nodi worker
            start_wait_time = time.time()
            while True:
                with self.servicer.lock:
                    received_count = len(self.servicer.received_weights.get(round_num, []))
                
                # Se tutti i nodi attivi hanno risposto
                if received_count >= len(active_nodes):
                    print(f"[Coordinator] Pesi ricevuti da tutti i nodi ({received_count}/{len(active_nodes)}).")
                    break
                
                # Handling Timeout
                if time.time() - start_wait_time > weight_timeout:
                    print(f"[Coordinator Timeout] Timeout raggiunto per il Round {round_num + 1}. Ricevuti pesi da {received_count}/{len(active_nodes)} nodi.")
                    break
                
                time.sleep(1)

            # Deserializzazione e Aggregazione (FedAvg)
            with self.servicer.lock:
                round_payloads = self.servicer.received_weights.get(round_num, [])

            print(f"[Coordinator] Esecuzione FedAvg su {len(round_payloads)} contributi...")

            round_payloads.sort(key=lambda x: x.sender_id)
            deserialized_models = []
            
            for request in round_payloads:
                state_dict = load_weights_from_bytes(request.model_weights)
                deserialized_models.append({
                    'sender_id': request.sender_id,
                    'weights': state_dict,
                    'num_samples': request.num_samples
                })
            
            # HASH PRIMA DI FEDAVG
            fc_hash_before = get_model_hash(global_model, only_trainable=True)
            print(f"[VERIFICATION] PRIMA DI FEDAVG round {round_num + 1} Model Classifier SHA-256: {fc_hash_before}")

            # Applicazione FedAvg
            global_model = apply_fedavg(global_model, deserialized_models)
            
            # Clear dei buffer del servicer
            with self.servicer.lock:
                self.servicer.received_weights.pop(round_num, None)

            # HASH DOPO FEDAVG
            fc_hash_after = get_model_hash(global_model, only_trainable=True)
            print(f"[VERIFICATION] DOPO FEDAVG round {round_num + 1} Model Classifier SHA-256: {fc_hash_after}")

            # Caricamento nel modello locale per verifica
            global_bytes = get_weights_as_bytes(global_model)
            
            with self.servicer.lock:
                self.servicer.latest_global_bytes = global_bytes

            print(f"[Coordinator] Round {round_num + 1} completato.")

            # Broadcast del nuovo modello ai worker
            print(f"[Coordinator] Invio del nuovo modello aggregato ai nodi...")
            self.broadcast_global_model(active_nodes, global_bytes, round_num)

        print("\n" + "="*40)
        print("  ADDESTRAMENTO FEDERATO COMPLETATO CON SUCCESSO ")
        print("="*40)


def start_coordinator_server(port: int):
    """Inizializza il server gRPC in background per il coordinatore."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedServerServicer(my_id='COORDINATOR')
    federated_pb2_grpc.add_FederatedServerServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"[Coordinator] Server gRPC in ascolto sulla porta {port}...")
    return server, servicer


def main():
    PORT = int(os.getenv("PORT", 50053))
    REGISTRY_ADDR = os.getenv("REGISTRY_ADDRESS", "centralized-registry-nlb-d298b040b4ebce81.elb.us-east-1.amazonaws.com:8080")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    my_ip = get_ecs_container_ip()
    coordinator_grpc_address = f"{my_ip}:{PORT}"
    training_nodes = int(os.getenv("TRAINING_NODES", 5))

    # Configurazione da distribuire a tutti i nodi
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

    # 1. Avvio gRPC Server del Coordinatore
    server, servicer = start_coordinator_server(PORT)

    # 2. Avvio della logica dell'Orchestratore
    coordinator = FederatedCoordinator(
        servicer=servicer,
        registry_addr=REGISTRY_ADDR,
        config=training_config,
        device=device
    )
    registry_client = RegistryClient(REGISTRY_ADDR, '1')
    coordinator.registry_client = registry_client

    try:
        coordinator.run_coordination_loop()
    except KeyboardInterrupt:
        print("\n[Coordinator] Arresto forzato da tastiera.")
    finally:
        server.stop(grace=5)
        print("[Coordinator] Server gRPC arrestato.")


if __name__ == "__main__":
    main()