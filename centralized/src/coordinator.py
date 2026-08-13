import os
import time
import json
import random
import threading
import torch
import grpc
import requests
from concurrent import futures

from utils import load_weights_from_bytes, get_weights_as_bytes, get_model_hash, get_ecs_container_ip
from model import SentimentPyTorch

import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc


class CoordinatorServicer(federated_pb2_grpc.FederatedNodeServicer):
    """
    Interfaccia gRPC del Coordinatore per la ricezione dei pesi dai Worker.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.received_weights = {}  # {sender_id: (model_bytes, num_samples)}
        self.current_round = -1
        self.latest_global_bytes = None

    def SendWeights(self, request, context):
        """Riceve i pesi inviati dai nodi worker a fine round locale."""
        with self.lock:
            if request.round_number == self.current_round:
                self.received_weights[request.sender_id] = (
                    request.model_weights,
                    request.num_samples
                )
                print(f"[Coordinator] Ricevuti pesi dal nodo '{request.sender_id}' per il Round {request.round_number + 1}.")
                return federated_pb2.WeightAck(status="SUCCESS", message="Pesi registrati con successo.")
            else:
                print(f"[Coordinator Warning] Scartati pesi da '{request.sender_id}' per Round {request.round_number + 1} (Round corrente: {self.current_round + 1}).")
                return federated_pb2.WeightAck(status="REJECTED", message="Round non sincronizzato.")

    def GetGlobalModel(self, request, context):
        """Permette ai nodi RESPAWNED di richiedere l'ultimo modello globale."""
        with self.lock:
            if self.latest_global_bytes:
                return federated_pb2.ModelResponse(
                    model_weights=self.latest_global_bytes,
                    round_number=self.current_round
                )
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Nessun modello globale ancora disponibile.")
                return federated_pb2.ModelResponse()


class FederatedCoordinator:
    def __init__(self, servicer, registry_addr, config, device):
        self.servicer = servicer
        self.registry_addr = registry_addr
        self.config = config
        self.device = device
        self.global_model = SentimentPyTorch(num_class=2).to(self.device)

    def fetch_active_nodes_from_registry(self):
        """
        Interroga il Service Registry per ottenere la lista dei nodi IDLE attivi.
        """
        url = f"http://{self.registry_addr}/get_nodes"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                nodes_data = response.json()
                # Esempio response: [{"id": "node_123", "ip": "10.0.1.5", "port": 50051, "status": "idle"}]
                active_nodes = [
                    f"{node['ip']}:{node['port']}" 
                    for node in nodes_data 
                    if node.get('status') == 'idle'
                ]
                print(f"[Registry] Trovati {len(active_nodes)} nodi in stato IDLE.")
                return active_nodes
            else:
                print(f"[Registry Error] Risposta non valida dal Registry: {response.status_code}")
                return []
        except Exception as e:
            print(f"[Registry Error] Impossibile contattare il Registry ({self.registry_addr}): {e}")
            return []

    def start_training_on_node(self, node_address, training_config):
        """
        Invia il segnale di START e la configurazione gRPC a un nodo worker.
        """
        try:
            channel = grpc.insecure_channel(node_address)
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            
            # Serializza la config da inviare
            req = federated_pb2.StartTrainingRequest(
                config_json=json.dumps(training_config)
            )
            stub.StartTraining(req, timeout=10)
            channel.close()
            return True
        except Exception as e:
            print(f"[Coordinator Error] Impossibile avviare il training sul nodo {node_address}: {e}")
            return False

    def broadcast_start_signal(self, node_addresses):
        """Invia in parallelo il comando di avvio a tutti i worker."""
        threads = []
        for addr in node_addresses:
            t = threading.Thread(
                target=self.start_training_on_node, 
                args=(addr, self.config)
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def fed_avg(self, weights_list):
        """
        Calcola la media pesata dei parametri del modello in base ai campioni (FedAvg).
        """
        total_samples = sum(num_samples for _, num_samples in weights_list)
        first_weights = weights_list[0][0]
        avg_weights = {}

        for key in first_weights.keys():
            avg_weights[key] = torch.zeros_like(first_weights[key], dtype=torch.float32)

        for weights, num_samples in weights_list:
            weight_factor = num_samples / total_samples
            for key in avg_weights.keys():
                avg_weights[key] += weights[key].to(self.device) * weight_factor

        return avg_weights

    def send_global_model_to_node(self, node_address, model_bytes, round_num):
        """Invia il nuovo modello globale aggregato ad un singolo worker."""
        try:
            channel = grpc.insecure_channel(node_address)
            stub = federated_pb2_grpc.FederatedNodeStub(channel)
            
            request = federated_pb2.ModelPayload(
                round_number=round_num,
                global_weights=model_bytes
            )
            stub.ReceiveGlobalModel(request, timeout=10)
            channel.close()
            return True
        except Exception as e:
            print(f"[Coordinator Error] Fallito l'invio del modello globale a {node_address}: {e}")
            return False

    def broadcast_global_model(self, node_addresses, model_bytes, round_num):
        """Invia in parallelo il modello aggregato a tutti i worker."""
        threads = []
        for addr in node_addresses:
            t = threading.Thread(
                target=self.send_global_model_to_node, 
                args=(addr, model_bytes, round_num)
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
        active_nodes = self.fetch_active_nodes_from_registry()
        if not active_nodes:
            print("[Coordinator Fatal] Nessun nodo disponibile per l'addestramento. Abort...")
            return

        print(f"[Coordinator] Nodi reclutati per l'addestramento: {active_nodes}")

        # 2. Avvio dell'addestramento sui nodi (Start Training Signal)
        print("[Coordinator] Invio segnale di avvio (StartTraining) a tutti i nodi...")
        self.broadcast_start_signal(active_nodes)

        total_rounds = self.config['total_rounds']
        weight_timeout = self.config['weight_wait_timeout']
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
                    received_count = len(self.servicer.received_weights)
                
                # Se tutti i nodi attivi hanno risposto
                if received_count >= len(active_nodes):
                    print(f"[Coordinator] Pesi ricevuti da tutti i nodi ({received_count}/{len(active_nodes)}).")
                    break
                
                # Handling Timeout
                if time.time() - start_wait_time > weight_timeout:
                    print(f"[Coordinator Timeout] Timeout raggiunto per il Round {round_num + 1}. Ricevuti pesi da {received_count}/{len(active_nodes)} nodi.")
                    break
                
                time.sleep(1)

            # Controllo del quorum minimo
            with self.servicer.lock:
                current_updates = dict(self.servicer.received_weights)

            if len(current_updates) < min_clients:
                print(f"[Coordinator Critical] Ricevuti meno contributi del quorum minimo ({len(current_updates)}/{min_clients}). Salto il round.")
                continue

            # Deserializzazione e Aggregazione (FedAvg)
            deserialized_weights = []
            for client_id, (w_bytes, n_samples) in current_updates.items():
                loaded_w = load_weights_from_bytes(w_bytes)
                deserialized_weights.append((loaded_w, n_samples))

            print(f"[Coordinator] Esecuzione FedAvg su {len(deserialized_weights)} contributi...")
            aggregated_weights = self.fed_avg(deserialized_weights)
            
            # Caricamento nel modello locale per verifica
            self.global_model.load_state_dict(aggregated_weights)
            global_bytes = get_weights_as_bytes(self.global_model)
            
            with self.servicer.lock:
                self.servicer.latest_global_bytes = global_bytes

            fc_hash = get_model_hash(self.global_model, only_trainable=True)
            print(f"[Coordinator] Round {round_num + 1} completato. Modello SHA-256 (FC): {fc_hash}")

            # Broadcast del nuovo modello ai worker
            print(f"[Coordinator] Invio del nuovo modello aggregato ai nodi...")
            self.broadcast_global_model(active_nodes, global_bytes, round_num)

        print("\n" + "="*40)
        print("  ADDESTRAMENTO FEDERATO COMPLETATO CON SUCCESSO ")
        print("="*40)


def start_coordinator_server(port: int):
    """Inizializza il server gRPC in background per il coordinatore."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = CoordinatorServicer()
    federated_pb2_grpc.add_FederatedNodeServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"[Coordinator] Server gRPC in ascolto sulla porta {port}...")
    return server, servicer


def main():
    PORT = int(os.getenv("PORT", 50052))
    REGISTRY_ADDR = os.getenv("REGISTRY_ADRESS", "registry-nlb-ba1dc354920c500b.elb.us-east-1.amazonaws.com:8080")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Indirizzo gRPC del Coordinatore
    # (Lo stesso che i nodi client leggono dal loro dict 'config')
    my_ip = get_ecs_container_ip()
    coordinator_grpc_address = f"{my_ip}:{PORT}"
    max_retries = int(os.getenv("MAX_DISCOVERY_RETRIES", 5))
    training_nodes = int(os.getenv("TRAINING_NODES", 5))

    # Configurazione da distribuire a tutti i nodi
    training_config = {
        'training_nodes': training_nodes,
        'total_rounds': int(os.getenv("TOTAL_ROUNDS", 5)),
        'start_round': int(os.getenv("START_ROUND", 0)),
        'start_index': int(os.getenv("START_INDEX", 0)),
        'num_peers_required': int(os.getenv("NUM_PEERS_REQUIRED", training_nodes - 1)),
        'timeout_sec': int(os.getenv("WEIGHT_WAIT_TIMEOUT_SECONDS", 60)),
        'training_set_percentage': float(os.getenv("TRAINING_SET_PERCENTAGE", 0.7)),
        'aggregator_address': coordinator_grpc_address,
        'num_epochs': int(os.getenv("NUM_EPOCHS", 1))
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

    try:
        coordinator.run_coordination_loop()
    except KeyboardInterrupt:
        print("\n[Coordinator] Arresto forzato da tastiera.")
    finally:
        server.stop(grace=5)
        print("[Coordinator] Server gRPC arrestato.")


if __name__ == "__main__":
    main()