import json
import math
import random
import grpc
from concurrent import futures
import time
import torch
import os
import urllib
import copy
import threading

# Import generated gRPC code
from rpc_calls import RegistryClient, send_weights_to_peer
from rpc_calls import FederatedNodeServicer
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

from model import SentimentPyTorch
from aggregator import apply_fedavg
from utils import calculate_k, get_weights_as_bytes, load_weights_from_bytes, get_model_hash

def start_grpc_server(port: int, my_id: str) -> tuple:
    """
    Initializes and starts the background gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedNodeServicer(my_id=my_id)
    federated_pb2_grpc.add_FederatedNodeServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"[Server] Background gRPC server listening on port {port}...")
    return server, servicer

def get_ecs_container_ip():
    metadata_url = os.getenv("ECS_CONTAINER_METADATA_URI_V4")
    
    if not metadata_url:
        return "Variable ECS_CONTAINER_METADATA_URI_V4 not found. Not running on ECS?"

    try:
        with urllib.request.urlopen(metadata_url) as response:
            body = response.read().decode('utf-8')
            metadata = json.loads(body)
            
            networks = metadata.get('Networks', [])
            if networks and 'IPv4Addresses' in networks[0]:
                return networks[0]['IPv4Addresses'][0]
                
    except Exception as e:
        return f"Error reading metadata: {e}"

    return "IP address not found in metadata"

def run_training_loop(config, global_model, servicer, registry_client, MY_ID, device, RESPAWNED):
    """
    Logica dell'addestramento federato con gestione Respawn
    """
    training_nodes = config['training_nodes']
    total_rounds = config['total_rounds']
    start_round = config['start_round']
    peers = config['peers']
    weight_wait_timeout = config['weight_wait_timeout']
    training_set_percentage = config['training_set_percentage']
    
    # 1. Preparazione Dataset
    bucket_name = "sdcc-dataset-771379920513-us-east-1-an"
    s3_key = "all_data_niid_05_keep_3_train_9.json"
    X_train, Mask_train, Y_train, X_val, Mask_val, Y_val = SentimentPyTorch.prepare_dataset(bucket_name=bucket_name, s3_key=s3_key, num_training_nodes=training_nodes, training_set_percentage=training_set_percentage)
    my_samples = len(Y_train)
    
    servicer.num_samples = my_samples
    servicer.peers = peers
    k = calculate_k(peers)
    servicer.fanout = k
    servicer.received_weights.clear()  # Clear any previous weights
    servicer.seen_messages.clear()  # Clear seen messages to avoid stale data
    servicer.latest_local_weights = None  # Reset latest local weights
    servicer.round_num = start_round  # Set the round number for the servicer
    
    # ==========================================
    # RESPAWN RECOVERY: Ripristino stato dai Peer
    # ==========================================
    if RESPAWNED:
        print("\n[Recovery] Nodo identificato come RESPAWNED. Avvio recupero pesi dai peer...")
        
        # 1. Chiedo i pesi a tutti i peer della lista
        for peer in peers: 
            servicer.get_weights_from_peer(peer['ip'], peer['port'], peer['id'], MY_ID, start_round)

        print("[Wait] In attesa dei pesi locali dai peer per completare il recovery...")
        start_wait_time = time.time()
        
        # Attesa dei pesi con timeout
        while True:
            with servicer.lock:
                rec_weights = servicer.received_weights.get(start_round, [])
                if len(rec_weights) > 0:
                    break
            if time.time() - start_wait_time > weight_wait_timeout:
                break
            time.sleep(0.5)

        with servicer.lock:
            round_payloads = servicer.received_weights.get(start_round, [])

        if not round_payloads:
            print("[Error] Nessun peso ricevuto dai peer durante il recovery. Fallimento recovery.")
            return

        print(f"[Aggregate] Recovery: Esecuzione FedAvg su {len(round_payloads)} modelli ricevuti dai peer...")
        round_payloads.sort(key=lambda x: x.sender_id)
        deserialized_models = []

        for request in round_payloads:
            state_dict = load_weights_from_bytes(request.model_weights)
            deserialized_models.append({
                'sender_id': request.sender_id,
                'weights': state_dict,
                'num_samples': request.num_samples
            })

        global_model = apply_fedavg(global_model, deserialized_models)
        servicer.received_weights.clear()
        
        # Avanziamo il round dato che abbiamo recuperato lo stato di quello precedente
        start_round += 1
        print(f"[Recovery] Modello ripristinato con successo. Il training ripartirà dal round {start_round + 1}.")

    print(f"\n[Training] Avvio sessione di addestramento. Peers: {len(peers)}, Fanout (k): {k}")

    # 2. Ciclo dei Round
    try:
        for round_num in range(start_round, total_rounds):
            print(f"\n{'='*10} ROUND {round_num + 1}/{total_rounds} {'='*10}")
            
            # A. Local Training
            global_model, my_samples = SentimentPyTorch.train_local(
                model=global_model, 
                X_train=X_train, Mask_train=Mask_train, Y_train=Y_train, 
                X_val=X_val, Mask_val=Mask_val, Y_val=Y_val, 
                device=device
            )

            # B. Serializzazione
            payload_bytes = get_weights_as_bytes(global_model)
            servicer.latest_local_weights = payload_bytes
            servicer.round_num = round_num
            servicer.seen_messages.add((MY_ID, round_num))
            
            # C. Gossip
            actual_k = min(k, len(peers))
            initial_gossip_peers = random.sample(peers, actual_k)
            
            for peer in initial_gossip_peers:
                payload = federated_pb2.WeightPayload(
                    sender_id=MY_ID,
                    round_number=round_num,
                    model_weights=payload_bytes,
                    num_samples=my_samples
                )
                if send_weights_to_peer(peer['ip'], peer['port'], peer['id'], payload) == 1:
                    print(f"[Warning] Peer {peer['id']} unresponsive. Cleanup...")
                    servicer.remove_peer_by_id(peer['id'])
                    peers = servicer.peers
                    registry_client.signal_unresponsive_node(
                        peer['ip'], peer['port'], peer['id'], 
                        training_nodes, total_rounds, start_round, 5, weight_wait_timeout, training_set_percentage
                    )

            # D. Wait Weights
            start_wait_time = time.time()
            print(f"[Wait] In attesa dei pesi dai peer per il round {round_num + 1} (Timeout: {weight_wait_timeout}s)...")
            enough_weights_received = False
            while True:
                with servicer.lock:
                    current_round_weights = servicer.received_weights.get(round_num, [])
                    if len(current_round_weights) >= len(peers):
                        print(f"[Info] Tutti i pesi ricevuti dai peer per il round {round_num + 1}.")
                        break
                    if len(current_round_weights) >= math.ceil(actual_k/2):
                        enough_weights_received = True

                if time.time() - start_wait_time > weight_wait_timeout:
                    if not enough_weights_received:
                        print(f"[Warning] Timeout! Proseguo con {len(current_round_weights)} modelli ricevuti.")
                    else:
                        print(f"[Info] Ricevuti abbastanza pesi per il round {round_num + 1}.")
                    break
                time.sleep(0.5)
                
            # E. Aggregazione (FedAvg)
            with servicer.lock:
                round_payloads = servicer.received_weights.get(round_num, [])

            round_payloads.sort(key=lambda x: x.sender_id)
            deserialized_models = []
            
            local_fc_only = {k: v.cpu() for k, v in global_model.state_dict().items() if k.startswith('fc.')}
            deserialized_models.append({
                'sender_id': MY_ID,
                'weights': copy.deepcopy(local_fc_only),
                'num_samples': my_samples
            })
            
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
            servicer.received_weights.clear()

            # HASH DOPO FEDAVG
            fc_hash_after = get_model_hash(global_model, only_trainable=True)
            print(f"[VERIFICATION] DOPO FEDAVG round {round_num + 1} Model Classifier SHA-256: {fc_hash_after}")

            # Update current round in servicer
            servicer.current_round += 1

        print("\nTRAINING HAS BEEN COMPLETED.")

        # ==========================================
        # VERIFICA DEGLI HASH FINALI
        # ==========================================
        fc_hash_final = get_model_hash(global_model, only_trainable=True)
        full_hash_final = get_model_hash(global_model, only_trainable=False)

        print("\n" + "="*10)
        print(f"[VERIFICATION] Final Model Classifier SHA-256: {fc_hash_final}")
        print(f"[VERIFICATION] Final Model Full SHA-256:       {full_hash_final}")
        print("="*10 + "\n")

        # Evaluation
        print("Valutazione globale su modello finale")
        try:
            X_full, Mask_full, Y_full = SentimentPyTorch.prepare_eval_dataset(bucket_name=bucket_name, s3_key=s3_key, training_set_percentage=training_set_percentage)
            SentimentPyTorch.evaluate_global(global_model, X_full, Mask_full, Y_full, device)
        except Exception as e:
            print(f"[Error] Valutazione globale fallita: {e}")

    except Exception as e:
        print(f"[Error] Eccezione durante il training loop: {e}")

def main():
    # 1. Lettura ENV
    MY_ID = str(os.getenv("CLIENT_ID", f"node_{random.randint(1000,9999)}"))
    MY_IP = get_ecs_container_ip()
    MY_PORT = int(os.getenv("PORT", 50051))
    REGISTRY_ADDR = os.getenv("REGISTRY_ADRESS", "registry-nlb-ba1dc354920c500b.elb.us-east-1.amazonaws.com:8080")
    STARTER = os.getenv("STARTER", "false").lower() == "true"
    RESPAWNED = os.getenv("RESPAWNED", "false").lower() == "true"
    
    # 2. Inizializzazione Server gRPC & Registry Client
    registry_client = RegistryClient(REGISTRY_ADDR, MY_ID)
    server, servicer = start_grpc_server(MY_PORT, MY_ID)
    registry_client.servicer = servicer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    global_model = SentimentPyTorch(num_class=2).to(device)

    # ==========================================
    # REGISTRAZIONE DIVERSIFICATA (RESPAWNED vs NORMAL)
    # ==========================================
    initial_status = "working" if STARTER else "idle"
    
    is_starter_execution = STARTER

    if not RESPAWNED:
        if not registry_client.register_node(MY_IP, MY_PORT, initial_status):
            print("[Fatal] Error connecting to Registry. Exiting.")
            server.stop(grace=0)
            return
    else:
        print("[Init] Nodo avviato in modalità RESPAWNED.")
        if not registry_client.register_respawned_node(MY_IP, MY_PORT):
            print("[Fatal] Error connecting to Registry for respawned node. Exiting.")
            server.stop(grace=0)
            return


    try:
        while True:
            config = None
            
            if is_starter_execution:
                # Flusso STARTER
                print("\n[Starter] Nodo STARTER. Fase discovery...")
                
                training_nodes = int(os.getenv("TRAINING_NODES", 5))
                total_rounds = int(os.getenv("TOTAL_ROUNDS", 5))
                start_round = int(os.getenv("START_ROUND", 0))
                num_peers_required = int(os.getenv("NUM_PEERS_REQUIRED", training_nodes - 1))
                max_retries = int(os.getenv("MAX_DISCOVERY_RETRIES", 5))
                timeout_sec = int(os.getenv("WEIGHT_WAIT_TIMEOUT_SECONDS", 30))
                training_set_percentage = float(os.getenv("TRAINING_SET_PERCENTAGE", 0.7))

                peers = []
                retries = 0
                while len(peers) < num_peers_required:
                    if retries >= max_retries:
                        print("[Error] Discovery timeout.")
                        break
                    print(f"[Discovery] Fetching peers ({retries+1}/{max_retries})...")
                    peers = registry_client.get_peer_list(node_request_count=num_peers_required)
                    if len(peers) < num_peers_required:
                        time.sleep(5)
                        retries += 1
                
                if len(peers) < num_peers_required:
                    print("[Aborting] Impossibile avviare il training per assenza peer.")
                else:
                    starter_peer = federated_pb2.NodeInfo(
                        node_id=MY_ID,
                        ip_address=str(MY_IP),
                        port=int(MY_PORT),
                        status="working"
                    )
                    config = {
                        'training_nodes': training_nodes,
                        'total_rounds': total_rounds,
                        'start_round': start_round,
                        'num_peers_required': num_peers_required,
                        'max_discovery_retries': max_retries,
                        'weight_wait_timeout': timeout_sec,
                        'training_set_percentage': training_set_percentage,
                        'peers': [*peers, starter_peer]  # Include the starter node itself in the peers list
                    }
                    
                    print("[Starter] Invio RPC StartTraining ai peer...")
                    for peer in peers:
                        servicer.send_start_training_signal(peer, config)

                is_starter_execution = False

            else:
                # Flusso SUPPORT NODE / IDLE
                print("\n[Idle] In attesa di richieste di addestramento (Timeout: 5 minuti)...")
                servicer.start_training_event.clear()
                
                received_signal = servicer.start_training_event.wait(timeout=300)

                if not received_signal:
                    print("\n[Timeout] Nessuna richiesta nei 5 minuti di idle. Spegnimento...")
                    break

                config = servicer.pending_training_config
                registry_client.update_status("working")

            # Esecuzione Training (passando la flag RESPAWNED)
            if config:
                try:
                    run_training_loop(
                        config, global_model, servicer, registry_client, MY_ID, device, RESPAWNED)
                except Exception as e:
                    print(f"[Error Main] Errore durante l'esecuzione del training: {e}")
                RESPAWNED = False

            # RESET DEL MODELLO GLOBALE PER NUOVE ESECUZIONI
            print("\n[Reset] Reinizializzazione del modello globale per future sessioni...")
            torch.manual_seed(42)
            global_model = SentimentPyTorch(num_class=2).to(device)

            # Ripristino stato IDLE
            print("\n[Status] Ripristino stato a IDLE per 5 minuti...")
            registry_client.update_status("idle")

    except KeyboardInterrupt:
        print("\n[Shutdown] Interruzione manuale.")
    finally:
        print("[Shutdown] Unregister e arresto gRPC...")
        registry_client.unregister_node(MY_IP, MY_PORT)
        server.stop(grace=5)
        print("[Shutdown] Done.")
    
    
if __name__ == "__main__":
    main()