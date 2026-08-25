import random
import grpc
from concurrent import futures
import time
import torch
import os
import datetime

# Import generated gRPC code
from rpc_calls import RegistryClient, send_weights_to_coordinator
from rpc_calls import FederatedNodeServicer
import federated_pb2 as federated_pb2
import federated_pb2_grpc as federated_pb2_grpc

from model import SentimentPyTorch
from utils import get_weights_as_bytes, load_weights_from_bytes, get_model_hash, get_ecs_container_ip

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

def run_training_loop(config, global_model, servicer, MY_ID, device, RESPAWNED):
    """
    Logica dell'addestramento federato con gestione Respawn
    """
    total_rounds = config['total_rounds']
    start_round = config['start_round']
    start_index = config['start_index']
    weight_wait_timeout = config['weight_wait_timeout']
    training_set_percentage = config['training_set_percentage']
    coordinator_address = config['aggregator_address']
    num_samples = config['num_samples']
    truncated_training_length = config['truncated_training_length']
    num_epochs = config['num_epochs']
    
    # 1. Preparazione Dataset
    bucket_name = "sdcc-dataset-264452429750-us-east-1-an"
    s3_key = "all_data_niid_05_keep_3_train_9.json"
    X_train, Mask_train, Y_train, X_val, Mask_val, Y_val = SentimentPyTorch.prepare_dataset(bucket_name, s3_key, num_samples, training_set_percentage, start_index, truncated_training_length, max_samples_per_client=1000, seed=42)
    my_samples = len(Y_train)
    
    servicer.num_samples = my_samples
    
    # ==========================================
    # RESPAWN RECOVERY: Ripristino stato dai Peer
    # ==========================================
    if RESPAWNED:
        print("\n[Recovery] Nodo identificato come RESPAWNED. Avvio recupero pesi dai peer...")
        
        # 1. Chiedo il modello aggiornato al coordinatore
        #ask model to coordinator

        print("[Wait] In attesa dei pesi locali dai peer per completare il recovery...")
        start_wait_time = time.time()
        
        # Attesa dei pesi con timeout
        while True:
            with servicer.lock:
                rec_model = servicer.received_model
                if rec_model is not None:
                    break
            if time.time() - start_wait_time > weight_wait_timeout:
                break
            time.sleep(0.5)

        with servicer.lock:
            current_model = servicer.received_model

        if not current_model:
            print("[Error] Nessun modello ricevuto dal coordinatore durante il recovery. Fallimento recovery.")
            return

        current_weights = load_weights_from_bytes(current_model)
        global_model.load_state_dict(current_weights)
        servicer.received_weights.clear()
        
        # Avanziamo il round dato che abbiamo recuperato lo stato di quello precedente
        start_round += 1
        print(f"[Recovery] Modello ripristinato con successo. Il training ripartirà dal round {start_round + 1}.")

    print(f"\n[Training] Avvio sessione di addestramento.")
    start_training_time = time.time()

    # 2. Ciclo dei Round
    try:
        for round_num in range(start_round, total_rounds):
            print(f"\n{'='*10} ROUND {round_num + 1}/{total_rounds} {'='*10}")
            
            # A. Local Training
            global_model, my_samples = SentimentPyTorch.train_local(
                model=global_model, 
                X_train=X_train, Mask_train=Mask_train, Y_train=Y_train, 
                X_val=X_val, Mask_val=Mask_val, Y_val=Y_val, 
                device=device,
                num_epochs=num_epochs
            )

            # B. Serializzazione
            payload_bytes = get_weights_as_bytes(global_model)
            servicer.latest_local_weights = payload_bytes
            servicer.round_num = round_num
            
            # C. Send weights to coordinator        
            payload = federated_pb2.WeightPayload(
                sender_id=MY_ID,
                round_number=round_num,
                model_weights=payload_bytes,
                num_samples=my_samples
            )
            if send_weights_to_coordinator(coordinator_address, payload) == 1:
                print(f"[Warning] coordinator unresponsive. Fault recovery initiated...")
                #implement

            # D. Wait Weights
            start_wait_time = time.time()
            print(f"[Wait] In attesa dei pesi dal coordinatore per il round {round_num + 1} (Timeout: {weight_wait_timeout}s)...")
            while True:
                with servicer.lock:
                    current_round_weights = servicer.received_model
                    if current_round_weights:
                        print(f"[Received] Pesi ricevuti per il round {round_num}.")
                        break
                if time.time() - start_wait_time > weight_wait_timeout:
                    print(f"[Warning] Timeout! Coordinatore non risponde.")
                    # ping coordinator and eventually fault detection
                    break
                time.sleep(0.5)
                
            # E. Aggregazione (FedAvg) fatta dal coordinatore
            current_round_weights_loaded = load_weights_from_bytes(current_round_weights)
            global_model.load_state_dict(current_round_weights_loaded, strict=False)

            # Clear dei buffer del servicer
            servicer.received_model = None

        total_training_time = time.time() - start_training_time
        print("\nTRAINING HAS BEEN COMPLETED.")
        print(f"Total training time: {str(datetime.timedelta(seconds = total_training_time))}\n")

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
            SentimentPyTorch.evaluate_global(global_model, X_val, Mask_val, Y_val, device)
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
    initial_status = "idle"

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
                        config, global_model, servicer, MY_ID, device, RESPAWNED)
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