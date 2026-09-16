# Decentralized-Federated-Learning-SDCC

## Descrizione Progetto

Progetto per i corsi di "Sistemi Distribuiti e Cloud Computing" e "Machine Learning" a cura di Federico Di Stefano, Cristiano Pera e Alfonso Maiorca per l’anno accademico 2025/2026.

Il progetto riguarda la progettazione, implementazione e valutazione di un sistema per l’addestramento federato di modelli di Machine Learning in modalità decentralizzata. Il sistema deve permettere a molteplici client, ciascuno con la sua porzione di dataset locale, di collaborare all’addestramento di un modello globale senza scambiare dati degli utenti, ma solo dati del modello stesso. E'anche presente una versione centralizzata da mettere a confronto con quella decentralizzata. "Relazione SDCC-ML.pdf" contiene tutti i dettagli sul progetto e le analisi compiute sui sistemi.

## Installazione e Configurazione

Per l'utilizzo del progetto è necessaria la seguente infrastruttura AWS:

### S3

Bucket:
* sdcc-dataset

AWS appenderà al bucket l'id dell utente e la regione (format -xxxxxxxxxxxx-xx-xxxx-x-xx). Su esso andrà caricato il file "all_data_niid_05_keep_3_train_9.json".

### Dynamo DB

Tabelle con primary key NodeId (String):
* Nodes
* Nodes_Centralized

### ECR

Le immagini devono essere create mediante comando mentre si è nella cartella root del progetto (es: docker build -t federated/client -f client/Dockerfile .)

Repositories:
* federated/client (immagine: client/Dockerfile)
* federated/registry (immagine: registry/Dockerfile)
* federated/registry-centralized (immagine: registry_centralized/Dockerfile)
* federated/training-centralized (immagine: centralized/Dockerfile)

### ECS

Task Definitions:
* client_task (ecr: federated/client)
* client_task_centralized (ecr: federated/training-centralized)
* coordinator_task (ecr: federated/training-centralized)
* registry_server_task (ecr: federated/registry)
* registry_server_task_centralized (ecr: federated/registry-centralized)

Clusters:
* federated_cluster (Fargate)
* centralized-cluster (Fargate)

Services:
* registry-service (cluster: federated_cluster, task: registry_server_task)
* centralized-registry-service (cluster: centralized-cluster, registry_server_task_centralized)

### Modifica parametri hardcoded

Utilizzare VPC su AWS per ottenere le subnet. Andare quindi in registry/src/utils/utils.go e sostituirle nella funzione LaunchTask. Ripetere lo stesso procedimento in registry_centralized/src/utils/utils.go.
Sostituire in client/src/main.py la variabile bucket_name con il nome del vostro bucket S3. Ripetere lo stesso in centralized/src/client.py.

## Esecuzione

Per lanciare il sistema decentralizzato andare in "federated_cluster" ed utilizzare "registry-service" per tenere attiva un'istanza di "registry_server_task" (desired amount = 1). Successivamente lanciare una "client_task" impostando in container overrides la variabile d'ambiente "STARTER = true". Modificando i seguenti parametri si può personalizzare il processo di addestramento:

* NUM_EPOCS per cambiare il numero di epoche in un round.
* TOTAL_ROUNDS per il numero di round. Alla fine di ogni round avverrà lo scambio dei pesi. Concluso l'ultimo round l'addestramento termina.
* TRAINING_NODES per il numero di nodi coinvolti nell' addestramento.
* NUM_PEERS_REQUIRED  numero minimo di peers per avviare il training. Può essere impostato al più a TRAINING_NODES - 1.
* WEIGHT_WAIT_TIMEOUT_SECONDS per impostare il tempo massimo di attesa per la ricezione dei pesi.

Per il sistema centralizzato il procedimento è analogo. Utilizzare il servizio "centralized-registry-service" in "centralized-cluster" per tenere attiva un'istanza di "registry_server_task_centralized". Infine lanciare una "coordinator_task" (STARTER non è presente in quanto il coordinatore avvia sempre il processo) con le variabili d'ambiente desiderate.
