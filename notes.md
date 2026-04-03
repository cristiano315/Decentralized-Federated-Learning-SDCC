# OBBIETTIVI:
Vedere traccia A4

# STRUTTURA:
1) Servise Registry fatto in go.
2) Tutto il resto in python(pytorch).
3) Comunicazione tramite GRPC.
4) AWS per inizializzare i container (possibile scelta: serverless sempre gestita da AWS (Lambda, DynamoDB, API Gateway) oppure insrastructure as a service). Usare Amazon Elastic Container Service e Fargate.
5) Dataset: Amazon S3.
6) Per il logging AWS Cloudwatch.

# Modello:
Il modello deve essere leggero. Da scegliere tra:

1) Modello 1 (FeMNIST): Rete Convoluzionale. Layers: 2x Layer Convoluzionali con Max pooling -> 2x Fully Connected Lineari (Per iniziare, da rivedere in seguito).
2) Modello 2 (Shakespare): RNN o LSTM (1-2x Layer LSTM -> 1x Fully Connected per mappare i caratteri).

# Funzionamento:
1) Inizialmente client singolo. Il client chiede all utente K client e successivamente contatta il service registry che gli fornisce la lista degli IP dei container gia inizializzati. Se non ve ne sono abbastanza vengono inizializzati degli altri. Ritorna una lista con tutti i K container.
2) Il database di test viene diviso tra i container. Ogni container scarica la propria parte da S3 (Potrebbe richiedere shuffle e/o pre-processamento).
3) Ogni client addestra il proprio modello localmente sulla propria parte di date e condivide i pesi (No API).
4) Ogni client riceve i pesi degli altri e li incorpora al proprio modello (No API).

# Ruoli:
Cristiano: comunicazione GRPC.
Federico: modello.
Alfonso: infrastruttura AWS.
