//MAIN CENTRALIZED SERVICE REGISTRY, TO REFINE

package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"strconv"

	pb "federate-registry/federated"

	utils "federate-registry/utils"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// =====================================================================
// SERVER STRUCT
// =====================================================================

// registryServer implements the RegistryService defined in the .proto file.
type registryServer struct {
	// Embedding this is required by gRPC for forward compatibility
	pb.UnimplementedRegistryServiceServer

	// RWMutex ensures thread-safe access to the map when multiple
	// clients register or discover concurrently.
	mu sync.RWMutex

	// Map to store active nodes. Key: node_id, Value: NodeInfo
	nodes map[string]*pb.NodeInfo

	// Avoids spawning multiple nodes with the same ID in case of respawn
	respawningNodes map[string]bool

	cond *sync.Cond

	currentClientID int

	pendingNodes int
}

// =====================================================================
// RPC METHODS IMPLEMENTATION
// =====================================================================

func (s *registryServer) WaitIdleNodes(x int, requesterID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for {
		idleCount := 0
		for id, node := range s.nodes {
			if id != requesterID && node.Status == "idle" {
				idleCount++
			}
		}

		if idleCount >= x {
			break
		}
		s.cond.Wait()
	}
}

// RegisterNode handles incoming registration requests from Python clients.
func (s *registryServer) RegisterNode(ctx context.Context, req *pb.NodeInfo) (*pb.RegisterResponse, error) {
	// Lock the map for writing to prevent race conditions
	s.mu.Lock()
	defer s.mu.Unlock()

	//WAKE UP any clients sleeping in DiscoverNodes
	defer s.cond.Broadcast()

	// Store the node information in the map
	delete(s.respawningNodes, req.NodeId) // Sblocca il deduping per il futuro
	s.nodes[req.NodeId] = req
	s.pendingNodes -= 1

	//call aws dynamodb to store node
	err := utils.AddNode(req)
	if err != nil {
		// Log the error but DO NOT crash the server
		log.Printf("[ERROR] Error adding node %s to DynamoDB: %v", req.NodeId, err)
		return &pb.RegisterResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to register node %s due to database error.", req.NodeId),
		}, nil
	}

	log.Printf("[REGISTER] Node joined: %s at %s:%d and added to DynamoDB\n ", req.NodeId, req.IpAddress, req.Port)

	return &pb.RegisterResponse{
		Success: true,
		Message: fmt.Sprintf("Node %s successfully registered.", req.NodeId),
	}, nil
}

func (s *registryServer) RegisterRespawnedNode(ctx context.Context, req *pb.NodeInfo) (*pb.RegisterResponse, error) {
    s.mu.Lock()
	delete(s.respawningNodes, req.NodeId) // Sblocca il deduping per il futuro
    s.nodes[req.NodeId] = req
    if s.pendingNodes > 0 {
        s.pendingNodes--
    }
    s.cond.Broadcast()

    // Fai uno snapshot dei peer esistenti (escludendo il nuovo arrivato)
    existingPeers := make([]*pb.NodeInfo, 0, len(s.nodes)-1)
    for id, n := range s.nodes {
        if id != req.NodeId {
            existingPeers = append(existingPeers, n)
        }
    }
    s.mu.Unlock()

    // Salva su DynamoDB
    if err := utils.AddNode(req); err != nil {
        log.Printf("[ERROR] Error adding node %s to DynamoDB: %v", req.NodeId, err)
    }

    log.Printf("[REGISTER] Node joined: %s at %s:%d", req.NodeId, req.IpAddress, req.Port)

    // Notifica in parallelo tutti gli altri nodi che c'è un NUOVO peer disponibile
    var wg sync.WaitGroup
    for _, peer := range existingPeers {
        wg.Add(1)
        go func(p *pb.NodeInfo) {
            defer wg.Done()
            signalNewNode(p, req) // Invia al peer 'p' le info sul nuovo nodo 'req'
        }(peer)
    }
    // Non blocchiamo la risposta gRPC se i ping sono lenti
    go func() {
        wg.Wait()
    }()

    return &pb.RegisterResponse{
        Success: true,
        Message: fmt.Sprintf("Node %s successfully registered.", req.NodeId),
    }, nil
}

// Discover handles requests from nodes asking for the list of peers.
func (s *registryServer) DiscoverNodes(ctx context.Context, req *pb.DiscoverRequest) (*pb.DiscoverResponse, error) {
	// 1. PING DEI NODI IN MEMORIA
	nodesToPing := make([]*pb.NodeInfo, 0, len(s.nodes))
	for id, node := range s.nodes {
		// Pingo SOLO i nodi in stato "idle" ed escludo chi fa la richiesta
		if id != req.NodeId && node.Status == "idle" {
			nodesToPing = append(nodesToPing, node)
		}
	}
	s.mu.RUnlock()

	var wgMem sync.WaitGroup
	var deadMemNodesMu sync.Mutex
	var deadMemNodeIDs []string

	for _, node := range nodesToPing {
		wgMem.Add(1)
		go func(n *pb.NodeInfo) {
			defer wgMem.Done()
			
			// Retry di cortesia prima di dichiarare morto un nodo
			isAlive := false
			for attempt := 1; attempt <= 3; attempt++ {
				if pingNode(n) {
					isAlive = true
					break
				}
				time.Sleep(500 * time.Millisecond)
			}

			if !isAlive {
				log.Printf("[DISCOVERY] In-memory idle peer %s unreachable. Removing.", n.NodeId)
				deadMemNodesMu.Lock()
				deadMemNodeIDs = append(deadMemNodeIDs, n.NodeId)
				deadMemNodesMu.Unlock()

				if err := utils.RemoveNode(n.NodeId); err != nil {
					log.Printf("[ERROR] Failed to remove dead node %s from DynamoDB: %v", n.NodeId, err)
				}
			}
		}(node)
	}
	wgMem.Wait()

	// Pulizia dei nodi non raggiungibili dalla mappa in memoria
	if len(deadMemNodeIDs) > 0 {
		s.mu.Lock()
		for _, id := range deadMemNodeIDs {
			delete(s.nodes, id)
		}
		s.mu.Unlock()
	}

	// 2. CONTEGGIO NODI IDLE IN MEMORIA
	s.mu.RLock()
	currentIdleCount := 0
	for id, node := range s.nodes {
		if id != req.NodeId && node.Status == "idle" {
			currentIdleCount++
		}
	}
	currentPending := s.pendingNodes
	s.mu.RUnlock()

	requiredPeers := int(req.RequestCount)
	missing := requiredPeers - (currentIdleCount + currentPending)

	// 3. RECUPERO NODI SE MANCANTI
	if missing > 0 {
		// Fetch solo dei nodi idle da DynamoDB
		dynamoNodes, err := utils.FetchIdleNodes()
		if err != nil {
			log.Printf("[ERROR] Error fetching idle nodes from DynamoDB: %v", err)
			return nil, fmt.Errorf("failed to fetch idle nodes from DynamoDB")
		}

		var newlyDiscovered []*pb.NodeInfo
		var wgDynamo sync.WaitGroup
		var aliveMu sync.Mutex

		for _, node := range dynamoNodes {
			// Se già presente in memoria, salta il ping
			s.mu.RLock()
			_, alreadyInMem := s.nodes[node.NodeId]
			s.mu.RUnlock()

			if alreadyInMem {
				continue
			}

			wgDynamo.Add(1)
			go func(n *pb.NodeInfo) {
				defer wgDynamo.Done()
				if pingNode(n) {
					aliveMu.Lock()
					newlyDiscovered = append(newlyDiscovered, n)
					aliveMu.Unlock()
				} else {
					log.Printf("[DISCOVERY] Node %s unreachable. Ignoring and removing from DynamoDB.", n.NodeId)
					utils.RemoveNode(n.NodeId)
				}
			}(node)
		}

		wgDynamo.Wait()

		// Aggiunge i nuovi nodi idle scoperti alla mappa in memoria
		s.mu.Lock()
		for _, node := range newlyDiscovered {
			s.nodes[node.NodeId] = node
		}

		// Ricalcola quanti nodi mancano ancora
		idleCountNow := 0
		for id, node := range s.nodes {
			if id != req.NodeId && node.Status == "idle" {
				idleCountNow++
			}
		}
		missingNow := requiredPeers - (idleCountNow + s.pendingNodes)
		s.mu.Unlock()

		// Se mancano ancora nodi, ne istanzia di nuovi tramite ECS
		if missingNow > 0 {
			s.raiseRequiredNodes(missingNow)
		}
	}

	// 4. ATTESA CHE CI SIANO ABBASTANZA NODI IDLE
	s.WaitIdleNodes(requiredPeers, req.NodeId)

	// 5. PREPARAZIONE DELLA LISTA DEI PEER (SOLO NODI IDLE)
	var peerList []*pb.NodeInfo
	s.mu.RLock()
	for _, node := range s.nodes {
		// Esclude lo starter richiedente e include SOLAMENTE i nodi con stato "idle"
		if node.NodeId != req.NodeId && node.Status == "idle" {
			peerList = append(peerList, node)
		}
	}
	s.mu.RUnlock()

	log.Printf("[DISCOVERY] Node %s requested peers. Returning %d IDLE peers.", req.NodeId, len(peerList))

	return &pb.DiscoverResponse{
		Nodes: peerList,
	}, nil
}

// ChangeNodeStatus aggiorna lo stato del nodo ("idle" o "working") in memoria e su DynamoDB.
func (s *registryServer) ChangeNodeStatus(ctx context.Context, req *pb.ChangeStatusRequest) (*pb.Ack, error) {
	s.mu.Lock()
	
	// 1. Verifichiamo se il nodo esiste nella mappa in memoria
	node, exists := s.nodes[req.NodeId]
	if !exists {
		s.mu.Unlock()
		log.Printf("[WARNING] tentato cambio stato per nodo inesistente: %s\n", req.NodeId)
		return &pb.Ack{
			Success: false,
			Message: fmt.Sprintf("Node %s not found in registry.", req.NodeId),
		}, nil
	}

	// 2. Aggiorniamo lo stato nella mappa in memoria
	oldStatus := node.Status
	node.Status = req.NewStatus
	
	// Risvegliamo eventuali goroutine in attesa (es. in DiscoverNodes o WaitNodes)
	s.cond.Broadcast()
	s.mu.Unlock()

	log.Printf("[STATUS_CHANGE] Node %s changed status from '%s' to '%s'\n", req.NodeId, oldStatus, req.NewStatus)

	// 3. Aggiorniamo lo stato su DynamoDB tramite la funzione in utils
	err := utils.ChangeStatus(req.NodeId, req.NewStatus)
	if err != nil {
		log.Printf("[ERROR] Failed to update status in DynamoDB for node %s: %v\n", req.NodeId, err)
		// Non facciamo fallire la chiamata gRPC se DynamoDB ha un ritardo, ma notifichiamo l'errore nel log
		return &pb.Ack{
			Success: true,
			Message: fmt.Sprintf("Node %s status updated in memory to '%s', but DynamoDB update failed.", req.NodeId, req.NewStatus),
		}, nil
	}

	return &pb.Ack{
		Success: true,
		Message: fmt.Sprintf("Node %s status successfully updated to '%s'.", req.NodeId, req.NewStatus),
	}, nil
}

// UnregisterNode allows nodes to gracefully leave the registry.
func (s *registryServer) UnregisterNode(ctx context.Context, req *pb.NodeInfo) (*pb.Ack, error) {
	// Lock the map for writing to prevent race conditions
	s.mu.Lock()
	defer s.mu.Unlock()
	defer s.cond.Broadcast() // Notify any waiting goroutines that a node has left

	// Remove the node from the in-memory map
	delete(s.nodes, req.NodeId)

	// Remove the node from dynamoDB and destroy it. NOT NECESSARY TO DESTROY IT IF NOT LEFT ON WAIT.
	err := utils.RemoveNode(req.NodeId)
	if err != nil {
		log.Printf("[ERROR] Error removing node %s from DynamoDB: %v", req.NodeId, err)
	} else {
		log.Printf("[INFO] Node %s removed from DynamoDB.", req.NodeId)
	}

	log.Printf("[UNREGISTER] Node left: %s at %s:%d\n", req.NodeId, req.IpAddress, req.Port)

	return &pb.Ack{
		Success: true,
		Message: fmt.Sprintf("Node %s successfully unregistered.", req.NodeId),
	}, nil
}

// SignalUnresponsiveNode allows nodes to signal the registry that a peer is unresponsive.
func (s *registryServer) SignalUnresponsiveNode(ctx context.Context, req *pb.FullNodeInfo) (*pb.Ack, error) {
	s.mu.Lock()

	// 1. VERIFICA DEDUPING: Se il nodo è già stato rimosso o è già in fase di respawn, ignora!
	_, existsInMap := s.nodes[req.NodeId]
	alreadyRespawning := s.respawningNodes[req.NodeId]

	if !existsInMap || alreadyRespawning {
		s.mu.Unlock()
		log.Printf("[SIGNAL_UNRESPONSIVE] Node %s already processed or being respawned. Ignoring duplicate signal.", req.NodeId)
		return &pb.Ack{
			Success: true,
			Message: fmt.Sprintf("Signal for node %s ignored (already processing).", req.NodeId),
		}, nil
	}

	// 2. Segna il nodo come "in fase di respawn" e rimuovilo dalla mappa
	s.respawningNodes[req.NodeId] = true
	delete(s.nodes, req.NodeId)
	s.pendingNodes++
	s.cond.Broadcast()
	s.mu.Unlock()

	// Remove the node from dynamoDB
	err := utils.RemoveNode(req.NodeId)
	if err != nil {
		log.Printf("[ERROR] Error removing node %s from DynamoDB: %v", req.NodeId, err)
	} else {
		log.Printf("[INFO] Node %s removed from DynamoDB.", req.NodeId)
	}

	log.Printf("[SIGNAL_UNRESPONSIVE] Node is unresponsive: %s at %s:%d\n", req.NodeId, req.IpAddress, req.Port)

	// Create a new node to replace the unresponsive one
	go s.raiseSpecificNode(
		req.NodeId,
		int(req.RequiredNodes),
		int(req.Port),
		int(req.TotalRounds),
		int(req.CurrentRound),
		int(req.MaxDiscoveryRetries),
		int(req.WeightWaitTimeoutSeconds),
	)

	// Signal all nodes that a node has been removed. It will be done when the new node is raised and registered, so it can be signaled to all nodes.

	return &pb.Ack{
		Success: true,
		Message: fmt.Sprintf("Node %s successfully signaled as unresponsive.", req.NodeId),
	}, nil
}

func (s *registryServer) raiseRequiredNodes(required int) {
	local := utils.GetFullLocalAdress()

	clientNumber := required

	peersRequired := 5

	for range clientNumber {

		s.mu.Lock()
		s.currentClientID++
		env := []utils.EnvVar{
			{Key: "CLIENT_ID", Value: strconv.Itoa(s.currentClientID)},
			{Key: "TRAINING_NODES", Value: strconv.Itoa(clientNumber)},
			{Key: "PORT", Value: "50051"},
			{Key: "TOTAL_ROUNDS", Value: "3"},
			{Key: "START_ROUND", Value: "0"},
			{Key: "NUM_PEERS_REQUIRED", Value: strconv.Itoa(peersRequired)},
			{Key: "MAX_DISCOVERY_RETRIES", Value: "5"},
			{Key: "WEIGHT_WAIT_TIMEOUT_SECONDS", Value: "1200"},
			{Key: "REGISTRY_ADRESS", Value: local},
			{Key: "RESPAWNED", Value: "false"},
			{Key: "STARTER", Value: "false"},
		}

		err := utils.LaunchTask("federated_cluster", "client_task", 1, "client_container", env)
		if err != nil {
			fmt.Printf("AWS Error: %s\n", err.Error())
		}
		s.pendingNodes += 1
		s.mu.Unlock()

	}

	fmt.Printf("Raised required nodes to %d\n", required)
}

func (s *registryServer) raiseSpecificNode(id string, requiredNodes int, port int, totalRounds int, startRound int, maxDiscoveryRetries int, weightWaitTimeoutSeconds int) {
	local := utils.GetFullLocalAdress()


	env := []utils.EnvVar{
		{Key: "CLIENT_ID", Value: id},
		{Key: "TRAINING_NODES", Value: strconv.Itoa(requiredNodes)},
		{Key: "PORT", Value: strconv.Itoa(port)},
		{Key: "TOTAL_ROUNDS", Value: strconv.Itoa(totalRounds)},
		{Key: "START_ROUND", Value: strconv.Itoa(startRound)},
		{Key: "NUM_PEERS_REQUIRED", Value: strconv.Itoa(requiredNodes)},
		{Key: "MAX_DISCOVERY_RETRIES", Value: strconv.Itoa(maxDiscoveryRetries)},
		{Key: "WEIGHT_WAIT_TIMEOUT_SECONDS", Value: strconv.Itoa(weightWaitTimeoutSeconds)},
		{Key: "REGISTRY_ADRESS", Value: local},
		{Key: "RESPAWNED", Value: "true"},
		{Key: "STARTER", Value: "false"},
	}

	err := utils.LaunchTask("federated_cluster", "client_task", 1, "client_container", env)
	if err != nil {
		log.Printf("[ERROR] AWS Error launching respawned node %s: %v\n", id, err)
		// Se il lancio fallisce, ripristiniamo il conteggio dei nodi pending
		s.mu.Lock()
		if s.pendingNodes > 0 {
			s.pendingNodes--
		}
		s.mu.Unlock()
		return
	}

	fmt.Printf("Raised node with id %s\n", id)
}

func pingNode(node *pb.NodeInfo) bool {
	address := fmt.Sprintf("%s:%d", node.IpAddress, node.Port)

	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return false
	}
	defer conn.Close()

	client := pb.NewFederatedNodeClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req := &pb.NodeInfo{
		NodeId:    node.NodeId,
		IpAddress: node.IpAddress,
		Port:      node.Port,
		Status:    node.Status,
	}

	resp, err := client.Ping(ctx, req)
	if err != nil {
		return false
	}

	return resp.Success
}

func signalNewNode(oldNode *pb.NodeInfo, newNode *pb.NodeInfo) bool {
	address := fmt.Sprintf("%s:%d", oldNode.IpAddress, oldNode.Port)

	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return false
	}
	defer conn.Close()

	client := pb.NewFederatedNodeClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req := &pb.NodeInfo{
		NodeId:    newNode.NodeId,
		IpAddress: newNode.IpAddress,
		Port:      newNode.Port,
		Status:    newNode.Status,
	}

	resp, err := client.NotifyUnresponsiveNode(ctx, req)
	if err != nil {
		return false
	}

	return resp.Success
}

// =====================================================================
// MAIN SERVER SETUP
// =====================================================================

func main() {

	// 1. Define the port the Go server will listen on
	port := ":8080"
	lis, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("[FATAL] Failed to listen on port %s: %v", port, err)
	}

	// 2. Create a new gRPC server instance
	grpcServer := grpc.NewServer()

	// 3. Instantiate our custom server struct with an initialized map
	myServer := &registryServer{
		nodes: make(map[string]*pb.NodeInfo),
		respawningNodes: make(map[string]bool),
	}
	myServer.cond = sync.NewCond(&myServer.mu)

	// 4. Register our server with the gRPC framework
	pb.RegisterRegistryServiceServer(grpcServer, myServer)

	// Raise the N nodes to start the training uncomment if needed
	//myServer.raiseRequiredNodes(1)

	// 5. Start serving incoming requests
	log.Printf("[INFO] Go Service Registry is running and listening on port %s...\n", port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("[FATAL] Failed to serve gRPC server: %v", err)
	}
}
