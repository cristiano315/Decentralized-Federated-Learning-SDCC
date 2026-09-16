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

	pb "federate-registry-centralized/federated"

	utils "federate-registry-centralized/utils"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// =====================================================================
// SERVER STRUCT
// =====================================================================

type CoordinatorCallback struct {
	IP   string
	Port int
}

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
	
	// Track nodes that have completed their tasks and unregistered gracefully
	completedNodes  map[string]bool

	coordinatorCallbacks map[string]CoordinatorCallback

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

	//Wake up any clients sleeping in DiscoverNodes
	defer s.cond.Broadcast()

	// Store the node information in the map
	delete(s.respawningNodes, req.NodeId) // Unlock deduping for future registrations
	s.nodes[req.NodeId] = req
	s.pendingNodes -= 1

	//call aws dynamodb to store node
	err := utils.AddNode(req)
	if err != nil {
		log.Printf("Error adding node %s to DynamoDB: %v", req.NodeId, err)
		return &pb.RegisterResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to register node %s due to database error.", req.NodeId),
		}, nil
	}

	log.Printf("Node joined: %s at %s:%d and added to DynamoDB\n ", req.NodeId, req.IpAddress, req.Port)

	return &pb.RegisterResponse{
		Success: true,
		Message: fmt.Sprintf("Node %s successfully registered.", req.NodeId),
	}, nil
}

func (s *registryServer) RegisterRespawnedNode(ctx context.Context, req *pb.NodeInfo) (*pb.RegisterResponse, error) {
	s.mu.Lock()
	delete(s.respawningNodes, req.NodeId) // Unlock deduping for future registrations
	s.nodes[req.NodeId] = req
	if s.pendingNodes > 0 {
		s.pendingNodes--
	}
	s.cond.Broadcast()

	// Get coordinator
	cb, hasCoordinator := s.coordinatorCallbacks[req.NodeId]
	if hasCoordinator {
		delete(s.coordinatorCallbacks, req.NodeId) // Consuma il callback
	}
	s.mu.Unlock()

	// Save on DynamoDB
	if err := utils.AddNode(req); err != nil {
		log.Printf("Error adding node %s to DynamoDB: %v", req.NodeId, err)
	}

	log.Printf("Node joined: %s at %s:%d", req.NodeId, req.IpAddress, req.Port)

	// Notify the coordinator
	if hasCoordinator {
		go func(coordIP string, coordPort int, respawned *pb.NodeInfo) {
			coordAddress := fmt.Sprintf("%s:%d", coordIP, coordPort)
			log.Printf("Notifying coordinator at %s of node respawn %s...", coordAddress, respawned.NodeId)

			conn, err := grpc.NewClient(coordAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				log.Printf("Error connecting gRPC to coordinator %s: %v", coordAddress, err)
				return
			}
			defer conn.Close()

			client := pb.NewFederatedServerClient(conn)
			ctxTimeout, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()

			_, err = client.NotifyUnresponsiveNode(ctxTimeout, respawned)
			if err != nil {
				log.Printf("Error calling NotifyUnresponsiveNode on coordinator %s: %v", coordAddress, err)
			} else {
				log.Printf("Coordinator %s notified successfully for node %s.", coordAddress, respawned.NodeId)
			}
		}(cb.IP, cb.Port, req)
	}

	return &pb.RegisterResponse{
		Success: true,
		Message: fmt.Sprintf("Node %s successfully registered.", req.NodeId),
	}, nil
}

// Discover handles requests from nodes asking for the list of idle peers.
func (s *registryServer) DiscoverNodes(ctx context.Context, req *pb.DiscoverRequest) (*pb.DiscoverResponse, error) {
	// Ping nodes in memory
	s.mu.RLock()
	nodesToPing := make([]*pb.NodeInfo, 0, len(s.nodes))
	for id, node := range s.nodes {
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

			isAlive := false
			for attempt := 1; attempt <= 3; attempt++ {
				if pingNode(n) {
					isAlive = true
					break
				}
				time.Sleep(500 * time.Millisecond)
			}

			if !isAlive {
				log.Printf("In-memory idle peer %s unreachable. Removing.", n.NodeId)
				deadMemNodesMu.Lock()
				deadMemNodeIDs = append(deadMemNodeIDs, n.NodeId)
				deadMemNodesMu.Unlock()

				if err := utils.RemoveNode(n.NodeId); err != nil {
					log.Printf("Failed to remove dead node %s from DynamoDB: %v", n.NodeId, err)
				}
			}
		}(node)
	}
	wgMem.Wait()

	// Cleanup dead nodes
	if len(deadMemNodeIDs) > 0 {
		s.mu.Lock()
		for _, id := range deadMemNodeIDs {
			delete(s.nodes, id)
		}
		s.mu.Unlock()
	}

	// Count idle and pending
	s.mu.Lock()
	currentIdleCount := 0
	for id, node := range s.nodes {
		if id != req.NodeId && node.Status == "idle" {
			currentIdleCount++
		}
	}

	if s.pendingNodes < 0 {
		s.pendingNodes = 0
	}

	requiredPeers := int(req.RequestCount)
	missing := requiredPeers - (currentIdleCount + s.pendingNodes)
	s.mu.Unlock()

	// Fetch from DynamoDB if missing
	if missing > 0 {
		dynamoNodes, err := utils.FetchIdleNodes()
		if err == nil && len(dynamoNodes) > 0 {
			var newlyDiscovered []*pb.NodeInfo
			var wgDynamo sync.WaitGroup
			var aliveMu sync.Mutex

			for _, node := range dynamoNodes {
				s.mu.RLock()
				_, alreadyInMem := s.nodes[node.NodeId]
				s.mu.RUnlock()

				if alreadyInMem || node.NodeId == req.NodeId {
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
						log.Printf("Node %s unreachable in DynamoDB. Removing.", n.NodeId)
						utils.RemoveNode(n.NodeId)
					}
				}(node)
			}
			wgDynamo.Wait()

			s.mu.Lock()
			for _, node := range newlyDiscovered {
				s.nodes[node.NodeId] = node
			}
			s.mu.Unlock()
		}

		// Recalculate after DynamoDB
		s.mu.Lock()
		idleCountNow := 0
		for id, node := range s.nodes {
			if id != req.NodeId && node.Status == "idle" {
				idleCountNow++
			}
		}
		missingNow := requiredPeers - (idleCountNow + s.pendingNodes)

		if missingNow > 0 {
			log.Printf("Required IDLE: %d. Found IDLE: %d. Pending: %d. Raising missing: %d",
				requiredPeers, idleCountNow, s.pendingNodes, missingNow)
			s.mu.Unlock()

			go s.raiseRequiredNodes(missingNow)
		} else {
			s.mu.Unlock()
		}
	}

	// Wait for nodes to register and become idle
	s.WaitIdleNodes(requiredPeers, req.NodeId)

	// Return peer list
	var peerList []*pb.NodeInfo
	s.mu.RLock()
	for _, node := range s.nodes {
		if node.NodeId != req.NodeId && node.Status == "idle" {
			peerList = append(peerList, node)
		}
	}
	s.mu.RUnlock()

	log.Printf("Node %s requested %d peers. Returning %d IDLE peers.", req.NodeId, requiredPeers, len(peerList))

	return &pb.DiscoverResponse{
		Nodes: peerList,
	}, nil
}

// ChangeNodeStatus updates node status ("idle" or "working") in memory and on DynamoDB.
func (s *registryServer) ChangeNodeStatus(ctx context.Context, req *pb.ChangeStatusRequest) (*pb.Ack, error) {
	s.mu.Lock()

	// Check if node already available in memory
	node, exists := s.nodes[req.NodeId]
	if !exists {
		s.mu.Unlock()
		log.Printf("Warning: Attempted to change status for non-existent node: %s\n", req.NodeId)
		return &pb.Ack{
			Success: false,
			Message: fmt.Sprintf("Node %s not found in registry.", req.NodeId),
		}, nil
	}

	// Update the status in memory
	oldStatus := node.Status
	node.Status = req.NewStatus

	// Wake up waiting goroutines
	s.cond.Broadcast()
	s.mu.Unlock()

	log.Printf("Node %s changed status from '%s' to '%s'\n", req.NodeId, oldStatus, req.NewStatus)

	// Update status in DynamoDB
	err := utils.ChangeStatus(req.NodeId, req.NewStatus)
	if err != nil {
		log.Printf("Error: Failed to update status in DynamoDB for node %s: %v\n", req.NodeId, err)
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
	s.completedNodes[req.NodeId] = true

	// Remove the node from dynamoDB and destroy it.
	err := utils.RemoveNode(req.NodeId)
	if err != nil {
		log.Printf("Error removing node %s from DynamoDB: %v", req.NodeId, err)
	} else {
		log.Printf("Node %s removed from DynamoDB.", req.NodeId)
	}

	log.Printf("Node left: %s at %s:%d\n", req.NodeId, req.IpAddress, req.Port)

	return &pb.Ack{
		Success: true,
		Message: fmt.Sprintf("Node %s successfully unregistered.", req.NodeId),
	}, nil
}

// SignalUnresponsiveNode allows nodes to signal the registry that a peer is unresponsive.
func (s *registryServer) SignalUnresponsiveNodeCoordinator(ctx context.Context, req *pb.DeadNodeInfo) (*pb.Ack, error) {
	s.mu.Lock()

	// Avoid recreating if node finished training and unregistered gracefully
	if s.completedNodes[req.NodeId] {
		s.mu.Unlock()
		log.Printf("Node %s has already completed and unregistered gracefully. Ignoring.", req.NodeId)
		return &pb.Ack{
			Success: true,
			Message: fmt.Sprintf("Node %s has already completed.", req.NodeId),
		}, nil
	}

	// Deduping
	nodeInfo, existsInMap := s.nodes[req.NodeId]
	if !existsInMap || s.respawningNodes[req.NodeId] {
		s.mu.Unlock()
		log.Printf("Node %s already processed or being respawned. Ignoring duplicate signal.", req.NodeId)
		return &pb.Ack{
			Success: true,
			Message: fmt.Sprintf("Signal for node %s ignored (already processing).", req.NodeId),
		}, nil
	}

	s.respawningNodes[req.NodeId] = true
	s.mu.Unlock()

	// Ping the node
	log.Printf("Peer signaled %s as unresponsive. Registry is verifying with a direct ping...", req.NodeId)
	targetNode := &pb.NodeInfo{
		NodeId:    req.NodeId,
		IpAddress: req.IpAddress,
		Port:      req.Port,
		Status:    nodeInfo.Status,
	}

	isAlive := pingNode(targetNode)

	s.mu.Lock()
	if isAlive {
		// Stop respawn and remove from respawningNodes
		delete(s.respawningNodes, req.NodeId)
		s.mu.Unlock()
		log.Printf("Node %s is actually ALIVE. False alarm, skipping respawn.", req.NodeId)
		return &pb.Ack{
			Success: false,
			Message: fmt.Sprintf("Node %s is alive and reachable by registry. Respawn aborted.", req.NodeId),
		}, nil
	}

	// Node is confirmed unresponsive, proceed with respawn
	delete(s.nodes, req.NodeId)
	s.pendingNodes++

	if req.CoordinatorIp != "" && req.CoordinatorPort != 0 {
		s.coordinatorCallbacks[req.NodeId] = CoordinatorCallback{
			IP:   req.CoordinatorIp,
			Port: int(req.CoordinatorPort),
		}
		log.Printf("Associato coordinatore %s:%d al worker %s in attesa di respawn", req.CoordinatorIp, req.CoordinatorPort, req.NodeId)
	}

	s.cond.Broadcast()
	s.mu.Unlock()

	// Remove the node from dynamoDB
	err := utils.RemoveNode(req.NodeId)
	if err != nil {
		log.Printf("Error: Failed to remove node %s from DynamoDB: %v", req.NodeId, err)
	} else {
		log.Printf("Node %s removed from DynamoDB.", req.NodeId)
	}

	log.Printf("Node is unresponsive: %s at %s:%d\n", req.NodeId, req.IpAddress, req.Port)

	// Create a new node to replace the unresponsive one
	go s.raiseSpecificNode(
		req.NodeId,
		int(req.RequiredNodes),
		int(req.Port),
		int(req.TotalRounds),
		int(req.CurrentRound),
		int(req.MaxDiscoveryRetries),
		int(req.WeightWaitTimeoutSeconds),
		float32(req.TrainingSetPercentage),
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
			{Key: "WEIGHT_WAIT_TIMEOUT_SECONDS", Value: "180"},
			{Key: "REGISTRY_ADRESS", Value: local},
			{Key: "RESPAWNED", Value: "false"},
			{Key: "STARTER", Value: "false"},
			{Key: "TRAINING_SET_PERCENTAGE", Value: "0.7"},
		}

		err := utils.LaunchTask("centralized-cluster", "client_task_centralized", 1, "Main", env)
		if err != nil {
			fmt.Printf("AWS Error: %s\n", err.Error())
		}
		s.pendingNodes += 1
		s.mu.Unlock()

	}

	fmt.Printf("Raised %d\n nodes", required)
}

func (s *registryServer) raiseSpecificNode(id string, requiredNodes int, port int, totalRounds int, startRound int, maxDiscoveryRetries int, weightWaitTimeoutSeconds int, trainingSetPercentage float32) {
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
		{Key: "TRAINING_SET_PERCENTAGE", Value: strconv.FormatFloat(float64(trainingSetPercentage), 'f', -1, 32)},
	}

	err := utils.LaunchTask("centralized-cluster", "client_task_centralized", 1, "Main", env)
	if err != nil {
		log.Printf("AWS Error launching respawned node %s: %v\n", id, err)
		// If the launch fails, decrement pendingNodes to avoid deadlock in WaitIdleNodes
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
		log.Printf("Error notifying unresponsive node %s: %v", address, err)
		return false
	}

	return resp.Success
}

// =====================================================================
// MAIN SERVER SETUP
// =====================================================================

func main() {

	// Define the port the Go server will listen on
	port := ":8080"
	lis, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("Failed to listen on port %s: %v", port, err)
	}

	// Create a new gRPC server instance
	grpcServer := grpc.NewServer()

	// Instantiate custom server struct
	myServer := &registryServer{
		nodes:           make(map[string]*pb.NodeInfo),
		respawningNodes: make(map[string]bool),
		completedNodes: make(map[string]bool),
		coordinatorCallbacks: make(map[string]CoordinatorCallback),
		currentClientID: 1,
	}
	myServer.cond = sync.NewCond(&myServer.mu)

	// Register server
	pb.RegisterRegistryServiceServer(grpcServer, myServer)

	// Start serving incoming requests
	log.Printf("Go Service Registry is running and listening on port %s with address %s...\n", port, utils.GetFullLocalAdress())
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve gRPC server: %v", err)
	}
}
