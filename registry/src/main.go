//MAIN CENTRALIZED SERVICE REGISTRY, TO REFINE

package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"sync"

	"strconv"

	pb "federate-registry/federated"

	utils "federate-registry/utils"

	"google.golang.org/grpc"
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

	cond *sync.Cond

	currentClientID int

	pendingNodes int
}

// =====================================================================
// RPC METHODS IMPLEMENTATION
// =====================================================================

func (s *registryServer) WaitNodes(x int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for len(s.nodes) < x {
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

// Discover handles requests from nodes asking for the list of peers.
func (s *registryServer) DiscoverNodes(ctx context.Context, req *pb.DiscoverRequest) (*pb.DiscoverResponse, error) {
	// Read-Lock the map (multiple clients can read simultaneously without blocking each other)
	s.mu.RLock()
	currentNodesLen := len(s.nodes)
    currentPending := s.pendingNodes
	s.mu.RUnlock()

	requiredPeers := int(req.RequestCount)

	// Check if there are enough registered nodes, if not, create them
	if requiredPeers > currentNodesLen + currentPending { 
		// Check if there are enough nodes in DynamoDB
		dynamoNodes, err := utils.FetchActiveNodes()      //CHANGE WITH DYNAMODB CALL
		if err != nil {
			log.Printf("[ERROR] Error fetching active nodes from DynamoDB: %v", err)
			return nil, fmt.Errorf("failed to fetch active nodes from DynamoDB")
		}
		if len(dynamoNodes) < requiredPeers { // Not enough nodes in DynamoDB either
			s.raiseRequiredNodes(requiredPeers - len(dynamoNodes))
		} else {
			for _, node := range dynamoNodes {
				s.mu.Lock()
				s.nodes[node.NodeId] = node
				s.mu.Unlock()
			}
			// REMEMBER TO PING THEM USING THE PING RPC TO CHECK IF THEY ARE ALIVE BEFORE RETURNING THEM
		}
	}


	//  Wait until enough nodes are registered (WaitNodes handles its own locks)
	s.WaitNodes(requiredPeers) // Wait until enough nodes are registered

	//Read-lock ONLY for reading the map
	var peerList []*pb.NodeInfo
	s.mu.RLock()
	for _, node := range s.nodes {
		// Do not include the node that made the request in the returned peer list
		if node.NodeId != req.NodeId {
			peerList = append(peerList, node)
		}
	}
	s.mu.RUnlock()

	log.Printf("[DISCOVERY] Node %s requested peers. Returning %d peers.\n", req.NodeId, len(peerList))

	return &pb.DiscoverResponse{
		Nodes: peerList,
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
			{Key: "NUM_PEERS_REQUIRED", Value: strconv.Itoa(peersRequired)},
			{Key: "MAX_DISCOVERY_RETRIES", Value: "5"},
			{Key: "WEIGHT_WAIT_TIMEOUT_SECONDS", Value: "30"},
			{Key: "REGISTRY_ADRESS", Value: local},
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
	}
	myServer.cond = sync.NewCond(&myServer.mu)

	// 4. Register our server with the gRPC framework
	pb.RegisterRegistryServiceServer(grpcServer, myServer)

	// Raise the N nodes to start the training
	myServer.raiseRequiredNodes(1)

	// 5. Start serving incoming requests
	log.Printf("[INFO] Go Service Registry is running and listening on port %s...\n", port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("[FATAL] Failed to serve gRPC server: %v", err)
	}
}
