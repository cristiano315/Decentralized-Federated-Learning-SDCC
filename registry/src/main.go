//MAIN CENTRALIZED SERVICE REGISTRY, TO REFINE

package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"sync"

	"google.golang.org/grpc"

	pb "federate-registry/federated"

	utils "federate-registry/utils"
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
}

// =====================================================================
// RPC METHODS IMPLEMENTATION
// =====================================================================

// RegisterNode handles incoming registration requests from Python clients.
func (s *registryServer) RegisterNode(ctx context.Context, req *pb.NodeInfo) (*pb.RegisterResponse, error) {
	// Lock the map for writing to prevent race conditions
	s.mu.Lock()
	defer s.mu.Unlock()

	// Store the node information in the map
	s.nodes[req.NodeId] = req

	//call aws dynamodb to store node
	err := utils.AddNode(req)
	if err != nil {
		log.Fatalf("Error adding node, %v", err)
	}

	log.Printf("[REGISTER] Node joined: %s at %s:%d\n", req.NodeId, req.IpAddress, req.Port)

	return &pb.RegisterResponse{
		Success: true,
		Message: fmt.Sprintf("Node %s successfully registered.", req.NodeId),
	}, nil
}

// Discover handles requests from nodes asking for the list of peers.
func (s *registryServer) Discover(ctx context.Context, req *pb.DiscoverRequest) (*pb.DiscoverResponse, error) {
	// Read-Lock the map (multiple clients can read simultaneously without blocking each other)
	s.mu.RLock()
	defer s.mu.RUnlock()

	var peerList []*pb.NodeInfo
	requiredPeers := int(req.RequestCount)

	// Check if there are enough registered nodes, if not, create them
	if requiredPeers > len(s.nodes) {
		// Check if there are enough nodes in DynamoDB
		dynamoNodes := 20                //CHANGE WITH DYNAMODB CALL
		if dynamoNodes < requiredPeers { // Not enough nodes in DynamoDB either
			// RAISE REQUIRED NODES
		} else {
			// GET REQUIRED NODES FROM DYNAMODB, ADD THEM TO THE LIST AND RETURN THEM
			// REMEMBER TO PING THEM USING THE PING RPC TO CHECK IF THEY ARE ALIVE BEFORE RETURNING THEM
		}
	}

	// Iterate over all registered nodes
	for _, node := range s.nodes {
		// Do not include the node that made the request in the returned peer list
		if node.NodeId != req.NodeId {
			peerList = append(peerList, node)
		}
	}

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

	// Remove the node from the in-memory map
	delete(s.nodes, req.NodeId)

	// Remove the node from dynamoDB and destroy it. NOT NECESSARY TO DESTROY IT IF NOT LEFT ON WAIT.

	log.Printf("[UNREGISTER] Node left: %s at %s:%d\n", req.NodeId, req.IpAddress, req.Port)

	return &pb.Ack{
		Success: true,
		Message: fmt.Sprintf("Node %s successfully unregistered.", req.NodeId),
	}, nil
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

	// 4. Register our server with the gRPC framework
	pb.RegisterRegistryServiceServer(grpcServer, myServer)

	// 5. Start serving incoming requests
	log.Printf("[INFO] Go Service Registry is running and listening on port %s...\n", port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("[FATAL] Failed to serve gRPC server: %v", err)
	}
}
