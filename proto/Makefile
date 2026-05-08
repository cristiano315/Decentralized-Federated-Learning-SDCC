# === VARIABLES ===
PROTO_FILE = federated.proto
CLIENT_DIR = ../client
SERVER_DIR = ../server
REGISTRY_DIR = ../registry/src

all: go python

# === GO TARGET ===
go:
	@echo "Compiling for Go (Registry)..."
	protoc --go_out=$(REGISTRY_DIR) --go-grpc_out=$(REGISTRY_DIR) $(PROTO_FILE)
	@echo "Go compilation completed."

# === PYTHON TARGET ===
python:
	@echo "Compiling for Python (Client and Server)..."
	python -m grpc_tools.protoc -I. --python_out=$(CLIENT_DIR) --grpc_python_out=$(CLIENT_DIR) $(PROTO_FILE)
	python -m grpc_tools.protoc -I. --python_out=$(SERVER_DIR) --grpc_python_out=$(SERVER_DIR) $(PROTO_FILE)
	@echo "Python compilation completed."

# === CLEANING TARGET ===
# Clean up old files
clean:
	@echo "Removing old generated files..."
	rm -f $(CLIENT_DIR)/*_pb2*.py
	rm -f $(SERVER_DIR)/*_pb2*.py
	rm -f $(REGISTRY_DIR)/*.pb.go
	@echo "Cleanup completed."