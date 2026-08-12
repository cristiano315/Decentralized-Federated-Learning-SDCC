# === VARIABLES ===
PROTO_FILE = federated.proto
CLIENT_DIR = ../client/src
SERVER_DIR = ../server
REGISTRY_DIR = ../registry/src
REGISTRY_SERVER_DIR = ../registry_centralized/src

all: go go-server python-client python-server

# === GO TARGET ===
go:
	@echo "Compiling for Go (Registry)..."
	protoc --go_out=$(REGISTRY_DIR) --go-grpc_out=$(REGISTRY_DIR) $(PROTO_FILE)
	@echo "Go compilation completed."
go-server:
	@echo "Compiling for Go (Server)..."
	protoc --go_out=$(REGISTRY_SERVER_DIR) --go-grpc_out=$(REGISTRY_SERVER_DIR) $(PROTO_FILE)
	@echo "Go compilation completed."

# === PYTHON TARGET ===
python-client:
	@echo "Compiling for Python (Client)..."
	python -m grpc_tools.protoc -I. --python_out=$(CLIENT_DIR) --grpc_python_out=$(CLIENT_DIR) $(PROTO_FILE)
	@echo "Python compilation completed."
python-server:
	@echo "Compiling for Python (Server)..."
	python -m grpc_tools.protoc -I. --python_out=$(SERVER_DIR) --grpc_python_out=$(SERVER_DIR) $(PROTO_FILE)
	@echo "Python compilation completed."

# === CLEANING TARGET ===
# Clean up old files
clean:
	@echo "Removing old generated files..."
	rm -f $(CLIENT_DIR)/_pb2.py
	rm -f $(SERVER_DIR)/_pb2.py
	rm -f $(REGISTRY_DIR)/*.pb.go
	rm -f $(REGISTRY_SERVER_DIR)/*.pb.go
	@echo "Cleanup completed."