# =====================================================================
# BUILDER
# =====================================================================
FROM golang:1.26-alpine AS builder

# Install Protocol Buffers and Go plugins
RUN apk add --no-cache protobuf make
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest \
    && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

ENV PATH="$PATH:/go/bin"

# Set working directory
WORKDIR /app

# Copy Go module files first for caching
COPY registry/src/go.mod registry/src/go.sum ./registry/src/
WORKDIR /app/registry/src
RUN go mod download

# Copy the rest of the project (proto and src)
WORKDIR /app
COPY proto/ ./proto/
COPY registry/src/ ./registry/src/

WORKDIR /app/proto
RUN make go

# Build the executable
WORKDIR /app/registry/src

# CGO_ENABLED=0 ensures it runs on any Linux environment without C dependencies
RUN CGO_ENABLED=0 GOOS=linux go build -o registry-server ./main.go



# =====================================================================
# FINAL CONTAINER
# =====================================================================
FROM alpine:latest

# Add CA certificates for external outbound HTTPS calls
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the compiled binary from the builder stage
COPY --from=builder /app/registry/src/registry-server .

# Expose the gRPC port configured in main.go
EXPOSE 8080

# Run the application
CMD ["./registry-server"]