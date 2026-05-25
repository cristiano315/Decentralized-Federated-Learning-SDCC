# =====================================================================
# PYTHON CLIENT DOCKERFILE (Location: client/Dockerfile)
# =====================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install make for compiling proto files
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY client/requirements.txt ./client/
RUN pip install --no-cache-dir -r client/requirements.txt

# Copy the proto folder and the client source code from the root context
COPY proto/ ./proto/
COPY client/src/ ./client/src/

# Compile the gRPC files for Python
WORKDIR /app/proto
RUN make python-client

# Move to the client directory where main.py is located
WORKDIR /app/client/src

EXPOSE 50051

# Run the application
CMD ["python", "main.py"]