FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install make
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Copy requirements & install
COPY centralized/requirements.txt ./centralized/
RUN pip install --default-timeout=10000 --no-cache-dir -r centralized/requirements.txt

# Copy source code
COPY proto/ ./proto/
COPY centralized/src/ ./centralized/src/

# Compile proto
WORKDIR /app/proto
RUN make python-server

WORKDIR /app/centralized/src

# Porta di default (esponiamo entrambe le porte nel layer dell'immagine)
EXPOSE 50051 50053

# Comando di default (es. il client)
CMD ["python", "client.py"]