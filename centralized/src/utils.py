#File for utility functions
import hashlib
import io
import json
import math
import os
import torch
import boto3
import json
import urllib

#Weight serialization
def get_weights_as_bytes(model):
    """Extract weights from the model and convert them to bytes for gRPC"""
    buffer = io.BytesIO()
    
    # 1. Get the full dictionary of all weights
    full_state_dict = model.state_dict()

    # 2. Filter out the heavy, frozen BERT layers. 
    # We only want keys that start with 'fc.' (like 'fc.weight' and 'fc.bias')
    fc_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith('fc.')}
    
    # 3. Save only the tiny classification head
    torch.save(fc_state_dict, buffer)
    return buffer.getvalue()

#Weight deserialization
def load_weights_from_bytes(weights_bytes):
    """Extract weights from bytes received via gRPC and convert them to a PyTorch dictionary"""
    buffer = io.BytesIO(weights_bytes)
    return torch.load(buffer, weights_only=True)

#Gossip
def calculate_k(num_peers):
    """Calculate the number of peers to gossip to based on the total number of peers"""
    
    N = len(num_peers) + 1 
    
    # k = ceil(ln(N)) + c
    c = 1  # Redundancy constant to ensure some overlap in gossiping
    k = math.ceil(math.log(N)) + c
    return k
    
#hash
def get_model_hash(model: torch.nn.Module, only_trainable: bool = True) -> str:
    """
    Calcola l'hash SHA-256 degli state_dict del modello PyTorch.
    - only_trainable=True: calcola l'hash solo dei parametri addestrati (es. 'fc.').
    - only_trainable=False: calcola l'hash di TUTTI i pesi del modello.
    """
    buffer = io.BytesIO()
    state_dict = model.state_dict()
    
    if only_trainable:
        # Considera solo la testa di classificazione (FC)
        state_dict = {k: v.cpu() for k, v in state_dict.items() if k.startswith('fc.')}
    else:
        state_dict = {k: v.cpu() for k, v in state_dict.items()}

    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()

def get_training_index_list(peers_list, bucket_name, s3_key, training_set_percentage, max_samples_per_client=1000):
    num_peers = len(peers_list)
    if num_peers == 0:
        return {}

    # Load JSON data from S3
    s3 = boto3.client('s3')
    bucket = bucket_name
    key = s3_key
    response = s3.get_object(Bucket=bucket, Key=key)

    # Read the JSON content from the S3 response
    raw_data = json.loads(response['Body'].read().decode('utf-8'))

    # given length
    texts = []
    for user in raw_data['users']:
        for tweet in raw_data['user_data'][user]['x']:
            texts.append(tweet[4])
    
    # save values
    total_original = len(texts)
    total_samples_per_client = total_original // num_peers
    truncated_total_length = total_samples_per_client * num_peers

    indexes = {}
    for i, peer in enumerate(peers_list):
        peer_id = peer['id'] if isinstance(peer, dict) else peer

        start = total_samples_per_client * i

        indexes[peer_id] = {
            'start_idx': int(start), 
            'num_samples': int(total_samples_per_client), 
            'truncated_training_length': int(truncated_total_length)
        }
    
    return indexes

def get_ecs_container_ip():
    metadata_url = os.getenv("ECS_CONTAINER_METADATA_URI_V4")
    
    if not metadata_url:
        return "Variable ECS_CONTAINER_METADATA_URI_V4 not found. Not running on ECS?"

    try:
        with urllib.request.urlopen(metadata_url) as response:
            body = response.read().decode('utf-8')
            metadata = json.loads(body)
            
            networks = metadata.get('Networks', [])
            if networks and 'IPv4Addresses' in networks[0]:
                return networks[0]['IPv4Addresses'][0]
                
    except Exception as e:
        return f"Error reading metadata: {e}"

    return "IP address not found in metadata"

