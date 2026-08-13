#File for utility functions
import hashlib
import io
import json
import math
import os
import torch
import boto3
import json

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
    total_original = sum(
        len(raw_data['user_data'][user]['x']) 
        for user in raw_data['users']
    )
    
    # save values
    training_length = int(total_original * training_set_percentage)
    idx_peers = training_length // num_peers
    samples_per_client = min(idx_peers, max_samples_per_client)

    indexes = {}
    for i, peer in enumerate(peers_list):
        peer_id = peer['id'] if isinstance(peer, dict) else peer

        start = idx_peers * i
        end = start + samples_per_client

        indexes[peer_id] = {'start_idx': int(start), 'end_idx': int(end), 'num_samples': int(samples_per_client), 'truncated_training_length': int(training_length)}
    
    return indexes

