#File for utility functions
import io
import math
import torch

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
    