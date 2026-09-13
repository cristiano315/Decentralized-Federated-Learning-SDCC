#File for utility functions
import hashlib
import io
import math
import torch

#Weight serialization
def get_weights_as_bytes(model):
    """Extract weights from the model and convert them to bytes for gRPC"""
    buffer = io.BytesIO()
    
    # Get the full dictionary of all weights
    full_state_dict = model.state_dict()

    # Filter out the heavy, frozen bert layers. 
    fc_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith('fc.')}
    
    # Save only the classification head
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
    """Calculate SHA-256 hash of pytorch model state_dicts."""

    buffer = io.BytesIO()
    state_dict = model.state_dict()
    
    if only_trainable:
        # Consider only the classification head (FC)
        state_dict = {k: v.cpu() for k, v in state_dict.items() if k.startswith('fc.')}
    else:
        state_dict = {k: v.cpu() for k, v in state_dict.items()}

    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()