import torch
import copy
from utils import load_weights_from_bytes

def apply_fedavg(global_model, received_payloads, local_samples):
    """
    Applies the Federated Averaging algorithm, including the local model.
    
    Args:
        global_model: The local PyTorch model instance (just trained).
        received_payloads: List of gRPC WeightPayload objects from peers.
        local_samples: Number of samples the local model was trained on.
    """
    # 1. Start the list with our OWN local model
    weights_list = [{
        'state_dict': copy.deepcopy(global_model.state_dict()),
        'num_samples': local_samples
    }]
    
    # 2. Add all received peer models (Deserializing them from bytes)
    for payload in received_payloads:
        # Extract bytes and decode into a PyTorch state_dict
        peer_state_dict = load_weights_from_bytes(payload.model_weights)
        weights_list.append({
            'state_dict': peer_state_dict,
            'num_samples': payload.num_samples
        })

    # 3. Calculate total samples for the weighted average denominator
    total_samples = sum([entry['num_samples'] for entry in weights_list])
    
    # 4. Initialize a new state dict with zeros
    averaged_weights = copy.deepcopy(global_model.state_dict())
    for key in averaged_weights.keys():
        averaged_weights[key] = torch.zeros_like(averaged_weights[key], dtype=torch.float32)

    # 5. Perform the weighted average
    for entry in weights_list:
        peer_weights = entry['state_dict']
        peer_samples = entry['num_samples']
        weight_factor = peer_samples / total_samples

        for key in averaged_weights.keys():
            averaged_weights[key] += peer_weights[key] * weight_factor

    # 6. Load the new averaged weights into the model
    global_model.load_state_dict(averaged_weights)
    
    return global_model