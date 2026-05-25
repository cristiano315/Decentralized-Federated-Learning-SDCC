# aggregator.py
import torch
import copy

def apply_fedavg(global_model, received_weights_list):
    """
    Applies the Federated Averaging algorithm.
    
    Args:
        global_model: The local PyTorch model instance to be updated.
        received_weights_list: A list of tuples/dicts containing 
                               (state_dict, num_samples) from peers.
    """
    if not received_weights_list:
        return global_model

    # Calculate total samples to use as the denominator for the weighted average
    total_samples = sum([entry['num_samples'] for entry in received_weights_list])
    
    # Initialize a new state dict with zeros, matching the model's architecture
    averaged_weights = copy.deepcopy(global_model.state_dict())
    for key in averaged_weights.keys():
        averaged_weights[key] = torch.zeros_like(averaged_weights[key], dtype=torch.float32)

    # Perform the weighted average
    for entry in received_weights_list:
        peer_weights = entry['state_dict']
        peer_samples = entry['num_samples']
        weight_factor = peer_samples / total_samples

        for key in averaged_weights.keys():
            # Add the weighted contribution from this peer
            averaged_weights[key] += peer_weights[key] * weight_factor

    # Load the new averaged weights into the model
    global_model.load_state_dict(averaged_weights)
    
    return global_model