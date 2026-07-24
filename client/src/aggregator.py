import torch
import copy

def apply_fedavg(global_model, received_payloads, local_samples):
    """
    Applies the Federated Averaging algorithm, including the local model.
        Args:
        global_model: The local PyTorch model instance (just trained).
        received_payloads: List of gRPC WeightPayload objects from peers.
        local_samples: Number of samples the local model was trained on.
    """
    # 1. Start the list with our OWN local model's classification head
    # (We must filter our own local dict just like we filtered the peers)
    local_fc_only = {k: v for k, v in global_model.state_dict().items() if k.startswith('fc.')}
    
    weights_list = [{
        'state_dict': copy.deepcopy(local_fc_only),
        'num_samples': local_samples
    }]
    
    # 2. Add all received peer models
    for payload in received_payloads:
        # payload is a Python dict created by rpc_calls.py, NOT a raw gRPC object
        weights_list.append({
            'state_dict': payload['weights'],
            'num_samples': payload['num_samples']
        })

    # 3. Calculate total samples
    total_samples = sum([entry['num_samples'] for entry in weights_list])
    
    # 4. Copy the full global model (including frozen BERT layers)
    averaged_weights = copy.deepcopy(global_model.state_dict())
    
    # Identify ONLY the keys we are averaging (e.g., 'fc.weight', 'fc.bias')
    keys_to_average = weights_list[0]['state_dict'].keys()
    
    # Zero out ONLY the classification head in our averaged_weights tracker
    for key in keys_to_average:
        averaged_weights[key] = torch.zeros_like(averaged_weights[key], dtype=torch.float32)

    # 5. Perform the weighted average on just the target keys
    for entry in weights_list:
        peer_weights = entry['state_dict']
        peer_samples = entry['num_samples']
        weight_factor = peer_samples / total_samples

        for key in keys_to_average:
            averaged_weights[key] += peer_weights[key] * weight_factor

    # 6. Load the new averaged weights into the model
    # (Because averaged_weights still contains the untouched BERT layers, this is safe)
    global_model.load_state_dict(averaged_weights)
    
    return global_model