import torch
import copy

def apply_fedavg(global_model, received_payloads, local_samples, my_id):
    """
    Applies the Federated Averaging algorithm, including the local model.
        Args:
        global_model: The local PyTorch model instance (just trained).
        received_payloads: List of gRPC WeightPayload objects from peers.
        local_samples: Number of samples the local model was trained on.
        my_id: The ID of the current client to ensure deterministic sorting.
    """
    # 1. Start the list with our OWN local model's classification head
    local_fc_only = {k: v.cpu() for k, v in global_model.state_dict().items() if k.startswith('fc.')}
    
    weights_list = [{
        'sender_id': my_id, 
        'state_dict': copy.deepcopy(local_fc_only),
        'num_samples': local_samples
    }]
    
    # 2. Add all received peer models
    for payload in received_payloads:
        weights_list.append({
            'sender_id': payload['sender_id'],
            'state_dict': payload['weights'],
            'num_samples': payload['num_samples']
        })

    # SORT THE LIST BASED ON SENDER_ID
    # This is the single most important step to guarantee bit-level identically hashes.
    weights_list.sort(key=lambda x: str(x['sender_id']))

    # 3. Calculate total samples
    total_samples = sum([entry['num_samples'] for entry in weights_list])
    
    # 4. Copy the full global model (including frozen BERT layers)
    averaged_weights = copy.deepcopy(global_model.state_dict())
    
    keys_to_average = weights_list[0]['state_dict'].keys()
    
    # 5. Perform the weighted average avoiding iterative summation drift
    for key in keys_to_average:
        # Stack all tensors for this specific key from all peers
        stacked_tensors = torch.stack([entry['state_dict'][key] for entry in weights_list])
        
        # Create a tensor of weight factors
        factors = torch.tensor([entry['num_samples'] / total_samples for entry in weights_list], dtype=torch.float32)
        
        # Multiply and sum across the stacked dimension (0)
        # This reduces precision differences compared to iterative +=
        averaged_weights[key] = torch.sum(stacked_tensors * factors.view(-1, *([1]*(stacked_tensors.dim()-1))), dim=0)

    # 6. Load the new averaged weights into the model
    global_model.load_state_dict(averaged_weights)
    
    return global_model