import torch
import copy

def apply_fedavg(global_model, received_payloads):
    """
    applies the Federated Averaging algorithm, including the local model
    recieved_payloads has all the models
    global_model is the local model, we use it just for the structure
    returns the new local model
    """

    # add the peer models and the local model in a list, they are all in received payloads
    weights_list = []
    for payload in received_payloads:
        weights_list.append({
            'sender_id': payload['sender_id'],
            'state_dict': payload['weights'],
            'num_samples': payload['num_samples']
        })

    # sort the list using the sender id to make the model calculation as uniform as possible
    weights_list.sort(key=lambda x: str(x['sender_id']))

    # calculate the total number of samples
    total_samples = sum([entry['num_samples'] for entry in weights_list])

    # copy the full global model
    averaged_weights = copy.deepcopy(global_model.state_dict())

    keys_to_average = weights_list[0]['state_dict'].keys()

    # calculate the weighted average
    for key in keys_to_average:
        stacked_tensors = torch.stack([entry['state_dict'][key] for entry in weights_list])

        factors = torch.tensor([entry['num_samples'] / total_samples for entry in weights_list], dtype=torch.float32)

        averaged_weights[key] = torch.sum(stacked_tensors * factors.view(-1, *([1]*(stacked_tensors.dim()-1))), dim=0)

    # load the new weights into the model
    global_model.load_state_dict(averaged_weights)

    return global_model