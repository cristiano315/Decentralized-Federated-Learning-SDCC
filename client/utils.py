#File for utility functions
import io
import torch

#Weight serialization
def get_weights_as_bytes(model):
    """Extract weights from the model and convert them to bytes for gRPC"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()

#Weight deserialization
def load_weights_from_bytes(weights_bytes):
    """Extract weights from bytes received via gRPC and convert them to a PyTorch dictionary"""
    buffer = io.BytesIO(weights_bytes)
    return torch.load(buffer, weights_only=True)