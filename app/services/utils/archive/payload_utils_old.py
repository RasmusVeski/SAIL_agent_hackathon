import torch
import torch.nn as nn
import logging
from io import BytesIO
import copy

def get_trainable_state_dict(model: nn.Module) -> dict:
    """
    Extracts ONLY the parameters that require gradients into a new state dict.
    This is the "smart" payload to send over the network.
    """
    trainable_state_dict = {}
    
    # Get the names of all parameters that are trainable
    trainable_names = {name for name, param in model.named_parameters() 
                         if param.requires_grad}
    
    if not trainable_names:
        logging.warning("get_trainable_state_dict: No trainable parameters found.")
        return {}

    # Create a new state dict containing only those parameters
    # We detach().clone() to avoid sending the entire computation graph
    full_state_dict = model.state_dict()
    for name, param in full_state_dict.items():
        if name in trainable_names:
            trainable_state_dict[name] = param.detach().clone()
            
    logging.debug(f"Extracted {len(trainable_state_dict)} trainable parameter groups.")
    return trainable_state_dict


def serialize_payload(state_dict: dict) -> bytes:
    """
    Serializes a state dict into bytes using torch.save.
    """
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def deserialize_and_verify_payload(payload_bytes: bytes, reference_model: nn.Module) -> dict:
    """
    Deserializes bytes into a state dict and verifies its keys and shapes
    against a reference model's trainable parameters..
    """
    try:
        buffer = BytesIO(payload_bytes)
        # Use weights_only=True for security, as in your example
        incoming_state_dict = torch.load(buffer, weights_only=True)
    except Exception as e:
        logging.error(f"Failed to deserialize payload: {e}")
        return None

    if not isinstance(incoming_state_dict, dict):
        logging.error("Deserialized payload is not a dictionary.")
        return None
        
    # --- Verification Step ---
    # Get the expected keys and shapes from our local reference model
    reference_trainable_dict = get_trainable_state_dict(reference_model)
    
    # 1. Verify Keys
    if set(incoming_state_dict.keys()) != set(reference_trainable_dict.keys()):
        logging.error("Payload verification failed: Key mismatch.")
        logging.error(f"Expected: {set(reference_trainable_dict.keys())}")
        logging.error(f"Received: {set(incoming_state_dict.keys())}")
        return None
        
    # 2. Verify Shapes
    for key in reference_trainable_dict:
        expected_shape = reference_trainable_dict[key].shape
        received_shape = incoming_state_dict[key].shape
        if received_shape != expected_shape:
            logging.error(f"Payload verification failed: Shape mismatch for key '{key}'.")
            logging.error(f"Expected: {expected_shape}, Received: {received_shape}")
            return None
            
    logging.debug("Payload deserialized and verified successfully.")
    return incoming_state_dict