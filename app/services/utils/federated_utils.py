import torch
import torch.nn as nn
import logging
import copy

def merge_payloads(payload_A: dict, payload_B: dict, alpha: float) -> dict:
    """
    Performs a weighted average on two *state_dict payloads*.
    
    merged_payload = (payload_A * alpha) + (payload_B * (1 - alpha))
    """
    if not payload_A: return payload_B
    if not payload_B: return payload_A
    
    beta = 1.0 - alpha
    merged_payload = {}
    
    for key in payload_A:
        if key in payload_B:
            merged_payload[key] = (payload_A[key] * alpha) + (payload_B[key] * beta)
        else:
            logging.warning(f"Key {key} in A but not B during merge.")
            
    return merged_payload

def update_global_model(global_model: nn.Module, merged_payload: dict):
    """
    Loads the merged payload (which only contains trainable params)
    into the full global_model.
    """
    if not merged_payload:
        logging.warning("Skipping model update, merged payload is empty.")
        return
        
    # Get the full state dict from the global model
    global_state_dict = global_model.state_dict()
    
    # Overwrite only the parameters that are in the merged payload
    # (which are the trainable ones)
    for key, param in merged_payload.items():
        if key in global_state_dict:
            global_state_dict[key].copy_(param)
        else:
            logging.error(f"Key {key} from payload not found in global model.")

    # Load the updated state dict back into the model
    global_model.load_state_dict(global_state_dict)
    logging.debug("Global model updated with merged payload.")