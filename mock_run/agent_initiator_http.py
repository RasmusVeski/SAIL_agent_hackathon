import torch
import os
import logging
import copy
import httpx
import json
import time

# Import all our existing utils
from app.services.utils.model import FoodClassifier
from app.services.utils.data_loader import create_dataloader
from app.services.utils.training import train, evaluate
from app.services.utils.logger_setup import setup_logging
from app.services.utils.federated_utils import merge_payloads, update_global_model
from app.services.utils.payload_utils import (
    get_trainable_state_dict, 
    serialize_payload_to_b64, 
    deserialize_payload_from_b64
)

def main():
    """
    AGENT INITIATOR (CLIENT)
    - Connects to the responder via HTTP.
    - Trains on its local data.
    - Initiates the weight exchange.
    """
    # --- 1. Setup ---
    agent_id = "INITIATOR (Agent A)"
    setup_logging(log_dir="logs", log_file="agent_initiator.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--- {agent_id} ---")
    
    # Use httpx.Client for persistent connections
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=120.0)

    # --- 2. Define Paths & HPs ---
    BASE_DIR = "./app/sharded_data"
    CLIENT_DATA_DIR = os.path.join(BASE_DIR, "client_0") # This agent uses client_0
    VAL_DIR = os.path.join(BASE_DIR, "test1")

    NUM_ROUNDS = 20
    agent_hps = {
        'epochs': 1, 
        'learning_rate': 1e-4, 
        'weight_decay': 1e-3, 
        'mu': 0.5,
        'val_frequency': 1,       
        'lr_scheduler_step_size': 999 
    }

    # --- 3. Load Data ---
    train_loader = create_dataloader(CLIENT_DATA_DIR, 32, shuffle=True, num_workers=0)
    val_loader = create_dataloader(VAL_DIR, 32, shuffle=False, num_workers=0)
    if not train_loader or not val_loader:
        logging.error("Failed to load data. Exiting.")
        return

    # --- 4. Handshake ---
    try:
        logging.info("Attempting handshake...")
        response = client.post("/handshake", json={"agent_id": agent_id})
        response.raise_for_status() # Raise an exception for bad status codes
        logging.info(f"Handshake successful. Responder is: {response.json()['responder_id']}")
    except httpx.ConnectError as e:
        logging.error(f"Connection failed. Is agent_responder_app.py running? Error: {e}")
        return
    except httpx.HTTPStatusError as e:
        logging.error(f"Handshake failed with status {e.response.status_code}: {e.response.text}")
        return

    # --- 5. Initialize Model & Evaluate ---
    global_model = FoodClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, _, _ = evaluate(global_model, val_loader, device, criterion)
    logging.info(f"Round 0 Global Accuracy: {val_acc:.2f}%")

    # --- 6. Synchronous Training Loop ---
    for round_num in range(NUM_ROUNDS):
        logging.info(f"\n--- {agent_id} | Round {round_num + 1}/{NUM_ROUNDS} ---")
        
        # 1. Train local model
        logging.info("Training local model...")
        local_model, _ = train(
            model=copy.deepcopy(global_model),
            train_loader=train_loader,
            val_loader=None,
            global_model=global_model,
            device=device,
            **agent_hps
        )

        # 2. Prepare *our* payload
        my_payload = get_trainable_state_dict(local_model)
        my_payload_b64 = serialize_payload_to_b64(my_payload)

        if not my_payload_b64:
            logging.error("Failed to serialize payload. Skipping round.")
            continue
            
        # 3. Send *our* payload and get *their* payload in one call
        logging.info("Sending initiator payload to /exchange_weights...")
        try:
            response = client.post(
                "/exchange_weights",
                json={
                    "agent_id": agent_id,
                    "payload_b64": my_payload_b64
                }
            )
            response.raise_for_status()
            
            # 4. Deserialize and verify *their* payload from the response
            logging.info("Received responder payload.")
            response_data = response.json()
            responder_payload = deserialize_payload_from_b64(
                response_data["payload_b64"], global_model
            )
            
            if not responder_payload:
                logging.error("Received corrupt payload from responder. Skipping merge.")
                continue

        except httpx.HTTPStatusError as e:
            logging.error(f"Weight exchange failed: {e.response.status_code} {e.response.text}")
            continue
        except Exception as e:
            logging.error(f"An error occurred during exchange: {e}")
            time.sleep(1) # Wait a second before retrying
            continue
            
        # 5. Merge! (We have both payloads)
        logging.info("Merging payloads...")
        merged_payload = merge_payloads(my_payload, responder_payload, alpha=0.5)

        # 6. Update our global model
        update_global_model(global_model, merged_payload)
        
        # 7. Evaluate
        val_loss, val_acc, _, _ = evaluate(global_model, val_loader, device, criterion)
        logging.info(f"Round {round_num + 1} Merged Accuracy: {val_acc:.2f}%")
        
    logging.info(f"--- {agent_id} Finished ---")
    client.close()

if __name__ == "__main__":
    main()