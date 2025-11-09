import torch
import os
import logging
import copy
import pprint
from utils.model import FoodClassifier
from utils.data_loader import create_dataloader
from utils.training import train, evaluate
from utils.logger_setup import setup_logging
from utils.federated_utils import merge_payloads, update_global_model
from utils.payload_utils import get_trainable_state_dict, serialize_payload, deserialize_and_verify_payload

def main():
    """
    Main function to simulate a MULTI-ROUND, HIGH-FREQUENCY merge loop.
    This version is adapted to use the efficient payload workflow.
    """
    # --- 1. Setup ---
    setup_logging(log_dir="logs", log_file="federated_training.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device: {device}")

    # --- 2. Define Paths ---
    BASE_DIR = "./sharded_data"
    CLIENT_0_DIR = os.path.join(BASE_DIR, "client_0")
    CLIENT_1_DIR = os.path.join(BASE_DIR, "client_1")
    VAL_DIR = os.path.join(BASE_DIR, "test1")

    # --- 3. Hyperparameters ---
    NUM_ROUNDS = 10        # Run 10 full rounds of merging
    EPOCHS_PER_ROUND = 1   # Clients ONLY train for 1 epoch before merging
    
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    LR_SCHEDULER_STEP = 999 # Disable LR scheduler for this test
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    MU = 0.5               

    # --- 4. Load Data ---
    logging.info(f"Loading data for Client 0 ({CLIENT_0_DIR})")
    train_loader_0 = create_dataloader(CLIENT_0_DIR, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    logging.info(f"Loading data for Client 1 ({CLIENT_1_DIR})")
    train_loader_1 = create_dataloader(CLIENT_1_DIR, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    logging.info(f"Loading validation data ({VAL_DIR})")
    val_loader = create_dataloader(VAL_DIR, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    if not train_loader_0 or not train_loader_1 or not val_loader:
        logging.error("Failed to load all dataloaders. Exiting.")
        return

    # --- 5. Initialize Global Model ---
    global_model = FoodClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    
    # --- 6. Pre-Training Evaluation ---
    logging.info("--- Evaluating Base Model (Round 0) ---")
    val_loss, val_acc, _, _ = evaluate(global_model, val_loader, device, criterion)
    logging.info(f"Round 0 Accuracy: {val_acc:.2f}%")
    
    # --- 7. Run Federated Learning Loop ---
    for round_num in range(NUM_ROUNDS):
        logging.info(f"\n--- STARTING FEDERATED ROUND {round_num + 1}/{NUM_ROUNDS} ---")
        
        # --- Client 0 Training ---
        logging.info("--- Training Client 0 (FedProx, 1 epoch) ---")
        model_0 = copy.deepcopy(global_model) # Client gets a copy
        model_0, _ = train(
            model=model_0,
            train_loader=train_loader_0,
            val_loader=None,
            epochs=EPOCHS_PER_ROUND,
            learning_rate=LEARNING_RATE,
            device=device,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_step_size=LR_SCHEDULER_STEP,
            global_model=global_model, 
            mu=MU
        )
        
        # --- Client 1 Training ---
        logging.info("--- Training Client 1 (FedProx, 1 epoch) ---")
        model_1 = copy.deepcopy(global_model) # Client gets a copy
        model_1, _ = train(
            model=model_1,
            train_loader=train_loader_1,
            val_loader=None,
            epochs=EPOCHS_PER_ROUND,
            learning_rate=LEARNING_RATE,
            device=device,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_step_size=LR_SCHEDULER_STEP,
            global_model=global_model,
            mu=MU
        )

        # --- Server-Side Payload Extraction & Verification ---
        logging.info("\n--- SERVER: Extracting and Verifying Payloads ---")
        
        # Extract payload from Client 0
        payload_0 = get_trainable_state_dict(model_0)
        bytes_0 = serialize_payload(payload_0)
        verified_payload_0 = deserialize_and_verify_payload(bytes_0, global_model)
        
        # Extract payload from Client 1
        payload_1 = get_trainable_state_dict(model_1)
        bytes_1 = serialize_payload(payload_1)
        verified_payload_1 = deserialize_and_verify_payload(bytes_1, global_model)
        
        if not verified_payload_0 or not verified_payload_1:
            logging.error("Payload verification failed. Skipping merge.")
            continue

        # --- Server-Side Merge Step ---
        logging.info("\n--- SERVER: Merging Payloads ---")
        merge_alpha = 0.5 
        logging.info(f"Agent strategy: merging with alpha = {merge_alpha}")
        
        # Use the NEW merge_payloads function
        merged_payload = merge_payloads(verified_payload_0, verified_payload_1, alpha=merge_alpha)

        # --- Server-Side Global Model Update ---
        # Use the NEW update_global_model function
        update_global_model(global_model, merged_payload)

        # --- Server-Side Evaluation Step ---
        logging.info(f"--- Evaluating Merged Global Model (Round {round_num + 1}) ---")
        merged_val_loss, merged_val_acc, _, _ = evaluate(global_model, val_loader, device, criterion)
        
        logging.info(f"Merged Global (Round {round_num + 1}) Accuracy: {merged_val_acc:.2f}%")
    
    logging.info("\nFederated simulation finished.")
    logging.info(f"Final model accuracy: {merged_val_acc:.2f}%")


if __name__ == "__main__":
    main()