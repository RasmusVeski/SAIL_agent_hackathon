import torch
import os
import logging
import copy
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

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

# --- 1. Agent State ---
# We create a class to hold our model and data in memory,
# since FastAPI is stateless by default.
class AgentState:
    def __init__(self):
        self.agent_id = "RESPONDER (Agent B)"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.round_num = 0
        self.agent_hps = {
            'epochs': 1, 
            'learning_rate': 1e-4, 
            'weight_decay': 1e-3, 
            'mu': 0.5,
            'val_frequency': 1,       
            'lr_scheduler_step_size': 999
        }

# Create a single, global instance of our agent's state
state = AgentState()

# --- 2. FastAPI App Setup ---
app = FastAPI()

# Pydantic models for request/response bodies
class HandshakeRequest(BaseModel):
    agent_id: str

class WeightExchangeRequest(BaseModel):
    agent_id: str
    payload_b64: str

class WeightExchangeResponse(BaseModel):
    agent_id: str
    payload_b64: str


@app.on_event("startup")
def startup_event():
    """
    On server startup, load data and initialize the model.
    """
    setup_logging(log_dir="logs", log_file="agent_responder.log")
    logging.info(f"--- {state.agent_id} | STARTING UP ---")
    
    BASE_DIR = "./app/sharded_data"
    CLIENT_DATA_DIR = os.path.join(BASE_DIR, "client_1") # This agent uses client_1
    VAL_DIR = os.path.join(BASE_DIR, "test1")

    state.train_loader = create_dataloader(CLIENT_DATA_DIR, 32, shuffle=True, num_workers=0)
    state.val_loader = create_dataloader(VAL_DIR, 32, shuffle=False, num_workers=0)
    
    if not state.train_loader or not state.val_loader:
        logging.error("Failed to load data. Server cannot function.")
        return

    state.global_model = FoodClassifier()
    state.criterion = torch.nn.CrossEntropyLoss()
    
    val_loss, val_acc, _, _ = evaluate(
        state.global_model, state.val_loader, state.device, state.criterion
    )
    logging.info(f"Round 0 Initial Accuracy: {val_acc:.2f}%")
    logging.info(f"--- {state.agent_id} | READY ---")


@app.post("/handshake")
def handshake(request: HandshakeRequest):
    """
    Simple handshake endpoint to confirm the server is live.
    """
    logging.info(f"Received HELLO from {request.agent_id}")
    return {"message": "HELLO_ACK", "responder_id": state.agent_id}


@app.post("/exchange_weights", response_model=WeightExchangeResponse)
def exchange_weights(request: WeightExchangeRequest):
    """
    This is the core synchronous logic.
    1. Receives initiator's payload.
    2. Trains its *own* model.
    3. Responds with its *own* payload.
    4. Merges both.
    """
    state.round_num += 1
    logging.info(f"\n--- {state.agent_id} | Round {state.round_num} ---")
    
    # 1. Deserialize and verify initiator's payload
    logging.info(f"Received payload from {request.agent_id}")
    initiator_payload = deserialize_payload_from_b64(
        request.payload_b64, state.global_model
    )
    if not initiator_payload:
        logging.error("Received corrupt payload.")
        raise HTTPException(status_code=400, detail="Corrupt payload")

    # 2. Train our *own* local model
    logging.info("Training local model...")
    local_model, _ = train(
        model=copy.deepcopy(state.global_model),
        train_loader=state.train_loader,
        val_loader=None,
        global_model=state.global_model,
        device=state.device,
        **state.agent_hps
    )

    # 3. Prepare *our* payload to send back
    logging.info("Preparing responder payload...")
    my_payload = get_trainable_state_dict(local_model)
    my_payload_b64 = serialize_payload_to_b64(my_payload)
    if not my_payload_b64:
        logging.error("Failed to serialize own payload.")
        raise HTTPException(status_code=500, detail="Failed to serialize payload")

    # 4. Merge! (We have both payloads now)
    logging.info("Merging payloads...")
    merged_payload = merge_payloads(my_payload, initiator_payload, alpha=0.5)
    
    # 5. Update our global model
    update_global_model(state.global_model, merged_payload)
    
    # 6. Evaluate
    val_loss, val_acc, _, _ = evaluate(
        state.global_model, state.val_loader, state.device, state.criterion
    )
    logging.info(f"Round {state.round_num} Merged Accuracy: {val_acc:.2f}%")
    
    # 7. Send our payload back as the response
    logging.info("Sending responder payload in response.")
    return WeightExchangeResponse(
        agent_id=state.agent_id,
        payload_b64=my_payload_b64
    )


if __name__ == "__main__":
    logging.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)