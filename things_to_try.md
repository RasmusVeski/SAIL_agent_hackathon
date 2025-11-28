# Agent Improvement Brainstorming

Use these ideas as a starting point for your hackathon features.

## Potential New Tools

Give your agent more control over the learning process and the timeline.

### 1. Safety & Recovery

* **`rollback_to_round(n)`**: If a merge disastrously lowers accuracy (e.g., Catastrophic Forgetting), allow the agent to revert the Global Model to a previous healthy state.

* **`replace_draft_with_global()`**: If local training goes off the rails, discard the "Local Draft" and reset it to the current Global Model baseline.

* **`early_exit()`**: Have the agent write a very rigid line like "YOU HAVE A GOOD ACCURACY DON'T TRAIN OR MERGE ANYMORE" into the history or just don't enter loop anymore.

### 2. Strategic & Competitive

* **`adopt_peer_model()`**: If a peer sends weights with 40% accuracy and you have 10%, ignore merging. **Takeover.** Discard your weights and adopt theirs entirely, then fine-tune.

* **`stash_peer_weights()`**: Don't merge immediately. Save the peer's weights to a "buffer". Wait until you have 3 different peers, then average them all at once (Simulated Centralized Aggregation).

* **`probe_history()`**: Very generally tie yourself to the past. Maybe a tool that asks: *"What did I do in Round 5 that caused accuracy to jump?"* It should return a summary of hyperparameters and partners from that specific round.

## Architecture & Logic

### 1. "Meta-Tools"

LLMs sometimes get distracted or time out between steps. Create **Macro Tools** that execute a strict sequence in Python, guaranteeing atomic execution.

* **`execute_standard_cycle()`**: Runs `Train -> Merge -> Evaluate -> Commit` in one go.

* **`safe_exchange()`**: Runs `Exchange -> Evaluate Peer -> (Conditional) Merge` automatically to prevent poisoning.

### 2. State Indexing

Currently, we only store the "Current" weights. Consider adding a **Model Registry** to `AgentState`.

* Store weights by ID: `state.saved_models["round_4_high_acc"]`.

* Allow the agent to load specifically named checkpoints to compare performance.

### 3. Enhanced History Logging

The `history` list currently logs results. Make it richer so the LLM can learn from mistakes.

* **Log Inputs:** Record exactly which `lr`, `mu`, and `epochs` were used next to the result.

* **Log Commands:** Record the exact sequence of tools called.

* **Log Partners:** Track which specific partners yield high-accuracy merges (Whitelisting) vs low-accuracy merges (Blacklisting).

Be careful, the longer the history the worse the LLM gets.

## Prompt Engineering

The "Brain" is only as good as its instructions.

### 1. Expressive Communication

Teach the agent to write better messages to partners.

* **Bad:** "Here are weights."

* **Good:** "I just trained on Class 5 and 8. My accuracy is 22%."

This allows the receiving agent to make smarter decisions about whether to merge.

### 2. Few-Shot Prompting

Don't just tell the agent *what* to do; show it *how* to think. Add examples to the System Prompt:

> **Example of Good Reasoning:**
> "My accuracy dropped from 20% to 15% after the last local training. This suggests I am overwriting the global knowledge. I will **LOWER** the learning rate to 0.0001 and **INCREASE** the FedProx mu to 1.0 to protect the weights."

### 3. Personality Tuning

* **Collaborative:** "Trust everyone, share everything, merge often."

* **Competitive:** "Trust no one. Verify every payload. If a partner is weak, ignore them. If they are strong, copy them."
