# Setup & Deployment Instructions

This guide covers running the prototype locally and maps each step to its production equivalent.

## Local Development (3 Terminals)

**Terminal 1 — Miner's Chute (the miner's scoring service)**
```bash
pip install fastapi uvicorn
python miner_chute.py
# Runs on port 8080 by default
# In prod: deployed via `chutes deploy miner_chute:chute --accept-fee`
```

**Terminal 2 — Miner Axon (responds to validator synapses)**
```bash
pip install fastapi uvicorn
export MINER_CHUTE_URL=http://localhost:8080   # points to Terminal 1
export MINER_CHUTE_ID=local-dev-chute
python miner_axon.py
# Runs on port 9000 by default
# In prod: this is the Bittensor axon registered on the network
```

**Terminal 3 — Validator Orchestrator**
```bash
pip install httpx
export MINER_URL=http://localhost:9000   # points to Terminal 2
python validator_orchestrator.py
```

Check validator orchestrator output for the final simulation result.

---

## Production Deployment

### 1. Miner: Deploy Chute on Chutes.ai

```bash
# Build and deploy the miner's chute
chutes build miner_chute:chute --wait --include-cwd
chutes deploy miner_chute:chute --accept-fee

# Note your chute_id from the deploy output

# Share code with the validator (validator-only access)
# This allows the validator to view the code at api.chutes.ai/code/{chute_id}
chutes chutes share <chute_id> <validator_user>
```

### 2. Miner: Run Axon on Bittensor Network

The miner registers their axon on Bittensor. When the validator sends a synapse, the miner responds with their `chute_id`. The validator then:
- Calls `api.chutes.ai/v1/chutes/{chute_id}/run/score` with input data
- Retrieves the miner's code from `api.chutes.ai/code/{chute_id}`

### 3. Validator: Run Orchestrator

```bash
export MINER_URL=<miner_axon_address>
export VALIDATOR_HOTKEY=<your_hotkey>
python validator_orchestrator.py
```

---

## How It Maps to Production

| Local Prototype | Production Equivalent |
|---|---|
| `python miner_chute.py` (port 8080) | `chutes deploy miner_chute:chute` on Chutes.ai |
| `python miner_axon.py` (port 9000) | Bittensor axon registered on network |
| `POST /synapse` to miner axon | `bt.dendrite` synapse forward |
| Miner responds with `chute_api_url` | Miner responds with `chute_id` in synapse |
| Miner responds with `chute_code` | `chutes chutes share` + `api.chutes.ai/code/{chute_id}` |
| `POST localhost:8080/score` | `POST api.chutes.ai/v1/chutes/{chute_id}/run/score` |
| Validator reads code from synapse response | Validator reads code from `api.chutes.ai/code/{chute_id}` |
