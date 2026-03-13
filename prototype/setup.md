# Terminal 1 — miner
pip install fastapi uvicorn cryptography
python miner_axon.py

# Terminal 2 — build + run chute locally
chutes build validator_chute:chute --local
# (follow the docker run instructions in the file header)
chutes run validator_chute:chute --dev --port 8080

# Terminal 3 — validator orchestrator
pip install httpx
export CHUTE_BASE_URL=http://localhost:8080
export MINER_URL=http://host.docker.internal:9000  # if chute is in Docker
python validator_orchestrator.py