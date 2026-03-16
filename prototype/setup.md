# Setup & Deployment Instructions

This guide covers running the components locally for development, and deploying them to `api.chutes.ai`.

## Local Development (Docker & Localhost)

**Terminal 1 — Miner Axon**
```bash
pip install fastapi uvicorn cryptography
python miner_axon.py (OR) pm2 start "PYTHONPATH=. python miner_axon.py" --name miner-axon
```

**Terminal 2 — Build + Run Chute Locally**
```bash
chutes build validator_chute:chute --local
export VALIDATOR_EXEC_PASSWORD="$(openssl rand -hex 32)"
echo $VALIDATOR_EXEC_PASSWORD  # copy this to use in terminal 3
# For running locally in a temporary container, follow the docker run instructions in the file header of validator_chute.py:
docker run --rm -it -e CHUTES_EXECUTION_CONTEXT=REMOTE -e "VALIDATOR_EXEC_PASSWORD=$VALIDATOR_EXEC_PASSWORD" -p 8080:8080 --add-host=host.docker.internal:host-gateway secure-validator:0.3 /bin/sh
# Inside the container:
chutes run validator_chute:chute --dev --port 8080
```

**Terminal 3 — Validator Orchestrator**
```bash
pip install httpx
export VALIDATOR_EXEC_PASSWORD= # paste `echo $VALIDATOR_EXEC_PASSWORD` output from terminal 2
export CHUTE_BASE_URL=http://localhost:8080
export MINER_URL=http://host.docker.internal:9000  # if chute is in Docker

python validator_orchestrator.py (OR) pm2 start "PYTHONPATH=. python validator_orchestrator.py" --name validator-orchestrator
```

Check validator orchestrator logs for final simulation result

---

## Production Deployment (api.chutes.ai)

### 1. Validator Chute (Deployed on TEE)

The TEE component must securely receive its configuration via Chutes secrets.

**1. Generate an Execution Password:**
```bash
export VALIDATOR_EXEC_PASSWORD="$(openssl rand -hex 32)"
```

**2. Configure Chutes Secrets:**
```bash
# Required: Administrator password for the orchestrator to call the chute
chutes secrets create --purpose secure-validator \
  --key VALIDATOR_EXEC_PASSWORD \
  --value "$VALIDATOR_EXEC_PASSWORD"

# Optional S3 Results Upload Credentials (TEE-only, never sent over wire):
chutes secrets create --purpose secure-validator \
  --key S3_ENDPOINT_URL --value "https://s3.amazonaws.com"   # or R2/MinIO endpoint
chutes secrets create --purpose secure-validator \
  --key S3_BUCKET       --value "my-validator-results"
chutes secrets create --purpose secure-validator \
  --key S3_ACCESS_KEY   --value "<write-access-key-id>"
chutes secrets create --purpose secure-validator \
  --key S3_SECRET_KEY   --value "<write-secret-access-key>"
```

**3. Deploy the Chute:**
```bash
chutes build validator_chute:chute --wait --include-cwd
chutes deploy validator_chute:chute --accept-fee
```
*(Note your `CHUTE_ID` and the image name you used, e.g., `youruser/secure-validator:0.3`)*

---

### 2. Miner Axon

The Miner Axon requires environment variables to securely verify that requests are coming from a legitimate, attested Chute.

Set these variables before running `miner_axon.py` on your miner instance:
```bash
export MINER_PORT=9000                             # Optional (Defaults to 9000)
export REQUIRE_ATTESTATION=true                    # Enforce genuine hardware TEE attestation
export CHUTES_API_BASE="https://api.chutes.ai"     # Optional, for cross-verification
export EXPECTED_CHUTE_NAME="secure-validator"
export EXPECTED_IMAGE_NAME="youruser/secure-validator:0.3"

python miner_axon.py
```

---

### 3. Validator Orchestrator (Validator VM)

The Validator Orchestrator runs on your cloud VM (outside the TEE) and coordinates the evaluation.

Set these variables before running `validator_orchestrator.py`:
```bash
export CHUTES_API_KEY="cpk_your_key"
export CHUTE_ID="your-chute-uuid"
export CHUTE_BASE_URL="https://api.chutes.ai"
export VALIDATOR_EXEC_PASSWORD="$VALIDATOR_EXEC_PASSWORD" # Must match step 1
export MINER_URL="http://your.miner.ip:9000"
export EXPECTED_IMAGE_NAME="youruser/secure-validator:0.3"

# Optional S3 setup (only needs to know bucket name + prefix for key generation)
export S3_BUCKET="my-validator-results"
export S3_RESULT_PREFIX="results"   # optional, default is "results"

python validator_orchestrator.py
```

---

## Password Rotation (No Redeploy)

The chute exposes a `/rotate-password` endpoint that is itself password-gated. The orchestrator has two convenience flags:

```bash
# Rotate an existing password:
python validator_orchestrator.py --rotate

# Bootstrap a password if none was ever set:
python validator_orchestrator.py --bootstrap
```