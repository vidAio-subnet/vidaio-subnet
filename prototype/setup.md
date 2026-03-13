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


```
# Orchestrator (validator VM):
export CHUTES_API_KEY=cpk_your_key
export CHUTE_BASE_URL=https://youruser-secure-validator.chutes.ai
export CHUTE_ID=your-chute-uuid

# 1. Generate the execution password
export VALIDATOR_EXEC_PASSWORD="$(openssl rand -hex 32)"

# 2. Store it as a Chutes secret (chute gets it as env var in TEE)
chutes secrets create --purpose secure-validator \
  --key VALIDATOR_EXEC_PASSWORD \
  --value "$VALIDATOR_EXEC_PASSWORD"

# 3. Store S3 write credentials as Chutes secrets (TEE-only, never sent over wire)
chutes secrets create --purpose secure-validator \
  --key S3_ENDPOINT_URL --value "https://s3.amazonaws.com"   # or R2/MinIO endpoint
chutes secrets create --purpose secure-validator \
  --key S3_BUCKET       --value "my-validator-results"
chutes secrets create --purpose secure-validator \
  --key S3_ACCESS_KEY   --value "<write-access-key-id>"
chutes secrets create --purpose secure-validator \
  --key S3_SECRET_KEY   --value "<write-secret-access-key>"

# 4. Deploy the chute
chutes deploy validator_chute:chute --accept-fee

# 5. Run the orchestrator — only needs to know bucket name + prefix for key generation
export VALIDATOR_EXEC_PASSWORD="$VALIDATOR_EXEC_PASSWORD"
export S3_BUCKET="my-validator-results"
export S3_RESULT_PREFIX="results"   # optional, default is "results"
python validator_orchestrator.py


# Miner:
export REQUIRE_ATTESTATION=true
export EXPECTED_CHUTE_NAME=secure-validator
export EXPECTED_IMAGE_NAME=youruser/secure-validator:0.3
```


Password Rotation (No Redeploy)
The chute exposes a /rotate-password endpoint that is itself password-gated (must provide current_password). The orchestrator has two convenience flags:
```
bash# Rotate an existing password:
python validator_orchestrator.py --rotate

# Bootstrap a password if none was ever set:
python validator_orchestrator.py --bootstrap
```