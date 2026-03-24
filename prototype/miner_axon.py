# miner_axon.py
#
# MINER AXON — simulates Bittensor synapse handling.
#
# In production, the miner registers on the Bittensor network and responds
# to validator synapse forwards. This prototype simulates that by exposing
# REST API endpoints that the validator orchestrator calls.
#
# When a validator sends a synapse (POST /synapse), the miner responds with:
#
#   (i)  chute_api_url — link to the miner's running chute instance.
#        In production, the validator receives the chute_id via the
#        Bittensor synapse response and calls the chute at:
#          https://api.chutes.ai/v1/chutes/{chute_id}/run/score
#
#   (ii) chute_code — the miner's chute source code (Python script).
#        In production, this is shared via:
#          `chutes chutes share <chute_id> <validator_user>`
#        and the validator retrieves the code at:
#          api.chutes.ai/code/{chute_id}
#
# IMPORTANT: Only the VALIDATOR is allowed to see the miner's code.
# The code is shared exclusively with the validator via Chutes' share
# mechanism — no other network participant can access it.
#
# Run with:
#   pip install fastapi uvicorn
#   python miner_axon.py
#
# Environment variables:
#   MINER_PORT       — listen port (default: 9000)
#   MINER_CHUTE_URL  — URL of the miner's deployed chute (default: http://localhost:8080)
#   MINER_CHUTE_ID   — the chute ID on Chutes.ai (used in prod references)

import os
import inspect

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


app = FastAPI()

# ── Configuration ────────────────────────────────────────────────────────────
MINER_CHUTE_URL = os.environ.get("MINER_CHUTE_URL", "http://localhost:8080")
MINER_CHUTE_ID = os.environ.get("MINER_CHUTE_ID", "local-dev-chute")

# Load the miner's chute source code to share with the validator.
# In production, this is handled by `chutes chutes share` and the validator
# retrieves it from api.chutes.ai/code/{chute_id}.
_CHUTE_CODE_PATH = os.path.join(os.path.dirname(__file__), "miner_chute.py")
try:
    with open(_CHUTE_CODE_PATH, "r") as f:
        _CHUTE_SOURCE_CODE = f.read()
except FileNotFoundError:
    _CHUTE_SOURCE_CODE = "# ERROR: miner_chute.py not found"


# ── Schemas ──────────────────────────────────────────────────────────────────
class SynapseRequest(BaseModel):
    """
    Simulates a Bittensor synapse forward from the validator.
    In production, this is a bt.Synapse subclass sent via dendrite.
    """
    validator_hotkey: str = "unknown"
    task_id: str = ""


class SynapseResponse(BaseModel):
    """
    Miner's response to the validator's synapse.

    Contains:
      - chute_id: The miner's chute ID on Chutes.ai
        (in prod: validator uses this to call api.chutes.ai/v1/chutes/{chute_id}/run/score)
      - chute_api_url: Direct API URL to the miner's chute
        (in local dev: http://localhost:8080/score)
        (in prod: https://api.chutes.ai/v1/chutes/{chute_id}/run/score)
      - chute_code: The miner's chute source code
        (in prod: retrieved via api.chutes.ai/code/{chute_id} after
         `chutes chutes share <chute_id> <validator_user>`)
    """
    status: str
    chute_id: str
    chute_api_url: str
    chute_code: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/synapse")
async def handle_synapse(req: SynapseRequest) -> SynapseResponse:
    """
    Handle a validator synapse forward.

    Simulates the Bittensor synapse response where the miner tells the
    validator about their deployed chute. The response includes:

    1. chute_api_url — the endpoint to call for scoring
       (simulates: validator receiving chute_id in prod and calling
        https://api.chutes.ai/v1/chutes/{chute_id}/run/score)

    2. chute_code — the miner's source code
       (simulates: `chutes chutes share` command in production and
        api.chutes.ai/code/{chute_id} in prod)
    """
    print(f"[MINER] Synapse received from validator: {req.validator_hotkey}")
    print(f"[MINER] Responding with chute_id={MINER_CHUTE_ID}, url={MINER_CHUTE_URL}")

    return SynapseResponse(
        status="ok",
        chute_id=MINER_CHUTE_ID,
        chute_api_url=f"{MINER_CHUTE_URL.rstrip('/')}/score",
        chute_code=_CHUTE_SOURCE_CODE,
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "chute_id": MINER_CHUTE_ID}


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("MINER_PORT", 9000))
    print(f"[MINER] Starting axon on port {port}")
    print(f"[MINER] Chute URL: {MINER_CHUTE_URL}")
    print(f"[MINER] Chute ID: {MINER_CHUTE_ID}")
    uvicorn.run(app, host="0.0.0.0", port=port)
