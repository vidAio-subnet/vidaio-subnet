# miner_chute.py
#
# MINER'S OWN CHUTE — deployed by the miner on Chutes.ai (or run locally).
#
# In production, this is deployed via:
#   chutes build miner_chute:chute --wait --include-cwd
#   chutes deploy miner_chute:chute --accept-fee
#
# The miner shares the chute with the validator via:
#   chutes chutes share <chute_id> <validator_user>
#   (validator can then view code at api.chutes.ai/code/{chute_id})
#
# For local dev, this runs as a standalone FastAPI server.
#
# Run with:
#   pip install fastapi uvicorn
#   python miner_chute.py
#
# Environment variables:
#   MINER_CHUTE_PORT  — listen port (default: 8080)

import os

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


app = FastAPI()


# ── Schemas ──────────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    input_data: dict


class ScoreResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None


# ── Miner's proprietary scoring logic ────────────────────────────────────────
# This is the miner's secret sauce. In production it runs inside the miner's
# own Chutes deployment. The validator can inspect this code ONLY via
# `chutes chutes share` / api.chutes.ai/code/{chute_id} — no one else can.

def score(data: dict) -> dict:
    """
    Example scoring function — computes basic stats on a list of values.
    In a real subnet this would be the miner's proprietary model/algorithm.
    """
    values = data.get("values", [])
    return {
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else 0,
        "count": len(values),
    }


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/score")
async def handle_score(req: ScoreRequest) -> ScoreResponse:
    """
    Execute the miner's scoring function on the provided input data.

    In production, this endpoint is exposed via the Chutes platform at:
      https://api.chutes.ai/v1/chutes/{chute_id}/run/score
    The validator calls this after receiving the chute_id from the miner's axon.
    """
    try:
        result = score(req.input_data)
        return ScoreResponse(status="ok", result=result)
    except Exception as e:
        return ScoreResponse(status="error", error=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("MINER_CHUTE_PORT", 8080))
    print(f"[MINER-CHUTE] Starting miner chute on port {port}")
    print(f"[MINER-CHUTE] Score endpoint: POST /score")
    uvicorn.run(app, host="0.0.0.0", port=port)
