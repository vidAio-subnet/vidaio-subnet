# validator_orchestrator.py
#
# VALIDATOR ORCHESTRATOR — runs on the validator's VM.
#
# FLOW:
#   1. Send a synapse forward to the miner's axon (simulates Bittensor dendrite)
#   2. Receive the miner's response containing:
#      - chute_api_url: endpoint to call for scoring
#      - chute_code: the miner's source code (ONLY validator sees this)
#   3. Call the miner's chute API with input data
#   4. Inspect the miner's code (optional — for audit/compliance)
#   5. Score the result
#
# In production (Bittensor network):
#   - Step 1 uses bt.dendrite to send a Synapse to the miner's axon
#   - The miner responds with their chute_id
#   - The validator calls the chute at:
#       https://api.chutes.ai/v1/chutes/{chute_id}/run/score
#   - The validator retrieves the miner's code via:
#       api.chutes.ai/code/{chute_id}
#     (requires the miner to have run `chutes chutes share <chute_id> <validator_user>`)
#
# SECURITY PROPERTY:
#   ONLY the validator can see the miner's code. The code is shared
#   exclusively with the validator via Chutes' share mechanism. No other
#   network participant (other miners, delegators, etc.) has access.
#
# Run with:
#   pip install httpx
#   python validator_orchestrator.py
#
# Environment variables:
#   MINER_URL          — miner axon URL (default: http://localhost:9000)
#   VALIDATOR_HOTKEY   — validator's Bittensor hotkey (default: "dev-validator")

import asyncio
import os
import sys

import httpx


# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_MINER_URL = os.environ.get("MINER_URL", "http://localhost:9000")
VALIDATOR_HOTKEY = os.environ.get("VALIDATOR_HOTKEY", "dev-validator")


# ── Send synapse to miner ────────────────────────────────────────────────────
async def send_synapse(
    miner_url: str,
    validator_hotkey: str = VALIDATOR_HOTKEY,
    task_id: str = "",
) -> dict:
    """
    Send a synapse forward to the miner's axon.

    In production, this is done via bt.dendrite:
        responses = await dendrite(
            axons=[miner_axon],
            synapse=ChuteInfoSynapse(),
        )

    The miner responds with their chute_id, chute_api_url, and chute_code.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{miner_url.rstrip('/')}/synapse",
            json={
                "validator_hotkey": validator_hotkey,
                "task_id": task_id,
            },
        )
        r.raise_for_status()
        return r.json()


# ── Call the miner's chute ───────────────────────────────────────────────────
async def call_miner_chute(
    chute_api_url: str,
    input_data: dict,
    timeout: float = 30.0,
) -> dict:
    """
    Call the miner's chute scoring endpoint.

    In production, this calls:
        https://api.chutes.ai/v1/chutes/{chute_id}/run/score
    with the Chutes API key for authentication.

    Locally, this calls the miner's chute directly (e.g. http://localhost:8080/score).
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            chute_api_url,
            json={"input_data": input_data},
        )
        r.raise_for_status()
        return r.json()


# ── Inspect miner code ──────────────────────────────────────────────────────
def inspect_miner_code(chute_code: str) -> dict:
    """
    Inspect the miner's chute source code.

    In production, the validator retrieves this via:
        api.chutes.ai/code/{chute_id}
    (requires the miner to have shared the chute with the validator via
     `chutes chutes share <chute_id> <validator_user>`)

    ONLY the validator sees this code. This enables:
      - Verifying the miner isn't running malicious/trivial code
      - Auditing the scoring logic for correctness
      - Checking for hardcoded values or gaming attempts
    """
    inspection = {
        "code_length": len(chute_code),
        "has_score_function": "def score(" in chute_code,
        "line_count": chute_code.count("\n") + 1,
    }

    # Basic checks
    if not inspection["has_score_function"]:
        inspection["warning"] = "No score() function found in miner code"

    return inspection


# ── Validation flow ─────────────────────────────────────────────────────────
async def validate_miner(
    miner_url: str,
    input_data: dict,
    miner_hotkey: str = "unknown",
) -> dict:
    """
    Full validation flow:
      1. Send synapse to miner → get chute info
      2. Call the miner's chute with input data
      3. Inspect miner code (validator-only)
      4. Score the result
    """
    print(f"[ORCH] Miner: {miner_url}")
    print(f"[ORCH] Input: {input_data}")

    # 1. Send synapse to miner (simulates Bittensor dendrite forward)
    print("[ORCH] Sending synapse to miner...")
    try:
        synapse_resp = await send_synapse(miner_url)
    except httpx.HTTPStatusError as e:
        print(f"[ORCH] Synapse HTTP {e.response.status_code}: {e.response.text}")
        return {"status": "error", "error": f"synapse HTTP {e.response.status_code}"}
    except Exception as e:
        print(f"[ORCH] Failed to reach miner: {e}")
        return {"status": "error", "error": str(e)}

    if synapse_resp.get("status") != "ok":
        print(f"[ORCH] Miner synapse error: {synapse_resp}")
        return {"status": "error", "error": "miner synapse failed"}

    chute_id = synapse_resp.get("chute_id", "")
    chute_api_url = synapse_resp.get("chute_api_url", "")
    chute_code = synapse_resp.get("chute_code", "")

    print(f"[ORCH] Miner chute_id: {chute_id}")
    # In prod: chute_api_url = https://api.chutes.ai/v1/chutes/{chute_id}/run/score
    print(f"[ORCH] Chute API URL: {chute_api_url}")

    if not chute_api_url:
        return {"status": "error", "error": "miner did not provide chute_api_url"}

    # 2. Call the miner's chute (simulates calling api.chutes.ai/v1/chutes/{chute_id}/run/score)
    print("[ORCH] Calling miner's chute...")
    try:
        chute_resp = await call_miner_chute(chute_api_url, input_data)
    except httpx.HTTPStatusError as e:
        print(f"[ORCH] Chute HTTP {e.response.status_code}: {e.response.text}")
        return {"status": "error", "error": f"chute HTTP {e.response.status_code}"}
    except Exception as e:
        print(f"[ORCH] Failed to reach miner chute: {e}")
        return {"status": "error", "error": str(e)}

    print(f"[ORCH] Chute status: {chute_resp.get('status')}")
    print(f"[ORCH] Chute result: {chute_resp.get('result')}")

    if chute_resp.get("error"):
        print(f"[ORCH] Chute error: {chute_resp['error']}")

    # 3. Inspect miner code (ONLY the validator sees this)
    # In prod: retrieved via api.chutes.ai/code/{chute_id}
    # (after miner ran `chutes chutes share <chute_id> <validator_user>`)
    if chute_code:
        inspection = inspect_miner_code(chute_code)
        print(f"[ORCH] Code inspection: {inspection}")
    else:
        print("[ORCH] WARNING: no chute code received — cannot inspect")

    # 4. Score the result
    if chute_resp.get("status") == "ok" and chute_resp.get("result"):
        result = chute_resp["result"]
        expected_sum = sum(input_data.get("values", []))
        actual_sum = result.get("sum")
        if actual_sum == expected_sum:
            print(f"[ORCH] PASS — sum={actual_sum}")
            return {"status": "pass", "result": result}
        else:
            print(f"[ORCH] FAIL — expected {expected_sum}, got {actual_sum}")
            return {"status": "fail", "result": result}

    return {"status": "error", "result": chute_resp}


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    miner_url = DEFAULT_MINER_URL
    input_data = {"values": [1, 2, 3, 4, 5]}

    print("=" * 60)
    print("VALIDATOR ORCHESTRATOR")
    print(f"  Miner:      {miner_url}")
    print(f"  Hotkey:     {VALIDATOR_HOTKEY}")
    print("=" * 60)

    result = await validate_miner(miner_url, input_data)
    print(f"\n[ORCH] Final: {result}")


if __name__ == "__main__":
    asyncio.run(main())
