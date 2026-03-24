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
    print()
    print(f"[ORCH] Miner URL:   {miner_url}")
    print(f"[ORCH] Input data:  {input_data}")

    # ── STEP 1: Send synapse to miner ────────────────────────────────
    print()
    print("─" * 60)
    print("[ORCH] STEP 1: Synapse Forward to Miner")
    print("─" * 60)
    print(f"[ORCH] Sending POST {miner_url}/synapse")
    print(f"[ORCH] (prod: bt.dendrite sends ChuteInfoSynapse to miner's axon)")
    try:
        synapse_resp = await send_synapse(miner_url)
    except httpx.HTTPStatusError as e:
        print(f"[ORCH] FAILED — HTTP {e.response.status_code}: {e.response.text}")
        return {"status": "error", "error": f"synapse HTTP {e.response.status_code}"}
    except Exception as e:
        print(f"[ORCH] FAILED — could not reach miner: {e}")
        return {"status": "error", "error": str(e)}

    if synapse_resp.get("status") != "ok":
        print(f"[ORCH] FAILED — miner returned: {synapse_resp}")
        return {"status": "error", "error": "miner synapse failed"}

    chute_id = synapse_resp.get("chute_id", "")
    chute_api_url = synapse_resp.get("chute_api_url", "")
    chute_code = synapse_resp.get("chute_code", "")

    print(f"[ORCH] Received synapse response:")
    print(f"[ORCH]   chute_id:      {chute_id}")
    print(f"[ORCH]   chute_api_url: {chute_api_url}")
    print(f"[ORCH]     (prod: api.chutes.ai/v1/chutes/{chute_id}/run/score)")
    print(f"[ORCH]   chute_code:    {len(chute_code)} chars received")
    print(f"[ORCH]     (prod: retrieved via api.chutes.ai/code/{chute_id}")
    print(f"[ORCH]      after miner runs `chutes chutes share {chute_id} <validator>`)")

    if not chute_api_url:
        return {"status": "error", "error": "miner did not provide chute_api_url"}

    # ── STEP 2: Call the miner's chute ───────────────────────────────
    print()
    print("─" * 60)
    print("[ORCH] STEP 2: Call Miner's Chute API")
    print("─" * 60)
    print(f"[ORCH] Sending POST {chute_api_url}")
    print(f"[ORCH]   payload: {{input_data: {input_data}}}")
    print(f"[ORCH] (prod: POST api.chutes.ai/v1/chutes/{chute_id}/run/score)")
    try:
        chute_resp = await call_miner_chute(chute_api_url, input_data)
    except httpx.HTTPStatusError as e:
        print(f"[ORCH] FAILED — HTTP {e.response.status_code}: {e.response.text}")
        return {"status": "error", "error": f"chute HTTP {e.response.status_code}"}
    except Exception as e:
        print(f"[ORCH] FAILED — could not reach miner chute: {e}")
        return {"status": "error", "error": str(e)}

    print(f"[ORCH] Response status: {chute_resp.get('status')}")
    print(f"[ORCH] Response result: {chute_resp.get('result')}")

    if chute_resp.get("error"):
        print(f"[ORCH] Chute error: {chute_resp['error']}")

    # ── STEP 3: Inspect miner code ───────────────────────────────────
    print()
    print("─" * 60)
    print("[ORCH] STEP 3: Inspect Miner Code (VALIDATOR-ONLY)")
    print("─" * 60)
    print(f"[ORCH] (prod: GET api.chutes.ai/code/{chute_id})")
    print(f"[ORCH] (requires miner to have run `chutes chutes share`)")
    print(f"[ORCH] Only the validator can see this code — no other participant has access.")
    if chute_code:
        inspection = inspect_miner_code(chute_code)
        print(f"[ORCH] Code inspection results:")
        for key, value in inspection.items():
            print(f"[ORCH]   {key}: {value}")
        # Show a preview of the score function
        lines = chute_code.split("\n")
        score_lines = []
        capture = False
        for line in lines:
            if "def score(" in line:
                capture = True
            if capture:
                score_lines.append(line)
                if line.strip().startswith("return "):
                    # grab the return block
                    break
        if score_lines:
            print(f"[ORCH] Score function preview:")
            for line in score_lines:
                print(f"[ORCH]   | {line}")
    else:
        print("[ORCH] WARNING: no chute code received — cannot inspect")

    # ── STEP 4: Score the result ─────────────────────────────────────
    print()
    print("─" * 60)
    print("[ORCH] STEP 4: Score & Validate Result")
    print("─" * 60)
    if chute_resp.get("status") == "ok" and chute_resp.get("result"):
        result = chute_resp["result"]
        expected_sum = sum(input_data.get("values", []))
        actual_sum = result.get("sum")
        print(f"[ORCH] Expected sum: {expected_sum}")
        print(f"[ORCH] Actual sum:   {actual_sum}")
        if actual_sum == expected_sum:
            print(f"[ORCH] PASS")
            return {"status": "pass", "result": result}
        else:
            print(f"[ORCH] FAIL")
            return {"status": "fail", "result": result}

    print(f"[ORCH] ERROR — unexpected chute response")
    return {"status": "error", "result": chute_resp}


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    miner_url = DEFAULT_MINER_URL
    input_data = {"values": [1, 2, 3, 4, 5]}

    print()
    print("=" * 60)
    print("  VALIDATOR ORCHESTRATOR — Prototype Simulation")
    print("=" * 60)
    print(f"  Miner axon:    {miner_url}")
    print(f"  Hotkey:        {VALIDATOR_HOTKEY}")
    print(f"  Input data:    {input_data}")
    print()
    print("  Simulation steps:")
    print("    1. Send synapse to miner (simulates bt.dendrite forward)")
    print("    2. Call miner's chute API (simulates api.chutes.ai call)")
    print("    3. Inspect miner code (simulates api.chutes.ai/code/{{id}})")
    print("    4. Score & validate result")
    print("=" * 60)

    result = await validate_miner(miner_url, input_data)

    print()
    print("=" * 60)
    print(f"  FINAL RESULT: {result['status'].upper()}")
    if result.get("result"):
        print(f"  Result data:  {result['result']}")
    if result.get("error"):
        print(f"  Error:        {result['error']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
