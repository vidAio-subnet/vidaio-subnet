# validator_orchestrator.py
#
# VALIDATOR VM COMPONENT — runs on your cloud VM (NOT inside TEE).
# This is the part the validator operator controls.
#
# SECURITY PROPERTY:
#   This code NEVER sees the miner's secret script. It only:
#     1. Decides which miner to query and what input_data to send
#     2. Calls the Chutes TEE chute via HTTPS API
#     3. Receives the execution result (a dict)
#
#   The miner ↔ chute crypto channel is end-to-end inside TDX.
#   Even if the validator operator adds logging to THIS file,
#   they can only see miner_url, input_data, and the result dict.
#
# USAGE:
#   # Dev mode (chute running locally via `chutes run --dev`):
#   export CHUTE_BASE_URL=http://localhost:8080
#   python validator_orchestrator.py
#
#   # Production (chute deployed on Chutes):
#   export CHUTES_API_KEY=your-api-key
#   export CHUTE_BASE_URL=https://youruser-secure-validator.chutes.ai
#   python validator_orchestrator.py

import asyncio
import os
import sys

import httpx


# ── Configuration ─────────────────────────────────────────────────────────────
CHUTE_BASE_URL = os.environ.get(
    "CHUTE_BASE_URL",
    "http://localhost:8080",  # default: local dev
)
CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY", "")

# Miner to query — in production this comes from the Bittensor metagraph
DEFAULT_MINER_URL = os.environ.get("MINER_URL", "http://localhost:9000")


async def call_chute_execute(
    miner_url: str,
    input_data: dict,
    *,
    chute_base_url: str = CHUTE_BASE_URL,
    api_key: str = CHUTES_API_KEY,
    timeout: float = 60.0,
) -> dict:
    """
    Call the TEE chute's /execute endpoint.

    This is the ONLY interface between the validator VM and the chute.
    The request contains: miner_url + input_data.
    The response contains: status + result (or error).

    The miner's script is decrypted and executed entirely inside the
    Chutes TEE — this function never sees it.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "miner_url": miner_url,
        "input_data": input_data,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{chute_base_url.rstrip('/')}/execute",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        return r.json()


async def validate_miner(miner_url: str, input_data: dict) -> dict:
    """
    High-level validation: send input to a miner via the TEE chute,
    get the result, and score it.

    In a real subnet you'd:
      - Pull miner_url from the Bittensor metagraph
      - Generate challenge input_data
      - Score the result against expected output
      - Update weights on-chain

    The key security property: no matter what logging or instrumentation
    the validator operator adds to THIS function, they only ever see
    input_data and result — never the miner's script.
    """
    print(f"[VALIDATOR] Querying miner: {miner_url}")
    print(f"[VALIDATOR] Input: {input_data}")

    try:
        response = await call_chute_execute(miner_url, input_data)
    except httpx.HTTPStatusError as e:
        print(f"[VALIDATOR] Chute returned HTTP {e.response.status_code}: {e.response.text}")
        return {"status": "error", "error": f"chute HTTP error: {e.response.status_code}"}
    except Exception as e:
        print(f"[VALIDATOR] Failed to reach chute: {e}")
        return {"status": "error", "error": str(e)}

    print(f"[VALIDATOR] Response status: {response.get('status')}")
    print(f"[VALIDATOR] Result: {response.get('result')}")

    if response.get("error"):
        print(f"[VALIDATOR] Error from TEE: {response['error']}")

    # ── Scoring logic (example) ──────────────────────────────────────
    # This is where you'd compare the result to ground truth.
    # The operator can see the result and score — that's fine.
    # What they CANNOT see is HOW the miner computed it.
    if response.get("status") == "ok" and response.get("result"):
        result = response["result"]
        # Example: verify the miner computed the correct sum
        expected_sum = sum(input_data.get("values", []))
        actual_sum = result.get("sum")
        if actual_sum == expected_sum:
            print(f"[VALIDATOR] PASS — miner result matches expected")
            return {"status": "pass", "result": result}
        else:
            print(f"[VALIDATOR] FAIL — expected sum={expected_sum}, got {actual_sum}")
            return {"status": "fail", "result": result}

    return {"status": "error", "result": response}


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    miner_url = DEFAULT_MINER_URL
    input_data = {"values": [1, 2, 3, 4, 5]}

    print("=" * 60)
    print("VALIDATOR ORCHESTRATOR")
    print(f"  Chute:  {CHUTE_BASE_URL}")
    print(f"  Miner:  {miner_url}")
    print("=" * 60)

    result = await validate_miner(miner_url, input_data)
    print(f"\n[VALIDATOR] Final verdict: {result}")


if __name__ == "__main__":
    asyncio.run(main())