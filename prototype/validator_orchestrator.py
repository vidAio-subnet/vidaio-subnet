# validator_orchestrator.py
#
# VALIDATOR VM COMPONENT — runs on your cloud VM (NOT inside TEE).
#
# ATTESTATION MODEL (Option A — Chutes platform attestation):
#   Before calling the chute, this orchestrator queries the Chutes API to:
#     1. Look up the chute's metadata (image hash, tee flag)
#     2. Identify which server instance is running the chute
#     3. Fetch that server's attestation status from Chutes
#     4. Bundle this into a "platform_attestation" proof
#     5. Pass the proof through the chute to the miner
#
#   The miner uses this proof to verify that the chute is running on
#   genuine TDX hardware with a cosign-signed image before releasing
#   the encrypted script.
#
# SECURITY PROPERTY:
#   This code NEVER sees the miner's secret script. It only sees:
#     - miner_url, input_data, result dict
#     - Chutes API metadata (public information)
#   The miner ↔ chute crypto channel is end-to-end inside TDX.
#
# USAGE:
#   # Dev mode (chute running locally):
#   export CHUTE_BASE_URL=http://localhost:8080
#   python validator_orchestrator.py
#
#   # Production:
#   export CHUTES_API_KEY=cpk_your_api_key
#   export CHUTE_BASE_URL=https://youruser-secure-validator.chutes.ai
#   export CHUTE_ID=your-chute-uuid-or-name
#   python validator_orchestrator.py

import asyncio
import os
import sys
from typing import Optional

import httpx


# ── Configuration ─────────────────────────────────────────────────────────────
CHUTES_API_BASE = "https://api.chutes.ai"
CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY", "")

CHUTE_BASE_URL = os.environ.get("CHUTE_BASE_URL", "http://localhost:8080")
CHUTE_ID = os.environ.get("CHUTE_ID", "")  # chute UUID or "username/name"

DEFAULT_MINER_URL = os.environ.get("MINER_URL", "http://localhost:9000")

# In production, set this to the expected image name:tag
# e.g. "youruser/secure-validator:0.3"
EXPECTED_IMAGE_NAME = os.environ.get("EXPECTED_IMAGE_NAME", "")


# ── Chutes Platform Attestation ──────────────────────────────────────────────
async def fetch_platform_attestation(
    chute_id: str,
    api_key: str,
    api_base: str = CHUTES_API_BASE,
) -> dict:
    """
    Query the Chutes platform API to build a platform attestation proof.

    This proof contains:
      - chute metadata (id, name, image, tee flag)
      - server attestation status (from Chutes' TDX verification)
      - image identity (the cosign-signed image hash)

    The miner can independently verify this against the Chutes API
    to confirm the chute is running on attested TDX hardware.

    Chutes API endpoints used:
      GET /chutes/{chute_id}            — chute metadata + image info
      GET /instances/                   — find which server runs the chute
      GET /servers/{server_id}/attestation/status — TDX attestation state
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(
        base_url=api_base, headers=headers, timeout=15.0
    ) as client:

        # 1. Get chute metadata
        try:
            r = await client.get(f"/chutes/{chute_id}")
            r.raise_for_status()
            chute_meta = r.json()
        except Exception as e:
            return {
                "type": "platform-error",
                "error": f"failed to fetch chute metadata: {e}",
            }

        chute_info = {
            "chute_id": chute_meta.get("id", chute_id),
            "chute_name": chute_meta.get("name", ""),
            "tee_enabled": chute_meta.get("tee", False),
            "image": chute_meta.get("image", {}),
        }

        # Verify the chute is actually TEE-enabled
        if not chute_info["tee_enabled"]:
            return {
                "type": "platform-error",
                "error": "chute does not have tee=True — cannot guarantee TEE",
                "chute": chute_info,
            }

        # Verify image matches expectation
        image_name = chute_info.get("image", {}).get("name", "")
        if EXPECTED_IMAGE_NAME and image_name != EXPECTED_IMAGE_NAME:
            return {
                "type": "platform-error",
                "error": (
                    f"image mismatch: expected {EXPECTED_IMAGE_NAME}, "
                    f"got {image_name}"
                ),
                "chute": chute_info,
            }

        # 2. Find which server instance is running this chute
        server_id = None
        try:
            r = await client.get("/instances/", params={"chute_id": chute_info["chute_id"]})
            r.raise_for_status()
            instances = r.json()
            if isinstance(instances, list) and len(instances) > 0:
                server_id = instances[0].get("server_id")
            elif isinstance(instances, dict):
                items = instances.get("items", instances.get("results", []))
                if items:
                    server_id = items[0].get("server_id")
        except Exception as e:
            # Non-fatal: we can still proceed, miner just won't get
            # server-level attestation status
            print(f"[ORCH] Warning: could not list instances: {e}")

        # 3. Get server attestation status
        attestation_status = None
        if server_id:
            try:
                r = await client.get(f"/servers/{server_id}/attestation/status")
                r.raise_for_status()
                attestation_status = r.json()
            except Exception as e:
                print(f"[ORCH] Warning: could not fetch attestation status: {e}")

        return {
            "type": "chutes-platform",
            "chute": chute_info,
            "server_id": server_id,
            "attestation_status": attestation_status,
            "api_base": api_base,
        }


async def fetch_dev_attestation() -> dict:
    """
    In dev mode (no API key / no chute ID), return a stub attestation.
    The miner will accept this only if REQUIRE_REAL_ATTESTATION is false.
    """
    return {
        "type": "dev-stub",
        "source": "orchestrator",
        "warning": "No Chutes API key — local dev mode, no platform attestation",
    }


# ── Call the Chute ────────────────────────────────────────────────────────────
async def call_chute_execute(
    miner_url: str,
    input_data: dict,
    platform_attestation: dict,
    *,
    chute_base_url: str = CHUTE_BASE_URL,
    api_key: str = CHUTES_API_KEY,
    timeout: float = 60.0,
) -> dict:
    """
    Call the TEE chute's /execute endpoint.

    Passes platform_attestation through to the chute, which forwards
    it to the miner during the ECDH handshake.

    The miner's script is decrypted and executed inside the chute's
    TDX enclave — this function never sees it.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "miner_url": miner_url,
        "input_data": input_data,
        "platform_attestation": platform_attestation,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{chute_base_url.rstrip('/')}/execute",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        return r.json()


# ── Validation flow ───────────────────────────────────────────────────────────
async def validate_miner(miner_url: str, input_data: dict) -> dict:
    """
    Full validation flow:
      1. Fetch platform attestation from Chutes API
      2. Call the chute (which does the miner handshake inside TEE)
      3. Score the result

    Even if the validator operator adds logging everywhere in this
    function, they only see: miner_url, input_data, platform attestation
    (public), and the result dict.
    """
    print(f"[ORCH] Miner: {miner_url}")
    print(f"[ORCH] Input: {input_data}")

    # 1. Get platform attestation
    if CHUTES_API_KEY and CHUTE_ID:
        print(f"[ORCH] Fetching platform attestation for chute {CHUTE_ID}...")
        platform_att = await fetch_platform_attestation(CHUTE_ID, CHUTES_API_KEY)
    else:
        print("[ORCH] No API key / chute ID — using dev stub attestation")
        platform_att = await fetch_dev_attestation()

    att_type = platform_att.get("type", "unknown")
    print(f"[ORCH] Attestation type: {att_type}")

    if att_type == "platform-error":
        print(f"[ORCH] Attestation error: {platform_att.get('error')}")
        return {"status": "error", "error": f"attestation: {platform_att.get('error')}"}

    # 2. Call the chute
    try:
        response = await call_chute_execute(miner_url, input_data, platform_att)
    except httpx.HTTPStatusError as e:
        print(f"[ORCH] Chute HTTP {e.response.status_code}: {e.response.text}")
        return {"status": "error", "error": f"chute HTTP {e.response.status_code}"}
    except Exception as e:
        print(f"[ORCH] Failed to reach chute: {e}")
        return {"status": "error", "error": str(e)}

    print(f"[ORCH] Status: {response.get('status')}")
    print(f"[ORCH] Result: {response.get('result')}")

    if response.get("error"):
        print(f"[ORCH] TEE error: {response['error']}")

    # 3. Score
    if response.get("status") == "ok" and response.get("result"):
        result = response["result"]
        expected_sum = sum(input_data.get("values", []))
        actual_sum = result.get("sum")
        if actual_sum == expected_sum:
            print(f"[ORCH] PASS — sum={actual_sum} matches expected")
            return {"status": "pass", "result": result}
        else:
            print(f"[ORCH] FAIL — expected {expected_sum}, got {actual_sum}")
            return {"status": "fail", "result": result}

    return {"status": "error", "result": response}


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    miner_url = DEFAULT_MINER_URL
    input_data = {"values": [1, 2, 3, 4, 5]}

    print("=" * 60)
    print("VALIDATOR ORCHESTRATOR (Option A — Chutes Platform Attestation)")
    print(f"  Chute endpoint:  {CHUTE_BASE_URL}")
    print(f"  Chute ID:        {CHUTE_ID or '(dev mode — none)'}")
    print(f"  Miner:           {miner_url}")
    print(f"  API key:         {'set' if CHUTES_API_KEY else 'not set (dev mode)'}")
    print("=" * 60)

    result = await validate_miner(miner_url, input_data)
    print(f"\n[ORCH] Final: {result}")


if __name__ == "__main__":
    asyncio.run(main())