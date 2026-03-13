# validator_orchestrator.py
#
# VALIDATOR VM COMPONENT — runs on your cloud VM (NOT inside TEE).
#
# ACCESS CONTROL:
#   This orchestrator is the ONLY entity that knows the chute's execution
#   password. It sends the password with every /execute call. Without it,
#   the chute rejects the request immediately — no miner contact, no
#   crypto, no script fetch.
#
#   Setup:
#     1. Generate a strong password:
#          export VALIDATOR_EXEC_PASSWORD="$(openssl rand -hex 32)"
#     2. Store it as a Chutes secret (so the chute gets it as env var):
#          chutes secrets create --purpose secure-validator \
#            --key VALIDATOR_EXEC_PASSWORD --value "$VALIDATOR_EXEC_PASSWORD"
#     3. Export it for the orchestrator:
#          export VALIDATOR_EXEC_PASSWORD="<same value>"
#     4. Run:
#          python validator_orchestrator.py
#
#   To rotate the password at runtime (no redeploy needed):
#     python validator_orchestrator.py --rotate
#
# SECURITY PROPERTY:
#   This code NEVER sees the miner's secret script. It only sees:
#     - miner_url, input_data, result dict
#     - Chutes API metadata (public info)
#     - The execution password (which it set itself)

import argparse
import asyncio
import os
import secrets
import sys
from typing import Optional

import httpx


# ── Configuration ─────────────────────────────────────────────────────────────
CHUTES_API_BASE = "https://api.chutes.ai"
CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY", "")

CHUTE_BASE_URL = os.environ.get("CHUTE_BASE_URL", "http://localhost:8080")
CHUTE_ID = os.environ.get("CHUTE_ID", "")

DEFAULT_MINER_URL = os.environ.get("MINER_URL", "http://localhost:9000")

EXPECTED_IMAGE_NAME = os.environ.get("EXPECTED_IMAGE_NAME", "")

# The execution password — must match what's in the Chutes secret
EXEC_PASSWORD = os.environ.get("VALIDATOR_EXEC_PASSWORD", "")


# ── Password rotation ────────────────────────────────────────────────────────
async def rotate_chute_password(
    chute_base_url: str = CHUTE_BASE_URL,
    api_key: str = CHUTES_API_KEY,
    current_password: str = "",
    new_password: str = "",
) -> bool:
    """
    Rotate the execution password on the chute at runtime.

    After this succeeds:
      1. Update your local VALIDATOR_EXEC_PASSWORD env var
      2. Update the Chutes secret for persistence across restarts:
           chutes secrets delete <old_secret_id>
           chutes secrets create --purpose secure-validator \
             --key VALIDATOR_EXEC_PASSWORD --value <new_password>

    Returns True on success.
    """
    if not new_password:
        new_password = secrets.token_hex(32)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{chute_base_url.rstrip('/')}/rotate-password",
            json={
                "current_password": current_password,
                "new_password": new_password,
            },
            headers=headers,
        )
        r.raise_for_status()
        resp = r.json()

    if resp.get("status") == "ok":
        print(f"[ORCH] Password rotated successfully.")
        print(f"[ORCH] New password: {new_password}")
        print(f"[ORCH] Update your environment:")
        print(f"  export VALIDATOR_EXEC_PASSWORD=\"{new_password}\"")
        print(f"[ORCH] And update the Chutes secret for persistence:")
        print(f"  chutes secrets create --purpose secure-validator \\")
        print(f"    --key VALIDATOR_EXEC_PASSWORD --value \"{new_password}\"")
        return True
    else:
        print(f"[ORCH] Password rotation failed: {resp.get('error')}")
        return False


# ── Chutes Platform Attestation ──────────────────────────────────────────────
async def fetch_platform_attestation(
    chute_id: str,
    api_key: str,
    api_base: str = CHUTES_API_BASE,
) -> dict:
    """
    Query the Chutes API to build a platform attestation proof.
    See previous version for full docstring — logic unchanged.
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(
        base_url=api_base, headers=headers, timeout=15.0
    ) as client:

        # 1. Chute metadata
        try:
            r = await client.get(f"/chutes/{chute_id}")
            r.raise_for_status()
            chute_meta = r.json()
        except Exception as e:
            return {"type": "platform-error", "error": f"chute metadata: {e}"}

        chute_info = {
            "chute_id": chute_meta.get("id", chute_id),
            "chute_name": chute_meta.get("name", ""),
            "tee_enabled": chute_meta.get("tee", False),
            "image": chute_meta.get("image", {}),
        }

        if not chute_info["tee_enabled"]:
            return {
                "type": "platform-error",
                "error": "chute does not have tee=True",
                "chute": chute_info,
            }

        image_name = chute_info.get("image", {}).get("name", "")
        if EXPECTED_IMAGE_NAME and image_name != EXPECTED_IMAGE_NAME:
            return {
                "type": "platform-error",
                "error": f"image mismatch: expected {EXPECTED_IMAGE_NAME}, got {image_name}",
                "chute": chute_info,
            }

        # 2. Find server running this chute
        server_id = None
        try:
            r = await client.get("/instances/", params={"chute_id": chute_info["chute_id"]})
            r.raise_for_status()
            instances = r.json()
            if isinstance(instances, list) and instances:
                server_id = instances[0].get("server_id")
            elif isinstance(instances, dict):
                items = instances.get("items", instances.get("results", []))
                if items:
                    server_id = items[0].get("server_id")
        except Exception as e:
            print(f"[ORCH] Warning: could not list instances: {e}")

        # 3. Server attestation status
        attestation_status = None
        if server_id:
            try:
                r = await client.get(f"/servers/{server_id}/attestation/status")
                r.raise_for_status()
                attestation_status = r.json()
            except Exception as e:
                print(f"[ORCH] Warning: attestation status: {e}")

        return {
            "type": "chutes-platform",
            "chute": chute_info,
            "server_id": server_id,
            "attestation_status": attestation_status,
            "api_base": api_base,
        }


async def fetch_dev_attestation() -> dict:
    return {
        "type": "dev-stub",
        "source": "orchestrator",
        "warning": "No Chutes API key — local dev mode",
    }


# ── Call the Chute ────────────────────────────────────────────────────────────
async def call_chute_execute(
    miner_url: str,
    input_data: dict,
    platform_attestation: dict,
    exec_password: str,
    *,
    chute_base_url: str = CHUTE_BASE_URL,
    api_key: str = CHUTES_API_KEY,
    timeout: float = 60.0,
) -> dict:
    """
    Call the TEE chute's /execute endpoint with the execution password.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "miner_url": miner_url,
        "input_data": input_data,
        "exec_password": exec_password,
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
      1. Check we have an execution password
      2. Fetch platform attestation
      3. Call the chute (password-gated)
      4. Score the result
    """
    if not EXEC_PASSWORD:
        print("[ORCH] ERROR: VALIDATOR_EXEC_PASSWORD not set")
        print("[ORCH] Generate one and set it as a Chutes secret:")
        print('  export VALIDATOR_EXEC_PASSWORD="$(openssl rand -hex 32)"')
        print("  chutes secrets create --purpose secure-validator \\")
        print("    --key VALIDATOR_EXEC_PASSWORD --value \"$VALIDATOR_EXEC_PASSWORD\"")
        return {"status": "error", "error": "no execution password configured"}

    print(f"[ORCH] Miner: {miner_url}")
    print(f"[ORCH] Input: {input_data}")

    # 1. Platform attestation
    if CHUTES_API_KEY and CHUTE_ID:
        print(f"[ORCH] Fetching attestation for chute {CHUTE_ID}...")
        platform_att = await fetch_platform_attestation(CHUTE_ID, CHUTES_API_KEY)
    else:
        print("[ORCH] No API key / chute ID — dev stub attestation")
        platform_att = await fetch_dev_attestation()

    att_type = platform_att.get("type", "unknown")
    print(f"[ORCH] Attestation type: {att_type}")

    if att_type == "platform-error":
        print(f"[ORCH] Attestation error: {platform_att.get('error')}")
        return {"status": "error", "error": f"attestation: {platform_att.get('error')}"}

    # 2. Call the chute (password-gated)
    try:
        response = await call_chute_execute(
            miner_url, input_data, platform_att, EXEC_PASSWORD
        )
    except httpx.HTTPStatusError as e:
        print(f"[ORCH] Chute HTTP {e.response.status_code}: {e.response.text}")
        return {"status": "error", "error": f"chute HTTP {e.response.status_code}"}
    except Exception as e:
        print(f"[ORCH] Failed to reach chute: {e}")
        return {"status": "error", "error": str(e)}

    # Check for password rejection
    if response.get("status") == "error" and response.get("error") == "unauthorized":
        print("[ORCH] ERROR: chute rejected our password — check VALIDATOR_EXEC_PASSWORD")
        return {"status": "error", "error": "password mismatch"}

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
            print(f"[ORCH] PASS — sum={actual_sum}")
            return {"status": "pass", "result": result}
        else:
            print(f"[ORCH] FAIL — expected {expected_sum}, got {actual_sum}")
            return {"status": "fail", "result": result}

    return {"status": "error", "result": response}


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Validator orchestrator")
    parser.add_argument(
        "--rotate", action="store_true",
        help="Rotate the chute execution password and exit",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Set initial password on a chute that has none configured",
    )
    args = parser.parse_args()

    if args.rotate:
        current = EXEC_PASSWORD
        if not current:
            print("[ORCH] ERROR: VALIDATOR_EXEC_PASSWORD not set — cannot rotate")
            print("[ORCH] Use --bootstrap if no password was ever set")
            sys.exit(1)
        new = secrets.token_hex(32)
        ok = await rotate_chute_password(
            current_password=current, new_password=new
        )
        sys.exit(0 if ok else 1)

    if args.bootstrap:
        new = secrets.token_hex(32)
        ok = await rotate_chute_password(
            current_password="BOOTSTRAP_NO_PASSWORD_SET",
            new_password=new,
        )
        sys.exit(0 if ok else 1)

    # Normal validation run
    miner_url = DEFAULT_MINER_URL
    input_data = {"values": [1, 2, 3, 4, 5]}

    print("=" * 60)
    print("VALIDATOR ORCHESTRATOR (password-gated, Option A attestation)")
    print(f"  Chute:     {CHUTE_BASE_URL}")
    print(f"  Chute ID:  {CHUTE_ID or '(dev mode)'}")
    print(f"  Miner:     {miner_url}")
    print(f"  Password:  {'set' if EXEC_PASSWORD else 'NOT SET'}")
    print(f"  API key:   {'set' if CHUTES_API_KEY else 'not set (dev mode)'}")
    print("=" * 60)

    result = await validate_miner(miner_url, input_data)
    print(f"\n[ORCH] Final: {result}")


if __name__ == "__main__":
    asyncio.run(main())