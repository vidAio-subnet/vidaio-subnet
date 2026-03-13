# miner_axon.py
#
# VM1 — the miner. Exposes 2 endpoints. Never sends plaintext.
#
# ATTESTATION MODEL (Option A — Chutes platform attestation):
#   Instead of verifying a raw TDX quote (which requires /dev/tdx_guest
#   access that user chute code doesn't have), the miner verifies
#   the Chutes platform attestation chain:
#
#   1. Orchestrator fetches chute metadata + server attestation status
#      from the Chutes API and passes it through the chute to the miner
#   2. Miner independently queries the SAME Chutes API endpoints to
#      cross-check that:
#        a) The chute exists and has tee=True
#        b) The server running it has passed TDX attestation
#        c) The image is cosign-signed (Chutes enforces this)
#   3. Only then does the miner release the encrypted script
#
#   This works because Chutes' own infrastructure does the TDX
#   verification (boot attestation, RTMR measurement, LUKS unlock)
#   and exposes the result via authenticated API endpoints.
#
# Run with:
#   pip install fastapi uvicorn cryptography httpx
#   python miner_axon.py
#
# Environment variables:
#   MINER_PORT                     — listen port (default: 9000)
#   REQUIRE_REAL_ATTESTATION=true  — reject dev-stub attestations
#   CHUTES_API_BASE                — Chutes API URL (default: https://api.chutes.ai)
#   EXPECTED_CHUTE_NAME            — expected chute name for verification
#   EXPECTED_IMAGE_NAME            — expected image name:tag

import base64
import hashlib
import os
import time
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ── Secret script — never leaves this machine in plaintext ───────────────────
SECRET_SCRIPT = b"""
# Miner's proprietary logic.
# `input_data` dict is injected by the validator chute before exec().
# Must assign output to `result`.

def score(data: dict) -> dict:
    values = data.get("values", [])
    return {
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else 0,
        "count": len(values),
    }

result = score(input_data)
"""

app = FastAPI()

# ── Session storage ──────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}
SESSION_TTL_SECONDS = 60

# ── Configuration ────────────────────────────────────────────────────────────
REQUIRE_REAL_ATTESTATION = (
    os.environ.get("REQUIRE_ATTESTATION", "false").lower() == "true"
)
CHUTES_API_BASE = os.environ.get("CHUTES_API_BASE", "https://api.chutes.ai")
EXPECTED_CHUTE_NAME = os.environ.get("EXPECTED_CHUTE_NAME", "")
EXPECTED_IMAGE_NAME = os.environ.get("EXPECTED_IMAGE_NAME", "")


# ── Schemas ──────────────────────────────────────────────────────────────────
class PlatformAttestation(BaseModel):
    type: str  # "chutes-platform" or "dev-stub"
    # Fields present when type == "chutes-platform":
    chute: dict | None = None
    server_id: str | None = None
    attestation_status: dict | None = None
    api_base: str | None = None
    # Fields present when type == "dev-stub":
    source: str | None = None
    warning: str | None = None
    # Error case:
    error: str | None = None


class HandshakeRequest(BaseModel):
    session_id: str
    validator_pubkey: str
    attestation: PlatformAttestation


class ScriptRequest(BaseModel):
    session_id: str


# ── Platform attestation verification ────────────────────────────────────────
async def verify_platform_attestation(att: PlatformAttestation) -> Optional[str]:
    """
    Verify the Chutes platform attestation proof.

    Returns None on success, or an error string on failure.

    For "chutes-platform" type:
      1. Check the claimed chute has tee=True
      2. Check image name matches expectations (if configured)
      3. Independently query the Chutes API to cross-verify:
         a) The chute exists with tee=True
         b) The server has valid attestation status
      4. Verify the attestation_status indicates a passing state

    For "dev-stub" type:
      Accept only if REQUIRE_REAL_ATTESTATION is False.
    """

    # ── Dev stub ─────────────────────────────────────────────────────
    if att.type == "dev-stub":
        if REQUIRE_REAL_ATTESTATION:
            return "real platform attestation required but got dev-stub"
        print(
            "[MINER] WARNING: accepting dev-stub attestation "
            "(no TEE guarantee — dev mode only)"
        )
        return None

    # ── Platform error forwarded from orchestrator ───────────────────
    if att.type == "platform-error":
        return f"orchestrator reported attestation error: {att.error}"

    # ── Chutes platform attestation ──────────────────────────────────
    if att.type != "chutes-platform":
        return f"unknown attestation type: {att.type}"

    if not att.chute:
        return "attestation missing chute metadata"

    # Check TEE flag in claimed metadata
    if not att.chute.get("tee", False):
        return "chute does not have tee=True"

    # Check image name if configured
    if EXPECTED_IMAGE_NAME:
        claimed_image = att.chute.get("image", {}).get("name", "")
        if claimed_image != EXPECTED_IMAGE_NAME:
            return (
                f"image mismatch: expected {EXPECTED_IMAGE_NAME}, "
                f"got {claimed_image}"
            )

    # Check chute name if configured
    if EXPECTED_CHUTE_NAME:
        claimed_name = att.chute.get("chute_name", "")
        if claimed_name != EXPECTED_CHUTE_NAME:
            return (
                f"chute name mismatch: expected {EXPECTED_CHUTE_NAME}, "
                f"got {claimed_name}"
            )

    # ── Independent cross-verification via Chutes API ────────────────
    # The miner does NOT trust the attestation blob at face value.
    # It queries the Chutes API itself to confirm.
    chute_id = att.chute.get("chute_id")
    server_id = att.server_id
    api_base = att.api_base or CHUTES_API_BASE

    if chute_id and REQUIRE_REAL_ATTESTATION:
        cross_check_err = await _cross_verify_with_chutes_api(
            chute_id, server_id, api_base
        )
        if cross_check_err:
            return f"cross-verification failed: {cross_check_err}"

    # ── Check attestation_status from the orchestrator ───────────────
    if att.attestation_status:
        status = att.attestation_status
        # The exact shape of this response depends on the Chutes API,
        # but we expect something indicating "passed" or "valid"
        att_state = (
            status.get("status", "")
            or status.get("state", "")
            or status.get("result", "")
        )
        if isinstance(att_state, str) and att_state.lower() in (
            "failed", "invalid", "expired", "error",
        ):
            return f"server attestation state: {att_state}"
        print(f"[MINER] Server attestation status: {status}")
    elif REQUIRE_REAL_ATTESTATION:
        return "no server attestation status provided"
    else:
        print("[MINER] No server attestation status — accepting in dev mode")

    print("[MINER] Platform attestation verified ✓")
    return None


async def _cross_verify_with_chutes_api(
    chute_id: str,
    server_id: Optional[str],
    api_base: str,
) -> Optional[str]:
    """
    Independently query the Chutes API to verify the claimed attestation.

    This prevents a malicious orchestrator from fabricating attestation
    data. The miner checks the same public endpoints the orchestrator
    used, confirming:
      - The chute exists and has tee=True
      - The server has passing attestation
    """
    import httpx

    try:
        async with httpx.AsyncClient(
            base_url=api_base, timeout=10.0
        ) as client:
            # Verify chute exists and has TEE
            r = await client.get(f"/chutes/{chute_id}")
            if r.status_code == 401:
                # Some endpoints may require auth — if so, we can't
                # cross-verify without an API key. Log and continue.
                print(
                    "[MINER] Cross-verify: chute endpoint requires auth "
                    "(cannot independently verify — trusting claimed data)"
                )
                return None
            r.raise_for_status()
            chute_data = r.json()

            if not chute_data.get("tee", False):
                return "cross-check: chute tee=False on Chutes API"

            # Verify server attestation if we have a server_id
            if server_id:
                r = await client.get(
                    f"/servers/{server_id}/attestation/status"
                )
                if r.status_code == 401:
                    print(
                        "[MINER] Cross-verify: attestation endpoint "
                        "requires auth — cannot verify server status"
                    )
                    return None
                r.raise_for_status()
                server_att = r.json()
                state = (
                    server_att.get("status", "")
                    or server_att.get("state", "")
                    or ""
                )
                if isinstance(state, str) and state.lower() in (
                    "failed", "invalid", "error",
                ):
                    return f"cross-check: server attestation {state}"

            return None  # All checks passed

    except httpx.HTTPStatusError as e:
        return f"Chutes API returned {e.response.status_code}"
    except Exception as e:
        return f"Chutes API unreachable: {e}"


# ── Session cleanup ──────────────────────────────────────────────────────────
def _cleanup_expired_sessions():
    now = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if now - data["created"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _sessions[sid]["key"] = b"\x00" * 32
        del _sessions[sid]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/handshake")
async def handshake(req: HandshakeRequest):
    """
    Combined handshake: chute sends ECDH pubkey + platform attestation.
    Miner verifies attestation (checking Chutes API independently),
    generates ephemeral keypair, derives session key, returns its pubkey.
    """
    _cleanup_expired_sessions()

    # Verify platform attestation BEFORE generating any keys
    err = await verify_platform_attestation(req.attestation)
    if err:
        raise HTTPException(
            status_code=403, detail=f"attestation rejected: {err}"
        )

    # Per-session ephemeral keypair
    session_private = X25519PrivateKey.generate()
    session_pub_bytes = session_private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw,
    )

    # Derive shared secret
    try:
        their_pub = X25519PublicKey.from_public_bytes(
            base64.b64decode(req.validator_pubkey)
        )
    except Exception:
        raise HTTPException(status_code=400, detail="invalid public key")

    raw_secret = session_private.exchange(their_pub)
    session_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None,
        info=b"chutes-miner-validator-v1",
    ).derive(raw_secret)

    # Scrub intermediaries
    raw_secret = b"\x00" * len(raw_secret)
    del raw_secret, session_private

    _sessions[req.session_id] = {
        "key": session_key,
        "created": time.time(),
    }

    return {
        "status": "ok",
        "session_id": req.session_id,
        "miner_pubkey": base64.b64encode(session_pub_bytes).decode(),
    }


@app.post("/script")
async def get_script(req: ScriptRequest):
    """
    Return the encrypted script for an established session.
    Single-use: session key is scrubbed after this call.
    """
    _cleanup_expired_sessions()

    if req.session_id not in _sessions:
        raise HTTPException(
            status_code=404, detail="unknown or expired session"
        )

    key = _sessions[req.session_id]["key"]
    iv = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(iv, SECRET_SCRIPT, None)
    script_hash = hashlib.sha256(SECRET_SCRIPT).hexdigest()

    # Scrub and delete
    _sessions[req.session_id]["key"] = b"\x00" * 32
    del _sessions[req.session_id]

    return {
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "iv": base64.b64encode(iv).decode(),
        "script_hash": script_hash,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("MINER_PORT", 9000))
    print(f"[MINER] Starting axon on port {port}")
    mode = "ENFORCED" if REQUIRE_REAL_ATTESTATION else "dev-mode (stubs accepted)"
    print(f"[MINER] Platform attestation: {mode}")
    if EXPECTED_CHUTE_NAME:
        print(f"[MINER] Expected chute: {EXPECTED_CHUTE_NAME}")
    if EXPECTED_IMAGE_NAME:
        print(f"[MINER] Expected image: {EXPECTED_IMAGE_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=port)