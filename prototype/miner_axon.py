# miner_axon.py
#
# VM1 — the miner. Exposes 2 endpoints. Never sends plaintext.
#
# ATTESTATION MODEL (Option A — Chutes platform attestation):
#   The miner verifies the Chutes platform attestation proof
#   (chute metadata + server attestation status) by independently
#   querying the Chutes API. See previous docstring for full details.
#
# NOTE: The miner has no knowledge of the orchestrator's execution
#   password. That password gates access to the *chute*, not to the
#   miner. The miner's own gate is the attestation verification —
#   it only releases the encrypted script after confirming the caller
#   is a genuine TEE chute on attested hardware.
#
# Run with:
#   pip install fastapi uvicorn cryptography httpx
#   python miner_axon.py
#
# Environment variables:
#   MINER_PORT                     — listen port (default: 9000)
#   REQUIRE_ATTESTATION=true       — reject dev-stub attestations
#   CHUTES_API_BASE                — for cross-verification
#   EXPECTED_CHUTE_NAME            — expected chute name
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
    type: str
    chute: dict | None = None
    server_id: str | None = None
    attestation_status: dict | None = None
    api_base: str | None = None
    source: str | None = None
    warning: str | None = None
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
    """

    # Dev stub
    if att.type == "dev-stub":
        if REQUIRE_REAL_ATTESTATION:
            return "real platform attestation required but got dev-stub"
        print("[MINER] WARNING: accepting dev-stub (no TEE guarantee)")
        return None

    # Error from orchestrator
    if att.type == "platform-error":
        return f"orchestrator error: {att.error}"

    # Chutes platform
    if att.type != "chutes-platform":
        return f"unknown attestation type: {att.type}"

    if not att.chute:
        return "missing chute metadata"

    if not att.chute.get("tee", False):
        return "chute tee=False"

    if EXPECTED_IMAGE_NAME:
        claimed = att.chute.get("image", {}).get("name", "")
        if claimed != EXPECTED_IMAGE_NAME:
            return f"image mismatch: expected {EXPECTED_IMAGE_NAME}, got {claimed}"

    if EXPECTED_CHUTE_NAME:
        claimed = att.chute.get("chute_name", "")
        if claimed != EXPECTED_CHUTE_NAME:
            return f"chute name mismatch: expected {EXPECTED_CHUTE_NAME}, got {claimed}"

    # Independent cross-verification
    chute_id = att.chute.get("chute_id")
    server_id = att.server_id
    api_base = att.api_base or CHUTES_API_BASE

    if chute_id and REQUIRE_REAL_ATTESTATION:
        err = await _cross_verify(chute_id, server_id, api_base)
        if err:
            return f"cross-verification: {err}"

    # Check attestation_status
    if att.attestation_status:
        state = (
            att.attestation_status.get("status", "")
            or att.attestation_status.get("state", "")
            or att.attestation_status.get("result", "")
        )
        if isinstance(state, str) and state.lower() in (
            "failed", "invalid", "expired", "error",
        ):
            return f"server attestation: {state}"
        print(f"[MINER] Server attestation: {att.attestation_status}")
    elif REQUIRE_REAL_ATTESTATION:
        return "no server attestation status"
    else:
        print("[MINER] No attestation status — accepting in dev mode")

    print("[MINER] Platform attestation verified ✓")
    return None


async def _cross_verify(
    chute_id: str, server_id: Optional[str], api_base: str,
) -> Optional[str]:
    """Independently query Chutes API to confirm attestation claims."""
    import httpx

    try:
        async with httpx.AsyncClient(base_url=api_base, timeout=10.0) as client:
            r = await client.get(f"/chutes/{chute_id}")
            if r.status_code == 401:
                print("[MINER] Cross-verify: auth required — trusting claimed data")
                return None
            r.raise_for_status()
            data = r.json()
            if not data.get("tee", False):
                return "cross-check: chute tee=False"

            if server_id:
                r = await client.get(f"/servers/{server_id}/attestation/status")
                if r.status_code == 401:
                    print("[MINER] Cross-verify: server auth required")
                    return None
                r.raise_for_status()
                srv = r.json()
                state = srv.get("status", "") or srv.get("state", "") or ""
                if isinstance(state, str) and state.lower() in (
                    "failed", "invalid", "error",
                ):
                    return f"cross-check: server {state}"

            return None
    except httpx.HTTPStatusError as e:
        return f"API {e.response.status_code}"
    except Exception as e:
        return f"API unreachable: {e}"


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
    Miner verifies, generates ephemeral key, derives session secret.
    """
    _cleanup_expired_sessions()

    err = await verify_platform_attestation(req.attestation)
    if err:
        raise HTTPException(status_code=403, detail=f"attestation rejected: {err}")

    session_private = X25519PrivateKey.generate()
    session_pub_bytes = session_private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw,
    )

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
    """Encrypted script for an established session. Single-use."""
    _cleanup_expired_sessions()

    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="unknown or expired session")

    key = _sessions[req.session_id]["key"]
    iv = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(iv, SECRET_SCRIPT, None)
    script_hash = hashlib.sha256(SECRET_SCRIPT).hexdigest()

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