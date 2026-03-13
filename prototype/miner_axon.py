# miner_axon.py
# VM1 — the miner. Exposes 2 endpoints. Never sends plaintext.
# Run with:
# pip install fastapi uvicorn cryptography
# python miner_axon.py (OR) pm2 start "PYTHONPATH=. python3 miner_axon.py" --name miner-axon
# → Listening on http://0.0.0.0:9000
#
# Changes from v1:
#   - /handshake now requires attestation evidence from the validator
#   - Miner verifies the TDX quote binds to the validator's ECDH pubkey
#   - Per-session ephemeral ECDH keys for forward secrecy
#   - /pubkey endpoint removed (keys are exchanged in /handshake)

import asyncio
import base64
import hashlib
import os
import sys
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

# ── Secret script — never leaves this machine in plaintext ───────────────────
SECRET_SCRIPT = b"""
# Miner's proprietary logic.
# `input_data` is injected by the validator chute before exec().
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
# session_id -> {"key": bytes, "created": float}
_sessions: dict[str, dict] = {}

# Sessions expire after 60 seconds — limits window for replay attacks
SESSION_TTL_SECONDS = 60

# ── Attestation configuration ────────────────────────────────────────────────
# Set to True to enforce real TDX attestation verification.
# In dev mode (False), the miner accepts dev-stub attestations.
REQUIRE_REAL_ATTESTATION = os.environ.get("REQUIRE_ATTESTATION", "false").lower() == "true"

# Expected image digest of the validator chute.
# This must match the digest the validator was built with:
#   chutes image digest youruser/secure-validator:0.2
# When set, the miner verifies this appears in the TDX quote's MRTD field.
EXPECTED_VALIDATOR_IMAGE_DIGEST = os.environ.get(
    "VALIDATOR_IMAGE_DIGEST", ""
)


class AttestationEvidence(BaseModel):
    type: str  # "tdx" or "dev-stub"
    report_data: str  # base64 — SHA-256(validator_ecdh_pubkey) padded to 64B
    pubkey: str  # base64 — the validator's ECDH pubkey
    quote: str | None = None  # base64 — the raw TDX quote (only for type=tdx)
    warning: str | None = None


class HandshakeRequest(BaseModel):
    session_id: str
    validator_pubkey: str  # base64 X25519 public key
    attestation: AttestationEvidence


class ScriptRequest(BaseModel):
    session_id: str


# ── Attestation verification ─────────────────────────────────────────────────
def verify_attestation(attestation: AttestationEvidence, validator_pubkey_b64: str) -> str | None:
    """
    Verify the validator's attestation evidence.

    Returns None on success, or an error string on failure.

    Checks performed:
      1. pubkey binding  — report_data == SHA-256(validator_pubkey)
      2. pubkey match    — attestation.pubkey == handshake pubkey
      3. quote validity  — (real TDX only) signature chain and MRTD
    """
    # ── 1. Verify pubkey binding ─────────────────────────────────────
    validator_pub_bytes = base64.b64decode(validator_pubkey_b64)
    expected_report_data = hashlib.sha256(validator_pub_bytes).digest().ljust(64, b"\x00")
    actual_report_data = base64.b64decode(attestation.report_data)

    if actual_report_data != expected_report_data:
        return "report_data does not bind to the presented pubkey"

    # ── 2. Verify pubkey consistency ─────────────────────────────────
    if attestation.pubkey != validator_pubkey_b64:
        return "attestation pubkey does not match handshake pubkey"

    # ── 3. Dev-stub: accept with warning ─────────────────────────────
    if attestation.type == "dev-stub":
        if REQUIRE_REAL_ATTESTATION:
            return "real TDX attestation required but got dev-stub"
        print(
            f"[MINER] WARNING: accepting dev-stub attestation "
            f"(pubkey binding verified, but no TEE guarantee)"
        )
        return None

    # ── 4. Real TDX quote verification ───────────────────────────────
    if attestation.type == "tdx":
        return _verify_tdx_quote(attestation)

    return f"unknown attestation type: {attestation.type}"


def _verify_tdx_quote(attestation: AttestationEvidence) -> str | None:
    """
    Verify a real TDX attestation quote.

    In production this should use Intel's DCAP (Data Center Attestation
    Primitives) library to:
      1. Verify the quote signature against Intel's root of trust
      2. Check the MRTD (measurement of the TD) matches the expected
         validator image digest
      3. Verify report_data inside the quote matches what we expect

    This is a structural placeholder — the verification points are marked
    with TODO comments where real DCAP calls would go.
    """
    if not attestation.quote:
        return "TDX attestation missing quote"

    quote_bytes = base64.b64decode(attestation.quote)
    if len(quote_bytes) < 48:
        return "TDX quote too short to be valid"

    # TODO: Replace with actual DCAP verification:
    #
    #   from intel_dcap import QuoteVerifier
    #   verifier = QuoteVerifier()
    #   result = verifier.verify(quote_bytes)
    #   if not result.valid:
    #       return f"quote signature invalid: {result.error}"
    #
    #   # Check the report_data embedded in the quote matches
    #   quote_report_data = result.report_data
    #   expected = base64.b64decode(attestation.report_data)
    #   if quote_report_data != expected:
    #       return "quote report_data mismatch"
    #
    #   # Check MRTD matches expected image
    #   if EXPECTED_VALIDATOR_IMAGE_DIGEST:
    #       if result.mrtd.hex() != EXPECTED_VALIDATOR_IMAGE_DIGEST:
    #           return "MRTD does not match expected validator image"

    print("[MINER] TDX quote received — DCAP verification placeholder (implement for production)")
    return None


def _cleanup_expired_sessions():
    """Remove sessions older than TTL to prevent accumulation."""
    now = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if now - data["created"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        # Scrub key material before deletion
        _sessions[sid]["key"] = b"\x00" * 32
        del _sessions[sid]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/handshake")
async def handshake(req: HandshakeRequest):
    """
    Combined handshake: validator sends ECDH pubkey + attestation evidence.
    Miner verifies attestation, generates ephemeral keypair, derives session
    key, and returns its ephemeral pubkey.

    This replaces the old /pubkey + /handshake two-step flow.
    """
    _cleanup_expired_sessions()

    # ── Verify attestation ───────────────────────────────────────────
    attestation_error = verify_attestation(req.attestation, req.validator_pubkey)
    if attestation_error:
        raise HTTPException(status_code=403, detail=f"attestation rejected: {attestation_error}")

    # ── Generate per-session ephemeral keypair ───────────────────────
    session_private = X25519PrivateKey.generate()
    session_pub_bytes = session_private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )

    # ── Derive shared secret ─────────────────────────────────────────
    try:
        their_pub_bytes = base64.b64decode(req.validator_pubkey)
        their_pub = X25519PublicKey.from_public_bytes(their_pub_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid public key")

    raw_secret = session_private.exchange(their_pub)
    session_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
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
    Validator requests the encrypted script for an established session.
    Ciphertext only — plaintext never leaves this machine.
    """
    _cleanup_expired_sessions()

    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="unknown or expired session")

    session = _sessions[req.session_id]
    key = session["key"]
    iv = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(iv, SECRET_SCRIPT, None)
    script_hash = hashlib.sha256(SECRET_SCRIPT).hexdigest()

    # Single-use: scrub and delete session key
    _sessions[req.session_id]["key"] = b"\x00" * 32
    del _sessions[req.session_id]

    return {
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "iv": base64.b64encode(iv).decode(),
        "script_hash": script_hash,
    }


if __name__ == "__main__":
    port = int(os.environ.get("MINER_PORT", 9000))
    print(f"[MINER] Starting axon on port {port}")
    if REQUIRE_REAL_ATTESTATION:
        print(f"[MINER] Real TDX attestation REQUIRED")
    else:
        print(f"[MINER] Dev mode — dev-stub attestations accepted")
    if EXPECTED_VALIDATOR_IMAGE_DIGEST:
        print(f"[MINER] Expected validator image: {EXPECTED_VALIDATOR_IMAGE_DIGEST}")
    uvicorn.run(app, host="0.0.0.0", port=port)