# miner_axon.py
#
# VM1 — the miner. Exposes 2 endpoints. Never sends plaintext.
#
# Security properties:
#   - Secret script ONLY leaves this machine encrypted with AES-256-GCM
#   - Per-session ephemeral ECDH keys → forward secrecy
#   - Miner verifies TDX attestation before releasing encrypted script
#   - DCAP verification confirms the validator chute runs in a genuine
#     Intel TDX enclave with the expected image (MRTD measurement)
#   - Sessions expire after 60s and keys are scrubbed on deletion
#
# Run with:
#   pip install fastapi uvicorn cryptography
#   python miner_axon.py
#
# Environment variables:
#   MINER_PORT                 — listen port (default: 9000)
#   REQUIRE_ATTESTATION=true   — reject dev-stub attestations
#   VALIDATOR_IMAGE_DIGEST     — expected MRTD hex of validator chute image

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
_sessions: dict[str, dict] = {}  # session_id → {"key": bytes, "created": float}
SESSION_TTL_SECONDS = 60

# ── Attestation config ───────────────────────────────────────────────────────
REQUIRE_REAL_ATTESTATION = (
    os.environ.get("REQUIRE_ATTESTATION", "false").lower() == "true"
)
EXPECTED_VALIDATOR_IMAGE_DIGEST = os.environ.get("VALIDATOR_IMAGE_DIGEST", "")


# ── Try to load Intel DCAP for real TDX quote verification ───────────────────
_dcap_available = False
try:
    from dcap_quote_verify import (  # type: ignore
        verify_quote_integrity,
        extract_report_data,
        extract_mrtd,
    )
    _dcap_available = True
except ImportError:
    # DCAP not installed — we'll use manual parsing as fallback.
    # Install via: pip install dcap-quote-verify
    # (requires Intel DCAP libraries on the system)
    pass


# ── Schemas ──────────────────────────────────────────────────────────────────
class AttestationEvidence(BaseModel):
    type: str                      # "tdx" or "dev-stub"
    report_data: str               # base64 — SHA-256(validator_ecdh_pubkey)
    pubkey: str                    # base64 — validator's ECDH pubkey
    quote: Optional[str] = None    # base64 — raw TDX quote (type=tdx only)
    warning: Optional[str] = None


class HandshakeRequest(BaseModel):
    session_id: str
    validator_pubkey: str          # base64 X25519 public key
    attestation: AttestationEvidence


class ScriptRequest(BaseModel):
    session_id: str


# ── DCAP / attestation verification ─────────────────────────────────────────
def verify_attestation(
    attestation: AttestationEvidence,
    validator_pubkey_b64: str,
) -> Optional[str]:
    """
    Verify the validator's attestation evidence.

    Returns None on success, or an error string on failure.

    Verification chain:
      1. Pubkey binding: report_data == SHA-256(validator_pubkey) ∥ padding
      2. Pubkey consistency: attestation.pubkey matches handshake pubkey
      3. For dev-stub: accept only if REQUIRE_REAL_ATTESTATION is false
      4. For TDX: verify quote signature via DCAP, check MRTD, confirm
         report_data inside the quote matches what we computed
    """
    # 1. Verify pubkey binding
    validator_pub_bytes = base64.b64decode(validator_pubkey_b64)
    expected_rd = hashlib.sha256(validator_pub_bytes).digest().ljust(64, b"\x00")
    actual_rd = base64.b64decode(attestation.report_data)

    if actual_rd != expected_rd:
        return "report_data does not bind to the presented pubkey"

    # 2. Verify pubkey consistency
    if attestation.pubkey != validator_pubkey_b64:
        return "attestation pubkey does not match handshake pubkey"

    # 3. Dev-stub
    if attestation.type == "dev-stub":
        if REQUIRE_REAL_ATTESTATION:
            return "real TDX attestation required but got dev-stub"
        print(
            "[MINER] WARNING: accepting dev-stub attestation "
            "(pubkey binding OK, but NO TEE guarantee)"
        )
        return None

    # 4. Real TDX
    if attestation.type == "tdx":
        return _verify_tdx_quote(attestation, expected_rd)

    return f"unknown attestation type: {attestation.type}"


def _verify_tdx_quote(
    attestation: AttestationEvidence,
    expected_report_data: bytes,
) -> Optional[str]:
    """
    Verify a TDX attestation quote.

    With DCAP library available:
      - Verifies the quote signature against Intel's root of trust
      - Extracts and checks report_data matches our expectation
      - Extracts MRTD and compares to EXPECTED_VALIDATOR_IMAGE_DIGEST

    Without DCAP (dev/fallback):
      - Performs structural validation on the raw quote bytes
      - Checks report_data at the known TDX quote offset (368..432)
      - Warns that signature verification is skipped
    """
    if not attestation.quote:
        return "TDX attestation missing quote"

    quote_bytes = base64.b64decode(attestation.quote)

    if len(quote_bytes) < 584:
        return f"TDX quote too short ({len(quote_bytes)} bytes, need ≥584)"

    # ── Path A: Full DCAP verification ───────────────────────────────
    if _dcap_available:
        try:
            if not verify_quote_integrity(quote_bytes):
                return "TDX quote signature verification failed (DCAP)"

            quote_rd = extract_report_data(quote_bytes)
            if quote_rd != expected_report_data:
                return "report_data inside TDX quote does not match"

            if EXPECTED_VALIDATOR_IMAGE_DIGEST:
                mrtd = extract_mrtd(quote_bytes)
                if mrtd.hex() != EXPECTED_VALIDATOR_IMAGE_DIGEST.lower():
                    return (
                        f"MRTD mismatch: got {mrtd.hex()}, "
                        f"expected {EXPECTED_VALIDATOR_IMAGE_DIGEST.lower()}"
                    )

            print("[MINER] TDX quote verified via DCAP ✓")
            return None

        except Exception as e:
            return f"DCAP verification error: {e}"

    # ── Path B: Structural validation (no DCAP) ─────────────────────
    # TDX quote v4 layout (simplified):
    #   Bytes 0-3:   version (should be 4)
    #   Bytes 4-5:   attestation key type
    #   Bytes 48-95: TD attributes
    #   Bytes 368-431: report_data (64 bytes)
    #   Bytes 432-479: MRTD (48 bytes) — TD measurement
    #
    # Without DCAP we can check structure + report_data, but NOT the
    # signature chain. This is acceptable for dev but not production.

    version = int.from_bytes(quote_bytes[0:2], "little")
    if version not in (4, 5):
        return f"unexpected TDX quote version: {version}"

    # Check report_data at offset 368
    quote_rd = quote_bytes[368:432]
    if quote_rd != expected_report_data:
        return "report_data inside TDX quote does not match (structural check)"

    # Check MRTD if configured
    if EXPECTED_VALIDATOR_IMAGE_DIGEST:
        mrtd = quote_bytes[432:480]
        if mrtd.hex() != EXPECTED_VALIDATOR_IMAGE_DIGEST.lower():
            return (
                f"MRTD mismatch: got {mrtd.hex()}, "
                f"expected {EXPECTED_VALIDATOR_IMAGE_DIGEST.lower()}"
            )

    print(
        "[MINER] TDX quote structural check passed "
        "(DCAP not available — signature NOT verified; "
        "install dcap-quote-verify for full verification)"
    )
    return None


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
    Combined handshake: validator sends ECDH pubkey + attestation.
    Miner verifies attestation, generates ephemeral key, derives
    session secret, and returns its ephemeral pubkey.
    """
    _cleanup_expired_sessions()

    # Verify attestation BEFORE generating any keys
    err = verify_attestation(req.attestation, req.validator_pubkey)
    if err:
        raise HTTPException(status_code=403, detail=f"attestation rejected: {err}")

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
        raise HTTPException(status_code=404, detail="unknown or expired session")

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
    print(f"[MINER] Attestation enforcement: {'REQUIRED' if REQUIRE_REAL_ATTESTATION else 'dev-mode (stubs accepted)'}")
    if EXPECTED_VALIDATOR_IMAGE_DIGEST:
        print(f"[MINER] Expected validator MRTD: {EXPECTED_VALIDATOR_IMAGE_DIGEST}")
    if _dcap_available:
        print("[MINER] Intel DCAP library loaded ✓")
    else:
        print("[MINER] Intel DCAP not available — structural validation only")
    uvicorn.run(app, host="0.0.0.0", port=port)