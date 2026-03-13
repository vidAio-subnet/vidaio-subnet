# miner_axon.py
# VM1 — the miner. Exposes 3 endpoints. Never sends plaintext.
# Run with:
# pip install fastapi uvicorn cryptography
# python miner_axon.py (OR) pm2 start "PYTHONPATH=. python3 miner_axon.py" --name miner-axon
# → Listening on http://0.0.0.0:9000

import asyncio
import base64
import hashlib
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey
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

# In-memory only. Keyed by session_id from the validator.
_sessions: dict[str, bytes] = {}  # session_id -> derived AES key

# Our ephemeral keypair — regenerated on startup, never persisted
_our_private: X25519PrivateKey = X25519PrivateKey.generate()
_our_pubkey_b64: str = base64.b64encode(
    _our_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
).decode()


class HandshakeRequest(BaseModel):
    session_id: str
    validator_pubkey: str  # base64 X25519 public key


class ScriptRequest(BaseModel):
    session_id: str


@app.get("/pubkey")
async def get_pubkey():
    """
    Step 1. Validator calls this first.
    Returns our ECDH public key. No secrets here — pubkey is safe to expose.
    """
    return {"miner_pubkey": _our_pubkey_b64}


@app.post("/handshake")
async def handshake(req: HandshakeRequest):
    """
    Step 2. Validator sends its ECDH pubkey + session_id.
    We derive the shared secret and store the session key.
    The raw shared secret is immediately discarded — only the HKDF-derived key kept.
    """
    try:
        their_pub_bytes = base64.b64decode(req.validator_pubkey)
        their_pub = X25519PublicKey.from_public_bytes(their_pub_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid public key")

    raw_secret = _our_private.exchange(their_pub)
    session_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"chutes-miner-validator-v1",
    ).derive(raw_secret)

    _sessions[req.session_id] = session_key
    return {"status": "ok", "session_id": req.session_id}


@app.post("/script")
async def get_script(req: ScriptRequest):
    """
    Step 3. Validator requests the encrypted script for an established session.
    We encrypt with the shared key derived during handshake.
    Ciphertext only ever travels on the wire — plaintext stays on this machine.
    """
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Unknown session")

    key = _sessions[req.session_id]
    iv = os.urandom(12)  # Fresh 96-bit nonce per request — never reuse
    ciphertext = AESGCM(key).encrypt(iv, SECRET_SCRIPT, None)

    # Commit to hash so validator can verify integrity after decryption
    script_hash = hashlib.sha256(SECRET_SCRIPT).hexdigest()

    # Session is single-use: delete key immediately after issuing ciphertext
    del _sessions[req.session_id]

    return {
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "iv": base64.b64encode(iv).decode(),
        "script_hash": script_hash,
    }


if __name__ == "__main__":
    port = int(os.environ.get("MINER_PORT", 9000))
    print(f"[MINER] Starting axon on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)