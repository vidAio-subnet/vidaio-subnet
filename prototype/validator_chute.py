# validator_chute.py
#
# TEE COMPONENT — runs ONLY inside Chutes (TDX enclave).
#
# ATTESTATION MODEL (Option A — Chutes platform attestation):
#   This chute does NOT access /dev/tdx_guest or generate TDX quotes.
#   Instead, attestation is handled by Chutes' infrastructure:
#     1. sek8s performs boot attestation (RTMR measurements → TD Quote)
#     2. Chutes validator verifies quote against Intel root of trust
#     3. LUKS passphrase released only if measurements match golden config
#     4. Pod admitted only if image is cosign-signed by Chutes forge
#   The miner verifies this chain via the Chutes API before sending
#   the encrypted script (see validator_orchestrator.py).
#
#   This chute's job is simpler: receive miner_url + input_data,
#   do ECDH handshake with the miner, decrypt script, execute it,
#   return only the result. All crypto happens in TDX-encrypted RAM.
#
# ARCHITECTURE:
#   Miner (VM1) ←─encrypted─→ Chute (TEE) ←─result only─→ Validator (VM2)
#
# LOCAL DEV:
#   Terminal 1:  python miner_axon.py
#   Terminal 2:  chutes build validator_chute:chute --local
#   Terminal 3:  docker run --rm -it \
#             -e CHUTES_EXECUTION_CONTEXT=REMOTE \
#             -p 8080:8080 \
#             --add-host=host.docker.internal:host-gateway \
#             secure-validator:0.3 /bin/sh
#   Terminal 4 (inside container):  chutes run validator_chute:chute --dev --port 8080
#   Terminal 5:  python validator_orchestrator.py
#
# DEPLOY:
#   chutes build validator_chute:chute --wait
#   chutes deploy validator_chute:chute --accept-fee

from chutes.image import Image
from chutes.chute import Chute
from pydantic import BaseModel

image = (
    Image(
        username="youruser",
        name="secure-validator",
        tag="0.3",
    )
    .from_base("parachutes/python:3.12")
    .run_command("pip install cryptography httpx")
)

chute = Chute(
    username="youruser",
    name="secure-validator",
    readme="## Secure validator — TEE-backed encrypted script executor",
    image=image,
    tee=True,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ExecuteRequest(BaseModel):
    miner_url: str
    input_data: dict
    # The orchestrator passes the platform attestation proof so the chute
    # can forward it to the miner during the handshake.
    platform_attestation: dict | None = None


class ExecuteResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None


# ── Sandbox helpers ───────────────────────────────────────────────────────────
def _restricted_builtins() -> dict:
    """
    Locked-down __builtins__ for the miner script sandbox.

    Allows: arithmetic, string ops, list/dict comprehensions, sum, len,
            range, enumerate, zip, map, filter, sorted, min, max, etc.

    Blocks: __import__, open, exec, eval, compile, getattr, setattr,
            delattr, type, globals, locals, vars, dir, breakpoint, etc.
    """
    import builtins

    BLOCKED = {
        "__import__", "open", "exec", "eval", "compile",
        "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr",
        "type", "__build_class__",
        "breakpoint", "exit", "quit", "input",
        "memoryview", "help",
    }

    safe = {}
    for name in dir(builtins):
        if name not in BLOCKED:
            safe[name] = getattr(builtins, name)
    return safe


# ── Startup ───────────────────────────────────────────────────────────────────
@chute.on_startup(priority=10)
async def startup(self):
    """Startup — no TDX device access needed."""
    import os

    self._is_tee = os.path.exists("/dev/tdx_guest") or os.environ.get(
        "CHUTES_EXECUTION_CONTEXT"
    ) == "REMOTE"
    env_label = "TEE (Chutes)" if self._is_tee else "local dev"
    print(f"[CHUTE] Started in {env_label} mode")


# ── Cord ──────────────────────────────────────────────────────────────────────
@chute.cord(
    public_api_path="/execute",
    public_api_method="POST",
    input_schema=ExecuteRequest,
    output_schema=ExecuteResponse,
)
async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
    """
    Full miner handshake → decrypt → execute → return result.

    Everything happens inside TDX memory. The validator VM that calls
    this via the Chutes API only ever sees ExecuteResponse — never the
    script, session keys, or any intermediate crypto state.
    """
    import base64
    import hashlib
    import sys
    import uuid

    import httpx
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey, X25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    # ── Per-session ephemeral key → forward secrecy ──────────────────
    session_private = X25519PrivateKey.generate()
    session_pub_bytes = session_private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw,
    )
    session_pub_b64 = base64.b64encode(session_pub_bytes).decode()

    session_id = str(uuid.uuid4())
    miner_url = request.miner_url.rstrip("/")

    # Build the attestation payload for the miner.
    # In production, this contains the Chutes platform attestation proof
    # (server_id, attestation status, chute image hash) that the
    # orchestrator fetched from the Chutes API and passed through.
    # In dev mode, it's a stub.
    attestation_payload = request.platform_attestation or {
        "type": "dev-stub",
        "source": "chute",
        "warning": "No platform attestation — local dev mode",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Step 1: Handshake — send ephemeral pubkey + platform attestation
        try:
            r = await client.post(f"{miner_url}/handshake", json={
                "session_id": session_id,
                "validator_pubkey": session_pub_b64,
                "attestation": attestation_payload,
            })
            r.raise_for_status()
            resp = r.json()
            miner_pubkey_b64 = resp["miner_pubkey"]
        except Exception as e:
            return ExecuteResponse(status="error", error=f"handshake failed: {e}")

        # Step 2: Derive shared secret
        try:
            miner_pub = X25519PublicKey.from_public_bytes(
                base64.b64decode(miner_pubkey_b64)
            )
            raw_secret = session_private.exchange(miner_pub)
            session_key = HKDF(
                algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"chutes-miner-validator-v1",
            ).derive(raw_secret)
        except Exception as e:
            return ExecuteResponse(status="error", error=f"key derivation failed: {e}")

        # Step 3: Get encrypted script
        try:
            r = await client.post(f"{miner_url}/script", json={
                "session_id": session_id,
            })
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            return ExecuteResponse(status="error", error=f"script fetch failed: {e}")

        # Step 4: Decrypt inside TDX
        try:
            script_bytes = AESGCM(session_key).decrypt(
                base64.b64decode(payload["iv"]),
                base64.b64decode(payload["ciphertext"]),
                None,
            )
        except Exception:
            return ExecuteResponse(status="error", error="decryption failed")

        if hashlib.sha256(script_bytes).hexdigest() != payload["script_hash"]:
            return ExecuteResponse(status="error", error="integrity check failed")

        # Step 5: Execute with restricted builtins
        sys.settrace(None)
        sys.setprofile(None)

        sandbox = {
            "__builtins__": _restricted_builtins(),
            "input_data": request.input_data,
            "result": None,
        }

        try:
            exec(  # noqa: S102
                compile(script_bytes, "<miner_script>", "exec"),
                sandbox,
            )
            return ExecuteResponse(status="ok", result=sandbox.get("result"))
        except Exception as e:
            return ExecuteResponse(status="error", error=type(e).__name__)
        finally:
            # Scrub all sensitive material from TDX memory
            script_bytes = b"\x00" * len(script_bytes)
            session_key = b"\x00" * 32
            raw_secret = b"\x00" * 32
            del script_bytes, session_key, raw_secret, sandbox, session_private