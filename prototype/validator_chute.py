# validator_chute.py
#
# TEE COMPONENT — runs ONLY inside Chutes (TDX enclave).
# This code decrypts and executes the miner's secret script.
# The validator VM operator cannot see the plaintext script because:
#   1. This code runs in hardware-encrypted memory (Intel TDX)
#   2. The Chutes image is cosign-signed and attested — no modifications
#   3. The validator VM only receives the execution *result*, never the script
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
#             secure-validator:0.2 /bin/sh
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
        tag="0.2",
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
    miner_url: str  # e.g. "http://192.168.1.10:9000"
    input_data: dict


class ExecuteResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None


# ── Attestation helpers ───────────────────────────────────────────────────────
def _build_attestation_evidence(pubkey_bytes: bytes) -> dict:
    """
    Collect TDX attestation evidence, binding the ECDH pubkey into the
    report_data field so the miner can verify key-to-enclave binding.

    On real TDX (deployed via Chutes on sek8s) this reads /dev/tdx_guest.
    In local dev mode it returns a clearly-labeled stub.
    """
    import base64
    import hashlib
    import os

    report_data = hashlib.sha256(pubkey_bytes).digest().ljust(64, b"\x00")

    tdx_device = "/dev/tdx_guest"
    if os.path.exists(tdx_device):
        quote_bytes = _get_tdx_quote(tdx_device, report_data)
        return {
            "type": "tdx",
            "quote": base64.b64encode(quote_bytes).decode(),
            "report_data": base64.b64encode(report_data).decode(),
            "pubkey": base64.b64encode(pubkey_bytes).decode(),
        }
    else:
        return {
            "type": "dev-stub",
            "report_data": base64.b64encode(report_data).decode(),
            "pubkey": base64.b64encode(pubkey_bytes).decode(),
            "warning": "NOT a real attestation — local dev mode only",
        }


def _get_tdx_quote(device_path: str, report_data: bytes) -> bytes:
    """
    Request a TDX quote via /dev/tdx_guest.

    On Chutes' sek8s infrastructure the quote is a signed blob containing
    RTMRs and our report_data, verifiable via Intel DCAP.
    """
    try:
        with open(device_path, "rb+") as f:
            f.write(report_data)
            f.flush()
            quote = f.read(8192)
        return quote
    except Exception:
        return b""


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
    """Pre-compute startup attestation for identity verification."""
    import base64
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    self._identity_private = X25519PrivateKey.generate()
    pub_bytes = self._identity_private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    self._identity_pub_b64 = base64.b64encode(pub_bytes).decode()
    self._startup_attestation = _build_attestation_evidence(pub_bytes)


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

    # Per-session ephemeral key → forward secrecy
    session_private = X25519PrivateKey.generate()
    session_pub_bytes = session_private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw,
    )
    session_pub_b64 = base64.b64encode(session_pub_bytes).decode()

    # Bind this ephemeral key to the TDX attestation
    session_attestation = _build_attestation_evidence(session_pub_bytes)

    session_id = str(uuid.uuid4())
    miner_url = request.miner_url.rstrip("/")

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Step 1: Handshake with attestation
        try:
            r = await client.post(f"{miner_url}/handshake", json={
                "session_id": session_id,
                "validator_pubkey": session_pub_b64,
                "attestation": session_attestation,
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
            script_bytes = b"\x00" * len(script_bytes)
            session_key = b"\x00" * 32
            raw_secret = b"\x00" * 32
            del script_bytes, session_key, raw_secret, sandbox, session_private