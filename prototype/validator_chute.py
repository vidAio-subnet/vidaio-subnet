# validator_chute.py
#
# LOCAL DEV:
#   Terminal 1:  python miner_axon.py (OR) pm2 start "PYTHONPATH=. python3 miner_axon.py" --name miner-axon
#   pip install chutes cryptography httpx
#   Terminal 2:  chutes build validator_chute:chute --local
#   Terminal 3:  docker run --rm -it \
#             -e CHUTES_EXECUTION_CONTEXT=REMOTE \
#             -p 8080:8080 \
#             --add-host=host.docker.internal:host-gateway \
#             secure-validator:0.1 /bin/sh
#   Terminal 4 (inside container):  chutes run validator_chute:chute --dev --port 8080
#   Terminal 5:  curl -X POST http://localhost:8080/execute \
#   -H "Content-Type: application/json" \
#   -d '{"miner_url":"http://host.docker.internal:9000","input_data":{"values":[1,2,3]}}'
# Expected response:
# {"status": "ok", "result": {"sum": 6, "mean": 2.0, "count": 3}}

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


# ── Image identity — hardcoded at build time ─────────────────────────────────
# SHA-256 of the expected Chutes image. The miner checks this against the
# attestation quote to confirm the validator is running the correct, unmodified
# image.  Recalculate whenever the image changes:
#   chutes image digest youruser/secure-validator:0.2
EXPECTED_IMAGE_DIGEST = "sha256:PLACEHOLDER_REPLACE_AFTER_BUILD"


# ── Startup ───────────────────────────────────────────────────────────────────
@chute.on_startup(priority=10)
async def generate_ephemeral_keypair(self):
    """
    Generate a fresh ECDH keypair on every startup.
    The public key is included in the attestation evidence so the miner
    can bind the TDX quote to *this* key — preventing replay or relay.
    """
    import base64
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    self._ecdh_private = X25519PrivateKey.generate()
    pub_bytes = self._ecdh_private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    self._ecdh_pubkey_bytes = pub_bytes
    self._ecdh_pubkey_b64 = base64.b64encode(pub_bytes).decode()

    # Pre-compute the attestation evidence once at startup so every
    # handshake can supply it without regenerating.
    self._attestation_evidence = _build_attestation_evidence(pub_bytes)


# ── Attestation helpers ───────────────────────────────────────────────────────
def _build_attestation_evidence(pubkey_bytes: bytes) -> dict:
    """
    Collect TDX attestation evidence, binding the ECDH pubkey into the
    report_data field so the miner can verify key-to-enclave binding.

    On real TDX hardware (deployed via Chutes) this reads from the
    /dev/tdx_guest device.  In local dev mode it returns a stub.
    """
    import base64
    import hashlib
    import os

    # report_data = SHA-256(ecdh_pubkey) padded to 64 bytes
    # This binds the attestation quote to the specific ephemeral key.
    report_data = hashlib.sha256(pubkey_bytes).digest().ljust(64, b"\x00")

    tdx_device = "/dev/tdx_guest"
    if os.path.exists(tdx_device):
        # ── Real TDX path ────────────────────────────────────────────
        quote_bytes = _get_tdx_quote(tdx_device, report_data)
        return {
            "type": "tdx",
            "quote": base64.b64encode(quote_bytes).decode(),
            "report_data": base64.b64encode(report_data).decode(),
            "pubkey": base64.b64encode(pubkey_bytes).decode(),
        }
    else:
        # ── Local dev stub ───────────────────────────────────────────
        return {
            "type": "dev-stub",
            "report_data": base64.b64encode(report_data).decode(),
            "pubkey": base64.b64encode(pubkey_bytes).decode(),
            "warning": "NOT a real attestation — local dev mode only",
        }


def _get_tdx_quote(device_path: str, report_data: bytes) -> bytes:
    """
    Request a TDX quote from the guest device.
    The kernel exposes /dev/tdx_guest; we write report_data and read
    back the signed quote.  This is a simplified wrapper — production
    code should use the full ioctl interface or Intel's DCAP library.
    """
    import struct

    # TDX_CMD_GET_REPORT ioctl (simplified — real implementation would
    # use fcntl.ioctl with the proper TDX structs)
    try:
        with open(device_path, "rb+") as f:
            # Write the 64-byte report_data
            f.write(report_data)
            f.flush()
            # Read back the quote (up to 8 KB typical)
            quote = f.read(8192)
        return quote
    except Exception as e:
        # Fallback: return empty quote so caller can handle gracefully
        return b""


def _restricted_builtins() -> dict:
    """
    Return a locked-down __builtins__ dict that removes dangerous
    functions.  The executed script can still do arithmetic, string ops,
    list comprehensions, etc., but cannot import modules, open files,
    eval arbitrary code, or access the runtime internals.
    """
    import builtins

    BLOCKED = {
        "__import__",
        "open",
        "exec",
        "eval",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "type",
        "__build_class__",
        "breakpoint",
        "exit",
        "quit",
        "input",
        "memoryview",
        "help",
    }

    safe = {}
    for name in dir(builtins):
        if name not in BLOCKED:
            safe[name] = getattr(builtins, name)

    return safe


# ── Cord ──────────────────────────────────────────────────────────────────────
@chute.cord(
    public_api_path="/execute",
    public_api_method="POST",
    input_schema=ExecuteRequest,
    output_schema=ExecuteResponse,
)
async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
    import base64
    import hashlib
    import sys
    import uuid

    import httpx
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    # ── Per-session ephemeral key (forward secrecy) ──────────────────
    session_private = X25519PrivateKey.generate()
    session_pub_bytes = session_private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw,
    )
    session_pub_b64 = base64.b64encode(session_pub_bytes).decode()

    session_id = str(uuid.uuid4())
    miner_url = request.miner_url.rstrip("/")

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Step 1: Send attestation + ephemeral pubkey, get miner's pubkey
        try:
            r = await client.post(
                f"{miner_url}/handshake",
                json={
                    "session_id": session_id,
                    "validator_pubkey": session_pub_b64,
                    "attestation": self._attestation_evidence,
                },
            )
            r.raise_for_status()
            resp = r.json()
            miner_pubkey_b64 = resp["miner_pubkey"]
        except Exception as e:
            return ExecuteResponse(
                status="error", error=f"handshake failed: {e}"
            )

        # Step 2: Derive shared secret from ephemeral keys
        try:
            miner_pub = X25519PublicKey.from_public_bytes(
                base64.b64decode(miner_pubkey_b64)
            )
            raw_secret = session_private.exchange(miner_pub)
            session_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"chutes-miner-validator-v1",
            ).derive(raw_secret)
        except Exception as e:
            return ExecuteResponse(
                status="error", error=f"key derivation failed: {e}"
            )

        # Step 3: Request encrypted script
        try:
            r = await client.post(
                f"{miner_url}/script",
                json={"session_id": session_id},
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            return ExecuteResponse(
                status="error", error=f"script fetch failed: {e}"
            )

        # Step 4: Decrypt inside TDX — never log script_bytes
        try:
            script_bytes = AESGCM(session_key).decrypt(
                base64.b64decode(payload["iv"]),
                base64.b64decode(payload["ciphertext"]),
                None,
            )
        except Exception:
            return ExecuteResponse(status="error", error="decryption failed")

        if hashlib.sha256(script_bytes).hexdigest() != payload["script_hash"]:
            return ExecuteResponse(
                status="error", error="integrity check failed"
            )

        # Step 5: Execute inside TDX with restricted builtins
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
            # Scrub sensitive material
            script_bytes = b"\x00" * len(script_bytes)
            session_key = b"\x00" * 32
            raw_secret = b"\x00" * 32
            del script_bytes, session_key, raw_secret, sandbox, session_private