# validator_chute.py
#
# TEE COMPONENT — runs ONLY inside Chutes (TDX enclave).
#
# ACCESS CONTROL:
#   The chute is gated by a "validator execution password" that ONLY the
#   orchestrator knows. This prevents random callers from triggering the
#   chute to contact miners and waste resources or probe for info.
#
#   Setup (one-time, before or after deploy):
#     chutes secrets create \
#       --purpose secure-validator \
#       --key VALIDATOR_EXEC_PASSWORD \
#       --value "$(openssl rand -hex 32)"
#
#   The password is injected as an env var inside the TEE. The
#   orchestrator must include it in every /execute request and can
#   rotate it at runtime via /rotate-password.
#
# ATTESTATION MODEL (Option A — Chutes platform attestation):
#   No /dev/tdx_guest access. Attestation is handled by sek8s
#   infrastructure. The orchestrator passes platform attestation
#   proof through the chute to the miner.
#
# ARCHITECTURE:
#   Miner (VM1) ←─encrypted─→ Chute (TEE) ←─result only─→ Validator (VM2)
#                                  ↑
#                          password-gated
#
# LOCAL DEV:
#   Terminal 1:  python miner_axon.py
#   Terminal 2:  chutes build validator_chute:chute --local
#   Terminal 3:  docker run --rm -it \
#             -e CHUTES_EXECUTION_CONTEXT=REMOTE \
#             -e VALIDATOR_EXEC_PASSWORD=devpass123 \
#             -p 8080:8080 \
#             --add-host=host.docker.internal:host-gateway \
#             secure-validator:0.3 /bin/sh
#   Terminal 4 (inside container):  chutes run validator_chute:chute --dev --port 8080
#   Terminal 5:  python validator_orchestrator.py
#
# DEPLOY:
#   chutes build validator_chute:chute --wait
#   chutes deploy validator_chute:chute --accept-fee
#   chutes secrets create --purpose secure-validator \
#     --key VALIDATOR_EXEC_PASSWORD --value "$(openssl rand -hex 32)"

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
    exec_password: str  # must match VALIDATOR_EXEC_PASSWORD
    platform_attestation: dict | None = None


class ExecuteResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None


class RotatePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class RotatePasswordResponse(BaseModel):
    status: str
    error: str | None = None


# ── Sandbox helpers ───────────────────────────────────────────────────────────
def _restricted_builtins() -> dict:
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
    """
    Load the execution password from the Chutes secret (env var).

    The password is stored in self._exec_password and lives only in
    TDX-encrypted memory. It can be rotated at runtime via /rotate-password.
    """
    import hashlib
    import os

    raw_password = os.environ.get("VALIDATOR_EXEC_PASSWORD", "")

    if not raw_password:
        print(
            "[CHUTE] WARNING: VALIDATOR_EXEC_PASSWORD not set. "
            "All /execute calls will be rejected until a password is set "
            "via /rotate-password. Set it with:\n"
            "  chutes secrets create --purpose secure-validator "
            "--key VALIDATOR_EXEC_PASSWORD --value <password>"
        )
        self._exec_password_hash = None
    else:
        # Store only the hash — never keep the plaintext password in memory
        # longer than necessary
        self._exec_password_hash = hashlib.sha256(
            raw_password.encode()
        ).hexdigest()
        print("[CHUTE] Execution password loaded from VALIDATOR_EXEC_PASSWORD")

    self._is_tee = os.path.exists("/dev/tdx_guest") or os.environ.get(
        "CHUTES_EXECUTION_CONTEXT"
    ) == "REMOTE"
    env_label = "TEE (Chutes)" if self._is_tee else "local dev"
    print(f"[CHUTE] Started in {env_label} mode")


# ── Password verification helper ─────────────────────────────────────────────
def _verify_password(self, password: str) -> bool:
    """Constant-time comparison of password against stored hash."""
    import hashlib
    import hmac

    if self._exec_password_hash is None:
        return False

    candidate_hash = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(candidate_hash, self._exec_password_hash)


# ── Rotate password cord ──────────────────────────────────────────────────────
@chute.cord(
    public_api_path="/rotate-password",
    public_api_method="POST",
    input_schema=RotatePasswordRequest,
    output_schema=RotatePasswordResponse,
)
async def rotate_password(
    self, request: RotatePasswordRequest
) -> RotatePasswordResponse:
    """
    Rotate the execution password at runtime without redeploying.

    Requires the current password to authorize the change.
    If no password was set at startup, current_password must be
    the special bootstrap value "BOOTSTRAP_NO_PASSWORD_SET".

    After rotating, update VALIDATOR_EXEC_PASSWORD in the orchestrator
    env and optionally update the Chutes secret for persistence across
    restarts:
      chutes secrets delete <old_secret_id>
      chutes secrets create --purpose secure-validator \
        --key VALIDATOR_EXEC_PASSWORD --value <new_password>
    """
    import hashlib

    # Authorize the rotation
    if self._exec_password_hash is None:
        # No password was set — allow bootstrap with magic string
        if request.current_password != "BOOTSTRAP_NO_PASSWORD_SET":
            return RotatePasswordResponse(
                status="error",
                error="no password set; use 'BOOTSTRAP_NO_PASSWORD_SET' as current_password to bootstrap",
            )
    else:
        if not _verify_password(self, request.current_password):
            return RotatePasswordResponse(status="error", error="invalid current password")

    if len(request.new_password) < 16:
        return RotatePasswordResponse(
            status="error", error="new password must be at least 16 characters"
        )

    self._exec_password_hash = hashlib.sha256(
        request.new_password.encode()
    ).hexdigest()

    print("[CHUTE] Execution password rotated successfully")
    return RotatePasswordResponse(status="ok")


# ── Execute cord ──────────────────────────────────────────────────────────────
@chute.cord(
    public_api_path="/execute",
    public_api_method="POST",
    input_schema=ExecuteRequest,
    output_schema=ExecuteResponse,
)
async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
    """
    Full miner handshake → decrypt → execute → return result.

    GATED: requires valid exec_password matching VALIDATOR_EXEC_PASSWORD.
    """
    # ── Password gate ────────────────────────────────────────────────
    if not _verify_password(self, request.exec_password):
        return ExecuteResponse(status="error", error="unauthorized")

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

    attestation_payload = request.platform_attestation or {
        "type": "dev-stub",
        "source": "chute",
        "warning": "No platform attestation — local dev mode",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Step 1: Handshake
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
            script_bytes = b"\x00" * len(script_bytes)
            session_key = b"\x00" * 32
            raw_secret = b"\x00" * 32
            del script_bytes, session_key, raw_secret, sandbox, session_private