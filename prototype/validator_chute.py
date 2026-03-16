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
# S3 RESULT UPLOAD (optional):
#   After executing the miner script the chute uploads the result JSON to any
#   S3-compatible bucket (AWS S3, Cloudflare R2, MinIO, …).  Credentials live
#   entirely as Chutes secrets — never sent over the wire by the orchestrator:
#
#   chutes secrets create --purpose secure-validator \
#     --key S3_ENDPOINT_URL --value "https://s3.amazonaws.com"
#   chutes secrets create --purpose secure-validator \
#     --key S3_BUCKET       --value "my-validator-results"
#   chutes secrets create --purpose secure-validator \
#     --key S3_ACCESS_KEY   --value "<access-key-id>"
#   chutes secrets create --purpose secure-validator \
#     --key S3_SECRET_KEY   --value "<secret-access-key>"
#
#   The orchestrator passes a `result_s3_key` string per /execute call
#   (e.g. "results/{miner_hotkey}/{session_id}.json").  The chute writes the
#   result there and returns the s3:// URI in the response.  If any credential
#   is absent the upload step is silently skipped.
#
# ATTESTATION MODEL (Option A — Chutes platform attestation):
#   No /dev/tdx_guest access. Attestation is handled by sek8s
#   infrastructure. The orchestrator passes platform attestation
#   proof through the chute to the miner.
#
# SCRIPT ISOLATION:
#   The miner's script executes inside a locked-down sandbox:
#     - Restricted builtins (no __import__, open, exec, eval, getattr, type …)
#     - Source-level audit rejects dunder escape hatches (__subclasses__,
#       __globals__, __bases__, __code__, __closure__, __mro__, …)
#     - Network kill-switch: socket monkey-patched + network modules purged
#       from sys.modules for the duration of the script.  The miner cannot
#       upload to external buckets, stream logs via wandb, or exfiltrate data.
#     - Network is restored AFTER the script returns so the chute's own
#       S3 upload (validator-owned credentials) still works.
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
#   export VALIDATOR_EXEC_PASSWORD="$(openssl rand -hex 32)"
#   chutes secrets create --purpose secure-validator \
#     --key VALIDATOR_EXEC_PASSWORD --value "$VALIDATOR_EXEC_PASSWORD"

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
    .run_command("pip install cryptography httpx boto3")
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
    # Optional S3 object key to store the result at.
    # The chute derives the full URI from S3_* secrets; the orchestrator only
    # supplies the key path (e.g. "results/{miner}/{session}.json").
    result_s3_key: str | None = None


class ExecuteResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None
    # Populated when the result was successfully written to S3.
    result_s3_uri: str | None = None


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


# ── Script safety audit ──────────────────────────────────────────────────────
#
# Conservative source-level scan.  Any match — even inside comments — causes
# rejection.  Intentionally strict: false-positives are preferred over
# false-negatives.

_BLOCKED_DUNDER_ATTRS = (
    b"__subclasses__",  # object introspection → sandbox escape
    b"__globals__",     # function.__globals__ → module namespace access
    b"__code__",        # code object manipulation
    b"__closure__",     # closure variable access
    b"__bases__",       # class hierarchy traversal
    b"__mro__",         # method-resolution-order traversal
    b"__loader__",      # module loader access
    b"__spec__",        # module spec access
    b"__import__",      # redundant (blocked in builtins), belt-and-suspenders
)


def _audit_script(src: bytes) -> str | None:
    """
    Reject scripts that reference dangerous dunder attributes.

    Returns an error string if the script is rejected, None if it passes.
    """
    for pat in _BLOCKED_DUNDER_ATTRS:
        if pat in src:
            return f"blocked attribute in script source: {pat.decode()}"
    return None


# ── Network isolation ────────────────────────────────────────────────────────
#
# Ensures the miner's script cannot reach the internet during execution —
# no uploading to external buckets, no streaming telemetry, no exfiltration.
#
# Layers (in order of execution):
#   1. Purge network-capable modules from sys.modules so the script can't
#      import-by-cache even though __import__ is blocked in builtins.
#   2. Monkey-patch socket.socket / getaddrinfo / create_connection to raise
#      RuntimeError so any residual path to a raw socket is cut.
#
# All mutations are reverted in the `finally` block so that the chute's own
# S3 upload (which runs AFTER the script) works normally.
#
# Production hardening (beyond this prototype):
#   - unshare(2) --net: empty network namespace, zero interfaces
#   - nsjail / gVisor: syscall-level sandboxing
#   - Dedicated no-network container sidecar for script execution

import contextlib

_PURGE_MODULE_PREFIXES = (
    "socket", "ssl", "http", "urllib", "requests", "httpx",
    "aiohttp", "boto3", "botocore", "wandb", "websocket",
    "ftplib", "smtplib", "imaplib", "poplib", "telnetlib",
    "xmlrpc", "socketserver", "subprocess", "ctypes",
    "multiprocessing",
)


@contextlib.contextmanager
def _network_disabled():
    """Kill network access for the duration of the block, then restore it."""
    import socket as _socket_mod
    import sys

    # ── save originals ────────────────────────────────────────────────
    _orig_socket = _socket_mod.socket
    _orig_getaddrinfo = _socket_mod.getaddrinfo
    _orig_create_conn = _socket_mod.create_connection

    # ── purge network modules from import cache ───────────────────────
    _purged: dict = {}
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name == p or mod_name.startswith(p + ".")
            for p in _PURGE_MODULE_PREFIXES
        ):
            _purged[mod_name] = sys.modules.pop(mod_name)

    # ── monkey-patch socket primitives ────────────────────────────────
    def _blocked(*_a, **_kw):
        raise RuntimeError("network access disabled during miner script execution")

    _socket_mod.socket = _blocked
    _socket_mod.getaddrinfo = _blocked
    _socket_mod.create_connection = _blocked

    try:
        yield
    finally:
        _socket_mod.socket = _orig_socket
        _socket_mod.getaddrinfo = _orig_getaddrinfo
        _socket_mod.create_connection = _orig_create_conn
        sys.modules.update(_purged)


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

    # ── S3 result-upload credentials ─────────────────────────────
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL", "")
    s3_bucket   = os.environ.get("S3_BUCKET", "")
    s3_access   = os.environ.get("S3_ACCESS_KEY", "")
    s3_secret   = os.environ.get("S3_SECRET_KEY", "")

    if s3_endpoint and s3_bucket and s3_access and s3_secret:
        self._s3_config = {
            "endpoint_url": s3_endpoint,
            "bucket":       s3_bucket,
            "access_key":   s3_access,
            "secret_key":   s3_secret,
        }
        print(f"[CHUTE] S3 upload configured → bucket '{s3_bucket}' @ {s3_endpoint}")
    else:
        self._s3_config = None
        missing = [
            k for k, v in {
                "S3_ENDPOINT_URL": s3_endpoint,
                "S3_BUCKET":       s3_bucket,
                "S3_ACCESS_KEY":   s3_access,
                "S3_SECRET_KEY":   s3_secret,
            }.items() if not v
        ]
        print(
            f"[CHUTE] WARNING: S3 upload disabled — missing secrets: {missing}. "
            "Results will only be returned in the response body."
        )

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


# ── S3 upload helper ─────────────────────────────────────────────────────────
async def _upload_result_to_s3(s3_config: dict, key: str, result: dict) -> str:
    """
    Upload *result* as UTF-8 JSON to the configured S3-compatible bucket.

    Runs boto3 (sync) in a thread so it doesn't block the async event loop.
    Returns the full s3:// URI on success, raises on failure.

    The credentials live inside TEE-encrypted memory (loaded at startup from
    Chutes secrets) and are never exposed to the orchestrator or the network.
    """
    import asyncio
    import json

    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    payload = json.dumps(result, separators=(",", ":")).encode()

    def _sync_put():
        client = boto3.client(
            "s3",
            endpoint_url=s3_config["endpoint_url"],
            aws_access_key_id=s3_config["access_key"],
            aws_secret_access_key=s3_config["secret_key"],
        )
        client.put_object(
            Bucket=s3_config["bucket"],
            Key=key,
            Body=payload,
            ContentType="application/json",
        )
        return f"s3://{s3_config['bucket']}/{key}"

    try:
        uri = await asyncio.to_thread(_sync_put)
        return uri
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 upload failed: {exc}") from exc


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

        # Step 4½: Static audit — reject scripts with sandbox-escape patterns
        audit_err = _audit_script(script_bytes)
        if audit_err:
            return ExecuteResponse(status="error", error=audit_err)

        # Step 5: Execute with restricted builtins + network disabled
        #
        # The miner's script defines `score(data)` and never references
        # input_data directly — the chute calls score() with input_data.
        #
        # _network_disabled() ensures the script cannot reach the internet
        # (no uploads to miner-owned buckets, no wandb, no exfiltration).
        # Network is restored AFTER the script finishes so the chute's own
        # S3 upload still works.
        sys.settrace(None)
        sys.setprofile(None)

        sandbox = {
            "__builtins__": _restricted_builtins(),
        }

        s3_uri: str | None = None
        try:
            with _network_disabled():
                exec(  # noqa: S102
                    compile(script_bytes, "<miner_script>", "exec"),
                    sandbox,
                )

                entry_fn = sandbox.get("score")
                if not callable(entry_fn):
                    return ExecuteResponse(
                        status="error",
                        error="script must define a callable named 'score(data)'",
                    )
                exec_result = entry_fn(request.input_data)

            # ── S3 upload (network restored — chute's own credentials) ──
            if (
                exec_result is not None
                and self._s3_config is not None
                and request.result_s3_key
            ):
                try:
                    s3_uri = await _upload_result_to_s3(
                        self._s3_config, request.result_s3_key, exec_result
                    )
                    print(f"[CHUTE] Result uploaded → {s3_uri}")
                except Exception as s3_err:
                    # Upload failure is non-fatal: result is still returned
                    print(f"[CHUTE] S3 upload error (non-fatal): {s3_err}")

            return ExecuteResponse(
                status="ok", result=exec_result, result_s3_uri=s3_uri
            )
        except Exception as e:
            return ExecuteResponse(status="error", error=type(e).__name__)
        finally:
            script_bytes = b"\x00" * len(script_bytes)
            session_key = b"\x00" * 32
            raw_secret = b"\x00" * 32
            del script_bytes, session_key, raw_secret, sandbox, session_private