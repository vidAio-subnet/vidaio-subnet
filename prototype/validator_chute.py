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
        tag="0.1",
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
    miner_url: str    # e.g. "http://192.168.1.10:9000"
    input_data: dict


class ExecuteResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None


# ── Startup ───────────────────────────────────────────────────────────────────
@chute.on_startup(priority=10)
async def generate_keypair(self):
    import base64
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    self._ecdh_private = X25519PrivateKey.generate()
    self._ecdh_pubkey_b64 = base64.b64encode(
        self._ecdh_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()


# ── Cord — input_schema/output_schema are required for dispatch to work ───────
@chute.cord(
    public_api_path="/execute",
    public_api_method="POST",
    input_schema=ExecuteRequest,    # ← tells dispatch how to parse the body
    output_schema=ExecuteResponse,  # ← tells dispatch how to serialise the response
)
async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
    import base64
    import hashlib
    import sys
    import uuid

    import httpx
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    session_id = str(uuid.uuid4())
    miner_url = request.miner_url.rstrip("/")

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Step 1: Get miner's ECDH pubkey
        try:
            r = await client.get(f"{miner_url}/pubkey")
            r.raise_for_status()
            miner_pubkey_b64 = r.json()["miner_pubkey"]
        except Exception as e:
            return ExecuteResponse(status="error", error=f"pubkey fetch failed: {e}")

        # Step 2: Send our pubkey — both sides derive the shared secret
        try:
            r = await client.post(f"{miner_url}/handshake", json={
                "session_id": session_id,
                "validator_pubkey": self._ecdh_pubkey_b64,
            })
            r.raise_for_status()
        except Exception as e:
            return ExecuteResponse(status="error", error=f"handshake failed: {e}")

        # Derive shared secret inside TDX RAM
        try:
            miner_pub = X25519PublicKey.from_public_bytes(
                base64.b64decode(miner_pubkey_b64)
            )
            raw_secret = self._ecdh_private.exchange(miner_pub)
            session_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"chutes-miner-validator-v1",
            ).derive(raw_secret)
        except Exception as e:
            return ExecuteResponse(status="error", error=f"key derivation failed: {e}")

        # Step 3: Request encrypted script
        try:
            r = await client.post(f"{miner_url}/script", json={
                "session_id": session_id,
            })
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            return ExecuteResponse(status="error", error=f"script fetch failed: {e}")

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
            return ExecuteResponse(status="error", error="integrity check failed")

        # Step 5: Execute inside TDX
        sys.settrace(None)

        sandbox = {
            "__builtins__": __builtins__,
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
            del script_bytes, sandbox