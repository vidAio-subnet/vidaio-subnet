# Secure Miner-Validator Architecture

## Three Scripts, One Rule: The Miner's Script Stays Secret

```
┌─────────────┐    encrypted     ┌──────────────┐    result only    ┌──────────────┐
│   MINER     │◄───────────────►│  CHUTE (TEE) │────────────────►│  VALIDATOR   │
│  (VM1)      │   script never   │ (Chutes.ai)  │  no script, no   │  (VM2)       │
│             │   leaves miner   │  hw-isolated  │  keys, nothing   │              │
│             │   unencrypted    │  memory       │                  │              │
└─────────────┘                  └──────┬───────┘                  └──────────────┘
                                        │                                  │
                                   password-gated                   calls chute via
                                   (only validator                  Chutes HTTPS API
                                    can trigger)
                                        │
                                   S3 upload
                                   (result JSON →
                                    S3-compatible
                                    bucket; creds
                                    TEE-only)
```

---

## How It Works (30-second version)

1. **Validator orchestrator** asks the Chute: "run this miner's script on this input data"
2. **Chute** (running inside Intel TDX hardware enclave) contacts the miner, does a cryptographic handshake, receives the encrypted script, decrypts it in hardware-isolated memory, runs it, and sends back *only the result*
3. **Validator gets back** `{"sum": 15, "mean": 3.0, "count": 5}` — never the script

The validator cannot intercept the script because it never passes through the validator's machine. The encrypted bytes travel directly between the miner and the Chute over a separate network connection that the validator doesn't participate in.

---

## Why the Validator Can't Cheat

The validator orchestrator makes a single HTTPS call to the Chute's `/execute` endpoint. That request contains the miner's URL and the input data. That's it. The response contains the result. There is no endpoint, parameter, log, or mechanism for the validator to request the script — encrypted or otherwise. The miner sends the encrypted script directly to the Chute over a connection the miner initiated with the Chute, not routed through the validator.

Even if the validator operator adds logging, packet capture, or middleware to their own VM, they see:

- The miner URL (they already know this)
- The input data (they chose it)
- The result dict
- **Nothing else**

---

## How TEE Verification Works

The scripts don't try to access TDX hardware directly (Chutes doesn't expose `/dev/tdx_guest` to user code). Instead, they rely on Chutes' own infrastructure attestation:

**Before any chute pod even starts**, Chutes' sek8s system performs hardware attestation — measuring firmware, bootloader, and kernel into TDX registers, generating a signed quote, and verifying it against Intel's root of trust. The disk only decrypts if this passes. The chute image only deploys if it's cosign-signed by Chutes' build system.

**The miner verifies this** by querying the Chutes API independently:

- `GET /chutes/{id}` → confirms `tee=True` and checks the image identity
- `GET /servers/{id}/attestation/status` → confirms TDX attestation passed

The miner doesn't trust the validator's word — it calls the Chutes API itself.

---

## How Miners Verify the Chute Code

The Chute image is built by Chutes' forge system, which hashes the filesystem and Python bytecode, then signs the image with cosign. The miner can:

1. Check the chute's **Source tab** on chutes.ai to see the exact code
2. Verify the **image name and tag** matches expectations via the API
3. Know that sek8s **only admits pods with cosign-verified images** — no modifications possible

The miner configures `EXPECTED_IMAGE_NAME=youruser/secure-validator:0.3` and the verification happens automatically during each handshake.

---

## What the Execution Password Does (and Doesn't Do)

The password **gates who can trigger the chute**. Without it, the chute rejects the call before doing anything — no miner contact, no crypto, no resources spent. Only the validator orchestrator knows the password.

The password has **nothing to do with script secrecy**. The script is protected by the TEE + encryption, not the password. The password simply prevents strangers from using your chute.

Setup:
```bash
chutes secrets create --purpose secure-validator \
  --key VALIDATOR_EXEC_PASSWORD --value "$(openssl rand -hex 32)"
```

Rotation (no redeploy):
```bash
python validator_orchestrator.py --rotate
```

---

## S3 Result Upload

After executing the miner script the chute optionally uploads the result JSON to any S3-compatible bucket (AWS S3, Cloudflare R2, MinIO, etc.).

### Credential flow

| Where stored | Who can read it | How |
|---|---|---|
| Chutes secrets (S3_ENDPOINT_URL, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY) | Only the TEE | Injected as env vars; TDX-encrypted memory |
| Orchestrator env (S3_BUCKET, S3_RESULT_PREFIX) | Validator VM only | Used only to construct the object key — never holds write credentials |

The orchestrator supplies a **key path** (`result_s3_key`, e.g. `results/{miner_hotkey}/{uuid}.json`) with each `/execute` call. The chute writes to that key using its own credentials. The orchestrator never touches the S3 write credentials.

### Key generation

```
S3_RESULT_PREFIX / miner_hotkey / <uuid4_hex>.json
```

Example: `results/5FHneW46.../a3f1b2c0d4e5f6a7b8c9d0e1f2a3b4c5.json`

### Response fields

| Field | Description |
|---|---|
| `result_s3_uri` | `s3://bucket/key` if upload succeeded; `null` if skipped or failed |

S3 upload failure is **non-fatal** — the result is still returned in the response body. The orchestrator logs a warning if a key was requested but no URI came back.

### Setup

```bash
# Store write credentials as Chutes secrets (TEE-only)
chutes secrets create --purpose secure-validator \
  --key S3_ENDPOINT_URL --value "https://s3.amazonaws.com"
chutes secrets create --purpose secure-validator \
  --key S3_BUCKET       --value "my-validator-results"
chutes secrets create --purpose secure-validator \
  --key S3_ACCESS_KEY   --value "<write-access-key-id>"
chutes secrets create --purpose secure-validator \
  --key S3_SECRET_KEY   --value "<write-secret-access-key>"

# Orchestrator only needs bucket name for key construction
export S3_BUCKET="my-validator-results"
export S3_RESULT_PREFIX="results"  # optional
```

---

## Can Anyone Get the Script? A Threat Checklist

| Who | Can they get the script? | Why |
|---|---|---|
| Validator operator | **No** | Script never passes through their VM |
| Random internet user | **No** | Chute is password-gated |
| Chutes platform operator | **No** | TDX hardware-encrypts memory; even the host can't read it |
| Someone sniffing the network | **No** | Script is AES-256-GCM encrypted with per-session ephemeral keys |
| The miner themselves | **Yes** | It's their script — they wrote it |