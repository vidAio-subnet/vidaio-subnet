# Miner-Validator Architecture

## Three Scripts, One Rule: Only the Validator Sees the Miner's Code

```
┌─────────────┐    synapse       ┌──────────────┐
│   MINER     │◄───────────────►│  VALIDATOR   │
│  (Axon)     │  responds with   │ (Orchestrator)│
│             │  chute_id +      │              │
│             │  chute_code      │  inspects    │
└──────┬──────┘                  │  miner code  │
       │                         └──────┬───────┘
       │ deploys                        │
       ▼                                │ calls
┌─────────────┐                         │
│ MINER CHUTE │◄────────────────────────┘
│ (Chutes.ai) │   POST /score
│  or local   │   with input_data
└─────────────┘
```

---

## How It Works (30-second version)

1. **Validator orchestrator** sends a synapse to the miner: "give me your chute info"
2. **Miner** responds with:
   - `chute_api_url` — endpoint to call for scoring (in prod: `api.chutes.ai/v1/chutes/{chute_id}/run/score`)
   - `chute_code` — the miner's source code (in prod: shared via `chutes chutes share` and retrieved at `api.chutes.ai/code/{chute_id}`)
3. **Validator** calls the miner's chute directly with input data and gets back the result
4. **Validator** can inspect the miner's code — only the validator has access

The miner deploys and controls their own chute. The validator never runs the miner's code — they only call the miner's API and inspect the source.

---

## Production Flow (Bittensor Network)

```
Validator                          Miner                        Chutes.ai
   │                                │                              │
   ├── bt.dendrite(synapse) ───────►│                              │
   │                                │                              │
   │◄── synapse response ──────────┤                              │
   │    (chute_id)                  │                              │
   │                                │── chutes deploy ────────────►│
   │                                │                              │
   │── GET api.chutes.ai/code/{id} ────────────────────────────────►│
   │◄── miner source code ─────────────────────────────────────────┤
   │    (via `chutes chutes share`)                                │
   │                                                               │
   │── POST api.chutes.ai/v1/chutes/{id}/run/score ───────────────►│
   │◄── scoring result ────────────────────────────────────────────┤
   │                                                               │
   ├── score & set weights                                         │
```

### Key production commands:

**Miner side:**
```bash
# Deploy chute
chutes build miner_chute:chute --wait --include-cwd
chutes deploy miner_chute:chute --accept-fee

# Share code with validator (validator-only access)
chutes chutes share <chute_id> <validator_user>
```

**Validator side:**
```bash
# Retrieve miner code (after miner shares it)
# GET api.chutes.ai/code/{chute_id}

# Call miner's chute for scoring
# POST api.chutes.ai/v1/chutes/{chute_id}/run/score
```

---

## Local Prototype Flow

The prototype simulates the production flow using local REST API calls:

| Production | Local Prototype |
|---|---|
| `bt.dendrite(synapse)` → miner axon | `POST localhost:9000/synapse` |
| Miner responds with `chute_id` via synapse | Miner responds with `chute_api_url` + `chute_code` |
| `api.chutes.ai/v1/chutes/{id}/run/score` | `POST localhost:8080/score` |
| `api.chutes.ai/code/{chute_id}` | `chute_code` field in synapse response |
| `chutes chutes share` | Miner reads `miner_chute.py` and includes it in response |

---

## Who Sees What

| Who | Sees miner's code? | Sees results? | Can call miner's chute? |
|---|---|---|---|
| Validator | **Yes** (via `chutes share` / `api.chutes.ai/code/{chute_id}`) | **Yes** | **Yes** |
| Other miners | **No** | **No** | **No** |
| Random internet user | **No** | **No** | **No** (Chutes API requires auth) |
| Chutes platform | Code is stored but access-controlled | **No** | N/A |
