# Miner-Validator Prototype

Simulates the Bittensor subnet flow where miners deploy their own chutes and validators call them for scoring.

## Quick Start

```bash
# Terminal 1: Miner's chute (scoring service)
python miner_chute.py

# Terminal 2: Miner axon (responds to validator synapses)
python miner_axon.py

# Terminal 3: Validator orchestrator
python validator_orchestrator.py
```

See [setup.md](setup.md) for full setup and production deployment instructions.
See [architecture.md](architecture.md) for the architecture overview.
