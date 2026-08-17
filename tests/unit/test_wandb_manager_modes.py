from types import SimpleNamespace
from unittest.mock import patch

from vidaio_subnet_core.utilities.wandb_manager import WandbManager


def validator_stub():
    return SimpleNamespace(
        uid=7,
        validator_mode="competition",
        config=SimpleNamespace(
            netuid=85,
            subtensor=SimpleNamespace(network="finney"),
            wandb=SimpleNamespace(off=False),
        ),
        wallet=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="validator-hotkey")
        ),
    )


def test_competition_wandb_identity_has_suffix_for_configured_values():
    environment = {
        "WANDB_RUN_ID": "shared-validator-run",
        "WANDB_RUN_NAME": "validator-seven",
    }
    with patch.dict("os.environ", environment, clear=True):
        manager = WandbManager(
            validator=validator_stub(),
            run_suffix="-competition",
        )
        run_id = manager._get_run_id()

    assert manager.run_name == "validator-seven-competition"
    assert manager.run_group.endswith("-competition")
    assert run_id == "shared-validator-run-competition"


def test_inference_wandb_identity_remains_unchanged():
    environment = {
        "WANDB_RUN_ID": "shared-validator-run",
        "WANDB_RUN_NAME": "validator-seven",
    }
    with patch.dict("os.environ", environment, clear=True):
        manager = WandbManager(validator=validator_stub())
        run_id = manager._get_run_id()

    assert manager.run_name == "validator-seven"
    assert not manager.run_group.endswith("-competition")
    assert run_id == "shared-validator-run"
