"""
Preview validator weight setting without running the validator loop.

Default usage stops before the final Bittensor set_weights extrinsic:

    python -m tests.manual.weight_setting_dry_run --netuid 85 --wallet.name <cold> --wallet.hotkey <hot>

To actually emit weights, pass --emit-set-weights. If both --dry-run and
--emit-set-weights are provided, --dry-run wins.

Use --alpha-stake-weigh-factor, --emission-liquidation-weigh-factor, and
--emission-liquidation-window-epochs to compare weighing settings locally.
"""

import argparse
from typing import Any, Iterable

import bittensor as bt
import pandas as pd
from loguru import logger

from vidaio_subnet_core.base.config import add_common_config
from vidaio_subnet_core.global_config import CONFIG
from vidaio_subnet_core.validating.managing.miner_manager import MinerManager


def _scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            return value
    return value


def _weights_df(uids: Iterable[Any], weights: Iterable[Any]) -> pd.DataFrame:
    rows = []
    for uid, weight in zip(uids, weights):
        uid_value = int(_scalar(uid))
        weight_value = float(_scalar(weight))
        rows.append(
            {
                "uid": uid_value,
                "weight": weight_value,
                "weight_percent": weight_value * 100,
            }
        )
    return pd.DataFrame(rows)


def _log_weight_table(title: str, uids: Iterable[Any], weights: Iterable[Any]) -> None:
    df = _weights_df(uids, weights)
    if df.empty:
        logger.info(f"{title}: no weights")
        return
    logger.info(f"{title}:\n{df.to_markdown(index=False, floatfmt='.10f')}")


def _build_config():
    parser = argparse.ArgumentParser(
        description="Preview miner metadata migrations and validator weight setting."
    )
    add_common_config(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop before subtensor.set_weights. This is also the default behavior.",
    )
    parser.add_argument(
        "--emit-set-weights",
        action="store_true",
        help="Actually call subtensor.set_weights after logging the preview.",
    )
    parser.add_argument(
        "--force-tempo",
        action="store_true",
        help="Allow emission even if the validator tempo gate has not elapsed.",
    )
    parser.add_argument(
        "--alpha-stake-weigh-factor",
        type=float,
        default=None,
        help=(
            "Override CONFIG.score.alpha_stake_weigh_factor for this run "
            "without editing config files."
        ),
    )
    parser.add_argument(
        "--emission-liquidation-weigh-factor",
        type=float,
        default=None,
        help=(
            "Override CONFIG.score.emission_liquidation_weigh_factor for this run "
            "without editing config files."
        ),
    )
    parser.add_argument(
        "--emission-liquidation-window-epochs",
        type=int,
        default=None,
        help=(
            "Override CONFIG.score.emission_liquidation_window_epochs for this run "
            "without editing config files."
        ),
    )
    return bt.Config(parser)


def main() -> int:
    config = _build_config()
    override_weigh_factor = getattr(config, "alpha_stake_weigh_factor", None)
    if override_weigh_factor is not None:
        CONFIG.score.alpha_stake_weigh_factor = override_weigh_factor
        logger.info(
            "Overriding alpha stake weigh factor for this run: "
            f"{override_weigh_factor}"
        )
    override_liquidation_weigh_factor = getattr(
        config,
        "emission_liquidation_weigh_factor",
        None,
    )
    if override_liquidation_weigh_factor is not None:
        CONFIG.score.emission_liquidation_weigh_factor = (
            override_liquidation_weigh_factor
        )
        logger.info(
            "Overriding emission liquidation weigh factor for this run: "
            f"{override_liquidation_weigh_factor}"
        )
    override_liquidation_window_epochs = getattr(
        config,
        "emission_liquidation_window_epochs",
        None,
    )
    if override_liquidation_window_epochs is not None:
        CONFIG.score.emission_liquidation_window_epochs = (
            override_liquidation_window_epochs
        )
        logger.info(
            "Overriding emission liquidation window epochs for this run: "
            f"{override_liquidation_window_epochs}"
        )

    logger.info(
        f"Initializing wallet/subtensor/metagraph for netuid={config.netuid} "
        f"on network={config.subtensor.network}"
    )
    wallet = bt.Wallet(config=config)
    subtensor = bt.Subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    metagraph.sync(subtensor=subtensor)

    hotkey = wallet.hotkey.ss58_address
    if hotkey not in metagraph.hotkeys:
        raise RuntimeError(
            f"Wallet hotkey {hotkey} is not registered on subnet {config.netuid}"
        )

    uid = metagraph.hotkeys.index(hotkey)
    logger.info(f"Validator hotkey {hotkey} resolved to UID {uid}")

    miner_manager = MinerManager(
        uid=uid,
        config=config,
        wallet=wallet,
        metagraph=metagraph,
    )
    logger.info(
        "MinerManager initialized; miner_metadata and "
        "miner_emission_epoch_snapshots tables have been created/migrated, "
        "and metagraph metadata has been synced."
    )

    raw_uids, raw_weights = miner_manager.weights
    _log_weight_table("Raw MinerManager weights after burn", raw_uids, raw_weights)

    processed_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
        uids=raw_uids,
        weights=raw_weights,
        netuid=config.netuid,
        subtensor=subtensor,
        metagraph=metagraph,
    )
    _log_weight_table(
        "Processed weights after Bittensor netuid constraints",
        processed_uids,
        processed_weights,
    )

    uint_uids, uint_weights = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
        uids=processed_uids,
        weights=processed_weights,
    )
    uint_df = pd.DataFrame(
        [
            {"uid": int(_scalar(uid)), "uint_weight": int(_scalar(weight))}
            for uid, weight in zip(uint_uids, uint_weights)
        ]
    )
    logger.info(
        "Final uint weights prepared for subtensor.set_weights:\n"
        f"{uint_df.to_markdown(index=False)}"
    )

    current_block = subtensor.get_current_block()
    last_update = int(_scalar(metagraph.last_update[uid]))
    tempo_ready = current_block > last_update + CONFIG.SUBNET_TEMPO
    logger.info(
        f"Tempo gate: current_block={current_block}, last_update={last_update}, "
        f"subnet_tempo={CONFIG.SUBNET_TEMPO}, ready={tempo_ready}"
    )

    emit_requested = bool(getattr(config, "emit_set_weights", False))
    explicit_dry_run = bool(getattr(config, "dry_run", False))
    dry_run = explicit_dry_run or not emit_requested
    if dry_run:
        logger.info(
            "DRY RUN: stopping immediately before subtensor.set_weights. "
            "Pass --emit-set-weights to emit on chain."
        )
        return 0

    if not tempo_ready and not bool(getattr(config, "force_tempo", False)):
        logger.warning(
            "Not setting weights because the tempo gate is not ready. "
            "Pass --force-tempo to bypass this manual-run guard."
        )
        return 0

    logger.warning(
        "Calling subtensor.set_weights because --emit-set-weights was provided."
    )
    success, message = subtensor.set_weights(
        netuid=config.netuid,
        wallet=wallet,
        uids=uint_uids,
        weights=uint_weights,
    )
    if success:
        logger.success(f"set_weights succeeded: {message}")
        return 0

    logger.error(f"set_weights failed: {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
