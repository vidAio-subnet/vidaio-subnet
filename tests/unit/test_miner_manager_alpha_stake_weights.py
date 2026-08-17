import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


bt = types.ModuleType("bittensor")
bt.Dendrite = lambda wallet=None: None
bt.Subtensor = lambda config=None: None
bt.logging = SimpleNamespace(
    debug=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
    info=lambda *args, **kwargs: None,
)
sys.modules["bittensor"] = bt

redis = types.ModuleType("redis")
redis.Redis = lambda *args, **kwargs: None
sys.modules["redis"] = redis

ROOT = Path(__file__).resolve().parents[2]


def install_package_stub(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    return package


install_package_stub("vidaio_subnet_core", ROOT / "vidaio_subnet_core")
install_package_stub(
    "vidaio_subnet_core.validating",
    ROOT / "vidaio_subnet_core" / "validating",
)
install_package_stub(
    "vidaio_subnet_core.validating.managing",
    ROOT / "vidaio_subnet_core" / "validating" / "managing",
)
install_package_stub(
    "vidaio_subnet_core.utilities",
    ROOT / "vidaio_subnet_core" / "utilities",
)

rate_limit = types.ModuleType("vidaio_subnet_core.utilities.rate_limit")
rate_limit.build_rate_limit = lambda *args, **kwargs: 0
sys.modules["vidaio_subnet_core.utilities.rate_limit"] = rate_limit

miner_manager_path = (
    ROOT / "vidaio_subnet_core" / "validating" / "managing" / "miner_manager.py"
)
spec = importlib.util.spec_from_file_location(
    "vidaio_subnet_core.validating.managing.miner_manager",
    miner_manager_path,
)
miner_manager_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = miner_manager_module
spec.loader.exec_module(miner_manager_module)
MinerManager = miner_manager_module.MinerManager
Base = miner_manager_module.Base
MinerEmissionEpochSnapshot = miner_manager_module.MinerEmissionEpochSnapshot
MinerMetadata = miner_manager_module.MinerMetadata


class BalanceValue:
    def __init__(self, tao):
        self.tao = tao


class TensorValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class MinerManagerAlphaStakeWeightTests(unittest.TestCase):
    def test_reuses_injected_subtensor_connection(self):
        subtensor = object()
        metagraph = SimpleNamespace(uids=[])

        with (
            patch.object(bt, "Subtensor") as subtensor_factory,
            patch.object(miner_manager_module, "create_engine") as create_engine,
            patch.object(miner_manager_module.Base.metadata, "create_all"),
            patch.object(miner_manager_module, "sessionmaker") as sessionmaker,
            patch.object(MinerManager, "_migrate_miner_metadata_table"),
            patch.object(MinerManager, "_migrate_miner_emission_epoch_snapshots_table"),
            patch.object(MinerManager, "sync_miner_chain_metadata"),
            patch.object(MinerManager, "initialize_serving_counter"),
        ):
            create_engine.return_value = object()
            sessionmaker.return_value.return_value = object()

            manager = MinerManager(
                uid=1,
                config=SimpleNamespace(),
                wallet=object(),
                metagraph=metagraph,
                subtensor=subtensor,
            )

        self.assertIs(manager.subtensor, subtensor)
        subtensor_factory.assert_not_called()

    def manager(self):
        manager = MinerManager.__new__(MinerManager)
        manager.alpha_stake_weigh_factor = 0.0
        manager.emission_liquidation_weigh_factor = 0.0
        manager.emission_liquidation_window_epochs = 10
        manager.burn_proportion = 0.0
        manager.competition_repository = None
        manager.compression_emission_allocation = 0.80
        manager.upscaling_emission_allocation = 0.20
        manager.competition_emission_allocation = 0.20
        manager.emission_rank_shares = [0.20, 0.20, 0.20, 0.20, 0.20]
        manager.metagraph = SimpleNamespace(
            alpha_stake=[0.0] * 100,
            E=[0.0] * 100,
            validator_permit=[False] * 100,
        )
        return manager

    def prepare_weight_run(self, manager, recipients):
        manager.metagraph.hotkeys = [f"hotkey-{uid}" for uid in range(100)]
        manager.competition_repository = SimpleNamespace(
            latest_competition_reward_context=lambda: recipients
        )
        manager.get_burn_uid = lambda: self.fail(
            "get_burn_uid should not be called when burn_proportion is 0"
        )
        manager.check_database_connection = lambda: None
        manager.sync_miner_chain_metadata = lambda: None
        manager.record_miner_emission_epoch_snapshots = lambda miners_by_uid: None
        manager.recent_emission_liquidation_stats = lambda uids: {}
        manager.query = lambda: {
            uid: SimpleNamespace(
                accumulate_score=float(20 - uid),
                processing_task_type=(
                    "compression" if uid <= 5 else "upscaling"
                ),
                alpha_stake=0.0,
            )
            for uid in range(1, 11)
        }

    def test_competition_podium_uses_70_20_10_and_merges_inference_uids(self):
        manager = self.manager()
        manager.metagraph.hotkeys = [f"hotkey-{uid}" for uid in range(100)]
        manager.metagraph.validator_permit[11] = True
        manager.competition_repository = SimpleNamespace(
            latest_competition_reward_context=lambda: (
                "competition-test",
                (
                    ("hotkey-1", 1, 0.70),
                    ("hotkey-6", 2, 0.20),
                    ("hotkey-11", 3, 0.10),
                ),
            )
        )
        manager.get_burn_uid = lambda: self.fail(
            "get_burn_uid should not be called when burn_proportion is 0"
        )
        manager.check_database_connection = lambda: None
        manager.sync_miner_chain_metadata = lambda: None
        manager.record_miner_emission_epoch_snapshots = lambda miners_by_uid: None
        manager.recent_emission_liquidation_stats = lambda uids: {}
        manager.query = lambda: {
            uid: SimpleNamespace(
                accumulate_score=float(20 - uid),
                processing_task_type=(
                    "compression" if uid <= 5 else "upscaling"
                ),
                alpha_stake=0.0,
            )
            for uid in range(1, 11)
        }

        with patch.object(miner_manager_module.logger, "info") as log_info:
            uids, scores = MinerManager.weights.fget(manager)
        scores_by_uid = {int(uid): float(score) for uid, score in zip(uids, scores)}

        self.assertEqual(len(uids), len(set(int(uid) for uid in uids)))
        self.assertAlmostEqual(sum(scores), 1.0)
        self.assertAlmostEqual(scores_by_uid[1], 0.12 + 0.14)
        self.assertAlmostEqual(scores_by_uid[6], 0.04 + 0.04)
        self.assertAlmostEqual(scores_by_uid[11], 0.02)
        self.assertAlmostEqual(sum(scores_by_uid[uid] for uid in range(1, 6)), 0.74)
        self.assertAlmostEqual(sum(scores_by_uid[uid] for uid in range(6, 11)), 0.24)
        overlap_logs = [
            call
            for call in log_info.call_args_list
            if call.args
            and "Inference/competition reward overlap" in call.args[0]
        ]
        self.assertEqual(len(overlap_logs), 2)
        self.assertEqual(
            {(call.args[1], call.args[2], call.args[3]) for call in overlap_logs},
            {
                ("competition-test", "hotkey-1", 1),
                ("competition-test", "hotkey-6", 6),
            },
        )
        overlap_by_uid = {call.args[3]: call.args for call in overlap_logs}
        self.assertEqual(overlap_by_uid[1][4:6], ("compression", 1))
        self.assertAlmostEqual(overlap_by_uid[1][6], 0.12)
        self.assertAlmostEqual(overlap_by_uid[1][7], 0.14)
        self.assertAlmostEqual(overlap_by_uid[1][8], 0.26)
        self.assertAlmostEqual(overlap_by_uid[1][9], 0.26)
        self.assertEqual(overlap_by_uid[6][4:6], ("upscaling", 2))
        self.assertAlmostEqual(overlap_by_uid[6][6], 0.04)
        self.assertAlmostEqual(overlap_by_uid[6][7], 0.04)
        self.assertAlmostEqual(overlap_by_uid[6][8], 0.08)
        self.assertAlmostEqual(overlap_by_uid[6][9], 0.08)
        self.assertTrue(all(call.args[-1] for call in overlap_logs))
        distribution_log = next(
            call.args[0]
            for call in log_info.call_args_list
            if call.args and call.args[0].startswith("Reward distribution:")
        )
        self.assertIn("competition_id=competition-test", distribution_log)
        self.assertIn("compression miners (0.600 weight)", distribution_log)
        self.assertIn("upscaling miners (0.200 weight)", distribution_log)
        self.assertIn("competition podium (0.200 weight)", distribution_log)

    def test_competition_podium_receives_full_weight_without_inference_metadata(self):
        manager = self.manager()
        manager.metagraph.hotkeys = [f"hotkey-{uid}" for uid in range(100)]
        manager.metagraph.validator_permit[1] = True
        manager.competition_repository = SimpleNamespace(
            latest_competition_reward_context=lambda: (
                "competition-test",
                (
                    ("hotkey-1", 1, 0.70),
                    ("hotkey-6", 2, 0.20),
                    ("hotkey-11", 3, 0.10),
                ),
            )
        )
        manager.get_burn_uid = lambda: self.fail(
            "get_burn_uid should not be called when burn_proportion is 0"
        )
        manager.check_database_connection = lambda: None
        manager.sync_miner_chain_metadata = lambda: None
        manager.record_miner_emission_epoch_snapshots = lambda miners_by_uid: None
        manager.recent_emission_liquidation_stats = lambda uids: {}
        manager.query = lambda: {}

        with patch.object(miner_manager_module.logger, "info") as log_info:
            uids, scores = MinerManager.weights.fget(manager)
        scores_by_uid = {int(uid): float(score) for uid, score in zip(uids, scores)}

        self.assertEqual(set(scores_by_uid), {1, 6, 11})
        self.assertAlmostEqual(scores_by_uid[1], 0.70)
        self.assertAlmostEqual(scores_by_uid[6], 0.20)
        self.assertAlmostEqual(scores_by_uid[11], 0.10)
        self.assertAlmostEqual(sum(scores), 1.0)
        distribution_log = next(
            call.args[0]
            for call in log_info.call_args_list
            if call.args and call.args[0].startswith("Reward distribution:")
        )
        self.assertIn("0 compression miners (0.000 weight)", distribution_log)
        self.assertIn("0 upscaling miners (0.000 weight)", distribution_log)
        self.assertIn("competition podium (1.000 weight)", distribution_log)
        self.assertIn("competition_id=competition-test", distribution_log)

    def test_pre_podium_keeps_legacy_80_20_inference_split(self):
        manager = self.manager()
        self.prepare_weight_run(manager, (None, ()))

        uids, scores = MinerManager.weights.fget(manager)
        scores_by_uid = {int(uid): float(score) for uid, score in zip(uids, scores)}

        self.assertAlmostEqual(sum(scores), 1.0)
        self.assertAlmostEqual(sum(scores_by_uid[uid] for uid in range(1, 6)), 0.80)
        self.assertAlmostEqual(sum(scores_by_uid[uid] for uid in range(6, 11)), 0.20)
        self.assertEqual(set(scores_by_uid), set(range(1, 11)))

    def test_missing_lower_podium_members_return_shares_to_compression(self):
        manager = self.manager()
        self.prepare_weight_run(
            manager,
            ("competition-test", (("hotkey-11", 1, 0.70),)),
        )

        uids, scores = MinerManager.weights.fget(manager)
        scores_by_uid = {int(uid): float(score) for uid, score in zip(uids, scores)}

        self.assertAlmostEqual(sum(scores), 1.0)
        self.assertAlmostEqual(sum(scores_by_uid[uid] for uid in range(1, 6)), 0.66)
        self.assertAlmostEqual(sum(scores_by_uid[uid] for uid in range(6, 11)), 0.20)
        self.assertAlmostEqual(scores_by_uid[11], 0.14)

    def test_missing_winner_disables_podium_and_restores_legacy_split(self):
        manager = self.manager()
        self.prepare_weight_run(
            manager,
            (
                "competition-test",
                (
                    ("missing-winner", 1, 0.70),
                    ("hotkey-6", 2, 0.20),
                ),
            ),
        )

        uids, scores = MinerManager.weights.fget(manager)
        scores_by_uid = {int(uid): float(score) for uid, score in zip(uids, scores)}

        self.assertAlmostEqual(sum(scores), 1.0)
        self.assertAlmostEqual(sum(scores_by_uid[uid] for uid in range(1, 6)), 0.80)
        self.assertAlmostEqual(sum(scores_by_uid[uid] for uid in range(6, 11)), 0.20)
        self.assertAlmostEqual(scores_by_uid[6], 0.04)

    def test_podium_hotkey_is_resolved_to_its_current_uid(self):
        manager = self.manager()
        self.prepare_weight_run(
            manager,
            ("competition-test", (("moving-winner", 1, 0.70),)),
        )
        manager.metagraph.hotkeys[42] = "moving-winner"

        uids, scores = MinerManager.weights.fget(manager)
        scores_by_uid = {int(uid): float(score) for uid, score in zip(uids, scores)}

        self.assertAlmostEqual(scores_by_uid[42], 0.14)
        self.assertNotIn(41, scores_by_uid)
        self.assertAlmostEqual(sum(scores), 1.0)

    def sqlite_manager(self, current_block=1000):
        manager = self.manager()
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)
        manager.engine = engine
        manager.session = session_factory()
        manager.subtensor = SimpleNamespace(get_current_block=lambda: current_block)
        return manager

    def test_reads_alpha_stake_from_metagraph_vector(self):
        manager = self.manager()
        manager.metagraph.alpha_stake = [BalanceValue(3.0), TensorValue(12.5)]

        self.assertEqual(manager._alpha_stake_for_uid(0), 3.0)
        self.assertEqual(manager._alpha_stake_for_uid(1), 12.5)

    def test_epoch_index_prefers_metagraph_tempo(self):
        manager = self.manager()
        manager.metagraph.tempo = TensorValue(360)

        self.assertEqual(manager._epoch_index_for_block(1080), 3)

    def test_record_snapshots_normalizes_legacy_epoch_index_scale_and_keeps_oldest(self):
        manager = self.sqlite_manager(current_block=8589355)
        manager.metagraph.tempo = TensorValue(360)
        manager.metagraph.E[1] = 2.5
        miner = MinerMetadata(
            uid=1,
            hotkey="hotkey-1",
            coldkey="coldkey-1",
            alpha_stake=25.0,
            processing_task_type="compression",
            accumulate_score=1.0,
        )
        manager.session.add(miner)
        manager.session.add(
            MinerEmissionEpochSnapshot(
                uid=1,
                hotkey="legacy-hotkey",
                coldkey="legacy-coldkey",
                task_type="compression",
                epoch_block=8589265,
                epoch_index=85892,
                alpha_stake=20.0,
                emission=1.5,
            )
        )
        manager.session.add(
            MinerEmissionEpochSnapshot(
                uid=1,
                hotkey="older-hotkey",
                coldkey="older-coldkey",
                task_type="compression",
                epoch_block=8588995,
                epoch_index=85889,
                alpha_stake=18.0,
                emission=1.0,
            )
        )
        manager.session.commit()

        manager.record_miner_emission_epoch_snapshots({1: miner})

        rows = (
            manager.session.query(MinerEmissionEpochSnapshot)
            .filter(MinerEmissionEpochSnapshot.uid == 1)
            .order_by(MinerEmissionEpochSnapshot.epoch_block)
            .all()
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.epoch_index for row in rows], [23858, 23859])
        self.assertEqual(rows[-1].epoch_block, 8589265)
        self.assertEqual(rows[-1].hotkey, "legacy-hotkey")
        self.assertAlmostEqual(rows[-1].emission, 1.5)

    def test_record_snapshots_preserves_first_observation_for_same_epoch(self):
        manager = self.sqlite_manager(current_block=1205)
        manager.metagraph.E[1] = 9.0
        miner = MinerMetadata(
            uid=1,
            hotkey="updated-hotkey",
            coldkey="updated-coldkey",
            alpha_stake=50.0,
            processing_task_type="compression",
            accumulate_score=1.0,
        )
        manager.session.add(miner)
        manager.session.add(
            MinerEmissionEpochSnapshot(
                uid=1,
                hotkey="first-hotkey",
                coldkey="first-coldkey",
                task_type="compression",
                epoch_block=1200,
                epoch_index=12,
                alpha_stake=25.0,
                emission=2.5,
            )
        )
        manager.session.commit()

        manager.record_miner_emission_epoch_snapshots({1: miner})

        rows = manager.session.query(MinerEmissionEpochSnapshot).all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].epoch_block, 1200)
        self.assertEqual(rows[0].hotkey, "first-hotkey")
        self.assertEqual(rows[0].coldkey, "first-coldkey")
        self.assertAlmostEqual(rows[0].alpha_stake, 25.0)
        self.assertAlmostEqual(rows[0].emission, 2.5)

    def test_zero_weigh_factor_keeps_equal_top_five_scores(self):
        manager = self.manager()
        miners = [
            (1, 5.0, 10.0),
            (2, 4.0, 20.0),
            (3, 3.0, 30.0),
            (4, 2.0, 40.0),
            (5, 1.0, 50.0),
        ]

        scores = dict(manager._apply_rank_curve(miners, 0.80))

        for uid in range(1, 6):
            self.assertAlmostEqual(scores[uid], 0.16)

    def test_alpha_stake_weighing_preserves_pool_total_and_rewards_higher_stake(self):
        manager = self.manager()
        manager.alpha_stake_weigh_factor = 1.0
        miners = [
            (1, 6.0, 10.0),
            (2, 5.0, 20.0),
            (3, 4.0, 30.0),
            (4, 3.0, 40.0),
            (5, 2.0, 0.0),
            (6, 1.0, 1000.0),
        ]

        scores = dict(manager._apply_rank_curve(miners, 0.80))

        self.assertAlmostEqual(sum(scores[uid] for uid in range(1, 6)), 0.80)
        self.assertEqual(scores[6], 0.0)
        self.assertGreater(scores[4], 0.16)
        self.assertLess(scores[5], 0.16)

    def test_weights_include_validator_permit_uids_from_miner_metadata(self):
        manager = self.manager()
        manager.alpha_stake_weigh_factor = 1.0
        manager.metagraph.validator_permit[2] = True
        manager.get_burn_uid = lambda: self.fail(
            "get_burn_uid should not be called when burn_proportion is 0"
        )
        manager.check_database_connection = lambda: None
        manager.sync_miner_chain_metadata = lambda: None
        manager.record_miner_emission_epoch_snapshots = lambda miners_by_uid: None
        manager.recent_emission_liquidation_stats = lambda uids: {}
        manager.query = lambda: {
            1: SimpleNamespace(
                accumulate_score=10.0,
                processing_task_type="compression",
                alpha_stake=100.0,
            ),
            2: SimpleNamespace(
                accumulate_score=9.0,
                processing_task_type="compression",
                alpha_stake=1000.0,
            ),
            3: SimpleNamespace(
                accumulate_score=8.0,
                processing_task_type="compression",
                alpha_stake=0.0,
            ),
            4: SimpleNamespace(
                accumulate_score=7.0,
                processing_task_type="upscaling",
                alpha_stake=0.0,
            ),
            5: SimpleNamespace(
                accumulate_score=6.0,
                processing_task_type="upscaling",
                alpha_stake=100.0,
            ),
        }

        uids, scores = MinerManager.weights.fget(manager)
        scores_by_uid = {int(uid): float(score) for uid, score in zip(uids, scores)}

        self.assertIn(2, scores_by_uid)
        self.assertGreater(scores_by_uid[2], scores_by_uid[1])
        self.assertGreater(scores_by_uid[1], scores_by_uid[3])
        self.assertGreater(scores_by_uid[5], scores_by_uid[4])
        self.assertAlmostEqual(
            scores_by_uid[1] + scores_by_uid[2] + scores_by_uid[3],
            0.48,
        )
        self.assertAlmostEqual(scores_by_uid[4] + scores_by_uid[5], 0.08)
        self.assertNotIn(99, scores_by_uid)

    def test_emission_snapshots_include_validator_permit_uids(self):
        manager = self.sqlite_manager(current_block=1200)
        manager.metagraph.validator_permit[1] = True
        manager.metagraph.E[1] = 2.5
        miner = MinerMetadata(
            uid=1,
            hotkey="hotkey-1",
            coldkey="coldkey-1",
            alpha_stake=25.0,
            processing_task_type="compression",
            accumulate_score=1.0,
        )
        manager.session.add(miner)
        manager.session.commit()

        manager.record_miner_emission_epoch_snapshots({1: miner})

        snapshot = manager.session.query(MinerEmissionEpochSnapshot).one()
        self.assertEqual(snapshot.uid, 1)
        self.assertAlmostEqual(snapshot.emission, 2.5)

    def test_emission_snapshot_records_hotkey_coldkey_and_prunes_window(self):
        manager = self.sqlite_manager(current_block=1200)
        manager.metagraph.E[1] = 2.5
        miner = MinerMetadata(
            uid=1,
            hotkey="hotkey-1",
            coldkey="coldkey-1",
            alpha_stake=25.0,
            processing_task_type="compression",
            accumulate_score=1.0,
        )
        manager.session.add(miner)
        for epoch_index in range(12):
            manager.session.add(
                MinerEmissionEpochSnapshot(
                    uid=1,
                    hotkey="old-hotkey",
                    coldkey="old-coldkey",
                    task_type="compression",
                    epoch_block=epoch_index * 100,
                    epoch_index=epoch_index,
                    alpha_stake=float(epoch_index),
                    emission=1.0,
                )
            )
        manager.session.commit()

        manager.record_miner_emission_epoch_snapshots({1: miner})

        rows = (
            manager.session.query(MinerEmissionEpochSnapshot)
            .filter(MinerEmissionEpochSnapshot.uid == 1)
            .order_by(MinerEmissionEpochSnapshot.epoch_index)
            .all()
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0].epoch_index, 3)
        self.assertEqual(rows[-1].epoch_index, 12)
        self.assertEqual(rows[-1].hotkey, "hotkey-1")
        self.assertEqual(rows[-1].coldkey, "coldkey-1")
        self.assertAlmostEqual(rows[-1].alpha_stake, 25.0)
        self.assertAlmostEqual(rows[-1].emission, 2.5)

    def test_recent_emission_liquidation_stats_from_snapshot_window(self):
        manager = self.sqlite_manager()
        for epoch_index, alpha_stake in [(1, 100.0), (2, 106.0), (3, 112.0)]:
            manager.session.add(
                MinerEmissionEpochSnapshot(
                    uid=1,
                    hotkey="hotkey-1",
                    coldkey="coldkey-1",
                    task_type="compression",
                    epoch_block=epoch_index * 100,
                    epoch_index=epoch_index,
                    alpha_stake=alpha_stake,
                    emission=10.0,
                )
            )
        manager.session.commit()

        stats = manager.recent_emission_liquidation_stats([1, 2])

        self.assertEqual(stats[1]["status"], "ok")
        self.assertAlmostEqual(stats[1]["first_excluded_emission"], 10.0)
        self.assertAlmostEqual(stats[1]["total_emission"], 20.0)
        self.assertAlmostEqual(stats[1]["retained_emission"], 12.0)
        self.assertAlmostEqual(stats[1]["liquidated_emission"], 8.0)
        self.assertAlmostEqual(stats[1]["liquidated_proportion"], 0.4)
        self.assertAlmostEqual(stats[1]["retained_proportion"], 0.6)
        self.assertEqual(stats[2]["status"], "new_or_insufficient_history")
        self.assertIsNone(stats[2]["retained_proportion"])

    def test_recent_emission_liquidation_excludes_initial_boundary_emission(self):
        manager = self.sqlite_manager()
        for epoch_index, alpha_stake, emission in [
            (1, 100.0, 10.0),
            (2, 108.0, 8.0),
            (3, 110.0, 2.0),
        ]:
            manager.session.add(
                MinerEmissionEpochSnapshot(
                    uid=1,
                    hotkey="hotkey-1",
                    coldkey="coldkey-1",
                    task_type="compression",
                    epoch_block=epoch_index * 100,
                    epoch_index=epoch_index,
                    alpha_stake=alpha_stake,
                    emission=emission,
                )
            )
        manager.session.commit()

        stats = manager.recent_emission_liquidation_stats([1])

        self.assertEqual(stats[1]["status"], "ok")
        self.assertAlmostEqual(stats[1]["first_excluded_emission"], 10.0)
        self.assertAlmostEqual(stats[1]["total_emission"], 10.0)
        self.assertAlmostEqual(stats[1]["retained_emission"], 10.0)
        self.assertAlmostEqual(stats[1]["liquidated_emission"], 0.0)
        self.assertAlmostEqual(stats[1]["liquidated_proportion"], 0.0)

    def test_recent_emission_liquidation_uses_two_snapshot_history(self):
        manager = self.sqlite_manager()
        for epoch_index, alpha_stake, emission in [
            (1, 100.0, 10.0),
            (2, 108.0, 8.0),
        ]:
            manager.session.add(
                MinerEmissionEpochSnapshot(
                    uid=1,
                    hotkey="hotkey-1",
                    coldkey="coldkey-1",
                    task_type="compression",
                    epoch_block=epoch_index * 100,
                    epoch_index=epoch_index,
                    alpha_stake=alpha_stake,
                    emission=emission,
                )
            )
        manager.session.commit()

        stats = manager.recent_emission_liquidation_stats([1])

        self.assertEqual(stats[1]["status"], "ok")
        self.assertAlmostEqual(stats[1]["first_excluded_emission"], 10.0)
        self.assertAlmostEqual(stats[1]["total_emission"], 8.0)
        self.assertAlmostEqual(stats[1]["retained_emission"], 8.0)
        self.assertAlmostEqual(stats[1]["liquidated_proportion"], 0.0)

    def test_recent_emission_liquidation_requires_prior_alpha_baseline(self):
        manager = self.sqlite_manager()
        manager.session.add(
            MinerEmissionEpochSnapshot(
                uid=1,
                hotkey="hotkey-1",
                coldkey="coldkey-1",
                task_type="compression",
                epoch_block=100,
                epoch_index=1,
                alpha_stake=100.0,
                emission=10.0,
            )
        )
        manager.session.commit()

        stats = manager.recent_emission_liquidation_stats([1])

        self.assertEqual(stats[1]["status"], "new_or_insufficient_history")
        self.assertIsNone(stats[1]["retained_proportion"])

    def test_emission_liquidation_weighing_preserves_total_and_assumes_unknown_half_liquidated(self):
        manager = self.manager()
        manager.emission_liquidation_weigh_factor = 2.0
        ranked_scores = [(1, 0.16), (2, 0.16), (3, 0.16)]
        stats = {
            1: {
                "retained_proportion": 0.0,
                "liquidated_proportion": 1.0,
                "snapshot_count": 10,
                "status": "ok",
            },
            2: {
                "retained_proportion": 1.0,
                "liquidated_proportion": 0.0,
                "snapshot_count": 10,
                "status": "ok",
            },
        }

        scores = dict(
            manager._weigh_scores_by_emission_liquidation(
                ranked_scores,
                "compression",
                stats,
            )
        )

        self.assertAlmostEqual(sum(scores.values()), 0.48)
        self.assertGreater(scores[2], scores[3])
        self.assertGreater(scores[3], scores[1])

    def test_emission_liquidation_weighing_keeps_all_unknown_pool_equal(self):
        manager = self.manager()
        manager.emission_liquidation_weigh_factor = 5.0
        ranked_scores = [(uid, 0.16) for uid in range(1, 6)]

        scores = dict(
            manager._weigh_scores_by_emission_liquidation(
                ranked_scores,
                "compression",
                {},
            )
        )

        self.assertAlmostEqual(sum(scores.values()), 0.80)
        for uid in range(1, 6):
            self.assertAlmostEqual(scores[uid], 0.16)


if __name__ == "__main__":
    unittest.main()
