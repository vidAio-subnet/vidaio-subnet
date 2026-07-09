import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


bt = types.ModuleType("bittensor")
bt.Dendrite = lambda wallet=None: None
bt.Subtensor = lambda config=None: None
bt.logging = SimpleNamespace(
    debug=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
    info=lambda *args, **kwargs: None,
)
sys.modules.setdefault("bittensor", bt)

redis = types.ModuleType("redis")
redis.Redis = lambda *args, **kwargs: None
sys.modules.setdefault("redis", redis)

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


class BalanceValue:
    def __init__(self, tao):
        self.tao = tao


class TensorValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class MinerManagerAlphaStakeWeightTests(unittest.TestCase):
    def manager(self):
        manager = MinerManager.__new__(MinerManager)
        manager.alpha_stake_weigh_factor = 0.0
        manager.burn_proportion = 0.6
        manager.compression_emission_allocation = 0.80
        manager.upscaling_emission_allocation = 0.20
        manager.emission_rank_shares = [0.20, 0.20, 0.20, 0.20, 0.20]
        manager.metagraph = SimpleNamespace(
            alpha_stake=[0.0] * 100,
            validator_permit=[False] * 100,
        )
        return manager

    def test_reads_alpha_stake_from_metagraph_vector(self):
        manager = self.manager()
        manager.metagraph.alpha_stake = [BalanceValue(3.0), TensorValue(12.5)]

        self.assertEqual(manager._alpha_stake_for_uid(0), 3.0)
        self.assertEqual(manager._alpha_stake_for_uid(1), 12.5)

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

    def test_weights_exclude_validators_and_weigh_within_each_task_pool(self):
        manager = self.manager()
        manager.alpha_stake_weigh_factor = 1.0
        manager.metagraph.validator_permit[2] = True
        manager.get_burn_uid = lambda: 99
        manager.check_database_connection = lambda: None
        manager.sync_miner_chain_metadata = lambda: None
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

        self.assertNotIn(2, scores_by_uid)
        self.assertGreater(scores_by_uid[1], scores_by_uid[3])
        self.assertGreater(scores_by_uid[5], scores_by_uid[4])
        self.assertAlmostEqual(scores_by_uid[1] + scores_by_uid[3], 0.128)
        self.assertAlmostEqual(scores_by_uid[4] + scores_by_uid[5], 0.032)
        self.assertAlmostEqual(scores_by_uid[99], 0.24)


if __name__ == "__main__":
    unittest.main()
