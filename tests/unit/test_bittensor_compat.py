import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vidaio_subnet_core" / "utilities" / "bittensor_compat.py"
SPEC = importlib.util.spec_from_file_location("bittensor_compat_under_test", MODULE_PATH)
bittensor_compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bittensor_compat)
apply_single_field_composite_encoding_patch = (
    bittensor_compat.apply_single_field_composite_encoding_patch
)
unwrap_singleton_tuples = bittensor_compat._unwrap_singleton_tuples


COMPOSITE_ERROR = (
    "Invalid type for data: 292 of type <class 'int'>, "
    "type_def: Composite(TypeDefComposite { fields: [...] })"
)


def runtime_with_type(type_id, type_definition):
    registry = {
        "types": [
            {
                "id": type_id,
                "type": {
                    "def": type_definition,
                },
            }
        ]
    }
    return SimpleNamespace(registry=SimpleNamespace(registry=json.dumps(registry)))


class BittensorCompatibilityPatchTests(unittest.TestCase):
    def install_patch_for(self, mixin, sync_interface=None):
        package = types.ModuleType("async_substrate_interface")
        package.__path__ = []
        types_module = types.ModuleType("async_substrate_interface.types")
        types_module.SubstrateMixin = mixin
        sync_module = types.ModuleType("async_substrate_interface.sync_substrate")
        if sync_interface is None:
            sync_interface = type(
                "FakeSyncInterface",
                (mixin,),
                {
                    "encode_scale": mixin._encode_scale,
                    "runtime_call": lambda self: SimpleNamespace(value=None),
                },
            )
        sync_module.SubstrateInterface = sync_interface
        async_module = types.ModuleType("async_substrate_interface.async_substrate")

        class FakeAsyncInterface:
            async def runtime_call(self):
                return SimpleNamespace(value=None)

        async_module.AsyncSubstrateInterface = FakeAsyncInterface

        modules = {
            "async_substrate_interface": package,
            "async_substrate_interface.async_substrate": async_module,
            "async_substrate_interface.sync_substrate": sync_module,
            "async_substrate_interface.types": types_module,
        }
        with patch.dict(sys.modules, modules):
            return apply_single_field_composite_encoding_patch()

    def test_retries_scalar_as_tuple_for_unnamed_single_field_composite(self):
        class FakeMixin:
            calls = []

            def _encode_scale(self, type_string, value, runtime=None):
                self.calls.append(value)
                if value == 292:
                    raise ValueError(COMPOSITE_ERROR)
                return b"encoded"

        runtime = runtime_with_type(
            99,
            {"composite": {"fields": [{"name": None, "type": 42}]}},
        )

        self.assertTrue(self.install_patch_for(FakeMixin))
        result = FakeMixin()._encode_scale("scale_info::99", 292, runtime=runtime)

        self.assertEqual(result, b"encoded")
        self.assertEqual(FakeMixin.calls, [292, (292,)])

    def test_patches_sync_interface_alias_used_by_runtime_call(self):
        class FakeMixin:
            calls = []

            def _encode_scale(self, type_string, value, runtime=None):
                self.calls.append(value)
                if value == 292:
                    raise ValueError(COMPOSITE_ERROR)
                return b"encoded"

        class FakeSyncInterface(FakeMixin):
            encode_scale = FakeMixin._encode_scale

            def runtime_call(self):
                return SimpleNamespace(value=None)

        runtime = runtime_with_type(
            99,
            {"composite": {"fields": [{"name": None, "type": 42}]}},
        )

        self.install_patch_for(FakeMixin, FakeSyncInterface)
        result = FakeSyncInterface().encode_scale(
            "scale_info::99", 292, runtime=runtime
        )

        self.assertEqual(result, b"encoded")
        self.assertEqual(FakeMixin.calls, [292, (292,)])

    def test_unwraps_decoded_newtypes_without_flattening_runtime_collections(self):
        decoded = [
            {
                "stake": [(((1, 2, 3),), (123,))],
                "tuple_backed_single_staker": ((((4, 5, 6),), (456,)),),
                "weights": [(7, 8)],
                "single_item_vector": [(9,)],
            }
        ]

        self.assertEqual(
            unwrap_singleton_tuples(decoded),
            [
                {
                    "stake": [(((1, 2, 3),), 123)],
                    "tuple_backed_single_staker": ((((4, 5, 6),), 456),),
                    "weights": [(7, 8)],
                    "single_item_vector": [9],
                }
            ],
        )

    def test_normalizes_sync_runtime_call_result(self):
        class FakeMixin:
            def _encode_scale(self, type_string, value, runtime=None):
                return b"encoded"

        class FakeSyncInterface(FakeMixin):
            encode_scale = FakeMixin._encode_scale

            def runtime_call(self):
                return SimpleNamespace(value={"stake": [((1, 2), (456,))]})

        self.install_patch_for(FakeMixin, FakeSyncInterface)
        result = FakeSyncInterface().runtime_call()

        self.assertEqual(result.value, {"stake": [((1, 2), 456)]})

    def test_does_not_retry_named_composite(self):
        class FakeMixin:
            calls = []

            def _encode_scale(self, type_string, value, runtime=None):
                self.calls.append(value)
                raise ValueError(COMPOSITE_ERROR)

        runtime = runtime_with_type(
            99,
            {"composite": {"fields": [{"name": "netuid", "type": 42}]}},
        )
        self.install_patch_for(FakeMixin)

        with self.assertRaisesRegex(ValueError, "Invalid type for data"):
            FakeMixin()._encode_scale("scale_info::99", 292, runtime=runtime)

        self.assertEqual(FakeMixin.calls, [292])

    def test_does_not_retry_unrelated_value_error(self):
        class FakeMixin:
            calls = []

            def _encode_scale(self, type_string, value, runtime=None):
                self.calls.append(value)
                raise ValueError("bad account address")

        runtime = runtime_with_type(
            99,
            {"composite": {"fields": [{"name": None, "type": 42}]}},
        )
        self.install_patch_for(FakeMixin)

        with self.assertRaisesRegex(ValueError, "bad account address"):
            FakeMixin()._encode_scale("scale_info::99", 292, runtime=runtime)

        self.assertEqual(FakeMixin.calls, [292])


if __name__ == "__main__":
    unittest.main()
