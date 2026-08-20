import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[2]
    / "vidaio_subnet_core"
    / "validating"
    / "task_warrants.py"
)
SPEC = importlib.util.spec_from_file_location("task_warrants", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
task_warrants = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task_warrants)
resolve_failed_task_warrant = task_warrants.resolve_failed_task_warrant


def test_failed_task_warrant_without_inference_history_remains_unresolved():
    assert resolve_failed_task_warrant(None) is None


def test_failed_task_warrant_with_invalid_history_remains_unresolved():
    assert resolve_failed_task_warrant("competition") is None


def test_failed_task_warrant_uses_known_inference_history():
    assert resolve_failed_task_warrant("compression") == "compression"
    assert resolve_failed_task_warrant("upscaling") == "upscaling"
