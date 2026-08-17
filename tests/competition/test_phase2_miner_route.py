from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "phase2_compression_app", ROOT / "miner" / "compression" / "app.py"
)
assert SPEC and SPEC.loader
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


class MinerRouteTests(unittest.TestCase):
    def test_cq_resolution_is_shared_by_tier_vmaf_and_explicit_override(self) -> None:
        self.assertEqual(APP.resolve_compression_cq(compression_type="Low"), 40)
        self.assertEqual(APP.resolve_compression_cq(compression_type="Medium"), 35)
        self.assertEqual(APP.resolve_compression_cq(compression_type="High"), 30)
        self.assertEqual(APP.resolve_compression_cq(vmaf_threshold=88.99), 40)
        self.assertEqual(APP.resolve_compression_cq(vmaf_threshold=89), 35)
        self.assertEqual(APP.resolve_compression_cq(vmaf_threshold=92.99), 35)
        self.assertEqual(APP.resolve_compression_cq(vmaf_threshold=93), 30)
        self.assertEqual(APP.resolve_compression_cq(), 35)
        self.assertEqual(
            APP.resolve_compression_cq(
                explicit_cq=27,
                compression_type="Low",
                vmaf_threshold=80,
            ),
            27,
        )
        request = APP.CompressRequest(
            video_paths=["/evaluation-inputs/input.mp4"],
            codec="AV1",
            vmaf_threshold=93,
        )
        ffmpeg_args = APP._build_ffmpeg_args(
            "/evaluation-inputs/input.mp4",
            "/output/result.mp4",
            request,
            "av1_nvenc",
        )
        self.assertEqual(ffmpeg_args[ffmpeg_args.index("-cq") + 1], "30")

        vbr_request = APP.CompressRequest(
            video_paths=["/evaluation-inputs/input.mp4"],
            codec="AV1",
            codec_mode="VBR",
            target_bitrate=8_000_000,
            vmaf_threshold=93,
        )
        vbr_args = APP._build_ffmpeg_args(
            "/evaluation-inputs/input.mp4",
            "/output/result-vbr.mp4",
            vbr_request,
            "av1_nvenc",
        )
        self.assertEqual(vbr_args[vbr_args.index("-b:v") + 1], "8000000")
        self.assertNotIn("-cq", vbr_args)

    def test_inference_contract_remains_backward_compatible(self) -> None:
        request = APP.CompressRequest(
            video_paths=["https://example.com/input.mp4"],
            task_id="legacy",
            codec="AV1",
        )
        self.assertEqual(request.video_paths, ["https://example.com/input.mp4"])
        self.assertEqual(request.codec, "AV1")
        self.assertEqual(
            APP.resolve_compression_cq(
                explicit_cq=request.cq,
                compression_type=request.compression_type,
                vmaf_threshold=request.vmaf_threshold,
            ),
            35,
        )

    def test_caller_output_paths_must_match_and_be_unique(self) -> None:
        with self.assertRaises(ValidationError):
            APP.CompressRequest(
                video_paths=["/evaluation-inputs/a.mp4", "/evaluation-inputs/b.mp4"],
                output_paths=["/output/a.mp4"],
            )
        with self.assertRaises(ValidationError):
            APP.CompressRequest(
                video_paths=["/evaluation-inputs/a.mp4", "/evaluation-inputs/b.mp4"],
                output_paths=["/output/result.mp4", "/output/result.mp4"],
            )

    def test_competition_contract_rejects_duplicate_output(self) -> None:
        item = {
            "evaluation_id": "first",
            "input_path": "/evaluation-inputs/first.mp4",
            "output_path": "/output/hotkey/result.mp4",
            "codec": "AV1",
            "vmaf_threshold": 90.0,
        }
        with self.assertRaises(ValidationError):
            APP.CompetitionCompressionRequest(
                competition_id="competition-1",
                hotkey="hotkey",
                batch_id="batch-1",
                items=[item, {**item, "evaluation_id": "second"}],
            )

    def test_path_guards_reject_url_escape_symlink_and_overwrite(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp,
            tempfile.TemporaryDirectory() as outside,
        ):
            root = Path(temp)
            inputs = root / "evaluation-inputs"
            outputs = root / "output"
            inputs.mkdir()
            outputs.mkdir()
            input_file = inputs / "input.mp4"
            input_file.write_bytes(b"fixture")
            existing = outputs / "existing.mp4"
            existing.write_bytes(b"existing")
            with (
                patch.object(APP, "COMPETITION_INPUT_ROOT", str(inputs)),
                patch.object(APP, "COMPETITION_OUTPUT_ROOT", str(outputs)),
            ):
                self.assertEqual(
                    APP._competition_input_path(str(input_file)), input_file.resolve()
                )
                with self.assertRaisesRegex(ValueError, "absolute local"):
                    APP._competition_input_path("https://example.com/input.mp4")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    APP._competition_output_path(str(existing))
                missing_parent = outputs / "missing" / "result.mp4"
                with self.assertRaisesRegex(ValueError, "parent must already exist"):
                    APP._competition_output_path(str(missing_parent))
                self.assertFalse(missing_parent.parent.exists())
                try:
                    os.symlink(Path(outside), outputs / "escape")
                except OSError:
                    self.skipTest("symlinks unavailable")
                with self.assertRaisesRegex(ValueError, "below /output"):
                    APP._competition_output_path(str(outputs / "escape" / "result.mp4"))

    def test_competition_route_returns_output_without_miner_side_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backing_input = root / "backing-input"
            backing_output = root / "backing-output"
            backing_input.mkdir()
            backing_output.mkdir()
            logical_input = root / "evaluation-inputs"
            logical_output = root / "output"
            try:
                logical_input.symlink_to(backing_input, target_is_directory=True)
                logical_output.symlink_to(backing_output, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks unavailable")
            input_path = logical_input / "warmup.mp4"
            input_path.write_bytes(b"input")
            requested_output = logical_output / "warmup.mp4"
            received = []

            async def compress_one(
                request, *_args, requested_output_path: str, **_kwargs
            ):
                received.append(request)
                Path(requested_output_path).write_bytes(b"output")
                return APP.CompressResponse(
                    success=True,
                    output_paths=[requested_output_path],
                    errors=[],
                )

            request = APP.CompetitionCompressionRequest(
                competition_id="competition-1",
                hotkey="hotkey",
                batch_id="batch-1",
                items=[
                    {
                        "evaluation_id": "warmup",
                        "input_path": str(input_path),
                        "output_path": str(requested_output),
                        "codec": "AV1",
                        "codec_mode": "VBR",
                        "target_bitrate": 10_000_000,
                        "vmaf_threshold": 93.0,
                    }
                ],
            )
            with (
                patch.object(APP, "DISABLE_REMOTE_IO", True),
                patch.object(APP, "COMPETITION_INPUT_ROOT", str(logical_input)),
                patch.object(APP, "COMPETITION_OUTPUT_ROOT", str(logical_output)),
                patch.object(APP, "_compress_one", side_effect=compress_one),
            ):
                response = asyncio.run(APP._compress_competition(request))

        self.assertEqual(response.results[0].output_path, str(requested_output))
        self.assertEqual(
            response.results[0].model_dump(), {"output_path": str(requested_output)}
        )
        self.assertNotIn("backing-output", response.results[0].output_path)
        self.assertEqual(received[0].codec_mode, "VBR")
        self.assertEqual(received[0].target_bitrate, 10_000_000)
        self.assertEqual(received[0].vmaf_threshold, 93)


if __name__ == "__main__":
    unittest.main()
