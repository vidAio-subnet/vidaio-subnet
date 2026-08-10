import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, call

from services.scoring.artifact_cleanup import cleanup_compression_artifacts


class CompressionArtifactCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_removes_all_distorted_files_when_collection_lengths_differ(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            distorted_paths = [root / f"distorted-{idx}.mp4" for idx in range(10)]
            reference_paths = [root / f"reference-{idx}.mp4" for idx in range(2)]
            for path in distorted_paths + reference_paths:
                path.write_bytes(b"video")

            delete_uploaded_object = AsyncMock()
            await cleanup_compression_artifacts(
                [str(path) for path in distorted_paths],
                [str(path) for path in reference_paths],
                ["challenge-0", "challenge-1"],
                delete_uploaded_object,
            )

            self.assertFalse(any(path.exists() for path in distorted_paths))
            self.assertFalse(any(path.exists() for path in reference_paths))
            self.assertEqual(
                delete_uploaded_object.await_args_list,
                [call("challenge-0"), call("challenge-1")],
            )

    async def test_cleanup_continues_after_individual_delete_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            distorted_path = root / "distorted.mp4"
            distorted_path.write_bytes(b"video")

            delete_uploaded_object = AsyncMock(
                side_effect=[RuntimeError("unavailable"), None]
            )
            with self.assertLogs(
                "services.scoring.artifact_cleanup", level="ERROR"
            ):
                await cleanup_compression_artifacts(
                    [str(distorted_path)],
                    [],
                    ["challenge-0", "challenge-1"],
                    delete_uploaded_object,
                )

            self.assertFalse(distorted_path.exists())
            self.assertEqual(delete_uploaded_object.await_count, 2)
