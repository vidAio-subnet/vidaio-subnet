#!/usr/bin/env python3
"""Extract multiple zero-timestamped clips from one or more videos.

The script deliberately uses ffmpeg stream copy (``-c copy``), so extraction is
fast and requires very little CPU. Cuts therefore occur near codec keyframes;
they are not guaranteed to be frame-exact. MP4 edit lists are disabled and
negative composition offsets are retained so every video timeline starts at
zero without re-encoding.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    REPO_ROOT / "vid1.mp4",
    REPO_ROOT / "vid2.mp4",
    REPO_ROOT / "vid3.mp4",
)


def video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def video_start_time(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=start_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def clip_windows(
    source_duration: float,
    count: int,
    min_duration: float,
    max_duration: float,
    rng: random.Random,
) -> list[tuple[float, float]]:
    """Create windows spread evenly over the usable source timeline."""
    if source_duration < min_duration:
        raise ValueError(
            f"source is {source_duration:.2f}s, shorter than the minimum "
            f"clip duration ({min_duration:.2f}s)"
        )

    longest = min(max_duration, source_duration)
    windows: list[tuple[float, float]] = []
    for index in range(count):
        duration = rng.uniform(min_duration, longest)
        latest_start = max(0.0, source_duration - duration)
        # Put one start in each equal timeline bucket, with seeded jitter.
        bucket_start = latest_start * index / count
        bucket_end = latest_start * (index + 1) / count
        start = rng.uniform(bucket_start, bucket_end) if bucket_end > bucket_start else 0.0
        windows.append((start, duration))
    return windows


def extract_clip(source: Path, output: Path, start: float, duration: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "disabled",
            "-movflags",
            "+faststart+negative_cts_offsets",
            "-use_editlist",
            "0",
            "-y",
            str(output),
        ],
        check=True,
    )


def extract_validated_clip(
    source: Path,
    output: Path,
    start: float,
    requested_duration: float,
    min_duration: float,
    max_duration: float,
) -> float:
    """Extract and compensate for keyframe-induced stream-copy duration drift."""
    duration = requested_duration
    for _ in range(3):
        extract_clip(source, output, start, duration)
        actual = video_duration(output)
        if min_duration - 0.01 <= actual <= max_duration + 0.01:
            start_time = video_start_time(output)
            if abs(start_time) > 0.000001:
                raise RuntimeError(
                    f"output video timeline starts at {start_time:.6f}s instead of zero"
                )
            return actual
        # With input-side seeking, an earlier keyframe can make the file longer
        # than -t. Adjusting -t retains stream copy and is still very cheap.
        boundary = min_duration if actual < min_duration else max_duration
        duration = max(0.1, duration + boundary - actual - 0.02)
    raise RuntimeError(f"output duration remained outside bounds (last: {actual:.3f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 10-30 second clips using ffmpeg stream copy."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="input videos (defaults to the three MP4s in the repository root)",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=REPO_ROOT / "video_clips")
    parser.add_argument("-n", "--clips-per-video", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=10.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2026, help="seed for reproducible selections")
    parser.add_argument("--dry-run", action="store_true", help="print selections without running ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clips_per_video < 1:
        raise SystemExit("--clips-per-video must be at least 1")
    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        raise SystemExit("durations must satisfy 0 < min-duration <= max-duration")
    if not args.dry_run:
        for command in ("ffmpeg", "ffprobe"):
            if shutil.which(command) is None:
                raise SystemExit(f"required command not found: {command}")

    output_dir = args.output_dir.expanduser().resolve()
    rng = random.Random(args.seed)
    failures = 0
    for raw_source in args.inputs:
        source = raw_source.expanduser().resolve()
        if not source.is_file():
            print(f"ERROR: input does not exist: {source}", file=sys.stderr)
            failures += 1
            continue
        try:
            duration = video_duration(source)
            windows = clip_windows(
                duration,
                args.clips_per_video,
                args.min_duration,
                args.max_duration,
                rng,
            )
        except (ValueError, subprocess.CalledProcessError) as error:
            print(f"ERROR: cannot process {source}: {error}", file=sys.stderr)
            failures += 1
            continue

        source_dir = output_dir / source.stem
        print(f"{source.name}: {duration:.2f}s -> {len(windows)} clips in {source_dir}")
        for number, (start, clip_duration) in enumerate(windows, start=1):
            output = source_dir / f"{source.stem}_clip_{number:02d}.mp4"
            print(f"  {number:02d}: start={start:9.3f}s duration={clip_duration:6.3f}s")
            if args.dry_run:
                continue
            try:
                actual = extract_validated_clip(
                    source,
                    output,
                    start,
                    clip_duration,
                    args.min_duration,
                    args.max_duration,
                )
                print(f"      wrote {output.name} ({actual:.3f}s)")
            except (RuntimeError, subprocess.CalledProcessError, ValueError) as error:
                print(f"ERROR: failed to create {output}: {error}", file=sys.stderr)
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
