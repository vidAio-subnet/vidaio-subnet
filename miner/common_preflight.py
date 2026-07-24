#!/usr/bin/env python3
"""Shared miner/validator preflight for one compression competition solution.

The runtime probe sends the same batch-of-one local-path request that the
validator will send to a contender service inside a Modal sandbox.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class PreflightError(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    duration_seconds: float
    frame_count: int
    codec: str
    container: str
    pixel_format: str
    sample_aspect_ratio: str


HttpJson = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]
MediaInspector = Callable[[Path], MediaInfo]
MAX_FRAME_COUNT_DELTA = 2


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def inspect_media(path: Path, *, ffprobe: str = "ffprobe") -> MediaInfo:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,pix_fmt,sample_aspect_ratio,duration,nb_read_frames:format=format_name,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise PreflightError("MEDIA_PROBE_FAILED", result.stderr.strip()[:300])
    try:
        payload = json.loads(result.stdout)
        stream = next(
            item for item in payload["streams"] if item.get("codec_type") == "video"
        )
        duration = stream.get("duration") or payload["format"].get("duration")
        return MediaInfo(
            width=int(stream["width"]),
            height=int(stream["height"]),
            duration_seconds=float(duration),
            frame_count=int(stream.get("nb_read_frames") or 0),
            codec=str(stream["codec_name"]),
            container=str(payload["format"]["format_name"]),
            pixel_format=str(stream.get("pix_fmt", "")),
            sample_aspect_ratio=str(stream.get("sample_aspect_ratio", "")),
        )
    except (
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise PreflightError(
            "MEDIA_PROBE_INVALID", "ffprobe returned incomplete video metadata"
        ) from exc


def validate_repository(repository: Path) -> dict[str, Any]:
    root = repository.resolve(strict=True)
    required = (
        "competition_solution.json",
        "requirements.txt",
        "miner/modal_workers.py",
        "miner/compression/app.py",
        "miner/common_preflight.py",
    )
    for relative in required:
        if not (root / relative).is_file():
            raise PreflightError(
                "REQUIRED_FILE_MISSING", f"submission is missing {relative}"
            )

    try:
        descriptor = json.loads(
            (root / "competition_solution.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(
            "INVALID_SOLUTION_DESCRIPTOR", "competition_solution.json is invalid"
        ) from exc
    expected_descriptor = {
        "competition_type": "COMPRESSION",
        "entrypoint": "miner/modal_workers.py",
        "local_path_io": True,
        "preflight": "miner/common_preflight.py",
        "schema_version": 2,
        "sdk": "miner/competition_sdk.py",
    }
    for key, expected in expected_descriptor.items():
        if descriptor.get(key) != expected:
            raise PreflightError(
                "INVALID_SOLUTION_DESCRIPTOR", f"{key} must equal {expected!r}"
            )
    routes = descriptor.get("routes")
    if not isinstance(routes, list) or not {"/health", "/compress"}.issubset(routes):
        raise PreflightError(
            "REQUIRED_ROUTE_MISSING", "descriptor must declare /health and /compress"
        )

    secret_patterns = (
        re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
        re.compile(rb"AKIA[0-9A-Z]{16}"),
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        if current != root and ".git" in names:
            raise PreflightError(
                "NESTED_REPOSITORY", f"nested Git metadata found below {current}"
            )
        names[:] = [
            name
            for name in sorted(names)
            if not (current == root and name == ".git")
            and name not in {"__pycache__", ".pytest_cache", ".ruff_cache"}
        ]
        for name in [*names, *sorted(filenames)]:
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                try:
                    resolved = path.resolve(strict=True)
                except (FileNotFoundError, RuntimeError) as exc:
                    raise PreflightError(
                        "BROKEN_SYMLINK", f"{relative} does not resolve"
                    ) from exc
                if not resolved.is_relative_to(root):
                    raise PreflightError(
                        "PATH_ESCAPE", f"{relative} resolves outside the repository"
                    )
                continue
            if not path.is_file():
                continue
            if relative == ".gitmodules" or name == ".git":
                raise PreflightError(
                    "SUBMODULE_NOT_ALLOWED",
                    f"Git submodule metadata found at {relative}",
                )
            if name == ".env" or (name.startswith(".env.") and name != ".env.template"):
                raise PreflightError(
                    "ENV_FILE_NOT_ALLOWED", f"remove local credential file {relative}"
                )
            size = path.stat().st_size
            if size > 50_000_000:
                raise PreflightError(
                    "FILE_TOO_LARGE", f"{relative} exceeds the 50 MB source-file limit"
                )
            content = path.read_bytes()
            if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise PreflightError(
                    "GIT_LFS_POINTER", f"Git LFS pointer found in {relative}"
                )
            if any(pattern.search(content) for pattern in secret_patterns):
                raise PreflightError(
                    "COMMITTED_SECRET", f"possible credential found in {relative}"
                )
            if path.suffix == ".py":
                try:
                    ast.parse(content, filename=relative)
                except (SyntaxError, ValueError) as exc:
                    raise PreflightError(
                        "INVALID_PYTHON", f"Python does not parse: {relative}"
                    ) from exc
            total_bytes += size
            file_count += 1
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(content).digest())

    if total_bytes > 2_000_000_000:
        raise PreflightError("REPOSITORY_TOO_LARGE", "repository source exceeds 2 GB")

    return {
        "status": "ACCEPTED",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "repository_tree_sha256": digest.hexdigest(),
    }


def validate_fixture(
    fixture: Path, inspector: MediaInspector = inspect_media
) -> MediaInfo:
    if not fixture.is_file():
        raise PreflightError("WARMUP_FIXTURE_MISSING", f"fixture not found: {fixture}")
    media = inspector(fixture)
    if (
        media.duration_seconds <= 0
        or media.duration_seconds > 5.5
        or media.width > 1280
        or media.height > 720
    ):
        raise PreflightError(
            "WARMUP_FIXTURE_INVALID",
            "fixture must contain a video no longer than 5.5 seconds and no larger than 720p",
        )
    return media


def build_batch_one_request(
    run_id: str,
    *,
    input_path: str | None = None,
) -> dict[str, Any]:
    return {
        "competition_id": "sdk-preflight",
        "hotkey": "sdk-local-miner",
        "batch_id": f"batch-{run_id}",
        "items": [
            {
                "evaluation_id": f"warmup-{run_id}",
                "input_path": input_path
                or f"/evaluation-inputs/sdk-preflight/{run_id}.mp4",
                "output_path": f"/output/sdk-preflight/{run_id}.mp4",
                "codec": "AV1",
                "vmaf_threshold": 90.0,
            }
        ],
    }


def _default_http_json(
    method: str, url: str, payload: dict[str, Any] | None
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with build_opener(_NoRedirect).open(request, timeout=1800) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise PreflightError(
            "SERVICE_HTTP_ERROR", f"HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise PreflightError("SERVICE_UNREACHABLE", str(exc)) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreflightError(
            "SERVICE_RESPONSE_INVALID", "service returned invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PreflightError(
            "SERVICE_RESPONSE_INVALID", "service response must be an object"
        )
    return value


def run_runtime_preflight(
    *,
    fixture: Path,
    service_url: str,
    host_input_root: Path,
    host_output_root: Path,
    allow_nonlocal_service: bool = False,
    http_json: HttpJson = _default_http_json,
    inspector: MediaInspector = inspect_media,
    run_id: str | None = None,
    input_relative_path: Path | None = None,
    prepositioned_input: bool = False,
) -> dict[str, Any]:
    parsed = urlparse(service_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PreflightError("SERVICE_URL_INVALID", "service URL must be HTTP(S)")
    if not allow_nonlocal_service and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise PreflightError(
            "SERVICE_URL_NOT_LOCAL",
            "preflight only calls localhost unless --allow-nonlocal-service is explicit",
        )

    run_id = run_id or uuid.uuid4().hex[:12]
    relative_input = input_relative_path or Path("sdk-preflight") / f"{run_id}.mp4"
    if relative_input.is_absolute() or ".." in relative_input.parts:
        raise PreflightError(
            "INPUT_PATH_INVALID", "preflight input path must stay below the input mount"
        )
    request_payload = build_batch_one_request(
        run_id,
        input_path=f"/evaluation-inputs/{relative_input.as_posix()}",
    )
    item = request_payload["items"][0]
    relative_output = Path("sdk-preflight") / f"{run_id}.mp4"
    host_input_root.mkdir(parents=True, exist_ok=True)
    host_output_root.mkdir(parents=True, exist_ok=True)
    host_input_root = host_input_root.resolve(strict=True)
    host_output_root = host_output_root.resolve(strict=True)
    host_input = host_input_root / relative_input
    host_output = host_output_root / relative_output
    if not prepositioned_input:
        host_input.parent.mkdir(parents=True, exist_ok=True)
    host_output.parent.mkdir(parents=True, exist_ok=True)
    if host_output.exists() or host_output.is_symlink():
        raise PreflightError(
            "OUTPUT_ALREADY_EXISTS",
            f"refusing to overwrite preflight output {host_output}",
        )
    if prepositioned_input:
        try:
            fixture_resolved = fixture.resolve(strict=True)
            host_input_resolved = host_input.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PreflightError(
                "WARMUP_FIXTURE_MISSING",
                "prepositioned warmup input is absent from the read-only mount",
            ) from exc
        if fixture_resolved != host_input_resolved:
            raise PreflightError(
                "INPUT_PATH_INVALID",
                "prepositioned fixture must be the exact requested input path",
            )
    else:
        shutil.copyfile(fixture, host_input)
    input_media = validate_fixture(host_input, inspector)
    before_outputs = {
        path.relative_to(host_output_root).as_posix()
        for path in host_output_root.rglob("*")
        if path.is_file()
    }

    base_url = service_url.rstrip("/")
    health = http_json("GET", f"{base_url}/health", None)
    if health.get("status") != "ok":
        raise PreflightError("HEALTH_CHECK_FAILED", "/health did not return status=ok")
    local_io = health.get("competition_local_io")
    if not isinstance(local_io, dict) or local_io.get("remote_io_disabled") is not True:
        raise PreflightError(
            "REMOTE_IO_ENABLED",
            "/health did not prove competition remote I/O is disabled",
        )

    response = http_json("POST", f"{base_url}/compress", request_payload)
    results = response.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise PreflightError(
            "COMPRESSION_RESPONSE_INVALID",
            "batch-of-one must return exactly one result",
        )
    result = results[0]
    if not isinstance(result, dict):
        raise PreflightError("COMPRESSION_RESPONSE_INVALID", "result must be an object")
    if (
        set(result) != {"output_path"}
        or result.get("output_path") != item["output_path"]
    ):
        detail = {
            "output_path": result.get("output_path"),
            "expected_output_path": item["output_path"],
        }
        raise PreflightError(
            "WARMUP_COMPRESSION_FAILED",
            json.dumps(detail, sort_keys=True),
        )
    if not host_input.is_file():
        raise PreflightError(
            "INPUT_WAS_REMOVED",
            "the contender service deleted the mounted evaluation input",
        )
    if not host_output.is_file() or host_output.stat().st_size <= 0:
        raise PreflightError(
            "OUTPUT_MISSING", "service did not create the requested local output"
        )

    after_outputs = {
        path.relative_to(host_output_root).as_posix()
        for path in host_output_root.rglob("*")
        if path.is_file()
    }
    expected_relative = relative_output.as_posix()
    unexpected = (after_outputs - before_outputs) - {expected_relative}
    if unexpected:
        raise PreflightError(
            "UNEXPECTED_OUTPUT_WRITE",
            f"service wrote unexpected output files: {sorted(unexpected)}",
        )

    output_media = inspector(host_output)
    if output_media.codec != "av1" or "mp4" not in output_media.container:
        raise PreflightError("OUTPUT_MEDIA_INVALID", "output must be AV1 in MP4")
    if (
        output_media.width != input_media.width
        or output_media.height != input_media.height
    ):
        raise PreflightError(
            "OUTPUT_DIMENSIONS_INVALID",
            "compression output must preserve input dimensions",
        )
    if host_output.stat().st_size >= host_input.stat().st_size:
        raise PreflightError(
            "OUTPUT_NOT_COMPRESSED", "compression output must be smaller than its input"
        )
    duration_tolerance = max(0.2, input_media.duration_seconds * 0.02)
    if (
        abs(output_media.duration_seconds - input_media.duration_seconds)
        > duration_tolerance
    ):
        raise PreflightError(
            "OUTPUT_DURATION_INVALID", "output duration differs from the input"
        )
    if (
        input_media.frame_count
        and output_media.frame_count
        and abs(output_media.frame_count - input_media.frame_count)
        > MAX_FRAME_COUNT_DELTA
    ):
        raise PreflightError(
            "OUTPUT_FRAME_COUNT_INVALID", "output frame count differs from the input"
        )
    if not output_media.pixel_format.startswith(
        "yuv"
    ) or output_media.sample_aspect_ratio not in {"", "1:1"}:
        raise PreflightError(
            "OUTPUT_MEDIA_INVALID",
            "output must use a YUV pixel format and square pixels",
        )
    return {
        "status": "ACCEPTED",
        "batch_size": 1,
        "request": request_payload,
        "input_media": asdict(input_media),
        "output_media": asdict(output_media),
        "host_input": str(host_input),
        "host_output": str(host_output),
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=default_root)
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Defaults to competitions/fixtures/compression_warmup_input.mp4",
    )
    parser.add_argument("--service-url", help="Run the exact batch-of-one HTTP probe")
    parser.add_argument(
        "--host-input-root",
        type=Path,
        default=Path("/tmp/vidaio-competition-preflight/evaluation-inputs"),
    )
    parser.add_argument(
        "--host-output-root",
        type=Path,
        default=Path("/tmp/vidaio-competition-preflight/output"),
    )
    parser.add_argument("--allow-nonlocal-service", action="store_true")
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Skip repository checks when executing inside a prevalidated sandbox image",
    )
    parser.add_argument(
        "--prepositioned-input",
        action="store_true",
        help="Consume an input already uploaded to a read-only input mount",
    )
    parser.add_argument(
        "--input-relative-path",
        type=Path,
        help="Input path below /evaluation-inputs; defaults to sdk-preflight/<run-id>.mp4",
    )
    parser.add_argument("--run-id", help="Stable identifier for the warmup request")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    fixture = args.fixture or (
        repository / "competitions/fixtures/compression_warmup_input.mp4"
    )
    try:
        report: dict[str, Any] = {"fixture": asdict(validate_fixture(fixture))}
        if not args.runtime_only:
            report["repository"] = validate_repository(repository)
        if args.service_url:
            report["runtime"] = run_runtime_preflight(
                fixture=fixture,
                service_url=args.service_url,
                host_input_root=args.host_input_root,
                host_output_root=args.host_output_root,
                allow_nonlocal_service=args.allow_nonlocal_service,
                run_id=args.run_id,
                input_relative_path=args.input_relative_path,
                prepositioned_input=args.prepositioned_input,
            )
            report["status"] = "ACCEPTED"
        else:
            report["status"] = "STATIC_READY"
            report["next_step"] = (
                "run miner/competition_sdk.py validate for the Modal qualification"
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except PreflightError as exc:
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "reason_code": exc.reason_code,
                    "detail": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
