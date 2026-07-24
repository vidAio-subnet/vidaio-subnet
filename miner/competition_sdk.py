#!/usr/bin/env python3
"""Prepare, validate, and privately publish a customized miner solution."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener


GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
EXPORT_MARKER = ".vidaio-sdk-export"
REPOSITORY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
MODAL_APP_NAME = "vidaio-compression-competition-sdk-preflight"
MODAL_INPUT_PATH = "/evaluation-inputs/compression_warmup_input.mp4"
MODAL_INPUT_VOLUME_PATH = "/compression_warmup_input.mp4"
MODAL_OUTPUT_ROOT = "/output"
MODAL_SERVICE_URL = "http://127.0.0.1:8003"
DEFAULT_MODAL_BUILD_TIMEOUT_SECONDS = 600.0
VALIDATION_RECEIPT_SCHEMA_VERSION = 4
DEFAULT_VALIDATION_MAX_AGE_HOURS = 24.0
CANONICAL_SDK_PATHS = (
    "miner/common_preflight.py",
    "miner/competition_sdk.py",
    "scripts/competition_modal_build.py",
)
REFRESH_WARNING = (
    "reusing a prepared export without --refresh; if this checkout contains a "
    "newer competition SDK, cancel and rerun with --refresh or the validator may "
    "reject the submission with SDK_TOOL_MODIFIED"
)
GPU_PRICE_PER_SECOND_USD = {
    "B300": Decimal("0.001972"),
    "B200": Decimal("0.001736"),
    "H200": Decimal("0.001261"),
    "H100": Decimal("0.001097"),
    "RTX-PRO-6000": Decimal("0.000842"),
    "A100-80GB": Decimal("0.000694"),
    "A100-40GB": Decimal("0.000583"),
    "L40S": Decimal("0.000542"),
    "A10": Decimal("0.000306"),
    "L4": Decimal("0.000222"),
    "T4": Decimal("0.000164"),
}
SUPPORTED_MODAL_GPU_REQUESTS = frozenset(GPU_PRICE_PER_SECOND_USD) | {
    "A100",
    "B200+",
    "H100!",
}
SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD = Decimal("0.00003942")
MODAL_COST_ATTRIBUTION_METHOD = (
    "MODAL_PUBLIC_PRICE_2026-07-21_ACTUAL_RESOURCES_SANDBOX_WALL_RUNTIME"
)
ISOLATION_PROBE = r"""
import json
import os
import pathlib
import socket
import urllib.request

checks = {}
try:
    socket.create_connection(("1.1.1.1", 443), timeout=3).close()
    checks["direct_ip_blocked"] = False
except Exception as exc:
    checks["direct_ip_blocked"] = True
    checks["direct_ip_error"] = type(exc).__name__
try:
    socket.getaddrinfo("example.com", 443)
    checks["dns_lookup_blocked"] = False
except Exception as exc:
    checks["dns_lookup_blocked"] = True
    checks["dns_lookup_error"] = type(exc).__name__
try:
    urllib.request.urlopen("https://example.com", timeout=3).read(1)
    checks["https_blocked"] = False
except Exception as exc:
    checks["https_blocked"] = True
    checks["https_error"] = type(exc).__name__

input_probe = pathlib.Path("/evaluation-inputs/.validator-write-probe")
try:
    input_probe.write_text("forbidden")
    checks["input_read_only"] = False
except OSError as exc:
    checks["input_read_only"] = True
    checks["input_write_error"] = type(exc).__name__

output_probe = pathlib.Path("/output/.validator-write-probe")
try:
    output_probe.write_text("ok")
    checks["output_writable"] = output_probe.read_text() == "ok"
finally:
    output_probe.unlink(missing_ok=True)

checks["reference_mount_absent"] = not any(
    pathlib.Path(path).exists()
    for path in ("/evaluation-references", "/references", "/validator-references")
)
checks["credentials_absent"] = not any(
    os.getenv(name)
    for name in (
        "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_IDENTITY_TOKEN",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN", "GH_TOKEN",
    )
)
checks["network_blocked"] = (
    checks["direct_ip_blocked"]
    and checks["dns_lookup_blocked"]
    and checks["https_blocked"]
)
checks["passed"] = all(
    checks[name]
    for name in (
        "network_blocked", "input_read_only", "output_writable",
        "reference_mount_absent", "credentials_absent",
    )
)
print(json.dumps(checks, sort_keys=True), flush=True)
"""


READINESS_PROBE = r"""
import json
import time
import urllib.request

deadline = time.monotonic() + 300
last = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8003/health", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if response.status == 200 and payload.get("status") == "ok":
            print(json.dumps(payload, sort_keys=True), flush=True)
            raise SystemExit(0)
        last = "invalid health response"
    except Exception as exc:
        last = f"{type(exc).__name__}: {exc}"
    time.sleep(1)
raise SystemExit(f"service readiness timed out: {last}")
"""
RESOURCE_PROBE = r"""
import json
import os
import pathlib
import subprocess

gpu_process = subprocess.run(
    ["/usr/bin/nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
)
gpu_names = [line.strip() for line in gpu_process.stdout.splitlines() if line.strip()]

def cpu_quota_cores():
    cpu_max = pathlib.Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        quota, period = cpu_max.read_text().strip().split()[:2]
        if quota != "max":
            return float(quota) / float(period)
    quota_path = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file():
        quota = float(quota_path.read_text().strip())
        period = float(period_path.read_text().strip())
        if quota > 0:
            return quota / period
    return None

logical_cpu_ids = sorted(os.sched_getaffinity(0))
physical_cores = set()
for cpu_id in logical_cpu_ids:
    topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
    try:
        physical_cores.add(
            (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        )
    except OSError:
        physical_cores.clear()
        break

quota_cores = cpu_quota_cores()
affinity_physical_cores = float(len(physical_cores)) if physical_cores else None
allocated_cpu_cores = quota_cores or affinity_physical_cores
if not gpu_names or not allocated_cpu_cores or allocated_cpu_cores <= 0:
    raise SystemExit("allocated Sandbox resources could not be identified")
print(json.dumps({
    "gpu_names": gpu_names,
    "allocated_cpu_cores": allocated_cpu_cores,
    "cpu_quota_cores": quota_cores,
    "affinity_logical_cpu_count": len(logical_cpu_ids),
    "affinity_physical_cpu_cores": affinity_physical_cores,
}, sort_keys=True), flush=True)
"""
MODAL_APP_BUILD_PLACEHOLDER = """\"\"\"SDK build-cache placeholder.

The customized service is added as a separate immutable Modal Image layer.
\"\"\"
"""


class SdkError(RuntimeError):
    pass


class GitHubApiError(SdkError):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        super().__init__(f"GitHub API returned HTTP {status}: {detail}")


def estimate_modal_compute_cost(
    allocated_gpu: str,
    allocated_cpu: float,
    runtime_seconds: float,
    *,
    allocated_gpu_count: int = 1,
) -> Decimal:
    """Estimate from the GPU and CPU allocation observed inside the Sandbox."""

    gpu_rate = GPU_PRICE_PER_SECOND_USD.get(allocated_gpu)
    if gpu_rate is None:
        raise SdkError(
            f"no locked Modal cost rate for GPU {allocated_gpu!r}; choose one of "
            f"{sorted(GPU_PRICE_PER_SECOND_USD)}"
        )
    if allocated_gpu_count <= 0 or allocated_cpu <= 0 or runtime_seconds < 0:
        raise SdkError(
            "Modal cost inputs must have positive allocated GPU/CPU and "
            "nonnegative runtime"
        )
    per_second = (gpu_rate * allocated_gpu_count) + (
        SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD * Decimal(str(allocated_cpu))
    )
    return per_second * Decimal(str(runtime_seconds))


def canonical_modal_gpu_type(nvidia_smi_name: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "-", nvidia_smi_name.upper()).strip("-")
    if "RTX-PRO-6000" in normalized:
        return "RTX-PRO-6000"
    if "A100" in normalized:
        if re.search(r"(?:^|-)80-?GB(?:-|$)", normalized):
            return "A100-80GB"
        if re.search(r"(?:^|-)40-?GB(?:-|$)", normalized):
            return "A100-40GB"
        raise SdkError(f"allocated A100 memory size is unknown: {nvidia_smi_name!r}")
    for sku in ("B300", "B200", "H200", "H100", "L40S", "A10", "L4", "T4"):
        if re.search(rf"(?:^|-){re.escape(sku)}(?:-|$)", normalized):
            return sku
    raise SdkError(f"allocated GPU has no locked Modal rate: {nvidia_smi_name!r}")


@dataclass(frozen=True)
class GitHubIdentity:
    login: str
    name: str
    email: str


@dataclass(frozen=True)
class GitHubRepositoryTarget:
    full_name: str
    repository_url: str
    default_branch: str
    existing: bool
    identity: GitHubIdentity


GitHubRequester = Callable[[str, str, str, dict[str, Any] | None], dict[str, Any]]
_LOG_LOCK = threading.Lock()


def _sdk_log(stage: str, message: str) -> None:
    """Emit human-readable progress without contaminating the final JSON stdout."""

    with _LOG_LOCK:
        print(
            f"[competition-sdk][{stage}] {message}",
            file=sys.stderr,
            flush=True,
        )


def _stream_sandbox_reader(reader: Any, channel: str) -> None:
    try:
        for chunk in reader:
            for line in str(chunk).splitlines():
                if line:
                    _sdk_log(f"sandbox-{channel}", line)
    except Exception as exc:
        _sdk_log(
            f"sandbox-{channel}",
            f"log stream closed with {type(exc).__name__}: {exc}",
        )


def _start_sandbox_log_streams(sandbox: Any) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    for channel in ("stdout", "stderr"):
        try:
            reader = getattr(sandbox, channel)
        except (AttributeError, RuntimeError):
            continue
        if not hasattr(reader, "__iter__"):
            continue
        thread = threading.Thread(
            target=_stream_sandbox_reader,
            args=(reader, channel),
            name=f"competition-sdk-{channel}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    return threads


def _validation_configuration(
    *,
    environment_name: str,
    gpu: str,
    requested_cpu: float,
    cpu_limit: float,
    sandbox_timeout: int,
    modal_build_timeout_seconds: float = DEFAULT_MODAL_BUILD_TIMEOUT_SECONDS,
) -> dict[str, object]:
    return {
        "environment_name": environment_name,
        "gpu": gpu,
        "requested_cpu": requested_cpu,
        "cpu_limit": cpu_limit,
        "sandbox_timeout": sandbox_timeout,
        "modal_build_timeout_seconds": modal_build_timeout_seconds,
    }


def _duration_seconds(value: Any) -> float:
    if not isinstance(value, str):
        raise SdkError(
            "modal_build_timeout must be a duration such as '10m' or 'PT10M'"
        )
    shorthand = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", value)
    if shorthand:
        amount = Decimal(shorthand.group(1))
        multiplier = {
            "s": Decimal(1),
            "m": Decimal(60),
            "h": Decimal(3600),
            "d": Decimal(86400),
        }[shorthand.group(2)]
        seconds = amount * multiplier
    else:
        iso = re.fullmatch(
            r"P(?:(\d+(?:\.\d+)?)D)?"
            r"(?:T(?:(\d+(?:\.\d+)?)H)?"
            r"(?:(\d+(?:\.\d+)?)M)?"
            r"(?:(\d+(?:\.\d+)?)S)?)?",
            value,
            flags=re.IGNORECASE,
        )
        if not iso or not any(iso.groups()):
            raise SdkError("modal_build_timeout must use s/m/h/d shorthand or ISO-8601")
        days, hours, minutes, iso_seconds = (
            Decimal(part or "0") for part in iso.groups()
        )
        seconds = (
            days * Decimal(86400)
            + hours * Decimal(3600)
            + minutes * Decimal(60)
            + iso_seconds
        )
    if seconds <= 0:
        raise SdkError("modal_build_timeout must be positive")
    return float(seconds)


def load_modal_build_timeout_seconds(manifest_path: Path | None) -> float:
    """Read the validator build deadline, defaulting to ten minutes."""

    if manifest_path is None:
        return DEFAULT_MODAL_BUILD_TIMEOUT_SECONDS
    path = manifest_path.resolve(strict=True)
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise SdkError("PyYAML is required to read YAML manifests") from exc
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise SdkError("competition manifest must be JSON or YAML")
    except (json.JSONDecodeError, OSError) as exc:
        raise SdkError(f"cannot read competition manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SdkError("competition manifest root must be an object")
    return _duration_seconds(payload.get("modal_build_timeout", "10m"))


def validation_receipt_path(export: Path) -> Path:
    export = export.resolve(strict=True)
    return export.parent / f".{export.name}.modal-validation.json"


def write_validation_receipt(
    export: Path,
    *,
    static_report: dict[str, Any],
    runtime_report: dict[str, Any],
    configuration: dict[str, object],
    now: datetime | None = None,
) -> Path:
    validated_at = now or datetime.now(timezone.utc)
    if validated_at.tzinfo is None or validated_at.utcoffset() is None:
        raise SdkError("validation receipt timestamp must be timezone-aware")
    tree_digest = static_report.get("repository", {}).get("repository_tree_sha256")
    if not tree_digest or runtime_report.get("status") != "ACCEPTED":
        raise SdkError("cannot write a receipt for an incomplete validation")
    receipt = {
        "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
        "status": "ACCEPTED",
        "validated_at": validated_at.astimezone(timezone.utc).isoformat(),
        "repository_tree_sha256": tree_digest,
        "configuration": configuration,
        "runtime_report": runtime_report,
    }
    path = validation_receipt_path(export)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    os.replace(temporary, path)
    _sdk_log("receipt", f"saved successful Modal validation to {path}")
    return path


def load_validation_receipt(
    export: Path,
    *,
    static_report: dict[str, Any],
    configuration: dict[str, object],
    max_age_hours: float = DEFAULT_VALIDATION_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if max_age_hours <= 0:
        raise SdkError("validation receipt maximum age must be positive")
    path = validation_receipt_path(export)
    if not path.is_file():
        return None, "no prior validation receipt"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validated_at = datetime.fromisoformat(str(receipt["validated_at"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, "prior validation receipt is malformed"
    if validated_at.tzinfo is None or validated_at.utcoffset() is None:
        return None, "prior validation receipt timestamp is not timezone-aware"
    if (
        receipt.get("schema_version") != VALIDATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "ACCEPTED"
        or receipt.get("runtime_report", {}).get("status") != "ACCEPTED"
    ):
        return None, "prior validation receipt was not successful"
    tree_digest = static_report.get("repository", {}).get("repository_tree_sha256")
    if receipt.get("repository_tree_sha256") != tree_digest:
        return None, "export content changed after validation"
    if receipt.get("configuration") != configuration:
        return None, "Modal validation resource settings changed"
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise SdkError("current receipt-check timestamp must be timezone-aware")
    age = current.astimezone(timezone.utc) - validated_at.astimezone(timezone.utc)
    if age < -timedelta(minutes=5):
        return None, "prior validation receipt timestamp is in the future"
    if age > timedelta(hours=max_age_hours):
        return None, "prior validation receipt expired"
    return receipt, "matching validation receipt"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        name
        for name in names
        if name
        in {
            ".DS_Store",
            ".competition-sdk",
            ".git",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
        }
        or name.endswith((".pyc", ".pyo"))
        or name == ".env"
        or (name.startswith(".env.") and name != ".env.template")
    }
    return ignored


def _safe_export_name(repository_name: str) -> str:
    name = repository_name.rsplit("/", 1)[-1]
    if not REPOSITORY_NAME_PATTERN.fullmatch(name):
        raise SdkError(
            "repository name must contain only letters, numbers, '.', '_' or '-'"
        )
    return name


@contextmanager
def stable_modal_build_context(source: Path):
    """Keep heavyweight Docker layers independent of Python service revisions."""

    with tempfile.TemporaryDirectory(prefix="vidaio-modal-build-context-") as temp:
        context = Path(temp) / "compression"
        shutil.copytree(source, context, ignore=_copy_ignore, symlinks=True)
        app_path = context / "app.py"
        if not app_path.is_file():
            raise SdkError("compression Docker context must contain app.py")
        app_path.write_text(MODAL_APP_BUILD_PLACEHOLDER, encoding="utf-8")
        yield context


def prepare_export(
    source_root: Path,
    destination: Path,
    *,
    refresh: bool = False,
) -> Path:
    source_root = source_root.resolve(strict=True)
    miner_source = source_root / "miner"
    descriptor = source_root / "competition_solution.json"
    fixture = source_root / "competitions/fixtures/compression_warmup_input.mp4"
    if not miner_source.is_dir() or not descriptor.is_file() or not fixture.is_file():
        raise SdkError(
            "source checkout is missing miner/, competition_solution.json, or the warmup fixture"
        )

    destination = destination.resolve()
    if destination.exists():
        marker = destination / EXPORT_MARKER
        if not marker.is_file():
            raise SdkError(f"refusing to reuse unmarked directory: {destination}")
        if not refresh:
            changed_tools = [
                relative
                for relative in CANONICAL_SDK_PATHS
                if (source_root / relative).is_file()
                and (destination / relative).is_file()
                and (source_root / relative).read_bytes()
                != (destination / relative).read_bytes()
            ]
            detail = (
                f"; detected newer/different source file(s): {changed_tools}"
                if changed_tools
                else ""
            )
            _sdk_log("warning", f"{REFRESH_WARNING}{detail}")
            return destination
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        miner_source,
        destination / "miner",
        ignore=_copy_ignore,
        symlinks=True,
    )
    shutil.copy2(descriptor, destination / "competition_solution.json")
    shutil.copy2(miner_source / "requirements.txt", destination / "requirements.txt")
    build_worker = source_root / "scripts/competition_modal_build.py"
    if not build_worker.is_file():
        build_worker = Path(__file__).resolve().parents[1] / build_worker.relative_to(
            source_root
        )
    if not build_worker.is_file():
        raise SdkError("source checkout is missing scripts/competition_modal_build.py")
    worker_destination = destination / "scripts/competition_modal_build.py"
    worker_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build_worker, worker_destination)
    fixture_destination = (
        destination / "competitions/fixtures/compression_warmup_input.mp4"
    )
    fixture_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture, fixture_destination)
    fixture_readme = source_root / "competitions/fixtures/README.md"
    if fixture_readme.is_file():
        shutil.copy2(fixture_readme, fixture_destination.parent / "README.md")

    (destination / EXPORT_MARKER).write_text(
        "Prepared by miner/competition_sdk.py.\n", encoding="utf-8"
    )
    (destination / ".gitignore").write_text(
        """**/__pycache__/
*.py[cod]
.env
.env.*
!.env.template
.pytest_cache/
.ruff_cache/
.DS_Store
""",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        """# Private Vidaio compression competition solution

This repository was prepared from a customized `miner/` checkout by
`miner/competition_sdk.py`. Keep it private and provide the validator only a
short-lived, contents-read-only credential after publishing.

Before each push, run the shared batch-of-one runtime preflight documented in
`miner/README.md`. Never commit `.env`, PAT, Modal, S3, wallet, or validator
credentials.
""",
        encoding="utf-8",
    )
    return destination


def run_common_preflight(
    export: Path,
) -> dict[str, Any]:
    _sdk_log("static", f"validating sanitized export at {export}")
    command = [
        sys.executable,
        str(export / "miner/common_preflight.py"),
        "--repository",
        str(export),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=1900,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    rendered = (result.stdout or result.stderr).strip()
    try:
        report = json.loads(rendered)
    except json.JSONDecodeError as exc:
        raise SdkError(
            f"common preflight returned invalid output: {rendered[:500]}"
        ) from exc
    if result.returncode != 0:
        raise SdkError(
            f"common preflight rejected the solution: {report.get('reason_code')}: "
            f"{report.get('detail')}"
        )
    repository = report.get("repository", {})
    _sdk_log(
        "static",
        "accepted "
        f"{repository.get('file_count', '?')} files / "
        f"{repository.get('total_bytes', '?')} bytes",
    )
    return report


def _load_modal():
    try:
        import modal
    except ImportError as exc:
        raise SdkError(
            "Modal SDK is not installed; install miner/requirements.txt and run "
            "`uvx modal setup`"
        ) from exc
    return modal


def _completed_process_output(process: Any, label: str) -> str:
    process.wait()
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    if stderr:
        for line in stderr.strip().splitlines():
            _sdk_log("sandbox-exec", f"{label}: {line}")
    if process.returncode != 0:
        detail = (stderr or stdout or "no process output").strip()[:1000]
        raise SdkError(f"{label} failed with exit {process.returncode}: {detail}")
    return stdout.strip()


def _stop_modal_app(
    environment_name: str,
    app_name: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str | None:
    try:
        completed = runner(
            [
                sys.executable,
                "-m",
                "modal",
                "app",
                "stop",
                "--yes",
                "--env",
                environment_name,
                app_name,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return f"remote App stop failed: {type(exc).__name__}: {str(exc)[:500]}"
    if completed.returncode != 0:
        detail = completed.stderr or completed.stdout or "Modal CLI returned failure"
        return f"remote App stop failed: {detail.strip()[:500]}"
    return None


def _build_modal_base_image(
    *,
    worker: Path,
    environment_name: str,
    app_name: str,
    dockerfile: Path,
    context_dir: Path,
    timeout_seconds: float,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    stopper: Callable[[str, str], str | None] | None = None,
) -> str:
    """Build untrusted Docker instructions in a client process we can kill."""

    if timeout_seconds <= 0:
        raise SdkError("modal_build_timeout must be positive")
    with tempfile.TemporaryDirectory(prefix="competition-sdk-modal-build-") as temp:
        result_path = Path(temp) / "result.json"
        command = [
            sys.executable,
            str(worker),
            "--environment",
            environment_name,
            "--app-name",
            app_name,
            "--dockerfile",
            str(dockerfile),
            "--context-dir",
            str(context_dir),
            "--result",
            str(result_path),
        ]
        try:
            completed = runner(
                command,
                check=False,
                stdout=sys.stderr,
                stderr=sys.stderr,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            stop = stopper or (
                lambda environment, name: _stop_modal_app(environment, name)
            )
            stop_failure = stop(environment_name, app_name)
            detail = (
                f"Modal image build exceeded modal_build_timeout "
                f"({timeout_seconds:g} seconds; app={app_name})"
            )
            if stop_failure:
                detail = f"{detail}; {stop_failure}"
            raise SdkError(detail) from exc
        if completed.returncode != 0:
            raise SdkError(
                f"Modal image build worker failed with exit {completed.returncode}"
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            image_id = str(payload["image_id"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SdkError("Modal image build worker returned no image ID") from exc
        if not image_id:
            raise SdkError("Modal image build worker returned an empty image ID")
        return image_id


def run_modal_preflight(
    export: Path,
    *,
    environment_name: str,
    gpu: str,
    requested_cpu: float = 16,
    cpu_limit: float = 32,
    sandbox_timeout: int = 1800,
    modal_build_timeout_seconds: float = DEFAULT_MODAL_BUILD_TIMEOUT_SECONDS,
    modal_api: Any | None = None,
) -> dict[str, Any]:
    """Run the batch-one qualification in the validator-shaped Modal sandbox."""

    if requested_cpu <= 0 or cpu_limit < requested_cpu or cpu_limit > 32:
        raise SdkError("Modal CPU values must satisfy 0 < request <= limit <= 32")
    if sandbox_timeout < 60:
        raise SdkError("Modal sandbox timeout must be at least 60 seconds")
    if modal_build_timeout_seconds <= 0:
        raise SdkError("modal_build_timeout must be positive")
    if gpu not in SUPPORTED_MODAL_GPU_REQUESTS:
        raise SdkError(
            f"unsupported Modal GPU request {gpu!r}; choose one of "
            f"{sorted(SUPPORTED_MODAL_GPU_REQUESTS)}"
        )

    export = export.resolve(strict=True)
    fixture = export / "competitions/fixtures/compression_warmup_input.mp4"
    dockerfile = export / "miner/compression/Dockerfile"
    service_app = export / "miner/compression/app.py"
    preflight = export / "miner/common_preflight.py"
    for required in (fixture, dockerfile, service_app, preflight):
        if not required.is_file():
            raise SdkError(f"Modal preflight input is missing: {required}")

    modal = modal_api or _load_modal()
    run_token = uuid.uuid4().hex[:12]
    app_name = f"{MODAL_APP_NAME}-{run_token}"
    _sdk_log(
        "modal",
        f"checking app={app_name} environment={environment_name}",
    )
    try:
        app = modal.App.lookup(
            app_name,
            environment_name=environment_name,
            create_if_missing=True,
        )
    except Exception as exc:
        raise SdkError(
            "Modal login/environment check failed; run `uvx modal setup` and "
            f"ensure the {environment_name!r} environment exists"
        ) from exc

    input_volume_name = f"vidaio-sdk-{run_token}-input"
    output_volume_name = f"vidaio-sdk-{run_token}-output"
    sandbox_name = f"vidaio-sdk-{run_token}"
    sandbox = None
    sandbox_usage_started: float | None = None
    result: dict[str, Any] | None = None
    resource_allocation: dict[str, Any] | None = None
    isolation_report: dict[str, Any] | None = None
    readiness_report: dict[str, Any] | None = None
    log_threads: list[threading.Thread] = []
    created_volume_names: list[str] = []
    try:
        _sdk_log(
            "volumes",
            f"creating input={input_volume_name} and output={output_volume_name}",
        )
        input_volume = modal.Volume.from_name(
            input_volume_name,
            environment_name=environment_name,
            create_if_missing=True,
        )
        created_volume_names.append(input_volume_name)
        output_volume = modal.Volume.from_name(
            output_volume_name,
            environment_name=environment_name,
            create_if_missing=True,
            version=2,
        )
        created_volume_names.append(output_volume_name)
        _sdk_log(
            "volumes",
            f"uploading {fixture.name} ({fixture.stat().st_size} bytes) read-only",
        )
        with input_volume.batch_upload(force=True) as upload:
            upload.put_file(fixture, MODAL_INPUT_VOLUME_PATH)
        _sdk_log("volumes", "warmup fixture upload complete")

        with stable_modal_build_context(dockerfile.parent) as build_context:
            _sdk_log(
                "modal-build",
                "resolving cached image layers and building changed layers; "
                f"deadline={modal_build_timeout_seconds:g}s; "
                "Modal's live build status follows",
            )
            if modal_api is None:
                worker = export / "scripts/competition_modal_build.py"
                if not worker.is_file():
                    raise SdkError(f"Modal build worker is missing: {worker}")
                image_id = _build_modal_base_image(
                    worker=worker,
                    environment_name=environment_name,
                    app_name=app_name,
                    dockerfile=build_context / "Dockerfile",
                    context_dir=build_context,
                    timeout_seconds=modal_build_timeout_seconds,
                )
                image = modal.Image.from_id(image_id)
            else:
                # Injected/test clients remain in-process; production always uses
                # the killable build worker above.
                image = modal.Image.from_dockerfile(
                    build_context / "Dockerfile",
                    context_dir=build_context,
                    secrets=[],
                )
            image = image.add_local_file(
                service_app, "/app/app.py", copy=True
            ).add_local_file(
                preflight,
                "/vidaio/common_preflight.py",
                copy=True,
            )
            enable_output = getattr(modal, "enable_output", None)
            output_context = (
                enable_output() if callable(enable_output) else nullcontext()
            )
            # Modal's public output manager may write to stdout. Redirect it so the
            # final command stdout remains a single machine-readable JSON document.
            with redirect_stdout(sys.stderr), output_context:
                sandbox = modal.Sandbox.create(
                    "python3",
                    "-m",
                    "uvicorn",
                    "app:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8003",
                    app=app,
                    name=sandbox_name,
                    tags={
                        "purpose": "competition-sdk-preflight",
                        "run_id": run_token,
                    },
                    image=image,
                    env={
                        "DISABLE_REMOTE_IO": "true",
                        "COMPETITION_INPUT_ROOT": "/evaluation-inputs",
                        "COMPETITION_OUTPUT_ROOT": MODAL_OUTPUT_ROOT,
                        "MINER_CLEANUP_ENABLED": "false",
                        "MINER_STORAGE_CLEANUP_ENABLED": "false",
                    },
                    secrets=[],
                    timeout=sandbox_timeout,
                    workdir="/app",
                    gpu=gpu,
                    cpu=(requested_cpu, cpu_limit),
                    block_network=True,
                    volumes={
                        "/evaluation-inputs": input_volume.with_mount_options(
                            read_only=True
                        ),
                        MODAL_OUTPUT_ROOT: output_volume.with_mount_options(
                            read_only=False
                        ),
                    },
                    include_oidc_identity_token=False,
                    encrypted_ports=[],
                    h2_ports=[],
                    unencrypted_ports=[],
                    verbose=True,
                )
        sandbox_usage_started = time.monotonic()

        sandbox_id = str(getattr(sandbox, "object_id", ""))
        _sdk_log(
            "sandbox",
            f"created id={sandbox_id or '<pending>'} name={sandbox_name}; "
            "streaming service stdout/stderr",
        )
        log_threads = _start_sandbox_log_streams(sandbox)

        _sdk_log("isolation", "running validator isolation probe")
        isolation_process = sandbox.exec(
            "python3", "-c", ISOLATION_PROBE, timeout=30
        )
        isolation_output = _completed_process_output(
            isolation_process, "active isolation probe"
        )
        try:
            isolation_report = json.loads(isolation_output)
        except json.JSONDecodeError as exc:
            raise SdkError("active isolation probe did not return JSON") from exc
        if not isolation_report.get("passed"):
            raise SdkError(
                "active isolation probe failed: "
                f"{json.dumps(isolation_report, sort_keys=True)}"
            )
        _sdk_log("isolation", "network, mounts, references, and credentials accepted")

        resource_process = sandbox.exec(
            "python3", "-I", "-S", "-c", RESOURCE_PROBE, timeout=30
        )
        resource_output = _completed_process_output(
            resource_process, "allocated resource probe"
        )
        try:
            resource_report = json.loads(resource_output)
            allocated_gpu_types = [
                canonical_modal_gpu_type(str(name))
                for name in resource_report["gpu_names"]
            ]
            allocated_cpu_cores = float(resource_report["allocated_cpu_cores"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SdkError("allocated resource probe returned invalid data") from exc
        if (
            not allocated_gpu_types
            or len(set(allocated_gpu_types)) != 1
            or allocated_cpu_cores <= 0
        ):
            raise SdkError("allocated GPU/CPU resources are missing or heterogeneous")
        resource_allocation = {
            "gpu": allocated_gpu_types[0],
            "gpu_count": len(allocated_gpu_types),
            "cpu_cores": allocated_cpu_cores,
            "probe": resource_report,
        }
        _sdk_log(
            "resources",
            f"allocated gpu={resource_allocation['gpu']} "
            f"count={resource_allocation['gpu_count']} "
            f"cpu_cores={allocated_cpu_cores:g}",
        )

        _sdk_log("readiness", "waiting for validator-compatible /health response")
        readiness_process = sandbox.exec(
            "python3",
            "-c",
            READINESS_PROBE,
            timeout=min(sandbox_timeout, 330),
        )
        readiness_output = _completed_process_output(
            readiness_process, "localhost readiness probe"
        )
        try:
            readiness_report = json.loads(readiness_output)
        except json.JSONDecodeError as exc:
            raise SdkError("localhost readiness probe did not return JSON") from exc
        _sdk_log("readiness", "/health returned status=ok")

        _sdk_log(
            "inference",
            "submitting batch=1 to /compress with AV1 and VMAF threshold 90",
        )
        runtime_process = sandbox.exec(
            "python3",
            "/vidaio/common_preflight.py",
            "--runtime-only",
            "--fixture",
            MODAL_INPUT_PATH,
            "--service-url",
            MODAL_SERVICE_URL,
            "--host-input-root",
            "/evaluation-inputs",
            "--host-output-root",
            MODAL_OUTPUT_ROOT,
            "--prepositioned-input",
            "--input-relative-path",
            "compression_warmup_input.mp4",
            "--run-id",
            "compression-warmup-input",
            timeout=max(60, sandbox_timeout - 30),
        )
        rendered = _completed_process_output(runtime_process, "Modal runtime preflight")
        try:
            report = json.loads(rendered)
        except json.JSONDecodeError as exc:
            raise SdkError(
                f"Modal runtime preflight returned invalid JSON: {rendered[:500]}"
            ) from exc
        if report.get("status") != "ACCEPTED":
            raise SdkError(f"Modal runtime preflight did not pass: {report}")
        _sdk_log("volumes", "committing v2 output Volume with sync")
        commit_process = sandbox.exec("sync", MODAL_OUTPUT_ROOT, timeout=30)
        _completed_process_output(commit_process, "v2 output Volume commit")
        runtime = report.get("runtime", {})
        output_media = runtime.get("output_media", {})
        _sdk_log(
            "inference",
            "accepted output "
            f"codec={output_media.get('codec', '?')} "
            f"size={output_media.get('width', '?')}x{output_media.get('height', '?')} "
            f"frames={output_media.get('frame_count', '?')} "
            f"duration={output_media.get('duration_seconds', '?')}s",
        )
        result = {
            "status": "ACCEPTED",
            "modal_environment": environment_name,
            "modal_app": app_name,
            "modal_build_timeout_seconds": modal_build_timeout_seconds,
            "sandbox_id": str(getattr(sandbox, "object_id", "")),
            "gpu": gpu,
            "cpu": {"request": requested_cpu, "limit": cpu_limit},
            "allocated_resources": resource_allocation,
            "isolation": isolation_report,
            "readiness": readiness_report,
            "network_blocked": True,
            "secrets_attached": False,
            "oidc_identity_attached": False,
            "mounts": {
                "/evaluation-inputs": {
                    "read_only": True,
                    "fixture": MODAL_INPUT_PATH,
                },
                MODAL_OUTPUT_ROOT: {"read_only": False},
            },
            "preflight": report,
        }
        return result
    except SdkError:
        raise
    except Exception as exc:
        raise SdkError(f"Modal sandbox preflight failed: {exc}") from exc
    finally:
        cleanup_errors: list[str] = []
        if sandbox is not None:
            try:
                _sdk_log("cleanup", "terminating Sandbox")
                sandbox.terminate(wait=True)
                _sdk_log("cleanup", "Sandbox terminated")
            except Exception as exc:
                cleanup_errors.append(f"sandbox termination: {exc}")
        if (
            result is not None
            and sandbox_usage_started is not None
            and resource_allocation is not None
        ):
            sandbox_runtime_seconds = max(0.0, time.monotonic() - sandbox_usage_started)
            estimated_cost = estimate_modal_compute_cost(
                resource_allocation["gpu"],
                resource_allocation["cpu_cores"],
                sandbox_runtime_seconds,
                allocated_gpu_count=resource_allocation["gpu_count"],
            )
            result["modal_cost_estimate"] = {
                "currency": "USD",
                "estimated_consumed_balance_usd": format(estimated_cost, "f"),
                "sandbox_runtime_seconds": sandbox_runtime_seconds,
                "allocated_gpu": resource_allocation["gpu"],
                "allocated_gpu_count": resource_allocation["gpu_count"],
                "allocated_cpu_cores": resource_allocation["cpu_cores"],
                "gpu_rate_per_second_usd": format(
                    GPU_PRICE_PER_SECOND_USD[resource_allocation["gpu"]], "f"
                ),
                "cpu_rate_per_core_second_usd": format(
                    SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD, "f"
                ),
                "attribution_method": MODAL_COST_ATTRIBUTION_METHOD,
                "scope": (
                    "estimated allocated Sandbox GPU and CPU capacity after creation "
                    "through termination; excludes memory, image build, storage, "
                    "credits, reservations, taxes, and CPU utilization reconciliation"
                ),
            }
            _sdk_log(
                "cost",
                "estimated consumed balance "
                f"${format(estimated_cost, 'f')} USD over "
                f"{sandbox_runtime_seconds:.3f}s",
            )
        for thread in log_threads:
            thread.join(timeout=5)
        for volume_name in reversed(created_volume_names):
            try:
                _sdk_log("cleanup", f"deleting Volume {volume_name}")
                modal.Volume.objects.delete(
                    volume_name,
                    allow_missing=True,
                    environment_name=environment_name,
                )
                _sdk_log("cleanup", f"deleted Volume {volume_name}")
            except Exception as exc:
                cleanup_errors.append(f"volume deletion ({volume_name}): {exc}")
        if cleanup_errors and sys.exc_info()[0] is None:
            raise SdkError("Modal cleanup failed: " + "; ".join(cleanup_errors))


def github_request(
    method: str,
    path: str,
    pat: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{GITHUB_API}{path}",
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {pat}",
            "Content-Type": "application/json",
            "User-Agent": "vidaio-competition-sdk",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
    )
    try:
        with build_opener(_NoRedirect).open(request, timeout=30) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(detail).get("message", detail)
        except json.JSONDecodeError:
            message = detail
        raise GitHubApiError(exc.code, str(message)[:500]) from exc
    except (URLError, TimeoutError) as exc:
        raise SdkError(f"GitHub API is unreachable: {exc}") from exc
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SdkError("GitHub API returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise SdkError("GitHub API response was not an object")
    return value


def ensure_private_repository(
    repository_reference: str,
    pat: str,
    *,
    use_existing: bool = False,
    requester: GitHubRequester = github_request,
) -> GitHubRepositoryTarget:
    user = requester("GET", "/user", pat, None)
    login = str(user.get("login") or "")
    if not login:
        raise SdkError("GitHub token did not resolve an authenticated user")
    user_id = user.get("id")
    author_name = str(user.get("name") or login)
    author_email = (
        f"{user_id}+{login}@users.noreply.github.com"
        if isinstance(user_id, int) and user_id > 0
        else f"{login}@users.noreply.github.com"
    )
    identity = GitHubIdentity(login, author_name, author_email)

    if "/" in repository_reference:
        owner, name = repository_reference.split("/", 1)
    else:
        owner, name = login, repository_reference
    if not REPOSITORY_NAME_PATTERN.fullmatch(
        owner
    ) or not REPOSITORY_NAME_PATTERN.fullmatch(name):
        raise SdkError("GitHub repository must be written as name or owner/name")

    if use_existing:
        repository = requester("GET", f"/repos/{quote(owner)}/{quote(name)}", pat, None)
    else:
        endpoint = (
            "/user/repos"
            if owner.lower() == login.lower()
            else f"/orgs/{quote(owner)}/repos"
        )
        try:
            repository = requester(
                "POST",
                endpoint,
                pat,
                {
                    "name": name,
                    "description": "Private Vidaio compression competition solution",
                    "private": True,
                    "has_issues": False,
                    "has_projects": False,
                    "has_wiki": False,
                },
            )
        except GitHubApiError as exc:
            if exc.status == 422:
                raise SdkError(
                    f"repository {owner}/{name} may already exist; rerun with "
                    "--use-existing (and --refresh after source changes)"
                ) from exc
            raise
    if repository.get("private") is not True:
        raise SdkError("GitHub repository is not private; refusing to push")
    full_name = str(repository.get("full_name") or f"{owner}/{name}")
    default_branch = str(repository.get("default_branch") or "main")
    return GitHubRepositoryTarget(
        full_name=full_name,
        repository_url=f"https://github.com/{full_name}.git",
        default_branch=default_branch,
        existing=use_existing,
        identity=identity,
    )


def _run_git(
    repository: Path,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> str:
    environment = {
        **os.environ,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    if env_overrides:
        environment.update(env_overrides)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "credential.helper=",
            "-c",
            "http.followRedirects=false",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    if result.returncode != 0:
        raise SdkError(
            f"git {' '.join(args[:2])} failed: {(result.stderr or result.stdout).strip()[:500]}"
        )
    return result.stdout


@contextmanager
def _git_askpass_environment(pat: str):
    with tempfile.TemporaryDirectory(prefix="vidaio-sdk-askpass-") as temp:
        askpass = Path(temp) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) printf \'%s\' "$VIDAIO_GIT_USERNAME" ;; *) printf \'%s\' "$VIDAIO_GIT_PASSWORD" ;; esac\n',
            encoding="utf-8",
        )
        askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        yield {
            **os.environ,
            "GIT_ASKPASS": str(askpass),
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "VIDAIO_GIT_USERNAME": "x-access-token",
            "VIDAIO_GIT_PASSWORD": pat,
        }


def initialize_export_git(
    export: Path,
    *,
    author_name: str,
    author_email: str,
    parent_repository_url: str | None = None,
    parent_branch: str = "main",
    pat: str = "",
) -> None:
    if (export / ".git").exists():
        raise SdkError(
            "export already contains Git metadata from an earlier SDK version; "
            "rerun with --refresh"
        )
    _run_git(export, "init", "-q", "-b", "main")
    _run_git(export, "config", "user.name", author_name)
    _run_git(export, "config", "user.email", author_email)
    parent_sha = ""
    if parent_repository_url:
        if not pat:
            raise SdkError("updating an existing repository requires a GitHub PAT")
        _sdk_log("github", f"fetching existing {parent_branch} branch as commit parent")
        with _git_askpass_environment(pat) as environment:
            _run_git(
                export,
                "fetch",
                "--quiet",
                "--no-tags",
                "--depth=1",
                "--",
                parent_repository_url,
                f"refs/heads/{parent_branch}",
                env_overrides=environment,
            )
        parent_sha = _run_git(export, "rev-parse", "FETCH_HEAD").strip()
    _run_git(export, "add", ".")
    if parent_sha:
        tree_sha = _run_git(export, "write-tree").strip()
        commit_sha = _run_git(
            export,
            "commit-tree",
            tree_sha,
            "-p",
            parent_sha,
            "-m",
            "Update compression competition solution",
        ).strip()
        _run_git(export, "update-ref", "refs/heads/main", commit_sha)
    else:
        _run_git(
            export,
            "commit",
            "-q",
            "-m",
            "Prepare compression competition solution",
        )


def push_with_askpass(
    export: Path,
    repository_url: str,
    pat: str,
    *,
    branch: str = "main",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    with _git_askpass_environment(pat) as env:
        result = runner(
            [
                "git",
                "-C",
                str(export),
                "-c",
                "credential.helper=",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "http.followRedirects=false",
                "push",
                "--",
                repository_url,
                f"HEAD:refs/heads/{branch}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).replace(pat, "[REDACTED]")
        raise SdkError(f"GitHub push failed: {detail.strip()[:500]}")


def cleanup_export_git(export: Path) -> None:
    if not (export / EXPORT_MARKER).is_file():
        raise SdkError("refusing to clean Git metadata from an unmarked export")
    git_directory = export / ".git"
    if git_directory.exists():
        shutil.rmtree(git_directory)


def _repository_prompt(value: str | None) -> str:
    repository = (
        value or input("Private GitHub repository (name or owner/name): ").strip()
    )
    if not repository:
        raise SdkError("GitHub repository name is required")
    _safe_export_name(repository)
    return repository


def _pat_from_environment_or_prompt(
    environment_name: str, *, warn_if_reusing_export: bool = False
) -> str:
    pat = os.getenv(environment_name, "")
    if not pat:
        if warn_if_reusing_export:
            _sdk_log("warning", f"{REFRESH_WARNING}; GitHub PAT prompt follows")
        pat = getpass.getpass(
            f"GitHub PAT ({environment_name}; input is hidden and never stored): "
        )
    if not pat:
        raise SdkError("GitHub PAT is required")
    return pat


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "publish"))
    parser.add_argument(
        "--repository", help="GitHub name or owner/name; prompted if omitted"
    )
    parser.add_argument("--source-root", type=Path, default=root)
    parser.add_argument("--export", type=Path, help="Prepared export directory")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild an SDK-marked export"
    )
    parser.add_argument(
        "--modal-environment",
        default="dev",
        help="Modal environment used for disposable validation resources",
    )
    parser.add_argument("--modal-gpu", default="L40S")
    parser.add_argument(
        "--modal-cpu", type=float, default=16, help="Requested Modal CPU cores"
    )
    parser.add_argument(
        "--modal-cpu-limit", type=float, default=32, help="Hard Modal CPU limit"
    )
    parser.add_argument(
        "--modal-timeout",
        type=int,
        default=1800,
        help="Sandbox lifetime and warmup deadline in seconds",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "competition JSON/YAML manifest supplying modal_build_timeout; "
            "defaults to 10 minutes when omitted or absent"
        ),
    )
    parser.add_argument("--pat-env", default="GITHUB_TOKEN")
    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help="run a new Modal validation even when a matching receipt exists",
    )
    parser.add_argument(
        "--validation-max-age-hours",
        type=float,
        default=DEFAULT_VALIDATION_MAX_AGE_HOURS,
        help="maximum age of a matching Modal validation receipt used by publish",
    )
    parser.add_argument(
        "--git-author-name",
        help="override the authenticated GitHub account's commit author name",
    )
    parser.add_argument(
        "--git-author-email",
        help="override the authenticated GitHub account's no-reply commit email",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pat = ""
    try:
        repository_reference = _repository_prompt(args.repository)
        export = args.export or (
            args.source_root
            / "miner/.competition-sdk"
            / _safe_export_name(repository_reference)
        )
        reusing_export_without_refresh = export.exists() and not args.refresh
        _sdk_log(
            "prepare",
            f"action={args.action} source={args.source_root} export={export} "
            f"refresh={args.refresh}",
        )
        export = prepare_export(args.source_root, export, refresh=args.refresh)
        _sdk_log("prepare", "sanitized standalone export ready")
        static_report = run_common_preflight(export)
        if args.action == "prepare":
            print(
                json.dumps(
                    {
                        "status": "STATIC_READY",
                        "export": str(export),
                        "preflight": static_report,
                        "next_step": "run `uvx modal setup`, then run validate or publish",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        manifest_path = getattr(args, "manifest", None)
        if manifest_path is not None and not manifest_path.is_absolute():
            manifest_path = args.source_root / manifest_path
        modal_build_timeout_seconds = load_modal_build_timeout_seconds(manifest_path)
        _sdk_log(
            "modal-build",
            f"using modal_build_timeout={modal_build_timeout_seconds:g}s "
            f"source={manifest_path or 'default (10m)'}",
        )

        validation_configuration = _validation_configuration(
            environment_name=args.modal_environment,
            gpu=args.modal_gpu,
            requested_cpu=args.modal_cpu,
            cpu_limit=args.modal_cpu_limit,
            sandbox_timeout=args.modal_timeout,
            modal_build_timeout_seconds=modal_build_timeout_seconds,
        )
        validation_reused = False
        receipt_path = validation_receipt_path(export)
        runtime_report: dict[str, Any]
        if args.action == "publish" and not args.revalidate:
            receipt, receipt_reason = load_validation_receipt(
                export,
                static_report=static_report,
                configuration=validation_configuration,
                max_age_hours=args.validation_max_age_hours,
            )
            if receipt is not None:
                runtime_report = receipt["runtime_report"]
                validation_reused = True
                _sdk_log(
                    "receipt",
                    "reusing successful Modal validation for this exact export; "
                    "no Sandbox will be started",
                )
            else:
                _sdk_log("receipt", f"running Modal validation: {receipt_reason}")
        if not validation_reused:
            runtime_report = run_modal_preflight(
                export,
                environment_name=args.modal_environment,
                gpu=args.modal_gpu,
                requested_cpu=args.modal_cpu,
                cpu_limit=args.modal_cpu_limit,
                sandbox_timeout=args.modal_timeout,
                modal_build_timeout_seconds=modal_build_timeout_seconds,
            )
            receipt_path = write_validation_receipt(
                export,
                static_report=static_report,
                runtime_report=runtime_report,
                configuration=validation_configuration,
            )
        if args.action == "validate":
            cost_estimate = runtime_report.get("modal_cost_estimate", {})
            print(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "export": str(export),
                        "preflight": runtime_report,
                        "modal_cost_estimate": cost_estimate,
                        "estimated_consumed_balance_usd": cost_estimate.get(
                            "estimated_consumed_balance_usd"
                        ),
                        "validation_receipt": str(receipt_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        pat = _pat_from_environment_or_prompt(
            args.pat_env,
            warn_if_reusing_export=reusing_export_without_refresh,
        )
        target = ensure_private_repository(
            repository_reference,
            pat,
            use_existing=args.use_existing,
        )
        author_name = args.git_author_name or target.identity.name
        author_email = args.git_author_email or target.identity.email
        _sdk_log(
            "github",
            f"publishing to {target.full_name}:{target.default_branch} as "
            f"{author_name} <{author_email}>",
        )
        try:
            initialize_export_git(
                export,
                author_name=author_name,
                author_email=author_email,
                parent_repository_url=(
                    target.repository_url if target.existing else None
                ),
                parent_branch=target.default_branch,
                pat=pat,
            )
            push_with_askpass(
                export,
                target.repository_url,
                pat,
                branch=target.default_branch,
            )
        finally:
            cleanup_export_git(export)
        print(
            json.dumps(
                {
                    "status": "PUBLISHED",
                    "repository": target.full_name,
                    "repository_url": target.repository_url,
                    "branch": target.default_branch,
                    "commit_author": {
                        "name": author_name,
                        "email": author_email,
                    },
                    "export": str(export),
                    "runtime_preflight": "ACCEPTED",
                    "validation_reused": validation_reused,
                    "validation_receipt": str(receipt_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (SdkError, OSError, subprocess.SubprocessError) as exc:
        detail = str(exc).replace(pat, "[REDACTED]") if pat else str(exc)
        print(
            json.dumps({"status": "FAILED", "detail": detail}, indent=2),
            file=sys.stderr,
        )
        return 2
    finally:
        pat = ""


if __name__ == "__main__":
    raise SystemExit(main())
