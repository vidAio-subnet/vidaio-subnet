"""Validator-owned, restart-safe Modal Sandbox supervisor.

Only immutable image IDs produced by the trusted build boundary are accepted.
Contender code never controls Modal resources, mounts, credentials, network
policy, ports, command, or lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any, Protocol

from .config import CompetitionManifest
from .contracts import CompetitionCompressionRequest, CompetitionCompressionResponse
from .models import CompetitionSandbox
from .pricing import canonical_modal_gpu_type
from .repository import CompetitionRepository, parse_utc


MAX_SANDBOX_LIFETIME = timedelta(hours=23, minutes=30)
ROLLOVER_MARGIN = timedelta(minutes=5)
SUPERVISOR_TIMEOUT_GRACE_SECONDS = 30


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


HTTP_COMPRESSION_CLIENT = r"""
import json
import sys
import urllib.request

body = sys.stdin.buffer.read()
request = urllib.request.Request(
    "http://127.0.0.1:8003/compress",
    data=body,
    method="POST",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=float(sys.argv[1])) as response:
    payload = response.read()
if response.status != 200:
    raise SystemExit(f"compression returned HTTP {response.status}")
sys.stdout.buffer.write(payload)
"""


class SandboxRunnerError(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class SandboxLaunchSpec:
    competition_id: str
    hotkey: str
    generation: int
    app_name: str
    sandbox_name: str
    image_id: str
    input_volume_name: str
    output_volume_name: str
    gpu_type: str
    cpu_request: float
    cpu_limit: float
    lifetime_seconds: int


@dataclass
class SandboxHandle:
    app_id: str
    sandbox_id: str
    raw: Any = None


@dataclass(frozen=True)
class SandboxResourceAllocation:
    gpu_type: str
    gpu_count: int
    cpu_cores: float
    report: dict[str, Any]


@dataclass
class SandboxSession:
    record: CompetitionSandbox
    handle: SandboxHandle


class SandboxBackend(Protocol):
    def create(self, spec: SandboxLaunchSpec) -> SandboxHandle: ...

    def attach(self, app_id: str, sandbox_id: str) -> SandboxHandle: ...

    def attach_by_name(self, app_name: str, sandbox_name: str) -> SandboxHandle: ...

    def verify_isolation(self, handle: SandboxHandle) -> dict[str, Any]: ...

    def inspect_resources(
        self, handle: SandboxHandle
    ) -> SandboxResourceAllocation: ...

    def health(self, handle: SandboxHandle, timeout_seconds: int) -> dict[str, Any]: ...

    def prepare_outputs(
        self, handle: SandboxHandle, output_paths: tuple[str, ...]
    ) -> None: ...

    def invoke(
        self, handle: SandboxHandle, payload_json: str, timeout_seconds: int
    ) -> str: ...

    def commit_outputs(self, handle: SandboxHandle) -> None: ...

    def terminate(self, handle: SandboxHandle) -> None: ...


def _completed_process_output(process: Any, label: str) -> str:
    process.wait()
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    if process.returncode != 0:
        detail = (stderr or stdout or "no process output").strip()[:1000]
        raise SandboxRunnerError(
            "SANDBOX_EXEC_FAILED",
            f"{label} exited {process.returncode}: {detail}",
        )
    return stdout.strip()


class ModalSandboxBackend:
    """Thin adapter around Modal; it accepts only validator-owned launch specs."""

    def __init__(self, *, environment_name: str = "dev", modal_api: Any = None) -> None:
        if modal_api is None:
            try:
                import modal as modal_api
            except ImportError as exc:
                raise RuntimeError(
                    "Modal SDK is required for competition execution"
                ) from exc
        self.modal = modal_api
        self.environment_name = environment_name

    def create(self, spec: SandboxLaunchSpec) -> SandboxHandle:
        modal = self.modal
        app = modal.App.lookup(
            spec.app_name,
            environment_name=self.environment_name,
            create_if_missing=True,
        )
        image = modal.Image.from_id(spec.image_id)
        input_volume = modal.Volume.from_name(
            spec.input_volume_name,
            environment_name=self.environment_name,
            create_if_missing=False,
        )
        try:
            output_volume = modal.Volume.from_name(
                spec.output_volume_name,
                environment_name=self.environment_name,
                create_if_missing=True,
                version=2,
            )
        except Exception as exc:
            detail = str(exc)
            if "exists but has version v1, not v2" in detail:
                raise SandboxRunnerError(
                    "OUTPUT_VOLUME_VERSION_MISMATCH",
                    f"output Volume {spec.output_volume_name!r} is v1; competition "
                    "Sandboxes require v2 for explicit sync commits. Stop users of "
                    "the empty v1 Volume, delete it, and recreate it with "
                    "`modal volume create --version=2`",
                ) from exc
            raise
        sandbox = modal.Sandbox.create(
            "python3",
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8003",
            app=app,
            name=spec.sandbox_name,
            tags={
                "purpose": "compression-competition",
                "competition_id": spec.competition_id,
                "hotkey_hash": _short_hash(spec.hotkey, 16),
                "generation": str(spec.generation),
            },
            image=image,
            env={
                "DISABLE_REMOTE_IO": "true",
                "COMPETITION_INPUT_ROOT": "/evaluation-inputs",
                "COMPETITION_OUTPUT_ROOT": "/output",
                "MINER_CLEANUP_ENABLED": "false",
                "MINER_STORAGE_CLEANUP_ENABLED": "false",
            },
            secrets=[],
            timeout=spec.lifetime_seconds,
            workdir="/app",
            gpu=spec.gpu_type,
            cpu=(spec.cpu_request, spec.cpu_limit),
            block_network=True,
            volumes={
                "/evaluation-inputs": input_volume.with_mount_options(read_only=True),
                "/output": output_volume.with_mount_options(read_only=False),
            },
            include_oidc_identity_token=False,
            encrypted_ports=[],
            h2_ports=[],
            unencrypted_ports=[],
        )
        return SandboxHandle(
            app_id=str(getattr(app, "app_id", "")),
            sandbox_id=str(getattr(sandbox, "object_id", "")),
            raw=sandbox,
        )

    def attach(self, app_id: str, sandbox_id: str) -> SandboxHandle:
        sandbox = self.modal.Sandbox.from_id(sandbox_id)
        return SandboxHandle(app_id=app_id, sandbox_id=sandbox_id, raw=sandbox)

    def attach_by_name(self, app_name: str, sandbox_name: str) -> SandboxHandle:
        app = self.modal.App.lookup(
            app_name,
            environment_name=self.environment_name,
            create_if_missing=False,
        )
        sandbox = self.modal.Sandbox.from_name(
            app_name,
            sandbox_name,
            environment_name=self.environment_name,
        )
        return SandboxHandle(
            app_id=str(getattr(app, "app_id", "")),
            sandbox_id=str(getattr(sandbox, "object_id", "")),
            raw=sandbox,
        )

    def verify_isolation(self, handle: SandboxHandle) -> dict[str, Any]:
        process = handle.raw.exec("python3", "-c", ISOLATION_PROBE, timeout=30)
        output = _completed_process_output(process, "active isolation probe")
        try:
            report = json.loads(output)
        except json.JSONDecodeError as exc:
            raise SandboxRunnerError(
                "ISOLATION_PROBE_INVALID", "probe did not return JSON"
            ) from exc
        return report

    def inspect_resources(self, handle: SandboxHandle) -> SandboxResourceAllocation:
        process = handle.raw.exec(
            "python3", "-I", "-S", "-c", RESOURCE_PROBE, timeout=30
        )
        output = _completed_process_output(process, "allocated resource probe")
        try:
            report = json.loads(output)
            gpu_names = report["gpu_names"]
            cpu_cores = float(report["allocated_cpu_cores"])
            gpu_types = [canonical_modal_gpu_type(str(name)) for name in gpu_names]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SandboxRunnerError(
                "RESOURCE_PROBE_INVALID", "allocated resource probe returned invalid data"
            ) from exc
        if not gpu_types or cpu_cores <= 0 or len(set(gpu_types)) != 1:
            raise SandboxRunnerError(
                "RESOURCE_PROBE_INVALID",
                "allocated GPU/CPU resources are missing or heterogeneous",
            )
        return SandboxResourceAllocation(
            gpu_type=gpu_types[0],
            gpu_count=len(gpu_types),
            cpu_cores=cpu_cores,
            report=report,
        )

    def health(self, handle: SandboxHandle, timeout_seconds: int) -> dict[str, Any]:
        process = handle.raw.exec(
            "python3", "-c", READINESS_PROBE, timeout=timeout_seconds
        )
        output = _completed_process_output(process, "localhost readiness probe")
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise SandboxRunnerError(
                "HEALTH_RESPONSE_INVALID", "health probe did not return JSON"
            ) from exc

    def prepare_outputs(
        self, handle: SandboxHandle, output_paths: tuple[str, ...]
    ) -> None:
        """Create validator-selected output parents before contender invocation."""

        output_root = PurePosixPath("/output")
        parents: set[str] = set()
        for raw_path in output_paths:
            path = PurePosixPath(raw_path)
            if (
                not path.is_absolute()
                or ".." in path.parts
                or path == output_root
                or not path.is_relative_to(output_root)
            ):
                raise SandboxRunnerError(
                    "OUTPUT_PATH_INVALID",
                    f"output path must be a file below /output: {raw_path!r}",
                )
            parents.add(str(path.parent))
        if not parents:
            raise SandboxRunnerError(
                "OUTPUT_PATH_INVALID", "compression request has no output paths"
            )
        process = handle.raw.exec("mkdir", "-p", "--", *sorted(parents), timeout=30)
        _completed_process_output(process, "competition output directory preparation")

    def invoke(
        self, handle: SandboxHandle, payload_json: str, timeout_seconds: int
    ) -> str:
        process = handle.raw.exec(
            "python3",
            "-c",
            HTTP_COMPRESSION_CLIENT,
            str(timeout_seconds),
            timeout=timeout_seconds + SUPERVISOR_TIMEOUT_GRACE_SECONDS,
        )
        process.stdin.write(payload_json.encode("utf-8"))
        # Modal buffers both data and EOF. Queue EOF before draining so the
        # in-Sandbox client can finish sys.stdin.buffer.read() and issue the POST.
        process.stdin.write_eof()
        process.stdin.drain()
        return _completed_process_output(process, "localhost /compress request")

    def commit_outputs(self, handle: SandboxHandle) -> None:
        process = handle.raw.exec("sync", "/output", timeout=30)
        _completed_process_output(process, "output Volume commit")

    def terminate(self, handle: SandboxHandle) -> None:
        handle.raw.terminate(wait=True)


class CompetitionModalRunner:
    def __init__(
        self,
        repository: CompetitionRepository,
        backend: SandboxBackend,
        *,
        actor: str,
        accepted_build_statuses: frozenset[str] = frozenset({"ACCEPTED"}),
        clock=lambda: datetime.now(timezone.utc),
    ) -> None:
        self.repository = repository
        self.backend = backend
        self.actor = actor
        self.accepted_build_statuses = accepted_build_statuses
        self.clock = clock

    def ensure_warm(
        self,
        manifest: CompetitionManifest,
        hotkey: str,
        *,
        now: datetime | None = None,
    ) -> SandboxSession:
        now = now or self.clock()
        contender = self.repository.get_contender(manifest.competition_id, hotkey)
        if contender is None:
            raise SandboxRunnerError("CONTENDER_NOT_FOUND", hotkey)
        if (
            contender.build_status not in self.accepted_build_statuses
            or not contender.image_id
            or not contender.image_digest
        ):
            raise SandboxRunnerError(
                "PINNED_IMAGE_REQUIRED",
                "contender has no build evidence accepted by this validator",
            )

        latest = self.repository.latest_sandbox(manifest.competition_id, hotkey)
        if latest is not None and latest.status in {"STARTING", "RUNNING"}:
            recovered = self._try_recover(manifest, latest, now)
            if recovered is not None:
                return recovered

        return self._start_generation(manifest, contender, hotkey, now)

    def invoke_batch(
        self,
        manifest: CompetitionManifest,
        request: CompetitionCompressionRequest,
        *,
        timeout_seconds: int | None = None,
        now: datetime | None = None,
    ) -> CompetitionCompressionResponse:
        if request.competition_id != manifest.competition_id:
            raise SandboxRunnerError(
                "COMPETITION_ID_MISMATCH", "request does not match manifest"
            )
        now = now or self.clock()
        timeout = timeout_seconds or int(
            manifest.evaluation_batched_run_timeout.total_seconds()
        )
        session = self.ensure_warm(manifest, request.hotkey, now=now)
        self.repository.update_sandbox_batch_timeout(
            session.record.id,
            batch_timeout_seconds=timeout,
            now=now,
        )
        try:
            self.backend.prepare_outputs(
                session.handle, tuple(item.output_path for item in request.items)
            )
            rendered = self.backend.invoke(
                session.handle, request.model_dump_json(), timeout
            )
            response = CompetitionCompressionResponse.model_validate_json(rendered)
            if len(response.results) != len(request.items):
                raise SandboxRunnerError(
                    "COMPRESSION_RESPONSE_ITEMS_MISMATCH",
                    "sandbox response count does not exactly match the request",
                )
            self.backend.commit_outputs(session.handle)
            self.repository.touch_sandbox(session.record.id, now=now)
            return response
        except Exception as exc:
            self._terminate(
                session.record,
                session.handle,
                status="FAILED",
                reason_code="BATCH_EXECUTION_FAILED",
                now=now,
                detail=str(exc),
            )
            if isinstance(exc, SandboxRunnerError):
                raise
            raise SandboxRunnerError("BATCH_EXECUTION_FAILED", str(exc)) from exc

    def terminate(
        self,
        manifest: CompetitionManifest,
        hotkey: str,
        *,
        now: datetime | None = None,
    ) -> None:
        row = self.repository.latest_sandbox(manifest.competition_id, hotkey)
        if row is None or row.status not in {"STARTING", "RUNNING"}:
            return
        handle = None
        if row.modal_sandbox_id:
            handle = self.backend.attach(row.modal_app_id or "", row.modal_sandbox_id)
        self._terminate(
            row,
            handle,
            status="TERMINATED",
            reason_code="SUPERVISOR_SHUTDOWN",
            now=now or self.clock(),
        )

    def _try_recover(
        self,
        manifest: CompetitionManifest,
        row: CompetitionSandbox,
        now: datetime,
    ) -> SandboxSession | None:
        contender = self.repository.get_contender(row.competition_id, row.hotkey)
        if (
            contender is None
            or row.image_id != contender.image_id
            or row.image_digest != contender.image_digest
        ):
            self._terminate(
                row,
                None,
                status="FAILED",
                reason_code="PINNED_IMAGE_MISMATCH",
                now=now,
            )
            return None
        if parse_utc(row.expires_at) - now <= ROLLOVER_MARGIN:
            handle = None
            if row.modal_sandbox_id:
                try:
                    handle = self.backend.attach(
                        row.modal_app_id or "", row.modal_sandbox_id
                    )
                except Exception:
                    handle = None
            self._terminate(
                row,
                handle,
                status="EXPIRED",
                reason_code="SANDBOX_ROLLOVER",
                now=now,
            )
            return None
        handle = None
        if not row.modal_sandbox_id and row.modal_app_name and row.modal_sandbox_name:
            try:
                handle = self.backend.attach_by_name(
                    row.modal_app_name, row.modal_sandbox_name
                )
                self.repository.bind_sandbox_identity(
                    row.id,
                    modal_app_id=handle.app_id,
                    modal_sandbox_id=handle.sandbox_id,
                    now=now,
                )
            except Exception:
                handle = None
        if not row.modal_sandbox_id and handle is None:
            self._terminate(
                row,
                None,
                status="ORPHANED",
                reason_code="SANDBOX_ID_NOT_PERSISTED",
                now=now,
            )
            return None
        try:
            if handle is None:
                handle = self.backend.attach(
                    row.modal_app_id or "", row.modal_sandbox_id
                )
            if row.status == "STARTING":
                isolation = self.backend.verify_isolation(handle)
                if not isolation.get("passed"):
                    raise SandboxRunnerError(
                        "ISOLATION_PROBE_FAILED", json.dumps(isolation, sort_keys=True)
                    )
            else:
                isolation = json.loads(row.isolation_report_json or "{}")
            needs_activation = row.status == "STARTING" or not (
                row.allocated_gpu_type
                and row.allocated_gpu_count
                and row.allocated_cpu_cores
            )
            allocation = (
                self.backend.inspect_resources(handle) if needs_activation else None
            )
            self.backend.health(handle, timeout_seconds=60)
            if needs_activation:
                assert allocation is not None
                isolation = {
                    **isolation,
                    "resource_allocation": allocation.report,
                }
                row = self.repository.activate_sandbox(
                    row.id,
                    modal_app_id=handle.app_id,
                    modal_sandbox_id=handle.sandbox_id,
                    isolation_report=isolation,
                    allocated_gpu_type=allocation.gpu_type,
                    allocated_gpu_count=allocation.gpu_count,
                    allocated_cpu_cores=allocation.cpu_cores,
                    now=now,
                    actor=self.actor,
                )
            else:
                self.repository.touch_sandbox(row.id, now=now)
            return SandboxSession(row, handle)
        except Exception as exc:
            self._terminate(
                row,
                locals().get("handle"),
                status="FAILED",
                reason_code="SANDBOX_RECOVERY_FAILED",
                now=now,
                detail=str(exc),
            )
            return None

    def _start_generation(
        self,
        manifest: CompetitionManifest,
        contender: Any,
        hotkey: str,
        now: datetime,
    ) -> SandboxSession:
        output_volume = _output_volume_name(
            manifest.output_volume_prefix, manifest.competition_id, hotkey
        )
        gpu = manifest.allowed_gpus[0]
        cpu = float(manifest.requested_cpu_cores)
        expires = now + MAX_SANDBOX_LIFETIME
        row = self.repository.reserve_sandbox_generation(
            competition_id=manifest.competition_id,
            hotkey=hotkey,
            image_id=contender.image_id,
            image_digest=contender.image_digest,
            output_volume_name=output_volume,
            gpu_type=gpu,
            requested_cpu_cores=manifest.requested_cpu_cores,
            max_cpu_cores=manifest.requested_cpu_cores,
            batch_timeout_seconds=manifest.evaluation_batched_run_timeout.total_seconds(),
            created_at=now,
            expires_at=expires,
            actor=self.actor,
        )
        token = (
            f"{_slug(manifest.competition_id, 20)}-"
            f"{_short_hash(manifest.competition_id, 8)}-"
            f"{_short_hash(hotkey, 12)}-g{row.generation}"
        )
        spec = SandboxLaunchSpec(
            competition_id=manifest.competition_id,
            hotkey=hotkey,
            generation=row.generation,
            app_name=f"vidaio-cmp-{token}"[:63].rstrip("-"),
            sandbox_name=f"cmp-{token}"[:63].rstrip("-"),
            image_id=contender.image_id,
            input_volume_name=manifest.evaluation_input_volume_name,
            output_volume_name=output_volume,
            gpu_type=gpu,
            cpu_request=cpu,
            cpu_limit=cpu,
            lifetime_seconds=int(MAX_SANDBOX_LIFETIME.total_seconds()),
        )
        self.repository.bind_sandbox_names(
            row.id,
            modal_app_name=spec.app_name,
            modal_sandbox_name=spec.sandbox_name,
            now=now,
        )
        handle = None
        try:
            legacy_token = (
                f"{_slug(manifest.competition_id, 20)}-"
                f"{_short_hash(hotkey, 12)}-g{row.generation}"
            )
            self._terminate_stale_named_sandbox(
                f"vidaio-cmp-{legacy_token}"[:63].rstrip("-"),
                f"cmp-{legacy_token}"[:63].rstrip("-"),
            )
            self._terminate_stale_named_sandbox(spec.app_name, spec.sandbox_name)
            handle = self.backend.create(spec)
            if not handle.app_id or not handle.sandbox_id:
                raise SandboxRunnerError(
                    "SANDBOX_ID_MISSING", "Modal did not return persistent IDs"
                )
            self.repository.bind_sandbox_identity(
                row.id,
                modal_app_id=handle.app_id,
                modal_sandbox_id=handle.sandbox_id,
                now=now,
            )
            isolation = self.backend.verify_isolation(handle)
            if not isolation.get("passed"):
                raise SandboxRunnerError(
                    "ISOLATION_PROBE_FAILED", json.dumps(isolation, sort_keys=True)
                )
            allocation = self.backend.inspect_resources(handle)
            isolation = {
                **isolation,
                "resource_allocation": allocation.report,
            }
            self.backend.health(handle, timeout_seconds=330)
            row = self.repository.activate_sandbox(
                row.id,
                modal_app_id=handle.app_id,
                modal_sandbox_id=handle.sandbox_id,
                isolation_report=isolation,
                allocated_gpu_type=allocation.gpu_type,
                allocated_gpu_count=allocation.gpu_count,
                allocated_cpu_cores=allocation.cpu_cores,
                now=now,
                actor=self.actor,
            )
            return SandboxSession(row, handle)
        except Exception as exc:
            self._terminate(
                row,
                handle,
                status="FAILED",
                reason_code=(
                    exc.reason_code
                    if isinstance(exc, SandboxRunnerError)
                    else "SANDBOX_START_FAILED"
                ),
                now=now,
                detail=str(exc),
            )
            if isinstance(exc, SandboxRunnerError):
                raise
            raise SandboxRunnerError("SANDBOX_START_FAILED", str(exc)) from exc

    def _terminate_stale_named_sandbox(
        self, app_name: str, sandbox_name: str
    ) -> None:
        """Remove an untracked Modal Sandbox that blocks deterministic creation."""

        try:
            stale = self.backend.attach_by_name(app_name, sandbox_name)
        except Exception:
            return
        try:
            self.backend.terminate(stale)
        except Exception as exc:
            raise SandboxRunnerError(
                "STALE_SANDBOX_TERMINATION_FAILED", str(exc)
            ) from exc

    def _terminate(
        self,
        row: CompetitionSandbox,
        handle: SandboxHandle | None,
        *,
        status: str,
        reason_code: str,
        now: datetime,
        detail: str | None = None,
    ) -> None:
        if handle is not None:
            try:
                self.backend.terminate(handle)
            except Exception as exc:
                reason_code = f"{reason_code}_TERMINATE_ERROR"
                detail = f"{detail or ''}; termination failed: {exc}".lstrip("; ")
        self.repository.close_sandbox(
            row.id,
            status=status,
            reason_code=reason_code,
            detail=detail,
            now=now,
            actor=self.actor,
        )


def _short_hash(value: str, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _slug(value: str, limit: int) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")[:limit]


def _readable_hotkey_label(hotkey: str) -> str:
    """Retain recognizable hotkey characters while bounding Modal name length."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", hotkey).strip("-")
    if len(cleaned) <= 28:
        return cleaned
    return f"{cleaned[:12]}-{cleaned[-8:]}-{_short_hash(hotkey, 6)}"


def _readable_competition_label(competition_id: str) -> str:
    cleaned = _slug(competition_id, 63)
    if len(cleaned) <= 24:
        return cleaned
    return f"{cleaned[:15].rstrip('-')}-{_short_hash(competition_id, 8)}"


def _bounded_prefix(value: str, limit: int) -> str:
    cleaned = _slug(value, 63) or "competition-output"
    if len(cleaned) <= limit:
        return cleaned
    bounded = cleaned[:limit].rstrip("-")
    # Avoid opaque partial words such as "vidaio-compet" where a complete
    # namespace segment fits in the same bound.
    if "-" in bounded:
        bounded = bounded.rsplit("-", 1)[0]
    return bounded or cleaned[:limit].rstrip("-")


def _output_volume_name(prefix: str, competition_id: str, hotkey: str) -> str:
    competition = _readable_competition_label(competition_id)
    miner = _readable_hotkey_label(hotkey)
    identity = f"{competition}-{miner}"
    namespace = _bounded_prefix(prefix, 63 - len(identity) - 1)
    return f"{namespace}-{identity}"


__all__ = [
    "CompetitionModalRunner",
    "ISOLATION_PROBE",
    "MAX_SANDBOX_LIFETIME",
    "ModalSandboxBackend",
    "RESOURCE_PROBE",
    "SandboxBackend",
    "SandboxHandle",
    "SandboxLaunchSpec",
    "SandboxResourceAllocation",
    "SandboxRunnerError",
    "SandboxSession",
]
