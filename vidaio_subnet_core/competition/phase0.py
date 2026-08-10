"""Executable Phase 0 feasibility and security gates for competition mode.

Nothing in this module launches a production workload.  It provides the policy
math, local enforcement probes, redaction boundary, and billing allocation
contract needed before the validator/Modal integration is implemented.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


GB = 1000**3
IMAGE_SIZE_LIMIT_BYTES = 25 * GB
DEFAULT_COST_FLOOR_USD = Decimal("0.000001")
REDACTED = "[REDACTED]"


class GateStatus(str, Enum):
    PASSED = "passed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    summary: str
    details: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "details": dict(self.details),
        }


class ImageSizeLimitExceeded(ValueError):
    def __init__(self, actual_bytes: int, limit_bytes: int = IMAGE_SIZE_LIMIT_BYTES):
        self.actual_bytes = actual_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"image size {actual_bytes} bytes exceeds limit {limit_bytes} bytes"
        )


def enforce_image_size(
    actual_bytes: int,
    limit_bytes: int = IMAGE_SIZE_LIMIT_BYTES,
) -> int:
    """Reject an image strictly above the configured measured size limit."""

    if isinstance(actual_bytes, bool) or not isinstance(actual_bytes, int):
        raise TypeError("actual_bytes must be an integer")
    if isinstance(limit_bytes, bool) or not isinstance(limit_bytes, int):
        raise TypeError("limit_bytes must be an integer")
    if actual_bytes < 0 or limit_bytes <= 0:
        raise ValueError("image sizes must be non-negative and limit must be positive")
    if actual_bytes > limit_bytes:
        raise ImageSizeLimitExceeded(actual_bytes, limit_bytes)
    return actual_bytes


def inspect_docker_image_size(image_ref: str) -> int:
    """Return Docker's measured image size without executing the image."""

    if not image_ref or any(ch.isspace() for ch in image_ref):
        raise ValueError("image_ref must be a non-empty Docker reference")
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Size}}", image_ref],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"docker image inspection failed: {detail[:500]}")
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("docker returned a non-integer image size") from exc


def _file_quota_probe(limit_bytes: int = 1024 * 1024) -> GateResult:
    """Prove the host can terminate a writer at a configurable file limit.

    This verifies a useful isolated-builder primitive, but deliberately does not
    claim that Docker's external daemon or Modal's builder inherits this limit.
    """

    if os.name != "posix":
        return GateResult(
            "builder_file_quota_primitive",
            GateStatus.BLOCKED,
            "RLIMIT_FSIZE is unavailable on this platform",
            {"platform": os.name},
        )

    child = """
import errno
import os
import resource
import signal
import sys

path = sys.argv[1]
limit = int(sys.argv[2])
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
try:
    with open(path, "wb", buffering=0) as handle:
        block = b"x" * 65536
        for _ in range((limit // len(block)) + 4):
            handle.write(block)
except OSError as exc:
    if exc.errno == errno.EFBIG:
        raise SystemExit(23)
    raise
raise SystemExit(0)
"""

    with tempfile.TemporaryDirectory(prefix="vidaio-phase0-quota-") as temp_dir:
        output_path = Path(temp_dir) / "hostile-layer.bin"
        result = subprocess.run(
            [sys.executable, "-c", child, str(output_path), str(limit_bytes)],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        actual_size = output_path.stat().st_size if output_path.exists() else 0

    passed = result.returncode == 23 and actual_size <= limit_bytes
    return GateResult(
        "builder_file_quota_primitive",
        GateStatus.PASSED if passed else GateStatus.FAILED,
        (
            "host writer was stopped at the configured file-size limit"
            if passed
            else "host writer was not reliably stopped at the configured limit"
        ),
        {
            "test_limit_bytes": limit_bytes,
            "written_bytes": actual_size,
            "child_returncode": result.returncode,
            "stderr": result.stderr.strip()[:500],
        },
    )


def _docker_capability() -> dict[str, Any]:
    executable = shutil.which("docker")
    if not executable:
        return {"cli_available": False, "daemon_available": False}
    try:
        result = subprocess.run(
            [executable, "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "cli_available": True,
            "daemon_available": False,
            "error": type(exc).__name__,
        }
    return {
        "cli_available": True,
        "daemon_available": result.returncode == 0,
        "server_version": result.stdout.strip() if result.returncode == 0 else None,
        "error": (result.stderr or result.stdout).strip()[:500]
        if result.returncode != 0
        else None,
    }


def image_limit_gate() -> GateResult:
    enforce_image_size(IMAGE_SIZE_LIMIT_BYTES)
    oversized_rejected = False
    try:
        enforce_image_size(IMAGE_SIZE_LIMIT_BYTES + 1)
    except ImageSizeLimitExceeded:
        oversized_rejected = True

    quota = _file_quota_probe()
    docker = _docker_capability()
    local_primitives_pass = oversized_rejected and quota.status == GateStatus.PASSED
    return GateResult(
        "hostile_build_size_enforcement",
        GateStatus.PARTIAL if local_primitives_pass else GateStatus.FAILED,
        (
            "25 GB policy math and a host file-quota primitive are proven; "
            "an end-to-end OCI/Modal builder quota is not yet proven"
            if local_primitives_pass
            else "one or more local image-limit primitives failed"
        ),
        {
            "limit_bytes": IMAGE_SIZE_LIMIT_BYTES,
            "limit_gb": 25,
            "exact_limit_accepted": True,
            "one_byte_over_rejected": oversized_rejected,
            "file_quota_probe": quota.as_dict(),
            "docker": docker,
            "end_to_end_builder_proven": False,
            "reason": (
                "Docker/Modal builders are external services; post-build inspection "
                "alone cannot prevent hostile build-time spend"
            ),
        },
    )


_TOKEN_VALUE_KEYS = {
    "authorization",
    "github_pat",
    "pat",
    "token",
    "token_id",
    "token_secret",
    "secret",
    "password",
}
_TOKEN_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
)


class SecretRedactor:
    """Redact explicit secrets and recognizable GitHub token forms."""

    def __init__(self, secrets: Iterable[str] = ()):
        self._secrets = tuple(
            sorted(
                {secret for secret in secrets if isinstance(secret, str) and secret},
                key=len,
                reverse=True,
            )
        )

    def redact_text(self, text: str) -> str:
        redacted = str(text)
        for secret in self._secrets:
            redacted = redacted.replace(secret, REDACTED)
        for pattern in _TOKEN_PATTERNS:
            redacted = pattern.sub(REDACTED, redacted)
        return redacted

    def redact(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            output: dict[Any, Any] = {}
            for key, item in value.items():
                key_name = str(key).lower()
                output[key] = (
                    REDACTED
                    if key_name in _TOKEN_VALUE_KEYS
                    or key_name.endswith("_token")
                    or key_name.endswith("_secret")
                    else self.redact(item)
                )
            return output
        if isinstance(value, tuple):
            return tuple(self.redact(item) for item in value)
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, set):
            return {self.redact(item) for item in value}
        if isinstance(value, BaseException):
            return self.redact_text(repr(value))
        if isinstance(value, str):
            return self.redact_text(value)
        return value

    def contains_secret(self, value: Any) -> bool:
        rendered = json.dumps(value, sort_keys=True, default=str)
        return any(secret in rendered for secret in self._secrets) or any(
            pattern.search(rendered) for pattern in _TOKEN_PATTERNS
        )


@dataclass(frozen=True)
class RawPatSubmission:
    competition_id: str
    hotkey: str
    repository_url: str
    github_pat: str
    commit_hint: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Raw transport payload. This is the sole intentional PAT boundary."""

        return asdict(self)

    def to_persistent_record(self) -> dict[str, Any]:
        """Return the only representation allowed in SQLite/Redis/W&B."""

        return {
            "competition_id": self.competition_id,
            "hotkey": self.hotkey,
            "repository_url": self.repository_url,
            "commit_hint": self.commit_hint,
            "credential_present": bool(self.github_pat),
        }


def pat_redaction_gate() -> GateResult:
    fake_pat = "github_pat_" + ("phase0A1" * 8)
    submission = RawPatSubmission(
        competition_id="compression-phase0",
        hotkey="5Phase0Hotkey",
        repository_url="https://github.com/example/private-compressor.git",
        github_pat=fake_pat,
    )
    redactor = SecretRedactor([fake_pat])
    wire = submission.to_wire()
    persistent = submission.to_persistent_record()

    simulated_surfaces = {
        "application": {"message": f"submission={wire}"},
        "bittensor_debug": {"synapse": wire},
        "pm2": f"validator exception while handling {wire}",
        "git": {
            "argv": ["git", "clone", submission.repository_url],
            "environment": {"GIT_ASKPASS_PAT": fake_pat},
        },
        "redis": persistent,
        "wandb": {"submission": persistent},
        "exception": RuntimeError(f"clone failed using {fake_pat}"),
    }
    sanitized = redactor.redact(simulated_surfaces)

    database_leaked = False
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE submission (competition_id TEXT, payload TEXT)"
        )
        connection.execute(
            "INSERT INTO submission VALUES (?, ?)",
            (submission.competition_id, json.dumps(persistent, sort_keys=True)),
        )
        database_dump = "\n".join(connection.iterdump())
        database_leaked = fake_pat in database_dump

    passed = (
        fake_pat in json.dumps(wire)
        and fake_pat not in json.dumps(persistent)
        and not redactor.contains_secret(sanitized)
        and not database_leaked
    )
    return GateResult(
        "raw_pat_redaction_boundary",
        GateStatus.PARTIAL if passed else GateStatus.FAILED,
        (
            "raw PAT is confined to the wire DTO and redacted from representative sinks; "
            "live Bittensor/PM2/Redis/W&B integration remains a Phase 1 repeat gate"
            if passed
            else "raw PAT leaked across a local persistence or logging boundary"
        ),
        {
            "wire_contains_raw_pat": fake_pat in json.dumps(wire),
            "persistent_record_contains_raw_pat": fake_pat in json.dumps(persistent),
            "sanitized_surfaces_contain_secret": redactor.contains_secret(sanitized),
            "sqlite_contains_raw_pat": database_leaked,
            "covered_surfaces": sorted(simulated_surfaces),
            "actual_protocol_integration_proven": False,
        },
    )


@dataclass(frozen=True)
class CompetitionItemResult:
    contender_id: str
    evaluation_id: str
    duration_seconds: float
    quality: float
    cost_usd: Decimal
    completed: bool

    def __post_init__(self) -> None:
        if not self.contender_id or not self.evaluation_id:
            raise ValueError("contender_id and evaluation_id are required")
        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, (int, float)
        ):
            raise TypeError("duration_seconds must be numeric")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive and finite")
        if isinstance(self.quality, bool) or not isinstance(self.quality, (int, float)):
            raise TypeError("quality must be numeric")
        if not math.isfinite(self.quality) or not 0 <= self.quality <= 1:
            raise ValueError("quality must be in [0, 1]")
        if not isinstance(self.cost_usd, Decimal):
            raise TypeError("cost_usd must be a Decimal")
        if not self.cost_usd.is_finite():
            raise ValueError("cost_usd must be finite")
        if self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a boolean")


@dataclass(frozen=True)
class CompetitionScore:
    contender_id: str
    quality_aggregate: float
    cost_aggregate: float
    length_coverage: float
    final_score: float
    item_components: Mapping[str, Mapping[str, float]]


def _duration_weight(
    duration_seconds: float,
    max_video_length_seconds: float,
    length_weight_exponent: float = 1.0,
) -> float:
    if (
        isinstance(max_video_length_seconds, bool)
        or not isinstance(max_video_length_seconds, (int, float))
        or not math.isfinite(max_video_length_seconds)
        or max_video_length_seconds <= 0
    ):
        raise ValueError("max_video_length_seconds must be positive and finite")
    if (
        isinstance(length_weight_exponent, bool)
        or not isinstance(length_weight_exponent, (int, float))
        or not math.isfinite(length_weight_exponent)
        or length_weight_exponent <= 0
        or length_weight_exponent > 10
    ):
        raise ValueError("length_weight_exponent must be finite and in (0, 10]")
    if duration_seconds > max_video_length_seconds:
        raise ValueError("item duration exceeds the manifest maximum")
    base_weight = math.log1p(duration_seconds) / math.log1p(
        max_video_length_seconds
    )
    return base_weight**length_weight_exponent


def calculate_competition_scores(
    results: Sequence[CompetitionItemResult],
    *,
    max_video_length_seconds: float,
    length_weight_exponent: float = 1.0,
    cost_floor_usd: Decimal = DEFAULT_COST_FLOOR_USD,
) -> dict[str, CompetitionScore]:
    """Calculate 60/25/15 media, relative-cost, and coverage scores."""

    if not results:
        raise ValueError("at least one result is required")
    if not isinstance(cost_floor_usd, Decimal) or not cost_floor_usd.is_finite():
        raise TypeError("cost_floor_usd must be a finite Decimal")
    if cost_floor_usd <= 0:
        raise ValueError("cost_floor_usd must be positive")
    by_contender: dict[str, dict[str, CompetitionItemResult]] = {}
    durations: dict[str, float] = {}
    for result in results:
        contender_items = by_contender.setdefault(result.contender_id, {})
        if result.evaluation_id in contender_items:
            raise ValueError("duplicate contender/evaluation result")
        contender_items[result.evaluation_id] = result
        prior_duration = durations.setdefault(
            result.evaluation_id, result.duration_seconds
        )
        if not math.isclose(prior_duration, result.duration_seconds):
            raise ValueError("evaluation duration differs between contenders")

    evaluation_ids = set(durations)
    for contender_id, contender_items in by_contender.items():
        if set(contender_items) != evaluation_ids:
            raise ValueError(
                f"contender {contender_id} does not cover the complete evaluation index"
            )

    weights = {
        evaluation_id: _duration_weight(
            duration,
            max_video_length_seconds,
            length_weight_exponent,
        )
        for evaluation_id, duration in durations.items()
    }
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("duration weights must have a positive sum")

    minimum_costs: dict[str, Decimal] = {}
    for evaluation_id in evaluation_ids:
        valid_costs = [
            max(contender_items[evaluation_id].cost_usd, cost_floor_usd)
            for contender_items in by_contender.values()
            if contender_items[evaluation_id].completed
        ]
        if valid_costs:
            minimum_costs[evaluation_id] = min(valid_costs)

    scores: dict[str, CompetitionScore] = {}
    for contender_id, contender_items in by_contender.items():
        quality_total = 0.0
        cost_total = 0.0
        completion_total = 0.0
        components: dict[str, Mapping[str, float]] = {}
        for evaluation_id in sorted(evaluation_ids):
            item = contender_items[evaluation_id]
            weight = weights[evaluation_id]
            completed = 1.0 if item.completed else 0.0
            quality = item.quality if item.completed else 0.0
            if not item.completed:
                cost_efficiency = 0.0
            else:
                contender_cost = max(item.cost_usd, cost_floor_usd)
                cost_efficiency = min(
                    1.0, float(minimum_costs[evaluation_id] / contender_cost)
                )

            quality_total += weight * quality
            cost_total += weight * cost_efficiency
            completion_total += weight * completed
            components[evaluation_id] = {
                "length_weight": weight,
                "quality": quality,
                "cost_efficiency": cost_efficiency,
                "completed": completed,
            }

        quality_aggregate = quality_total / denominator
        cost_aggregate = cost_total / denominator
        length_coverage = completion_total / denominator
        final_score = (
            0.6 * quality_aggregate
            + 0.25 * cost_aggregate
            + 0.15 * length_coverage
        )
        scores[contender_id] = CompetitionScore(
            contender_id=contender_id,
            quality_aggregate=quality_aggregate,
            cost_aggregate=cost_aggregate,
            length_coverage=length_coverage,
            final_score=final_score,
            item_components=components,
        )
    return scores


def score_formula_gate() -> GateResult:
    results = [
        CompetitionItemResult(
            "long-specialist", "short", 5, 0.2, Decimal("0.10"), True
        ),
        CompetitionItemResult(
            "long-specialist", "long", 3600, 1.0, Decimal("0.10"), True
        ),
        CompetitionItemResult(
            "short-specialist", "short", 5, 1.0, Decimal("0.10"), True
        ),
        CompetitionItemResult(
            "short-specialist", "long", 3600, 0.2, Decimal("0.10"), True
        ),
        CompetitionItemResult(
            "long-failure", "short", 5, 1.0, Decimal("0.05"), True
        ),
        CompetitionItemResult(
            "long-failure", "long", 3600, 1.0, Decimal("0.05"), False
        ),
    ]
    scores = calculate_competition_scores(
        results, max_video_length_seconds=3600
    )
    long_dominates = (
        scores["long-specialist"].final_score
        > scores["short-specialist"].final_score
    )
    long_failure_penalized = (
        scores["long-failure"].length_coverage
        < scores["long-specialist"].length_coverage
    )
    bounded = all(0 <= score.final_score <= 1 for score in scores.values())
    passed = long_dominates and long_failure_penalized and bounded
    return GateResult(
        "competition_score_formula",
        GateStatus.PASSED if passed else GateStatus.FAILED,
        (
            "60/25/15 formula is bounded and longer entries dominate as approved"
            if passed
            else "score formula did not satisfy its approved invariants"
        ),
        {
            "long_specialist_score": scores["long-specialist"].final_score,
            "short_specialist_score": scores["short-specialist"].final_score,
            "long_failure_coverage": scores["long-failure"].length_coverage,
            "long_dominates": long_dominates,
            "long_failure_penalized": long_failure_penalized,
            "all_scores_bounded": bounded,
        },
    )


def allocate_batch_cost(
    total_cost_usd: Decimal,
    runtimes_seconds: Mapping[str, float],
) -> dict[str, Decimal]:
    """Allocate a batch bill to items while preserving the exact total."""

    if not isinstance(total_cost_usd, Decimal):
        raise TypeError("total_cost_usd must be a Decimal")
    if not total_cost_usd.is_finite():
        raise ValueError("total_cost_usd must be finite")
    if total_cost_usd < 0:
        raise ValueError("total_cost_usd cannot be negative")
    if not runtimes_seconds:
        raise ValueError("at least one runtime is required")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in runtimes_seconds.values()
    ):
        raise ValueError("runtimes must be finite and non-negative")

    item_ids = sorted(runtimes_seconds)
    total_runtime = sum(runtimes_seconds.values())
    allocations: dict[str, Decimal] = {}
    allocated = Decimal("0")
    for item_id in item_ids[:-1]:
        share = (
            Decimal(str(runtimes_seconds[item_id] / total_runtime))
            if total_runtime > 0
            else Decimal(1) / Decimal(len(item_ids))
        )
        item_cost = total_cost_usd * share
        allocations[item_id] = item_cost
        allocated += item_cost
    allocations[item_ids[-1]] = total_cost_usd - allocated
    return allocations


def billing_capability_gate() -> GateResult:
    allocations = allocate_batch_cost(
        Decimal("1.00"), {"short": 1.0, "long": 3.0}
    )
    allocation_preserves_total = sum(allocations.values()) == Decimal("1.00")
    return GateResult(
        "modal_billing_granularity",
        GateStatus.PARTIAL if allocation_preserves_total else GateStatus.FAILED,
        (
            "hourly object/resource billing and deterministic item allocation are documented; "
            "workspace access and per-input reconciliation are not yet proven"
            if allocation_preserves_total
            else "batch cost allocation failed to preserve the billed total"
        ),
        {
            "minimum_report_resolution": "hourly",
            "native_attribution": "Modal object/app plus resource type and tags",
            "native_per_input_cost": False,
            "availability": "Team and Enterprise workspaces",
            "collection_delay_expected": True,
            "allocation_method": "active-runtime share; equal share when all runtimes are zero",
            "allocation_preserves_total": allocation_preserves_total,
            "live_workspace_access_proven": False,
        },
    )


def modal_runtime_gate() -> GateResult:
    credential_present = bool(
        os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")
    ) or (Path.home() / ".modal.toml").is_file()
    sdk_present = shutil.which("modal") is not None
    return GateResult(
        "modal_offline_runtime",
        GateStatus.BLOCKED,
        "live Modal network isolation has not been executed",
        {
            "credentials_present": credential_present,
            "modal_cli_present": sdk_present,
            "required_probe": "scripts/competition_modal_phase0_probe.py",
            "requires_live_dev_workspace": True,
        },
    )


def run_local_phase0_gates() -> list[GateResult]:
    return [
        image_limit_gate(),
        pat_redaction_gate(),
        billing_capability_gate(),
        score_formula_gate(),
        modal_runtime_gate(),
    ]


def phase0_report(results: Sequence[GateResult]) -> dict[str, Any]:
    statuses = [result.status for result in results]
    if any(status == GateStatus.FAILED for status in statuses):
        overall = GateStatus.FAILED
    elif any(status == GateStatus.BLOCKED for status in statuses):
        overall = GateStatus.BLOCKED
    elif any(status == GateStatus.PARTIAL for status in statuses):
        overall = GateStatus.PARTIAL
    else:
        overall = GateStatus.PASSED
    return {
        "phase": 0,
        "overall_status": overall.value,
        "production_ready": overall == GateStatus.PASSED,
        "gates": [result.as_dict() for result in results],
    }
