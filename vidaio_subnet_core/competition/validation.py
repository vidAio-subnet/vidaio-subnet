"""Fail-closed static validation for untrusted competition repositories."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import zipfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class ValidationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class ValidationReason(str, Enum):
    ACCEPTED = "ACCEPTED"
    REQUIRED_FILE_MISSING = "REQUIRED_FILE_MISSING"
    INVALID_SOLUTION_DESCRIPTOR = "INVALID_SOLUTION_DESCRIPTOR"
    REQUIRED_ROUTE_MISSING = "REQUIRED_ROUTE_MISSING"
    PATH_ESCAPE = "PATH_ESCAPE"
    BROKEN_SYMLINK = "BROKEN_SYMLINK"
    SUBMODULE_NOT_ALLOWED = "SUBMODULE_NOT_ALLOWED"
    NESTED_REPOSITORY = "NESTED_REPOSITORY"
    GIT_LFS_POINTER = "GIT_LFS_POINTER"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    REPOSITORY_TOO_LARGE = "REPOSITORY_TOO_LARGE"
    INVALID_PYTHON = "INVALID_PYTHON"
    COMMITTED_SECRET = "COMMITTED_SECRET"
    DYNAMIC_EXECUTION = "DYNAMIC_EXECUTION"
    ENCRYPTED_ARCHIVE = "ENCRYPTED_ARCHIVE"
    COMPILED_ONLY_CODE = "COMPILED_ONLY_CODE"
    REMOTE_DOWNLOAD_REVIEW = "REMOTE_DOWNLOAD_REVIEW"
    OBFUSCATION_REVIEW = "OBFUSCATION_REVIEW"
    SDK_TOOL_MODIFIED = "SDK_TOOL_MODIFIED"


@dataclass(frozen=True)
class ValidationFinding:
    reason_code: ValidationReason
    path: str
    detail: str
    rejects: bool


@dataclass(frozen=True)
class ValidationReport:
    status: ValidationStatus
    reason_code: ValidationReason
    repository_tree_sha256: str
    file_count: int
    total_bytes: int
    findings: tuple[ValidationFinding, ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["status"] = self.status.value
        value["reason_code"] = self.reason_code.value
        value["findings"] = [
            {**asdict(item), "reason_code": item.reason_code.value}
            for item in self.findings
        ]
        return value


_SECRET_PATTERNS = (
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_DOWNLOAD_PATTERN = re.compile(
    rb"(?:requests|httpx)\.(?:get|post)|urllib\.request|(?:^|[;&|])\s*(?:curl|wget)\s+https?://",
    re.IGNORECASE | re.MULTILINE,
)
_DYNAMIC_PATTERN = re.compile(rb"\b(?:eval|exec)\s*\(")
_KNOWN_OBFUSCATOR_PATTERN = re.compile(
    rb"(?:__pyarmor__|pyarmor_runtime|pytransform|cython_runtime|nuitka)",
    re.IGNORECASE,
)
_SOURCE_CODE_EXTENSIONS = frozenset(
    {
        ".bash",
        ".cjs",
        ".js",
        ".jsx",
        ".mjs",
        ".ps1",
        ".py",
        ".pyw",
        ".sh",
        ".ts",
        ".tsx",
        ".zsh",
    }
)
_OPAQUE_CODE_EXTENSIONS = frozenset(
    {
        ".class",
        ".dll",
        ".dylib",
        ".enc",
        ".exe",
        ".jar",
        ".jsc",
        ".obf",
        ".packed",
        ".pyd",
        ".pye",
        ".pyc",
        ".pyo",
        ".pyz",
        ".so",
        ".wasm",
    }
)
_ENCODED_LOADER_NAMES = frozenset(
    {
        "a2b_base64",
        "b16decode",
        "b32decode",
        "b64decode",
        "b85decode",
        "cloudpickle.loads",
        "codecs.decode",
        "decodebytes",
        "decompress",
        "fromhex",
        "marshal.loads",
        "pickle.loads",
        "unhexlify",
    }
)
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
_TRUSTED_SDK_DIGESTS = {
    "miner/common_preflight.py": frozenset(
        {
            # Rich contender-authored result metadata (legacy contract).
            "76841c91939673fd18949511015a13d58b7bcb03fabd45892c284db45a3a7812",
            # Output-path-only result contract.
            "b1c33f183bba6fe9f2a3441af06d9c7feec3489bb109511e4725dd93037cbc54",
        }
    ),
    # Retain audited canonical releases that contenders may already have published.
    "miner/competition_sdk.py": frozenset(
        {
            # Validation receipt schema v2, before the 2026-07-17 pricing refresh.
            "249c0579a661f4f701271976f4fd04d9f3c85e88580831ff5ff1afc76b84aafa",
            # Validation receipt schema v3 with the 2026-07-17 Modal prices.
            "80ef435e028cc49787c03b9dc303f311d4f22fe99183783a02a2036a589a7207",
            # Schema v3 SDK with stale-export and PAT prompt refresh warnings.
            "be79bb2880a213cb2044d207f9912e12ea6dc8781a66229bd693ad107f804c5e",
            # Schema v3 SDK with manifest-bounded, killable Modal image builds.
            "7192b27922be51c10dd78c4eb6c89cea0d5ec15e3b619fbd3ddfac754ea0c2e2",
            # Schema v4 SDK with actual-allocation Sandbox pricing.
            "522e204bb9ef5916486482510150c5ea86f4b43ff3cac0662df1438a2a5333c3",
            # Schema v4 SDK with validator-parity activation probes.
            "ea87877762c084c60055555ab735e2a0afcf3d8f034ddf4a2f25f7abdf16248d",
        }
    ),
}


class RepositoryStaticValidator:
    """Validate source without importing or executing any contender code."""

    def __init__(
        self,
        *,
        max_file_bytes: int = 50_000_000,
        max_repository_bytes: int = 2_000_000_000,
    ) -> None:
        self.max_file_bytes = max_file_bytes
        self.max_repository_bytes = max_repository_bytes

    def validate(self, root: Path) -> ValidationReport:
        root = root.resolve(strict=True)
        findings: list[ValidationFinding] = []
        files: list[Path] = []
        total_bytes = 0

        if (root / ".gitmodules").exists():
            findings.append(
                self._reject(
                    ValidationReason.SUBMODULE_NOT_ALLOWED,
                    ".gitmodules",
                    "Git submodules are not permitted",
                )
            )

        for directory, names, filenames in os.walk(root, followlinks=False):
            current = Path(directory)
            names[:] = [
                name
                for name in names
                if not (current == root and name == ".git")
                and name
                not in {
                    "__pycache__",
                    ".competition-sdk",
                    ".pytest_cache",
                    ".ruff_cache",
                }
            ]
            for name in [*names, *filenames]:
                path = current / name
                relative = path.relative_to(root).as_posix()
                try:
                    mode = path.lstat().st_mode
                except FileNotFoundError:
                    findings.append(
                        self._reject(
                            ValidationReason.BROKEN_SYMLINK,
                            relative,
                            "path disappeared during validation",
                        )
                    )
                    continue
                if stat.S_ISLNK(mode):
                    try:
                        resolved = path.resolve(strict=True)
                    except (FileNotFoundError, RuntimeError):
                        findings.append(
                            self._reject(
                                ValidationReason.BROKEN_SYMLINK,
                                relative,
                                "symlink target does not resolve",
                            )
                        )
                        continue
                    if not resolved.is_relative_to(root):
                        findings.append(
                            self._reject(
                                ValidationReason.PATH_ESCAPE,
                                relative,
                                "symlink resolves outside repository",
                            )
                        )
                    continue
                if name == ".git" and current != root:
                    findings.append(
                        self._reject(
                            ValidationReason.NESTED_REPOSITORY,
                            relative,
                            "nested Git metadata is not permitted",
                        )
                    )
                if not stat.S_ISREG(mode):
                    continue
                size = path.stat().st_size
                total_bytes += size
                files.append(path)
                if size > self.max_file_bytes:
                    findings.append(
                        self._reject(
                            ValidationReason.FILE_TOO_LARGE,
                            relative,
                            f"file is {size} bytes; limit is {self.max_file_bytes}",
                        )
                    )

        if total_bytes > self.max_repository_bytes:
            findings.append(
                self._reject(
                    ValidationReason.REPOSITORY_TOO_LARGE,
                    ".",
                    f"repository is {total_bytes} bytes; limit is {self.max_repository_bytes}",
                )
            )

        required = (
            "miner/modal_workers.py",
            "miner/compression/app.py",
            "miner/common_preflight.py",
            "miner/competition_sdk.py",
            "competition_solution.json",
        )
        for relative in required:
            if not (root / relative).is_file():
                findings.append(
                    self._reject(
                        ValidationReason.REQUIRED_FILE_MISSING,
                        relative,
                        "required competition template file is missing",
                    )
                )
        if not any(
            (root / name).is_file()
            for name in ("requirements.txt", "uv.lock", "poetry.lock")
        ):
            findings.append(
                self._reject(
                    ValidationReason.REQUIRED_FILE_MISSING,
                    "requirements.txt|uv.lock|poetry.lock",
                    "a locked or explicit dependency file is required",
                )
            )

        findings.extend(self._validate_descriptor(root / "competition_solution.json"))
        for relative, trusted_digests in _TRUSTED_SDK_DIGESTS.items():
            path = root / relative
            if (
                path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() not in trusted_digests
            ):
                findings.append(
                    self._reject(
                        ValidationReason.SDK_TOOL_MODIFIED,
                        relative,
                        "canonical miner SDK tooling cannot be modified in a submission",
                    )
                )
        for path in files:
            findings.extend(self._validate_file(root, path))

        digest = hashlib.sha256()
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())

        rejecting = [item for item in findings if item.rejects]
        review = [item for item in findings if not item.rejects]
        if rejecting:
            status = ValidationStatus.REJECTED
            reason = rejecting[0].reason_code
        elif review:
            status = ValidationStatus.REVIEW_REQUIRED
            reason = review[0].reason_code
        else:
            status = ValidationStatus.ACCEPTED
            reason = ValidationReason.ACCEPTED
        return ValidationReport(
            status, reason, digest.hexdigest(), len(files), total_bytes, tuple(findings)
        )

    def _validate_descriptor(self, path: Path) -> Iterable[ValidationFinding]:
        if not path.is_file():
            return ()
        try:
            descriptor = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return (
                self._reject(
                    ValidationReason.INVALID_SOLUTION_DESCRIPTOR,
                    path.name,
                    "descriptor must be valid UTF-8 JSON",
                ),
            )
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("competition_type") != "COMPRESSION"
            or descriptor.get("schema_version") != 2
        ):
            return (
                self._reject(
                    ValidationReason.INVALID_SOLUTION_DESCRIPTOR,
                    path.name,
                    "descriptor must declare COMPRESSION schema version 2",
                ),
            )
        routes = descriptor.get("routes")
        if not isinstance(routes, list) or not {"/health", "/compress"}.issubset(
            routes
        ):
            return (
                self._reject(
                    ValidationReason.REQUIRED_ROUTE_MISSING,
                    path.name,
                    "descriptor must declare /health and /compress",
                ),
            )
        if descriptor.get("local_path_io") is not True:
            return (
                self._reject(
                    ValidationReason.INVALID_SOLUTION_DESCRIPTOR,
                    path.name,
                    "local_path_io must be true",
                ),
            )
        if descriptor.get("entrypoint") != "miner/modal_workers.py":
            return (
                self._reject(
                    ValidationReason.INVALID_SOLUTION_DESCRIPTOR,
                    path.name,
                    "entrypoint must be miner/modal_workers.py",
                ),
            )
        if descriptor.get("preflight") != "miner/common_preflight.py":
            return (
                self._reject(
                    ValidationReason.INVALID_SOLUTION_DESCRIPTOR,
                    path.name,
                    "preflight must be miner/common_preflight.py",
                ),
            )
        if descriptor.get("sdk") != "miner/competition_sdk.py":
            return (
                self._reject(
                    ValidationReason.INVALID_SOLUTION_DESCRIPTOR,
                    path.name,
                    "sdk must be miner/competition_sdk.py",
                ),
            )
        return ()

    def _validate_file(self, root: Path, path: Path) -> Iterable[ValidationFinding]:
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_bytes()
        except OSError:
            return (
                self._reject(
                    ValidationReason.INVALID_SOLUTION_DESCRIPTOR,
                    relative,
                    "file could not be read",
                ),
            )
        findings: list[ValidationFinding] = []
        if content.startswith(_LFS_HEADER):
            findings.append(
                self._reject(
                    ValidationReason.GIT_LFS_POINTER,
                    relative,
                    "Git LFS pointers are not accepted",
                )
            )
        if any(pattern.search(content) for pattern in _SECRET_PATTERNS):
            findings.append(
                self._reject(
                    ValidationReason.COMMITTED_SECRET,
                    relative,
                    "possible committed credential",
                )
            )
        suffix = path.suffix.lower()
        executable_scope = relative not in _TRUSTED_SDK_DIGESTS and (
            suffix in _SOURCE_CODE_EXTENSIONS
            or suffix in _OPAQUE_CODE_EXTENSIONS
            or path.name.lower().startswith("dockerfile")
        )
        if executable_scope and (
            suffix in _OPAQUE_CODE_EXTENSIONS
            or _KNOWN_OBFUSCATOR_PATTERN.search(content)
        ):
            detail = (
                f"opaque executable extension {suffix!r} is not reviewable"
                if suffix in _OPAQUE_CODE_EXTENSIONS
                else "source references a known code-obfuscation runtime"
            )
            findings.append(
                self._reject(
                    ValidationReason.OBFUSCATION_REVIEW,
                    relative,
                    detail,
                )
            )
        if executable_scope and _DYNAMIC_PATTERN.search(content):
            findings.append(
                self._reject(
                    ValidationReason.DYNAMIC_EXECUTION,
                    relative,
                    "eval/exec is not permitted",
                )
            )
        if executable_scope and _DOWNLOAD_PATTERN.search(content):
            findings.append(
                self._review(
                    ValidationReason.REMOTE_DOWNLOAD_REVIEW,
                    relative,
                    "remote download behavior requires review",
                )
            )
        if executable_scope and suffix in _SOURCE_CODE_EXTENSIONS:
            findings.extend(self._validate_text_obfuscation(relative, content))
        if suffix in {".py", ".pyw"}:
            try:
                tree = ast.parse(content, filename=relative)
            except (SyntaxError, ValueError):
                findings.append(
                    self._reject(
                        ValidationReason.INVALID_PYTHON,
                        relative,
                        "Python source does not parse",
                    )
                )
            else:
                if executable_scope:
                    findings.extend(
                        self._validate_code_obfuscation(relative, content, tree)
                    )
        if path.suffix.lower() in {".zip", ".jar"}:
            try:
                with zipfile.ZipFile(path) as archive:
                    if any(info.flag_bits & 0x1 for info in archive.infolist()):
                        findings.append(
                            self._reject(
                                ValidationReason.ENCRYPTED_ARCHIVE,
                                relative,
                                "encrypted archives are not permitted",
                            )
                        )
            except zipfile.BadZipFile:
                pass
        return findings

    def _validate_text_obfuscation(
        self, relative: str, content: bytes
    ) -> Iterable[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if b"\0" in content:
            findings.append(
                self._reject(
                    ValidationReason.OBFUSCATION_REVIEW,
                    relative,
                    "source contains NUL bytes",
                )
            )
        if any(len(line) > 1000 for line in content.splitlines()):
            findings.append(
                self._reject(
                    ValidationReason.OBFUSCATION_REVIEW,
                    relative,
                    "source contains a line longer than 1000 bytes",
                )
            )
        return findings

    def _validate_code_obfuscation(
        self, relative: str, content: bytes, tree: ast.AST
    ) -> Iterable[ValidationFinding]:
        """Reject clear code-obfuscation patterns without executing the source."""

        findings: list[ValidationFinding] = []
        statements_by_line: dict[int, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt) and hasattr(node, "lineno"):
                statements_by_line[node.lineno] = (
                    statements_by_line.get(node.lineno, 0) + 1
                )
        if any(count > 12 for count in statements_by_line.values()):
            findings.append(
                self._reject(
                    ValidationReason.OBFUSCATION_REVIEW,
                    relative,
                    "source packs more than 12 statements onto one line",
                )
            )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function_name = self._call_name(node.func)
            short_name = function_name.rsplit(".", 1)[-1] if function_name else None
            if not {
                function_name,
                short_name,
            }.intersection(_ENCODED_LOADER_NAMES):
                continue
            payload = node.args[0]
            if not isinstance(payload, ast.Constant) or not isinstance(
                payload.value, (str, bytes)
            ):
                continue
            if len(payload.value) < 128:
                continue
            findings.append(
                self._reject(
                    ValidationReason.OBFUSCATION_REVIEW,
                    relative,
                    f"{function_name} loads an embedded encoded payload",
                )
            )
            break
        return findings

    @staticmethod
    def _call_name(function: ast.expr) -> str | None:
        if isinstance(function, ast.Name):
            return function.id
        if isinstance(function, ast.Attribute):
            owner = RepositoryStaticValidator._call_name(function.value)
            return f"{owner}.{function.attr}" if owner else function.attr
        return None

    @staticmethod
    def _reject(reason: ValidationReason, path: str, detail: str) -> ValidationFinding:
        return ValidationFinding(reason, path, detail, True)

    @staticmethod
    def _review(reason: ValidationReason, path: str, detail: str) -> ValidationFinding:
        return ValidationFinding(reason, path, detail, False)


def write_validation_report(report: ValidationReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "RepositoryStaticValidator",
    "ValidationFinding",
    "ValidationReason",
    "ValidationReport",
    "ValidationStatus",
    "write_validation_report",
]
