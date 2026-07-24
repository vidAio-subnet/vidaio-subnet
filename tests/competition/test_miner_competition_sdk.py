from __future__ import annotations

import ast
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = load_module(
    "miner_common_preflight_test", ROOT / "miner/common_preflight.py"
)
SDK = load_module("miner_competition_sdk_test", ROOT / "miner/competition_sdk.py")


class CompetitionSdkTests(unittest.TestCase):
    def test_modal_probes_match_validator_probes(self) -> None:
        runner_tree = ast.parse(
            (ROOT / "vidaio_subnet_core/competition/modal_runner.py").read_text(
                encoding="utf-8"
            )
        )
        runner_probes = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in runner_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in {"ISOLATION_PROBE", "RESOURCE_PROBE", "READINESS_PROBE"}
        }
        for name in ("ISOLATION_PROBE", "RESOURCE_PROBE", "READINESS_PROBE"):
            with self.subTest(name=name):
                self.assertEqual(getattr(SDK, name), runner_probes[name])

    def make_source(self, root: Path) -> Path:
        source = root / "source"
        files = {
            "competition_solution.json": json.dumps(
                {
                    "competition_type": "COMPRESSION",
                    "entrypoint": "miner/modal_workers.py",
                    "local_path_io": True,
                    "preflight": "miner/common_preflight.py",
                    "routes": ["/health", "/compress"],
                    "schema_version": 2,
                    "sdk": "miner/competition_sdk.py",
                }
            ),
            "miner/modal_workers.py": "VALUE = 1\n",
            "miner/compression/app.py": "def compress():\n    return None\n",
            "miner/compression/Dockerfile": "FROM scratch\n",
            "miner/requirements.txt": "fastapi==0.110.3\n",
            "miner/.env": "SECRET=do-not-copy\n",
            "miner/.env.template": "SECRET=\n",
            "competitions/fixtures/compression_warmup_input.mp4": "fixture",
        }
        for relative, content in files.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        (source / "miner/common_preflight.py").write_text(
            (ROOT / "miner/common_preflight.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (source / "miner/competition_sdk.py").write_text(
            (ROOT / "miner/competition_sdk.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return source

    def test_prepare_export_is_standalone_and_excludes_local_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_source(root)
            export = SDK.prepare_export(source, root / "export")
            self.assertTrue((export / SDK.EXPORT_MARKER).is_file())
            self.assertTrue((export / "miner/common_preflight.py").is_file())
            self.assertTrue((export / "scripts/competition_modal_build.py").is_file())
            self.assertTrue((export / "competition_solution.json").is_file())
            self.assertTrue((export / "requirements.txt").is_file())
            self.assertFalse((export / "miner/.env").exists())
            self.assertTrue((export / "miner/.env.template").is_file())
            report = PREFLIGHT.validate_repository(export)
            self.assertEqual(report["status"], "ACCEPTED")

    def test_prepare_refuses_to_reuse_unmarked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_source(root)
            destination = root / "not-an-export"
            destination.mkdir()
            with self.assertRaisesRegex(SDK.SdkError, "unmarked"):
                SDK.prepare_export(source, destination)

    def test_reusing_stale_export_warns_that_refresh_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_source(root)
            export = SDK.prepare_export(source, root / "export")
            with (source / "miner/competition_sdk.py").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("\n# newer canonical SDK release\n")

            progress = io.StringIO()
            with redirect_stderr(progress):
                reused = SDK.prepare_export(source, export)

            self.assertEqual(reused, export)
            warning = progress.getvalue()
            self.assertIn("--refresh", warning)
            self.assertIn("SDK_TOOL_MODIFIED", warning)
            self.assertIn("miner/competition_sdk.py", warning)

    def test_pat_prompt_repeats_refresh_warning_for_reused_export(self) -> None:
        progress = io.StringIO()
        with (
            patch.object(SDK.os, "getenv", return_value=""),
            patch.object(SDK.getpass, "getpass", return_value="test-token"),
            redirect_stderr(progress),
        ):
            pat = SDK._pat_from_environment_or_prompt(
                "GITHUB_TOKEN", warn_if_reusing_export=True
            )

        self.assertEqual(pat, "test-token")
        self.assertIn("--refresh", progress.getvalue())
        self.assertIn("GitHub PAT prompt follows", progress.getvalue())

    def test_common_preflight_sends_exact_batch_of_one_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_root = root / "evaluation-inputs"
            input_root.mkdir()
            fixture = input_root / "compression_warmup_input.mp4"
            fixture.write_bytes(b"fixture")
            output_root = root / "output"
            captured: list[dict] = []

            def inspector(path: Path):
                is_output = path.as_posix().endswith("/output/sdk-preflight/fixed.mp4")
                return PREFLIGHT.MediaInfo(
                    width=320,
                    height=180,
                    duration_seconds=5,
                    frame_count=118 if is_output else 120,
                    codec="av1" if is_output else "h264",
                    container="mov,mp4",
                    pixel_format="yuv420p",
                    sample_aspect_ratio="1:1",
                )

            def http_json(method: str, url: str, payload: dict | None):
                if url.endswith("/health"):
                    return {
                        "status": "ok",
                        "competition_local_io": {"remote_io_disabled": True},
                    }
                self.assertEqual(method, "POST")
                self.assertIsNotNone(payload)
                captured.append(payload)
                item = payload["items"][0]
                host_output = output_root / "sdk-preflight/fixed.mp4"
                host_output.write_bytes(b"output")
                return {
                    "results": [{"output_path": item["output_path"]}],
                }

            result = PREFLIGHT.run_runtime_preflight(
                fixture=fixture,
                service_url="http://127.0.0.1:8003",
                host_input_root=input_root,
                host_output_root=output_root,
                http_json=http_json,
                inspector=inspector,
                run_id="fixed",
                input_relative_path=Path("compression_warmup_input.mp4"),
                prepositioned_input=True,
            )
            self.assertEqual(result["status"], "ACCEPTED")
            self.assertEqual(result["batch_size"], 1)
            self.assertEqual(len(captured[0]["items"]), 1)
            self.assertEqual(
                captured[0]["items"][0]["input_path"],
                "/evaluation-inputs/compression_warmup_input.mp4",
            )
            self.assertEqual(
                set(captured[0]["items"][0]),
                {
                    "evaluation_id",
                    "input_path",
                    "output_path",
                    "codec",
                    "vmaf_threshold",
                },
            )

    def test_github_repository_creation_is_private(self) -> None:
        calls = []

        def requester(method: str, path: str, pat: str, payload: dict | None):
            calls.append((method, path, pat, payload))
            if path == "/user":
                return {"login": "miner-user", "id": 12345, "name": "Real Miner"}
            return {
                "full_name": "miner-user/private-compressor",
                "private": True,
                "default_branch": "main",
            }

        target = SDK.ensure_private_repository(
            "private-compressor", "test-token-value", requester=requester
        )
        self.assertEqual(target.full_name, "miner-user/private-compressor")
        self.assertEqual(
            target.repository_url,
            "https://github.com/miner-user/private-compressor.git",
        )
        self.assertEqual(target.identity.name, "Real Miner")
        self.assertEqual(
            target.identity.email,
            "12345+miner-user@users.noreply.github.com",
        )
        self.assertIs(target.existing, False)
        self.assertEqual(calls[-1][1], "/user/repos")
        self.assertIs(calls[-1][3]["private"], True)

    def test_existing_repository_is_looked_up_instead_of_created(self) -> None:
        calls = []

        def requester(method: str, path: str, pat: str, payload: dict | None):
            calls.append((method, path, payload))
            if path == "/user":
                return {"login": "miner-user", "id": 42}
            return {
                "full_name": "miner-user/private-compressor",
                "private": True,
                "default_branch": "trunk",
            }

        target = SDK.ensure_private_repository(
            "miner-user/private-compressor",
            "test-token-value",
            use_existing=True,
            requester=requester,
        )
        self.assertEqual(
            calls[-1], ("GET", "/repos/miner-user/private-compressor", None)
        )
        self.assertEqual(target.default_branch, "trunk")
        self.assertIs(target.existing, True)

    def test_existing_repository_creation_error_has_recovery_command(self) -> None:
        def requester(method: str, path: str, pat: str, payload: dict | None):
            if path == "/user":
                return {"login": "miner-user", "id": 42}
            raise SDK.GitHubApiError(422, "Repository creation failed")

        with self.assertRaisesRegex(SDK.SdkError, "--use-existing"):
            SDK.ensure_private_repository(
                "private-compressor", "test-token-value", requester=requester
            )

    def test_validation_receipt_is_bound_to_tree_resources_and_age(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            export = Path(temp) / "export"
            export.mkdir()
            static = {"repository": {"repository_tree_sha256": "a" * 64}}
            runtime = {"status": "ACCEPTED", "sandbox_id": "sb-fixed"}
            configuration = SDK._validation_configuration(
                environment_name="dev",
                gpu="L40S",
                requested_cpu=16,
                cpu_limit=32,
                sandbox_timeout=1800,
            )
            validated_at = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)
            path = SDK.write_validation_receipt(
                export,
                static_report=static,
                runtime_report=runtime,
                configuration=configuration,
                now=validated_at,
            )
            self.assertEqual(path.parent, export.parent.resolve())
            self.assertFalse(path.is_relative_to(export))

            receipt, reason = SDK.load_validation_receipt(
                export,
                static_report=static,
                configuration=configuration,
                now=validated_at + timedelta(hours=1),
            )
            self.assertIsNotNone(receipt)
            self.assertEqual(reason, "matching validation receipt")

            changed_static = {"repository": {"repository_tree_sha256": "b" * 64}}
            receipt, reason = SDK.load_validation_receipt(
                export,
                static_report=changed_static,
                configuration=configuration,
                now=validated_at + timedelta(hours=1),
            )
            self.assertIsNone(receipt)
            self.assertIn("content changed", reason)

            changed_configuration = {**configuration, "gpu": "A100"}
            receipt, reason = SDK.load_validation_receipt(
                export,
                static_report=static,
                configuration=changed_configuration,
                now=validated_at + timedelta(hours=1),
            )
            self.assertIsNone(receipt)
            self.assertIn("settings changed", reason)

            receipt, reason = SDK.load_validation_receipt(
                export,
                static_report=static,
                configuration=configuration,
                now=validated_at + timedelta(hours=25),
            )
            self.assertIsNone(receipt)
            self.assertIn("expired", reason)

    def test_modal_build_timeout_comes_from_manifest_with_ten_minute_default(
        self,
    ) -> None:
        self.assertEqual(SDK.load_modal_build_timeout_seconds(None), 600)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "competition.json"
            manifest.write_text(
                json.dumps({"modal_build_timeout": "90s"}), encoding="utf-8"
            )
            self.assertEqual(SDK.load_modal_build_timeout_seconds(manifest), 90)
            manifest.write_text("{}", encoding="utf-8")
            self.assertEqual(SDK.load_modal_build_timeout_seconds(manifest), 600)
            manifest.write_text(
                json.dumps({"modal_build_timeout": "PT2M"}), encoding="utf-8"
            )
            self.assertEqual(SDK.load_modal_build_timeout_seconds(manifest), 120)

    def test_modal_build_timeout_stops_app_and_rejects_validation(self) -> None:
        stopped = []

        def timeout_runner(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("modal-build", 3)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text("pass\n", encoding="utf-8")
            dockerfile = root / "Dockerfile"
            dockerfile.write_text("FROM scratch\n", encoding="utf-8")
            with self.assertRaisesRegex(
                SDK.SdkError, "exceeded modal_build_timeout.*3 seconds"
            ):
                SDK._build_modal_base_image(
                    worker=worker,
                    environment_name="dev",
                    app_name="sdk-build-fixed",
                    dockerfile=dockerfile,
                    context_dir=root,
                    timeout_seconds=3,
                    runner=timeout_runner,
                    stopper=lambda environment, app: stopped.append((environment, app)),
                )
        self.assertEqual(stopped, [("dev", "sdk-build-fixed")])

    def test_publish_reuses_matching_validation_without_modal_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export = root / "export"
            export.mkdir()
            (export / SDK.EXPORT_MARKER).write_text("sdk\n", encoding="utf-8")
            static = {"repository": {"repository_tree_sha256": "c" * 64}}
            runtime = {"status": "ACCEPTED", "sandbox_id": "sb-fixed"}
            configuration = SDK._validation_configuration(
                environment_name="dev",
                gpu="L40S",
                requested_cpu=16,
                cpu_limit=32,
                sandbox_timeout=1800,
            )
            SDK.write_validation_receipt(
                export,
                static_report=static,
                runtime_report=runtime,
                configuration=configuration,
            )
            args = SimpleNamespace(
                action="publish",
                repository="miner-user/private-compressor",
                source_root=root,
                export=export,
                refresh=False,
                modal_environment="dev",
                modal_gpu="L40S",
                modal_cpu=16,
                modal_cpu_limit=32,
                modal_timeout=1800,
                pat_env="GITHUB_TOKEN",
                use_existing=True,
                revalidate=False,
                validation_max_age_hours=24,
                git_author_name=None,
                git_author_email=None,
            )
            target = SDK.GitHubRepositoryTarget(
                full_name="miner-user/private-compressor",
                repository_url="https://github.com/miner-user/private-compressor.git",
                default_branch="main",
                existing=True,
                identity=SDK.GitHubIdentity(
                    "miner-user",
                    "Real Miner",
                    "42+miner-user@users.noreply.github.com",
                ),
            )
            output = io.StringIO()
            with (
                patch.object(SDK, "parse_args", return_value=args),
                patch.object(SDK, "prepare_export", return_value=export),
                patch.object(SDK, "run_common_preflight", return_value=static),
                patch.object(SDK, "run_modal_preflight") as modal_preflight,
                patch.object(
                    SDK,
                    "_pat_from_environment_or_prompt",
                    return_value="test-token-value",
                ),
                patch.object(SDK, "ensure_private_repository", return_value=target),
                patch.object(SDK, "initialize_export_git"),
                patch.object(SDK, "push_with_askpass"),
                patch.object(SDK, "cleanup_export_git"),
                patch("sys.stdout", output),
            ):
                status = SDK.main()

            self.assertEqual(status, 0)
            modal_preflight.assert_not_called()
            result = json.loads(output.getvalue())
            self.assertIs(result["validation_reused"], True)
            self.assertEqual(result["commit_author"]["name"], "Real Miner")

    def test_modal_preflight_uses_validator_shaped_isolated_mounts(self) -> None:
        captured: dict[str, object] = {"deleted": [], "exec": []}

        class Process:
            def __init__(self, stdout: str = "") -> None:
                self.returncode = 0
                self.stdout = SimpleNamespace(read=lambda: stdout)
                self.stderr = SimpleNamespace(read=lambda: "")

            def wait(self) -> None:
                return None

        class Upload:
            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def put_file(self, local_path: Path, remote_path: str) -> None:
                captured["upload"] = (local_path.name, remote_path)

        class VolumeHandle:
            def __init__(self, name: str) -> None:
                self.name = name

            def batch_upload(self, *, force: bool):
                captured["upload_force"] = force
                return Upload()

            def with_mount_options(self, *, read_only: bool):
                return {"volume": self.name, "read_only": read_only}

        class VolumeApi:
            objects = None

            @staticmethod
            def from_name(name: str, **kwargs):
                captured.setdefault("volume_creates", []).append((name, kwargs))
                return VolumeHandle(name)

        class VolumeObjectsApi:
            @staticmethod
            def delete(name: str, **kwargs) -> None:
                captured["deleted"].append((name, kwargs))

        VolumeApi.objects = VolumeObjectsApi()

        class ImageHandle:
            def add_local_file(self, local_path: Path, remote_path: str, *, copy: bool):
                captured.setdefault("image_layers", []).append(
                    (local_path.name, remote_path, copy)
                )
                return self

        class ImageApi:
            @staticmethod
            def from_dockerfile(path: Path, **kwargs):
                captured["dockerfile"] = (
                    path.read_text(encoding="utf-8"),
                    (Path(kwargs["context_dir"]) / "app.py").read_text(
                        encoding="utf-8"
                    ),
                    kwargs,
                )
                return ImageHandle()

        class SandboxHandle:
            object_id = "sb-test"
            stdout = ("uvicorn ready\n", "compression complete\n")
            stderr = ()

            def exec(self, *args: str, **kwargs):
                captured["exec"].append((args, kwargs))
                if SDK.ISOLATION_PROBE in args:
                    return Process(
                        json.dumps(
                            {
                                "network_blocked": True,
                                "input_read_only": True,
                                "output_writable": True,
                                "reference_mount_absent": True,
                                "credentials_absent": True,
                                "passed": True,
                            }
                        )
                    )
                if SDK.RESOURCE_PROBE in args:
                    return Process(
                        json.dumps(
                            {
                                "gpu_names": ["NVIDIA H200"],
                                "allocated_cpu_cores": 8.0,
                                "cpu_quota_cores": 8.0,
                                "affinity_logical_cpu_count": 16,
                                "affinity_physical_cpu_cores": 8.0,
                            }
                        )
                    )
                if SDK.READINESS_PROBE in args:
                    return Process(json.dumps({"status": "ok"}))
                if "/vidaio/common_preflight.py" in args:
                    return Process(json.dumps({"status": "ACCEPTED"}))
                return Process()

            def terminate(self, *, wait: bool) -> None:
                captured["terminated"] = wait

        class SandboxApi:
            @staticmethod
            def create(*args: str, **kwargs):
                captured["sandbox_create"] = (args, kwargs)
                return SandboxHandle()

        class ModalOutputContext:
            def __enter__(self):
                captured["modal_output_enabled"] = True
                print("latest Modal build line")
                return self

            def __exit__(self, *_args) -> None:
                captured["modal_output_closed"] = True

        modal = SimpleNamespace(
            App=SimpleNamespace(
                lookup=lambda *args, **kwargs: captured.setdefault(
                    "app_lookup", (args, kwargs)
                )
            ),
            Volume=VolumeApi,
            Image=ImageApi,
            Sandbox=SandboxApi,
            Probe=SimpleNamespace(with_tcp=lambda port: ("tcp", port)),
            enable_output=ModalOutputContext,
        )

        with tempfile.TemporaryDirectory() as temp:
            source = self.make_source(Path(temp))
            export = SDK.prepare_export(source, Path(temp) / "export")
            progress = io.StringIO()
            with (
                redirect_stderr(progress),
                patch.object(SDK.time, "monotonic", side_effect=(100.0, 110.0)),
            ):
                report = SDK.run_modal_preflight(
                    export,
                    environment_name="dev",
                    gpu="L40S",
                    modal_api=modal,
                )

        self.assertEqual(report["status"], "ACCEPTED")
        self.assertEqual(
            report["modal_cost_estimate"]["estimated_consumed_balance_usd"],
            "0.0157636000",
        )
        self.assertEqual(report["modal_cost_estimate"]["sandbox_runtime_seconds"], 10.0)
        self.assertEqual(
            report["modal_cost_estimate"]["attribution_method"],
            SDK.MODAL_COST_ATTRIBUTION_METHOD,
        )
        self.assertEqual(report["modal_cost_estimate"]["allocated_gpu"], "H200")
        self.assertEqual(report["modal_cost_estimate"]["allocated_cpu_cores"], 8.0)
        self.assertEqual(
            captured["upload"],
            ("compression_warmup_input.mp4", SDK.MODAL_INPUT_VOLUME_PATH),
        )
        _, create = captured["sandbox_create"]
        self.assertIs(create["block_network"], True)
        self.assertEqual(create["secrets"], [])
        self.assertIs(create["include_oidc_identity_token"], False)
        self.assertEqual(create["cpu"], (16, 32))
        self.assertEqual(report["modal_build_timeout_seconds"], 600)
        self.assertTrue(report["isolation"]["passed"])
        self.assertEqual(report["readiness"], {"status": "ok"})
        self.assertEqual(create["encrypted_ports"], [])
        self.assertEqual(create["h2_ports"], [])
        self.assertEqual(create["unencrypted_ports"], [])
        self.assertIs(create["verbose"], True)
        self.assertIs(captured["modal_output_enabled"], True)
        self.assertIs(captured["modal_output_closed"], True)
        self.assertNotIn("readiness_probe", create)
        self.assertEqual(captured["dockerfile"][1], SDK.MODAL_APP_BUILD_PLACEHOLDER)
        self.assertEqual(
            captured["image_layers"],
            [
                ("app.py", "/app/app.py", True),
                ("common_preflight.py", "/vidaio/common_preflight.py", True),
            ],
        )
        self.assertIs(create["volumes"]["/evaluation-inputs"]["read_only"], True)
        self.assertIs(create["volumes"]["/output"]["read_only"], False)
        output_volume_create = next(
            kwargs
            for name, kwargs in captured["volume_creates"]
            if name.endswith("-output")
        )
        self.assertEqual(output_volume_create["version"], 2)
        runtime_args = next(
            args
            for args, _kwargs in captured["exec"]
            if "/vidaio/common_preflight.py" in args
        )
        self.assertIn("--prepositioned-input", runtime_args)
        self.assertIn("compression_warmup_input.mp4", runtime_args)
        probe_args = [args for args, _kwargs in captured["exec"]]
        self.assertLess(
            next(i for i, args in enumerate(probe_args) if SDK.ISOLATION_PROBE in args),
            next(i for i, args in enumerate(probe_args) if SDK.RESOURCE_PROBE in args),
        )
        self.assertLess(
            next(i for i, args in enumerate(probe_args) if SDK.RESOURCE_PROBE in args),
            next(i for i, args in enumerate(probe_args) if SDK.READINESS_PROBE in args),
        )
        self.assertIn(
            (("sync", SDK.MODAL_OUTPUT_ROOT), {"timeout": 30}), captured["exec"]
        )
        self.assertIs(captured["terminated"], True)
        self.assertEqual(len(captured["deleted"]), 2)
        rendered_progress = progress.getvalue()
        self.assertIn("[competition-sdk][modal-build]", rendered_progress)
        self.assertIn("latest Modal build line", rendered_progress)
        self.assertIn(
            "[competition-sdk][sandbox-stdout] uvicorn ready", rendered_progress
        )
        self.assertIn("[competition-sdk][inference] accepted output", rendered_progress)
        self.assertIn(
            "[competition-sdk][cost] estimated consumed balance $0.0157636000 USD",
            rendered_progress,
        )
        self.assertIn(
            "[competition-sdk][cleanup] Sandbox terminated", rendered_progress
        )

    def test_modal_cost_estimate_rejects_gpu_without_validator_rate(self) -> None:
        with self.assertRaisesRegex(SDK.SdkError, "no locked Modal cost rate"):
            SDK.estimate_modal_compute_cost("A100", 16, 1)

    def test_git_push_uses_ephemeral_askpass_not_token_in_argv(self) -> None:
        captured = {}

        def runner(command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            self.assertTrue(Path(kwargs["env"]["GIT_ASKPASS"]).is_file())
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as temp:
            token = "test-token-value"
            SDK.push_with_askpass(
                Path(temp),
                "https://github.com/miner-user/private-compressor.git",
                token,
                runner=runner,
            )
        self.assertNotIn(token, " ".join(captured["command"]))
        self.assertEqual(captured["env"]["VIDAIO_GIT_PASSWORD"], token)

    def test_existing_repository_update_is_a_fast_forward_with_real_author(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            remote = root / "remote.git"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)

            first = root / "first"
            first.mkdir()
            (first / SDK.EXPORT_MARKER).write_text("sdk\n", encoding="utf-8")
            (first / "solution.txt").write_text("v1\n", encoding="utf-8")
            SDK.initialize_export_git(
                first,
                author_name="First Miner",
                author_email="1+first@example.invalid",
            )
            SDK.push_with_askpass(first, str(remote), "test-token-value")
            SDK.cleanup_export_git(first)

            second = root / "second"
            second.mkdir()
            (second / SDK.EXPORT_MARKER).write_text("sdk\n", encoding="utf-8")
            (second / "solution.txt").write_text("v2\n", encoding="utf-8")
            SDK.initialize_export_git(
                second,
                author_name="Real Miner",
                author_email="42+miner-user@users.noreply.github.com",
                parent_repository_url=str(remote),
                parent_branch="main",
                pat="test-token-value",
            )
            SDK.push_with_askpass(second, str(remote), "test-token-value")

            commit_count = subprocess.check_output(
                ["git", "--git-dir", str(remote), "rev-list", "--count", "main"],
                text=True,
            ).strip()
            latest_author = subprocess.check_output(
                [
                    "git",
                    "--git-dir",
                    str(remote),
                    "show",
                    "-s",
                    "--format=%an|%ae",
                    "main",
                ],
                text=True,
            ).strip()
            latest_content = subprocess.check_output(
                [
                    "git",
                    "--git-dir",
                    str(remote),
                    "show",
                    "main:solution.txt",
                ],
                text=True,
            )
            self.assertEqual(commit_count, "2")
            self.assertEqual(
                latest_author,
                "Real Miner|42+miner-user@users.noreply.github.com",
            )
            self.assertEqual(latest_content, "v2\n")


if __name__ == "__main__":
    unittest.main()
