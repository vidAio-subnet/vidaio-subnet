#!/usr/bin/env python3
"""Export miner Compose services as pull-only prebuilt images.

The exporter builds services with local Dockerfiles, pushes them to Docker Hub,
and writes a docker-compose.yml that contains image references only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPOSE_NAMES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)
DOCKER_HUB_API = "https://hub.docker.com"


class ExportError(RuntimeError):
    """Raised for user-facing export failures."""


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def env_value(env_file: dict[str, str], name: str, default: str = "") -> str:
    return os.environ.get(name) or env_file.get(name) or default


def is_placeholder(value: str) -> bool:
    return value.startswith("your-") or value.endswith("-api-key")


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    shown = " ".join(shlex.quote(part) for part in args)
    print(f"+ {shown}")
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=check,
    )


def prompt_project_root(default_root: Path) -> Path:
    print("Compose project root examples:")
    print(f"  {SCRIPT_DIR}")
    print(f"  {SCRIPT_DIR / 'upscaling'}")
    print(f"  {SCRIPT_DIR / 'upscaling' / 'ffmpeg'}")
    print(f"  {SCRIPT_DIR / 'compression'}")
    answer = input(f"Compose project root [{default_root}]: ").strip()
    return Path(answer or default_root).expanduser().resolve()


def find_compose_file(project_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise ExportError(f"Compose file does not exist: {path}")
        return path
    for name in DEFAULT_COMPOSE_NAMES:
        candidate = project_root / name
        if candidate.exists():
            return candidate
    return None


def compose_config(compose_file: Path, project_root: Path) -> dict[str, Any]:
    command = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "--project-directory",
        str(project_root),
        "config",
        "--format",
        "json",
        "--no-interpolate",
    ]
    result = run(command, capture=True)
    return json.loads(result.stdout)


def build_context_path(service: dict[str, Any]) -> Path | None:
    build = service.get("build")
    if isinstance(build, str):
        return Path(build).resolve()
    if isinstance(build, dict) and build.get("context"):
        return Path(str(build["context"])).resolve()
    return None


def select_config(project_root: Path, compose_file: Path | None) -> tuple[Path, Path, dict[str, Any], list[str]]:
    if compose_file:
        data = compose_config(compose_file, project_root)
        services = [name for name, svc in data.get("services", {}).items() if "build" in svc]
        return compose_file, project_root, data, services

    miner_compose = SCRIPT_DIR / "docker-compose.yml"
    if not miner_compose.exists():
        raise ExportError(f"No compose file found in {project_root} and {miner_compose} is missing.")

    data = compose_config(miner_compose, SCRIPT_DIR)
    requested = project_root.resolve()
    services = [
        name
        for name, svc in data.get("services", {}).items()
        if build_context_path(svc) == requested
    ]
    if not services:
        raise ExportError(
            f"No compose file found in {project_root}, and no miner service build context matches it."
        )
    return miner_compose, SCRIPT_DIR, data, services


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", value)
    value = re.sub(r"([._-]){2,}", r"\1", value)
    if len(value) < 2:
        value = f"{value}x"
    return value[:255]


def default_tag() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    try:
        result = run(["git", "rev-parse", "--short", "HEAD"], cwd=SCRIPT_DIR, capture=True, check=False)
    except FileNotFoundError:
        return stamp
    git_sha = (result.stdout or "").strip()
    return f"{git_sha}-{stamp}" if result.returncode == 0 and git_sha else stamp


def image_reference(registry: str, namespace: str, repository: str, tag: str) -> str:
    registry = registry.strip().rstrip("/")
    namespace = namespace.strip("/")
    ref = f"{namespace}/{repository}:{tag}"
    if registry and registry not in {"docker.io", "index.docker.io"}:
        ref = f"{registry}/{ref}"
    return ref


def dockerhub_login(username: str, api_key: str) -> str:
    body = {"username": username, "password": api_key}
    data, _ = dockerhub_request("POST", "/v2/users/login/", body=body, bearer=None)
    token = data.get("token") or data.get("access_token")
    if not token:
        raise ExportError("Docker Hub login succeeded but no bearer token was returned.")
    return str(token)


def dockerhub_request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    bearer: str | None,
    expected: set[int] | None = None,
) -> tuple[dict[str, Any], int]:
    expected = expected or {200, 201}
    url = f"{DOCKER_HUB_API}{path}"
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    request = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {"detail": raw}
        if exc.code not in expected:
            detail = data.get("detail") or data.get("message") or raw or exc.reason
            raise ExportError(f"Docker Hub API {method} {path} returned {exc.code}: {detail}")
        return data, exc.code
    except urllib.error.URLError as exc:
        raise ExportError(f"Could not reach Docker Hub API: {exc.reason}") from exc

    if status not in expected:
        raise ExportError(f"Docker Hub API {method} {path} returned unexpected status {status}.")
    return data, status


def get_repository(namespace: str, repository: str, bearer: str) -> tuple[dict[str, Any], int]:
    escaped_namespace = urllib.parse.quote(namespace, safe="")
    escaped_repository = urllib.parse.quote(repository, safe="")
    return dockerhub_request(
        "GET",
        f"/v2/namespaces/{escaped_namespace}/repositories/{escaped_repository}",
        bearer=bearer,
        expected={200, 401, 403, 404},
    )


def create_repository(namespace: str, repository: str, bearer: str, private: bool) -> None:
    escaped_namespace = urllib.parse.quote(namespace, safe="")
    body = {
        "name": repository,
        "namespace": namespace,
        "description": "Vidaio miner exported service image",
        "full_description": "Prebuilt image generated by miner/export_compose.py.",
        "registry": "docker.io",
        "is_private": private,
    }
    dockerhub_request(
        "POST",
        f"/v2/namespaces/{escaped_namespace}/repositories",
        body=body,
        bearer=bearer,
        expected={200, 201},
    )


def ensure_repositories(namespace: str, repositories: list[str], bearer: str, private: bool) -> None:
    for repository in repositories:
        existing, status = get_repository(namespace, repository, bearer)
        if status == 404:
            mode = "private" if private else "public"
            print(f"Creating Docker Hub {mode} repository {namespace}/{repository}")
            create_repository(namespace, repository, bearer, private)
            continue
        if status in {401, 403}:
            raise ExportError(
                f"Write API key cannot read or create Docker Hub repository {namespace}/{repository}."
            )
        if private and existing.get("is_private") is not True:
            raise ExportError(
                f"Docker Hub repository {namespace}/{repository} already exists but is public. "
                "Make it private or export with --public."
            )
        print(f"Using existing Docker Hub repository {namespace}/{repository}")


def docker_login_cli(registry: str, username: str, api_key: str, docker_env: dict[str, str]) -> None:
    command = ["docker", "login"]
    if registry and registry not in {"docker.io", "index.docker.io"}:
        command.append(registry)
    command.extend(["--username", username, "--password-stdin"])
    shown_registry = registry or "docker.io"
    print(f"+ docker login {shown_registry} --username {shlex.quote(username)} --password-stdin")
    subprocess.run(
        command,
        input=f"{api_key}\n",
        text=True,
        env=docker_env,
        check=True,
    )


def convert_ports(ports: Any) -> Any:
    if not isinstance(ports, list):
        return ports
    converted: list[Any] = []
    for item in ports:
        if not isinstance(item, dict) or "target" not in item:
            converted.append(item)
            continue
        target = str(item["target"])
        published = str(item.get("published") or "")
        host_ip = item.get("host_ip")
        protocol = item.get("protocol", "tcp")
        parts = []
        if host_ip:
            parts.append(str(host_ip))
        if published:
            parts.append(published)
        parts.append(target)
        value = ":".join(parts)
        if protocol and protocol != "tcp":
            value = f"{value}/{protocol}"
        converted.append(value)
    return converted


def convert_volumes(volumes: Any) -> Any:
    if not isinstance(volumes, list):
        return volumes
    converted: list[Any] = []
    for item in volumes:
        if not isinstance(item, dict) or "source" not in item or "target" not in item:
            converted.append(item)
            continue
        value = f"{item['source']}:{item['target']}"
        if item.get("read_only"):
            value = f"{value}:ro"
        converted.append(value)
    return converted


def convert_networks(networks: Any) -> Any:
    if not isinstance(networks, dict):
        return networks
    if all(value is None for value in networks.values()):
        return list(networks.keys())
    return networks


def normalize_service(service: dict[str, Any], image: str, include_build: bool) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if include_build and "build" in service:
        normalized["build"] = service["build"]
    normalized["image"] = image
    for key, value in service.items():
        if key in {"build", "image"}:
            continue
        if key == "ports":
            normalized[key] = convert_ports(value)
        elif key == "volumes":
            normalized[key] = convert_volumes(value)
        elif key == "networks":
            normalized[key] = convert_networks(value)
        elif key == "gpus" and isinstance(value, list) and len(value) == 1 and value[0].get("count") == "all":
            normalized[key] = "all"
        else:
            normalized[key] = value
    return normalized


def referenced_networks(services: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for service in services.values():
        networks = service.get("networks")
        if isinstance(networks, list):
            names.update(str(item) for item in networks)
        elif isinstance(networks, dict):
            names.update(str(item) for item in networks)
    return names


def export_compose_data(
    config: dict[str, Any],
    services: list[str],
    images: dict[str, str],
    *,
    include_build: bool,
) -> dict[str, Any]:
    exported_services = {
        service_name: normalize_service(config["services"][service_name], images[service_name], include_build)
        for service_name in services
    }
    data: dict[str, Any] = {"services": exported_services}

    used_networks = referenced_networks(exported_services)
    if used_networks:
        networks: dict[str, Any] = {}
        for name in sorted(used_networks):
            source = dict(config.get("networks", {}).get(name) or {})
            source.pop("name", None)
            networks[name] = source or None
        data["networks"] = networks
    return data


def yaml_quote(value: str) -> str:
    if value == "":
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_./-]+", value) and value.lower() not in {"true", "false", "null"}:
        return value
    return json.dumps(value)


def yaml_key(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        return value
    return yaml_quote(value)


def to_yaml(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            rendered_key = yaml_key(str(key))
            if isinstance(item, (dict, list)):
                if not item:
                    lines.append(f"{prefix}{rendered_key}: {{}}")
                else:
                    lines.append(f"{prefix}{rendered_key}:")
                    lines.extend(to_yaml(item, indent + 2))
            elif item is None:
                lines.append(f"{prefix}{rendered_key}: null")
            else:
                lines.append(f"{prefix}{rendered_key}: {scalar_to_yaml(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {scalar_to_yaml(item)}")
        return lines
    return [f"{prefix}{scalar_to_yaml(value)}"]


def scalar_to_yaml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    return yaml_quote(str(value))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    rendered = "\n".join(to_yaml(data)) + "\n"
    path.write_text(rendered, encoding="utf-8")


def build_and_push(
    project_root: Path,
    build_compose: Path,
    services: list[str],
    docker_env: dict[str, str],
    *,
    skip_build: bool,
    skip_push: bool,
) -> None:
    if skip_build:
        print("Skipping docker compose build because --skip-build was set.")
        return
    build_cmd = [
        "docker",
        "compose",
        "-f",
        str(build_compose),
        "--project-directory",
        str(project_root),
        "build",
        *services,
    ]
    run(build_cmd, env=docker_env)
    if skip_push:
        print("Skipping docker push because --skip-push was set.")
        return
    push_cmd = [
        "docker",
        "compose",
        "-f",
        str(build_compose),
        "--project-directory",
        str(project_root),
        "push",
        *services,
    ]
    run(push_cmd, env=docker_env)


def verify_readonly_token(
    registry: str,
    username: str,
    api_key: str,
    namespace: str,
    repositories: list[str],
    images: list[str],
    docker_env: dict[str, str],
    *,
    skip_push_checks: bool,
) -> None:
    bearer = dockerhub_login(username, api_key)
    for repository in repositories:
        existing, status = get_repository(namespace, repository, bearer)
        if status != 200:
            raise ExportError(
                f"Read-only API key cannot read Docker Hub repository {namespace}/{repository}."
            )
        if existing.get("permissions", {}).get("write"):
            raise ExportError(
                f"Read-only API key reports write permission for {namespace}/{repository}."
            )

    docker_login_cli(registry, username, api_key, docker_env)
    if skip_push_checks:
        print("Skipping read-only pull/push probes because push checks were disabled.")
        return

    for image in images:
        run(["docker", "manifest", "inspect", image], env=docker_env, capture=True)

    probe = run(["docker", "push", images[0]], env=docker_env, capture=True, check=False)
    combined = f"{probe.stdout or ''}\n{probe.stderr or ''}".lower()
    denied_markers = (
        "denied",
        "unauthorized",
        "forbidden",
        "insufficient_scope",
        "requested access",
        "push access denied",
    )
    if probe.returncode == 0:
        raise ExportError(
            "Read-only API key was able to push an image. Provide a Docker Hub read-only token."
        )
    if not any(marker in combined for marker in denied_markers):
        raise ExportError(
            "Read-only write probe failed for a reason other than an access denial:\n"
            f"{probe.stdout or ''}{probe.stderr or ''}"
        )
    print("Read-only API key can pull image metadata and cannot push.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build miner Docker services, push private images, and export image-only docker-compose.yml."
    )
    parser.add_argument("--project-root", help="Compose root or miner service root to export.")
    parser.add_argument("--compose-file", help="Compose file to use. Defaults to a compose file in --project-root.")
    parser.add_argument("--env-file", default=str(SCRIPT_DIR / ".env"), help="Env file with Docker Hub credentials.")
    parser.add_argument("--output-dir", help="Directory for the exported docker-compose.yml.")
    parser.add_argument("--registry", help="Registry hostname. Docker Hub defaults to docker.io.")
    parser.add_argument("--namespace", help="Docker Hub namespace or organization.")
    parser.add_argument("--image-prefix", help="Repository prefix, for example vidaio-miner.")
    parser.add_argument("--tag", help="Image tag. Defaults to git short SHA plus UTC timestamp.")
    parser.add_argument("--public", action="store_true", help="Create/use public repositories instead of private.")
    parser.add_argument("--yes", action="store_true", help="Use defaults for prompts.")
    parser.add_argument("--skip-build", action="store_true", help="Write compose output without building images.")
    parser.add_argument("--skip-push", action="store_true", help="Build and write compose output without pushing.")
    parser.add_argument(
        "--skip-registry-check",
        action="store_true",
        help="Skip Docker Hub API key verification. Intended only for local/offline compose generation tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_file = load_env_file(Path(args.env_file).expanduser().resolve())

    default_root = SCRIPT_DIR
    project_root = (
        Path(args.project_root).expanduser().resolve()
        if args.project_root
        else default_root if args.yes else prompt_project_root(default_root)
    )
    compose_file = find_compose_file(project_root, args.compose_file)
    actual_compose, actual_project_root, config, services = select_config(project_root, compose_file)
    if not services:
        raise ExportError("No services with local Docker build contexts were found.")

    username = env_value(env_file, "DOCKERHUB_USERNAME") or (getpass.getuser() if args.skip_registry_check else "")
    namespace = args.namespace or env_value(env_file, "DOCKERHUB_NAMESPACE") or username
    registry = args.registry or env_value(env_file, "DOCKER_EXPORT_REGISTRY", "docker.io")
    image_prefix = args.image_prefix or env_value(env_file, "DOCKER_EXPORT_IMAGE_PREFIX", "vidaio-miner")
    tag = args.tag or env_value(env_file, "DOCKER_EXPORT_TAG") or default_tag()
    private = not args.public and env_value(env_file, "DOCKER_EXPORT_VISIBILITY", "private").lower() != "public"

    repositories = {
        service: slugify(f"{image_prefix}-{service}")
        for service in services
    }
    images = {
        service: image_reference(registry, namespace, repository, tag)
        for service, repository in repositories.items()
    }

    env_output_dir = env_value(env_file, "DOCKER_EXPORT_OUTPUT_DIR")
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif env_output_dir:
        output_dir = Path(env_output_dir).expanduser().resolve()
    else:
        relative = project_root.relative_to(SCRIPT_DIR) if project_root.is_relative_to(SCRIPT_DIR) else project_root.name
        slug = slugify(str(relative).replace(os.sep, "-")) if str(relative) != "." else "all"
        output_dir = SCRIPT_DIR / "exported" / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    export_data = export_compose_data(config, services, images, include_build=False)
    final_compose = output_dir / "docker-compose.yml"
    build_compose = output_dir / ".docker-compose.build.yml"
    write_yaml(final_compose, export_data)
    write_yaml(build_compose, export_compose_data(config, services, images, include_build=True))

    print(f"Exporting services from {actual_compose}: {', '.join(services)}")
    print("Image mapping:")
    for service in services:
        print(f"  {service}: {images[service]}")

    with tempfile.TemporaryDirectory(prefix="miner-export-docker-") as docker_config:
        docker_env = os.environ.copy()
        docker_env["DOCKER_CONFIG"] = docker_config

        if not args.skip_registry_check:
            if registry not in {"docker.io", "index.docker.io"}:
                raise ExportError("Docker Hub API key verification currently requires DOCKER_EXPORT_REGISTRY=docker.io.")
            write_key = env_value(env_file, "DOCKERHUB_WRITE_API_KEY")
            read_key = env_value(env_file, "DOCKERHUB_READONLY_API_KEY")
            if (
                not username
                or not write_key
                or not read_key
                or is_placeholder(username)
                or is_placeholder(namespace)
                or is_placeholder(write_key)
                or is_placeholder(read_key)
            ):
                raise ExportError(
                    "Set DOCKERHUB_USERNAME, DOCKERHUB_WRITE_API_KEY, and "
                    "DOCKERHUB_READONLY_API_KEY in miner/.env or the environment."
                )
            print("Verifying Docker Hub write API key.")
            write_bearer = dockerhub_login(username, write_key)
            ensure_repositories(namespace, list(repositories.values()), write_bearer, private)
            docker_login_cli(registry, username, write_key, docker_env)
        else:
            print("Skipping Docker Hub API key verification.")
            write_key = env_value(env_file, "DOCKERHUB_WRITE_API_KEY")
            if write_key:
                docker_login_cli(registry, username, write_key, docker_env)

        build_and_push(
            actual_project_root,
            build_compose,
            services,
            docker_env,
            skip_build=args.skip_build,
            skip_push=args.skip_push,
        )

        if not args.skip_registry_check:
            read_key = env_value(env_file, "DOCKERHUB_READONLY_API_KEY")
            print("Verifying Docker Hub read-only API key.")
            verify_readonly_token(
                registry,
                username,
                read_key,
                namespace,
                list(repositories.values()),
                [images[service] for service in services],
                docker_env,
                skip_push_checks=args.skip_push or args.skip_build,
            )

    build_compose.unlink(missing_ok=True)
    print(f"Exported image-only compose file: {final_compose}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
