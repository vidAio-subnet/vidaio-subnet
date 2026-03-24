#!/usr/bin/env python3
"""
Deploy a Vidaio miner chute to the Chutes platform.

  1. Upload miner files (miner.py + chute_config.yml) to a HuggingFace repo
  2. Render the chute template with user-specific variables
  3. Build and deploy the chute via the `chutes` CLI
  4. Optionally warm up the deployed chute

Usage:
    python scripts/deploy_chute.py \
        --task compression \
        --hf-username your-hf-username \
        --hf-token hf_xxx \
        --chutes-api-key cpk_xxx \
        --chutes-username your-chutes-username \
        --model-path example_miners/compression

    # Redeploy existing HF repo (skip upload)
    python scripts/deploy_chute.py \
        --task upscaling \
        --hf-username your-hf-username \
        --hf-token hf_xxx \
        --chutes-api-key cpk_xxx \
        --chutes-username your-chutes-username \
        --no-upload

    # Skip warmup
    python scripts/deploy_chute.py ... --no-warmup
"""

import argparse
import asyncio
import os
import sys
import tempfile
from asyncio import create_subprocess_exec, subprocess
from pathlib import Path

from huggingface_hub import HfApi
from jinja2 import Template


REPO_ROOT = Path(__file__).resolve().parent.parent

CHUTE_TEMPLATES = {
    "compression": REPO_ROOT / "chutes" / "compression" / "vidaio_compression_chute.py.j2",
    "upscaling": REPO_ROOT / "chutes" / "upscaling" / "vidaio_upscaling_chute.py.j2",
}


# ── HuggingFace helpers ──────────────────────────────────────────────


def get_hf_repo_name(hf_username: str, task: str) -> str:
    return f"{hf_username}/vidaio-{task}"


def get_uploadable_files(path_dir: Path) -> list[Path]:
    paths = []
    for p in path_dir.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(path_dir).parts):
            continue
        if p.name.endswith(".lock"):
            continue
        paths.append(p)
    return paths


async def upload_to_hf(
    hf_api: HfApi,
    repo_name: str,
    model_path: Path,
    concurrency: int = 4,
) -> str:
    hf_api.create_repo(repo_id=repo_name, repo_type="model", private=True, exist_ok=True)

    files = get_uploadable_files(model_path)
    if not files:
        raise ValueError(f"No files found in {model_path}")

    print(f"  uploading {len(files)} files in a single commit...")
    await asyncio.to_thread(
        lambda: hf_api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            repo_type="model",
            commit_message="vidaio: upload miner files",
        )
    )
    for p in files:
        print(f"  uploaded {p.relative_to(model_path)}")

    # Make repo public so the chute can access it
    try:
        hf_api.update_repo_settings(repo_id=repo_name, repo_type="model", private=False)
    except Exception as e:
        print(f"  warning: could not make repo public: {e}")

    info = hf_api.repo_info(repo_id=repo_name, repo_type="model")
    revision = getattr(info, "sha", "") or ""
    print(f"  HF revision: {revision}")
    return revision


# ── Chute helpers ────────────────────────────────────────────────────


def get_chute_name(hf_username: str, task: str) -> str:
    return f"vidaio-{hf_username.replace('/', '-')}-{task}".lower()


def render_template(
    task: str,
    hf_repo_name: str,
    hf_revision: str,
    chutes_username: str,
    chute_name: str,
) -> str:
    template_path = CHUTE_TEMPLATES[task]
    template = Template(template_path.read_text())
    return template.render(
        huggingface_repository_name=hf_repo_name,
        huggingface_repository_revision=hf_revision,
        chute_username=chutes_username,
        chute_name=chute_name,
    )


async def run_chutes_cli(args: list[str], chutes_api_key: str, cwd: str | None = None) -> None:
    env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
    proc = await create_subprocess_exec(
        *args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    if proc.stdin:
        proc.stdin.write(b"y\n")
        await proc.stdin.drain()
        proc.stdin.close()

    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(f"  [{args[0]}] {line.decode(errors='ignore').rstrip()}")

    returncode = await proc.wait()
    if returncode != 0:
        raise RuntimeError(f"Command failed (exit {returncode}): {' '.join(args)}")


async def build_and_deploy(script_path: Path, chutes_api_key: str) -> None:
    print("Building chute...")
    await run_chutes_cli(
        ["chutes", "build", f"{script_path.stem}:chute", 
        #  "--public",   # keep chute private
         "--wait", 
         "--debug",
         "--include-cwd"
        ],
        chutes_api_key=chutes_api_key,
        cwd=str(script_path.parent),
    )
    print("Deploying chute...")
    await run_chutes_cli(
        ["chutes", "deploy", f"{script_path.stem}:chute", "--accept-fee", "--debug"],
        chutes_api_key=chutes_api_key,
        cwd=str(script_path.parent),
    )


async def warmup(chute_name: str, chutes_api_key: str) -> None:
    print(f"Warming up chute '{chute_name}'...")
    await run_chutes_cli(
        ["chutes", "warmup", chute_name],
        chutes_api_key=chutes_api_key,
    )


# ── Main ─────────────────────────────────────────────────────────────


async def deploy(args: argparse.Namespace) -> None:
    task = args.task
    hf_username = args.hf_username
    hf_token = args.hf_token
    chutes_api_key = args.chutes_api_key
    chutes_username = args.chutes_username

    hf_repo_name = get_hf_repo_name(hf_username, task)
    chute_name = get_chute_name(hf_username, task)
    hf_api = HfApi(token=hf_token)

    # 1. Upload to HuggingFace
    if not args.no_upload:
        model_path = Path(args.model_path).resolve()
        if not model_path.is_dir():
            print(f"Error: --model-path '{model_path}' is not a directory")
            sys.exit(1)
        print(f"Uploading {model_path} to HF repo '{hf_repo_name}'...")
        hf_revision = await upload_to_hf(hf_api, hf_repo_name, model_path)
    else:
        info = hf_api.repo_info(repo_id=hf_repo_name, repo_type="model")
        hf_revision = getattr(info, "sha", "") or ""
        print(f"Using existing HF repo '{hf_repo_name}' at revision {hf_revision}")

    # 2. Render template and deploy
    print(f"Rendering {task} chute template...")
    rendered = render_template(
        task=task,
        hf_repo_name=hf_repo_name,
        hf_revision=hf_revision,
        chutes_username=chutes_username,
        chute_name=chute_name,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"vidaio_{task}_", delete=False, dir=REPO_ROOT / "chutes"
    ) as f:
        f.write(rendered)
        script_path = Path(f.name)

    try:
        await build_and_deploy(script_path, chutes_api_key)
    finally:
        script_path.unlink(missing_ok=True)

    # 3. Warmup
    if not args.no_warmup:
        await warmup(chute_name, chutes_api_key)

    # 4. Print slug for env config
    slug = f"{chutes_username.replace('_', '-')}-{chute_name}"
    print()
    print("=" * 60)
    print(f"Chute deployed: {chute_name}")
    print(f"Expected slug:  {slug}")
    print()
    if task == "compression":
        print(f'  export CHUTES__COMPRESSION_SLUG="{slug}"')
    else:
        print(f'  export CHUTES__UPSCALING_SLUG="{slug}"')
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a Vidaio miner chute")
    parser.add_argument("--task", required=True, choices=["compression", "upscaling"])
    parser.add_argument("--hf-username", required=True, help="HuggingFace username")
    parser.add_argument("--hf-token", required=True, help="HuggingFace API token")
    parser.add_argument("--chutes-api-key", required=True, help="Chutes API key")
    parser.add_argument("--chutes-username", required=True, help="Chutes username")
    parser.add_argument("--model-path", default=None, help="Path to miner directory (miner.py + chute_config.yml)")
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload, use existing repo")
    parser.add_argument("--no-warmup", action="store_true", help="Skip chute warmup after deploy")
    args = parser.parse_args()

    if not args.no_upload and not args.model_path:
        parser.error("--model-path is required unless --no-upload is set")

    asyncio.run(deploy(args))


if __name__ == "__main__":
    main()
