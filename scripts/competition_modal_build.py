#!/usr/bin/env python3
"""Run one Modal contender image build in a killable client process."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path


def _dev_mode_enabled() -> bool:
    return os.getenv("DEV_MODE", "False").strip().lower() == "true"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    import modal

    app = modal.App.lookup(
        args.app_name,
        environment_name=args.environment,
        create_if_missing=True,
    )
    image = modal.Image.from_dockerfile(
        args.dockerfile,
        context_dir=args.context_dir,
        secrets=[],
    )
    output_context = nullcontext()
    if _dev_mode_enabled():
        enable_output = getattr(modal, "enable_output", None)
        if callable(enable_output):
            output_context = enable_output()
    with output_context:
        built_image = image.build(app)
    image_id = str(
        getattr(built_image, "object_id", "") or getattr(image, "object_id", "")
    )
    if not image_id:
        raise RuntimeError("Modal returned no immutable image ID")
    args.result.write_text(json.dumps({"image_id": image_id}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
