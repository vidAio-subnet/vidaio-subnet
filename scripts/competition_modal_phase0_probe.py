#!/usr/bin/env python3
"""Run the live Modal network-isolation and billing-capability Phase 0 probe.

This creates one short-lived CPU Sandbox with no secrets and no GPU.  It is
intentionally separate from the local gate runner so CI can run without Modal
credentials or cloud spend.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timedelta, timezone


REMOTE_PROBE = r'''
import json
import os
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

checks["modal_account_credentials_absent"] = not any(
    os.getenv(name)
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_IDENTITY_TOKEN")
)
checks["network_blocked"] = (
    checks["direct_ip_blocked"]
    and checks["dns_lookup_blocked"]
    and checks["https_blocked"]
)
print(json.dumps(checks, sort_keys=True), flush=True)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="dev")
    parser.add_argument("--app-name", default="vidaio-competition-phase0")
    parser.add_argument(
        "--skip-billing",
        action="store_true",
        help="Skip the workspace billing API capability request",
    )
    return parser.parse_args()


def _credentials_present() -> bool:
    return bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")) or os.path.isfile(
        os.path.expanduser("~/.modal.toml")
    )


def _billing_probe(modal) -> dict:
    now = datetime.now(timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=2)
    try:
        workspace = modal.Workspace.from_context()
        rows = workspace.billing.report(
            start=start,
            end=end,
            resolution="h",
            tag_names=["competition_id", "hotkey"],
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
            "note": "Billing reports require a supported Team/Enterprise workspace.",
        }

    resource_types = sorted(
        {
            resource
            for row in rows
            for resource in getattr(row, "cost_by_resource", {}).keys()
        }
    )
    return {
        "status": "passed",
        "row_count": len(rows),
        "resolution": "hourly",
        "resource_types": resource_types,
        "native_per_input_cost": False,
    }


def main() -> int:
    args = parse_args()
    if not _credentials_present():
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "Modal credentials are absent",
                    "required_environment_variables": [
                        "MODAL_TOKEN_ID",
                        "MODAL_TOKEN_SECRET",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    os.environ["MODAL_ENVIRONMENT"] = args.environment
    try:
        import modal
    except ImportError:
        print(json.dumps({"status": "blocked", "reason": "modal SDK is not installed"}))
        return 2

    network_result: dict = {}
    try:
        app = modal.App(args.app_name)
        with app.run(environment_name=args.environment):
            sandbox = None
            try:
                image = modal.Image.debian_slim(python_version="3.12")
                sandbox = modal.Sandbox.create(
                    "python3",
                    "-c",
                    REMOTE_PROBE,
                    app=app,
                    image=image,
                    name=f"offline-network-probe-{uuid.uuid4().hex[:8]}",
                    tags={
                        "competition_id": "phase0",
                        "hotkey": "probe-only",
                        "purpose": "network-isolation",
                    },
                    cpu=(0.125, 0.25),
                    memory=(128, 256),
                    timeout=60,
                    block_network=True,
                    include_oidc_identity_token=False,
                )
                stdout = sandbox.stdout.read()
                sandbox.wait()
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                output_lines = [
                    line for line in str(stdout).splitlines() if line.strip()
                ]
                if not output_lines:
                    raise RuntimeError("Modal Sandbox returned no probe output")
                network_result = json.loads(output_lines[-1])
            finally:
                if sandbox is not None:
                    try:
                        sandbox.terminate(wait=True)
                    except Exception:
                        pass
                    try:
                        sandbox.detach()
                    except Exception:
                        pass
    except Exception as exc:
        network_result = {
            "network_blocked": False,
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        }

    billing_result = (
        {"status": "skipped"} if args.skip_billing else _billing_probe(modal)
    )
    network_passed = bool(
        network_result.get("network_blocked")
        and network_result.get("modal_account_credentials_absent")
    )
    result = {
        "status": "passed" if network_passed else "failed",
        "environment": args.environment,
        "modal_sdk_version": getattr(modal, "__version__", "unknown"),
        "network_isolation": network_result,
        "billing": billing_result,
        "image_size_api": {
            "status": "blocked",
            "reason": "Public Image API exposes an ID but no enforceable image-size quota/measurement.",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if network_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
