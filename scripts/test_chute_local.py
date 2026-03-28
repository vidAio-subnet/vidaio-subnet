#!/usr/bin/env python3
"""
Test a Vidaio chute — either locally or against a deployed Chute on chutes.ai.

Local mode (default):
    python scripts/test_chute_local.py \
        --task compression \
        --video-url https://example.com/video.mp4

    python scripts/test_chute_local.py \
        --task upscaling \
        --video-url https://example.com/video.mp4 \
        --no-s3

Remote mode (deployed Chute):
    python scripts/test_chute_local.py \
        --task compression \
        --video-url https://example.com/video.mp4 \
        --remote

Environment variables (S3 — local mode with upload, or remote mode):
    BUCKET_TYPE                 backblaze | amazon_s3 | cloudflare | hippius
    BUCKET_COMPATIBLE_ENDPOINT  e.g. s3.us-west-004.backblazeb2.com
    BUCKET_COMPATIBLE_ACCESS_KEY
    BUCKET_COMPATIBLE_SECRET_KEY
    BUCKET_NAME

Environment variables (remote mode):
    CHUTES_API_KEY              API key for chutes.ai
    CHUTE_SLUG                  Chute slug (e.g. "username/my-compression-chute")
"""

import argparse
import base64
import datetime
import json
import os
import sys
import uuid

import requests


# ---------------------------------------------------------------------------
# S3 presigned URL helpers
# ---------------------------------------------------------------------------

def get_s3_clients(bucket_type: str, endpoint: str, access_key: str, secret_key: str, bucket_name: str):
    """Return (presigned_put_fn, presigned_get_fn) callables for the given backend."""

    expires = 3600  # 1 hour for testing

    if bucket_type in ("backblaze", "hippius"):
        from minio import Minio

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=True,
        )

        def put_url(obj: str) -> str:
            return client.presigned_put_object(bucket_name, obj, datetime.timedelta(seconds=expires))

        def get_url(obj: str) -> str:
            return client.presigned_get_object(bucket_name, obj, datetime.timedelta(seconds=expires))

        return put_url, get_url

    else:  # amazon_s3, cloudflare
        import boto3
        from botocore.config import Config

        kwargs = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "config": Config(signature_version="s3v4"),
        }
        if bucket_type == "cloudflare":
            kwargs["endpoint_url"] = endpoint if endpoint.startswith("http") else f"https://{endpoint}"
            kwargs["region_name"] = "auto"
        else:
            kwargs["region_name"] = "us-east-1"

        client = boto3.client("s3", **kwargs)

        def put_url(obj: str) -> str:
            return client.generate_presigned_url(
                "put_object",
                Params={"Bucket": bucket_name, "Key": obj},
                ExpiresIn=expires,
            )

        def get_url(obj: str) -> str:
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": obj},
                ExpiresIn=expires,
            )

        return put_url, get_url


def require_s3_env():
    """Read and validate S3 env vars, return the tuple or exit."""
    bucket_type = os.environ.get("BUCKET_TYPE", "")
    endpoint = os.environ.get("BUCKET_COMPATIBLE_ENDPOINT", "")
    access_key = os.environ.get("BUCKET_COMPATIBLE_ACCESS_KEY", "")
    secret_key = os.environ.get("BUCKET_COMPATIBLE_SECRET_KEY", "")
    bucket_name = os.environ.get("BUCKET_NAME", "")

    if not all([bucket_type, endpoint, access_key, secret_key, bucket_name]):
        print("S3 env vars not set. Required:")
        print("  BUCKET_TYPE, BUCKET_COMPATIBLE_ENDPOINT, BUCKET_COMPATIBLE_ACCESS_KEY,")
        print("  BUCKET_COMPATIBLE_SECRET_KEY, BUCKET_NAME")
        sys.exit(1)

    return bucket_type, endpoint, access_key, secret_key, bucket_name


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def build_payload(task: str, video_url: str, upload_url: str) -> dict:
    if task == "compression":
        return {
            "video_url": video_url,
            "vmaf_threshold": 90.0,
            "target_codec": "h264",
            "codec_mode": "CRF",
            "target_bitrate": 10.0,
            "upload_url": upload_url,
        }
    else:
        return {
            "video_url": video_url,
            "task_type": "SD2HD",
            "upload_url": upload_url,
        }


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def display_result(result: dict, task: str, get_url_fn, object_name: str | None) -> None:
    print(f"\nResult:")
    print(f"  success: {result.get('success')}")

    if not result.get("success"):
        print(f"  error: {result.get('error')}")
        sys.exit(1)

    if result.get("output_url") and get_url_fn and object_name:
        download_url = get_url_fn(object_name)
        print(f"  output uploaded to S3")
        print(f"\n  Download URL (presigned, 1h expiry):")
        print(f"  {download_url}")
    elif result.get("output_video_b64"):
        out_path = f"test_output_{task}.mp4"
        video_bytes = base64.b64decode(result["output_video_b64"])
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        print(f"  No S3 upload -- saved result to: {out_path} ({len(video_bytes)} bytes)")
    else:
        print(f"  output_url: {result.get('output_url')}")
        print("  (no presigned GET URL generated)")


# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------

def run_local(args) -> None:
    base_url = f"http://localhost:{args.port}"

    # Health check
    print(f"Checking health at {base_url}/health ...")
    try:
        resp = requests.post(f"{base_url}/health", json={}, timeout=10)
        resp.raise_for_status()
        health = resp.json()
        print(f"  Status: {health.get('status')}")
        print(f"  Miner:  {health.get('miner')}")
        if health.get("status") != "healthy":
            print("  WARNING: chute is not healthy, proceeding anyway...")
    except Exception as e:
        print(f"  Health check failed: {e}")
        print("  Is the chute running? Proceeding anyway...")

    # Generate presigned URLs
    object_name = None
    upload_url = ""
    get_url_fn = None

    if not args.no_s3:
        bucket_type, endpoint, access_key, secret_key, bucket_name = require_s3_env()
        suffix = "compressed" if args.task == "compression" else "upscaled"
        object_name = f"test_{uuid.uuid4()}_{suffix}.mp4"

        print(f"Generating presigned URLs for '{object_name}' ...")
        put_url_fn, get_url_fn = get_s3_clients(bucket_type, endpoint, access_key, secret_key, bucket_name)
        upload_url = put_url_fn(object_name)
        print(f"  PUT URL generated: {upload_url}")

    if args.only_put_url:
        print("Only generating presigned URLs, exiting...")
        sys.exit(0)
    # Send /process request
    payload = build_payload(args.task, args.video_url, upload_url)
    print(f"\nSending /process request ({args.task}) ...")
    print(f"  video_url: {args.video_url}")
    if args.task == "compression":
        print(f"  codec: {payload['target_codec']}, mode: {payload['codec_mode']}, vmaf: {payload['vmaf_threshold']}")
    else:
        print(f"  task_type: {payload['task_type']}")

    try:
        resp = requests.post(f"{base_url}/process", json=payload, timeout=600)
        resp.raise_for_status()
        result = resp.json()
    except requests.Timeout:
        print("ERROR: Request timed out (600s)")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Request failed: {e}")
        sys.exit(1)

    display_result(result, args.task, get_url_fn, object_name)


# ---------------------------------------------------------------------------
# Remote mode — deployed Chute on chutes.ai
# ---------------------------------------------------------------------------

def run_remote(args) -> None:
    chutes_api_key = os.environ.get("CHUTES_API_KEY", "")
    chute_slug = os.environ.get("CHUTE_SLUG", "")

    if not chutes_api_key:
        print("ERROR: CHUTES_API_KEY env var is required for --remote mode")
        sys.exit(1)
    if not chute_slug:
        print("ERROR: CHUTE_SLUG env var is required for --remote mode")
        sys.exit(1)

    base_url = f"https://{chute_slug}.chutes.ai"

    # S3 is required for remote — the chute uploads the result there
    bucket_type, endpoint, access_key, secret_key, bucket_name = require_s3_env()
    suffix = "compressed" if args.task == "compression" else "upscaled"
    object_name = f"test_{uuid.uuid4()}_{suffix}.mp4"

    print(f"Generating presigned URLs for '{object_name}' ...")
    put_url_fn, get_url_fn = get_s3_clients(bucket_type, endpoint, access_key, secret_key, bucket_name)
    upload_url = put_url_fn(object_name)
    print(f"  PUT URL generated: {upload_url}")

    # Build and send request
    payload = build_payload(args.task, args.video_url, upload_url)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {chutes_api_key}",
    }

    print(f"\nSending /process request to deployed chute ...")
    print(f"  url:  {base_url}/process")
    print(f"  slug: {chute_slug}")
    print(f"  video_url: {args.video_url}")
    if args.task == "compression":
        print(f"  codec: {payload['target_codec']}, mode: {payload['codec_mode']}, vmaf: {payload['vmaf_threshold']}")
    else:
        print(f"  task_type: {payload['task_type']}")

    try:
        resp = requests.post(
            f"{base_url}/process",
            json=payload,
            headers=headers,
            timeout=600,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.Timeout:
        print("ERROR: Request timed out (600s)")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"ERROR: HTTP {e.response.status_code}")
        try:
            print(f"  body: {e.response.text[:500]}")
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Request failed: {e}")
        sys.exit(1)

    display_result(result, args.task, get_url_fn, object_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test a Vidaio chute (local or remote)")
    parser.add_argument("--task", required=True, choices=["compression", "upscaling"])
    parser.add_argument("--video-url", required=True, help="Public URL of the test video")
    parser.add_argument("--remote", action="store_true",
                        help="Query a deployed Chute on chutes.ai instead of localhost")
    parser.add_argument("--port", type=int, default=8000,
                        help="Local chute port (default: 8000, ignored with --remote)")
    parser.add_argument("--no-s3", action="store_true",
                        help="Skip S3, let chute return base64 inline (local mode only)")
    parser.add_argument("--only-put-url", action="store_true",
                        help="Only print the S3 upload URL (local mode only)")
    args = parser.parse_args()

    if args.remote:
        if args.no_s3:
            print("WARNING: --no-s3 is ignored in remote mode (S3 upload is required)")
        run_remote(args)
    else:
        run_local(args)


if __name__ == "__main__":
    main()
