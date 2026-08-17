#!/usr/bin/env bash
set -euo pipefail


profile="${1:-profile}"
volume="${2:-vidaio-compression-comp-inp-volume}"
destination="${3:-./output}"
source="${4:-./batches/000010/inputs}"

command -v modal >/dev/null || {
  echo "Modal CLI not found. Install it with: pip install modal" >&2
  exit 1
}

mkdir -p "$destination"
MODAL_PROFILE="$profile" modal volume get \
  --env dev --force "$volume" "$source" "$destination"