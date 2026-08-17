"""Locked public Modal Sandbox rates and allocated-resource cost estimates."""

from __future__ import annotations

import re
from decimal import Decimal


# https://modal.com/pricing, fetched 2026-07-21. GPU rates are the standard
# per-GPU rates. Sandboxes have a distinct CPU rate from normal Modal Functions.
GPU_PRICE_PER_SECOND_USD = {
    "B300": Decimal("0.001972"),
    "B200": Decimal("0.001736"),
    "H200": Decimal("0.001261"),
    "H100": Decimal("0.001097"),
    "RTX-PRO-6000": Decimal("0.000842"),
    "A100-80GB": Decimal("0.000694"),
    "A100-40GB": Decimal("0.000583"),
    "L40S": Decimal("0.000542"),
    "A10": Decimal("0.000306"),
    "L4": Decimal("0.000222"),
    "T4": Decimal("0.000164"),
}
SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD = Decimal("0.00003942")
COST_ATTRIBUTION_METHOD = (
    "MODAL_PUBLIC_PRICE_2026-07-21_ACTUAL_RESOURCES_BATCH_WALL_EQUAL_SHARE"
)


def canonical_modal_gpu_type(nvidia_smi_name: str) -> str:
    """Map an allocated NVIDIA device name to Modal's public pricing SKU."""

    normalized = re.sub(r"[^A-Z0-9]+", "-", nvidia_smi_name.upper()).strip("-")
    if "RTX-PRO-6000" in normalized:
        return "RTX-PRO-6000"
    if "A100" in normalized:
        if re.search(r"(?:^|-)80-?GB(?:-|$)", normalized):
            return "A100-80GB"
        if re.search(r"(?:^|-)40-?GB(?:-|$)", normalized):
            return "A100-40GB"
        raise ValueError(
            f"allocated A100 memory size is not identifiable: {nvidia_smi_name!r}"
        )
    for sku in ("B300", "B200", "H200", "H100", "L40S", "A10", "L4", "T4"):
        if re.search(rf"(?:^|-){re.escape(sku)}(?:-|$)", normalized):
            return sku
    raise ValueError(f"allocated GPU has no locked Modal rate: {nvidia_smi_name!r}")


def estimate_sandbox_cost(
    *,
    allocated_gpu_type: str,
    allocated_gpu_count: int,
    allocated_cpu_cores: float,
    runtime_seconds: float,
) -> Decimal:
    """Estimate runtime cost exclusively from hardware observed in the Sandbox."""

    gpu_rate = GPU_PRICE_PER_SECOND_USD.get(allocated_gpu_type)
    if gpu_rate is None:
        raise ValueError(
            f"no locked Modal cost rate for allocated GPU {allocated_gpu_type!r}"
        )
    if allocated_gpu_count <= 0:
        raise ValueError("allocated GPU count must be positive")
    if allocated_cpu_cores <= 0 or runtime_seconds < 0:
        raise ValueError(
            "allocated CPU cores must be positive and runtime must be nonnegative"
        )
    per_second = (gpu_rate * allocated_gpu_count) + (
        SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD
        * Decimal(str(allocated_cpu_cores))
    )
    return per_second * Decimal(str(runtime_seconds))


__all__ = [
    "COST_ATTRIBUTION_METHOD",
    "GPU_PRICE_PER_SECOND_USD",
    "SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD",
    "canonical_modal_gpu_type",
    "estimate_sandbox_cost",
]
