"""Narrow compatibility patches for the project's pinned Bittensor stack."""

from __future__ import annotations

from functools import wraps
import json
from typing import Any


_PATCH_MARKER = "_vidaio_single_field_composite_patch"
_RUNTIME_CALL_PATCH_MARKER = "_vidaio_single_field_composite_decode_patch"
_COMPOSITE_ERROR_PARTS = ("Invalid type for data:", "type_def: Composite(")


def _is_single_unnamed_field_composite(type_string: str, runtime: Any) -> bool:
    """Return whether ``type_string`` is an unnamed one-field SCALE composite."""
    if not type_string.startswith("scale_info::"):
        return False

    try:
        type_id = int(type_string.removeprefix("scale_info::"))
        registry = json.loads(runtime.registry.registry)
        type_entry = next(
            entry for entry in registry["types"] if entry["id"] == type_id
        )
        fields = type_entry["type"]["def"]["composite"]["fields"]
    except (AttributeError, KeyError, TypeError, ValueError, StopIteration):
        return False

    return len(fields) == 1 and fields[0].get("name") is None


def _unwrap_singleton_tuples(value: Any) -> Any:
    """Recursively unwrap scalar Rust newtypes decoded as one-element tuples.

    Tuple-backed collections and account ID wrappers can also contain one
    element, so containers are retained when their only item is another
    tuple, list, or dict.
    """
    if isinstance(value, tuple):
        normalized = tuple(_unwrap_singleton_tuples(item) for item in value)
        if len(normalized) == 1 and not isinstance(
            normalized[0], (tuple, list, dict)
        ):
            return normalized[0]
        return normalized
    if isinstance(value, list):
        return [_unwrap_singleton_tuples(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _unwrap_singleton_tuples(item) for key, item in value.items()
        }
    return value


def _normalize_runtime_call_result(result: Any) -> Any:
    if hasattr(result, "value"):
        result.value = _unwrap_singleton_tuples(result.value)
    return result


def apply_single_field_composite_encoding_patch() -> bool:
    """Teach async-substrate-interface 1.x to handle Rust newtype parameters.

    Some newer Subtensor runtime metadata represents values such as ``NetUid``
    as tuple/newtype composites with one unnamed field.  The 1.x bt-decode
    encoder rejects the scalar value passed by Bittensor 10.1.0, and the
    decoder returns nested newtypes as singleton tuples. Their wire
    representation is just that of the inner value.

    The retry is deliberately limited to the exact composite type error and a
    metadata-confirmed one-field unnamed composite.  Other SCALE errors retain
    their original behaviour.

    Returns ``True`` when the patch is installed and ``False`` if it was
    already installed.
    """
    from async_substrate_interface.async_substrate import AsyncSubstrateInterface
    from async_substrate_interface.sync_substrate import SubstrateInterface
    from async_substrate_interface.types import SubstrateMixin

    original_encode_scale = SubstrateMixin._encode_scale
    if getattr(original_encode_scale, _PATCH_MARKER, False):
        encode_scale_with_newtype_retry = original_encode_scale
        patch_installed = False
    else:
        @wraps(original_encode_scale)
        def encode_scale_with_newtype_retry(
            self: Any,
            type_string: str,
            value: Any,
            runtime: Any = None,
        ) -> bytes:
            try:
                return original_encode_scale(
                    self, type_string, value, runtime=runtime
                )
            except ValueError as error:
                is_composite_type_error = all(
                    part in str(error) for part in _COMPOSITE_ERROR_PARTS
                )
                is_unwrapped_value = not isinstance(value, (list, tuple, dict))
                if (
                    runtime is not None
                    and is_composite_type_error
                    and is_unwrapped_value
                    and _is_single_unnamed_field_composite(type_string, runtime)
                ):
                    return original_encode_scale(
                        self, type_string, (value,), runtime=runtime
                    )
                raise

        setattr(encode_scale_with_newtype_retry, _PATCH_MARKER, True)
        SubstrateMixin._encode_scale = encode_scale_with_newtype_retry
        patch_installed = True

    # async-substrate-interface 1.x copies the mixin method onto the sync
    # class at import time rather than resolving it through inheritance.
    # Replace that stale alias as well; runtime_call invokes this public name.
    if SubstrateInterface.encode_scale is not encode_scale_with_newtype_retry:
        SubstrateInterface.encode_scale = encode_scale_with_newtype_retry
        patch_installed = True

    sync_runtime_call = SubstrateInterface.runtime_call
    if not getattr(sync_runtime_call, _RUNTIME_CALL_PATCH_MARKER, False):
        @wraps(sync_runtime_call)
        def runtime_call_with_newtype_normalization(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            result = sync_runtime_call(self, *args, **kwargs)
            return _normalize_runtime_call_result(result)

        setattr(
            runtime_call_with_newtype_normalization,
            _RUNTIME_CALL_PATCH_MARKER,
            True,
        )
        SubstrateInterface.runtime_call = runtime_call_with_newtype_normalization
        patch_installed = True

    async_runtime_call = AsyncSubstrateInterface.runtime_call
    if not getattr(async_runtime_call, _RUNTIME_CALL_PATCH_MARKER, False):
        @wraps(async_runtime_call)
        async def async_runtime_call_with_newtype_normalization(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            result = await async_runtime_call(self, *args, **kwargs)
            return _normalize_runtime_call_result(result)

        setattr(
            async_runtime_call_with_newtype_normalization,
            _RUNTIME_CALL_PATCH_MARKER,
            True,
        )
        AsyncSubstrateInterface.runtime_call = (
            async_runtime_call_with_newtype_normalization
        )
        patch_installed = True

    return patch_installed