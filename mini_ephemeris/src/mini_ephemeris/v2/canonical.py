"""Deterministic canonical encoding used by v2 identities and fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from .errors import V2FoundationError


def finite_binary64_hex(value: float, field: str) -> str:
    """Return an exact, locale-independent binary64 encoding for a finite value."""

    number = float(value)
    if not math.isfinite(number):
        raise V2FoundationError(f"{field} must be finite")
    return number.hex()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode a canonical payload as sorted compact UTF-8 JSON.

    Material floating-point values must already be represented by exact hex
    strings. This boundary rejects NaN and infinity and never depends on mapping
    insertion order, locale, object identity, or process hash randomization.
    """

    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V2FoundationError("payload is not canonically serializable") from exc
    return text.encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact bytes."""

    return hashlib.sha256(payload).hexdigest()
