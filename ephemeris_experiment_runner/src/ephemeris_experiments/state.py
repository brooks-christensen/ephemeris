from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def default_state() -> dict[str, Any]:
    return {"stages": {}, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_state()
    try:
        return json.loads(path.read_text())
    except Exception:
        return default_state()


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def load_approvals(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        return set(data.get("approved_stages", []))
    except Exception:
        return set()


def save_approvals(path: Path, approvals: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"approved_stages": sorted(approvals)}, indent=2) + "\n")
