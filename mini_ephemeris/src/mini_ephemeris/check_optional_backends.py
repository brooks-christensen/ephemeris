from __future__ import annotations

import importlib
import json
from typing import Any


BACKENDS = ("rebound", "reboundx", "numba", "cupy")


def backend_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {
            "name": name,
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    version = getattr(module, "__version__", None)
    if version is None and name == "rebound":
        version = getattr(module, "version", None)
    return {
        "name": name,
        "available": True,
        "version": str(version) if version is not None else "unknown",
        "error": None,
    }


def main() -> None:
    statuses = [backend_status(name) for name in BACKENDS]
    for status in statuses:
        if status["available"]:
            print(f"{status['name']}: available (version {status['version']})")
        else:
            print(f"{status['name']}: not installed ({status['error']})")
    print(json.dumps({"optional_backends": statuses}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
