from __future__ import annotations

import argparse
import json

from .gr_potential_tangent_c import build_c_backend, load_c_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and ABI-check the compiled GR tangent callback.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when source/header metadata match.")
    parser.add_argument("--cc", default=None, help="C compiler executable (defaults to CC or cc).")
    args = parser.parse_args()
    metadata = build_c_backend(force=args.force, compiler=args.cc)
    backend = load_c_backend()
    print(json.dumps({"build": metadata, "abi": backend.abi_metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
