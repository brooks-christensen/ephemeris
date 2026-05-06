from __future__ import annotations

import argparse

from .american_ephemeris import generate_jpl_book_style_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate American Ephemeris-style JPL apparent geocentric longitude tables."
    )
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    generate_jpl_book_style_csv(
        kernel_path=args.kernel_path,
        year=args.year,
        month=args.month,
        output_path=args.output,
    )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()