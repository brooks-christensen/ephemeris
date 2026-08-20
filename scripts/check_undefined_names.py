#!/usr/bin/env python3
"""Report names read inside a function that resolve to nothing at module level
and are not builtins -- i.e. NameError candidates at runtime.

WHY THIS EXISTS
    `py_compile` passing says nothing about names: Python does not resolve them
    until execution, so a NameError compiles cleanly. One was shipped into this
    repository that way (an out-of-scope `invariant_extrema` in
    long_term_stability_cli). Run this before handing over any edit.

A NOTE ON THE IMPLEMENTATION
    In `symtable`, an unbound name referenced inside a function is classified as
    an implicit GLOBAL. The first version of this checker excluded globals, and
    therefore reported a deliberate NameError as clean. Its --self-test mode
    exists so that never silently happens again.

USAGE
    python3 scripts/check_undefined_names.py --self-test
    python3 scripts/check_undefined_names.py mini_ephemeris/src/mini_ephemeris/*.py

Exit code 0 if clean, 1 if any suspect name is found.
"""

from __future__ import annotations

import builtins
import symtable
import sys
import tempfile
from pathlib import Path


def _module_bindings(table: symtable.SymbolTable) -> set[str]:
    return {
        sym.get_name()
        for sym in table.get_symbols()
        if sym.is_assigned() or sym.is_imported() or sym.is_namespace()
    }


def check(path: str) -> list[tuple[str, str, int]]:
    source = Path(path).read_text(encoding="utf-8")
    top = symtable.symtable(source, path, "exec")
    bound = _module_bindings(top)
    known_builtins = set(dir(builtins))
    problems: list[tuple[str, str, int]] = []

    def walk(table: symtable.SymbolTable, enclosing: set[str]) -> None:
        for child in table.get_children():
            if child.get_type() != "function":
                walk(child, enclosing)
                continue
            local = {
                s.get_name()
                for s in child.get_symbols()
                if s.is_local() or s.is_parameter() or s.is_imported()
            }
            for sym in child.get_symbols():
                name = sym.get_name()
                if not sym.is_referenced():
                    continue
                if name in local or name in enclosing or sym.is_free():
                    continue
                if name in bound or name in known_builtins:
                    continue
                problems.append((child.get_name(), name, child.get_lineno()))
            walk(child, enclosing | local)

    walk(top, set())
    return problems


def _self_test() -> int:
    """Confirm the checker catches known failures before any result is trusted."""
    cases = [
        ("catches a bare undefined name",
         "def f():\n    return undefined_thing + 1\n", True),
        ("catches an out-of-scope parameter (the bug that shipped)",
         'def g(a):\n    return invariant_extrema.get("x")\n', True),
        ("passes legitimate code",
         "import math\nK = 3\ndef h(x):\n    y = x + K\n    return math.sqrt(y)\n", False),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, source, should_flag in cases:
            path = Path(tmp) / "case.py"
            path.write_text(source, encoding="utf-8")
            flagged = bool(check(str(path)))
            ok = flagged == should_flag
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")
            failures += not ok
    if failures:
        print("\nSELF-TEST FAILED. Do not trust this checker's output.")
        return 1
    print("\nSelf-test passed; results from this checker can be trusted.")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return _self_test()
    if not argv:
        print(__doc__)
        return 2
    total = 0
    for path in argv:
        problems = check(path)
        total += len(problems)
        name = Path(path).name
        if problems:
            print(f"  {name}: {len(problems)} suspect")
            for func, symbol, line in problems[:8]:
                print(f"      {func}() near line {line}: '{symbol}'")
        else:
            print(f"  {name}: clean")
    print(f"TOTAL SUSPECT: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
