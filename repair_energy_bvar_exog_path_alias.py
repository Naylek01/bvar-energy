"""Repair the ragged-edge level smoother's ``exog_path`` keyword alias.

Run this script from the root of the ``bvar-energy`` project:

    python repair_energy_bvar_exog_path_alias.py

It updates:

    src/model/energy_bvar_model.py

The repair is idempotent: running it again does nothing if the alias is already
present. A backup is created before any modification.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
import py_compile
import shutil
import tempfile


DEFAULT_TARGET = Path("src/model/energy_bvar_model.py")
FUNCTION_NAME = "_durbin_koopman_level_draw"
ALIAS_LINE = '            ("exog_path", False),\n'


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level function {name!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def repair(target: Path) -> tuple[Path, Path | None, bool]:
    target = target.resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Model file not found: {target}")

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = _find_function(tree, FUNCTION_NAME)

    function_lines = source.splitlines(keepends=True)[
        function.lineno - 1:function.end_lineno
    ]
    function_source = "".join(function_lines)

    if '("exog_path", False)' in function_source:
        return target, None, False

    anchor = '            ("future_exog", False),\n'
    if anchor not in function_source:
        raise RuntimeError(
            "Could not find the future_exog alias inside "
            f"{FUNCTION_NAME}. The source differs from the expected patch."
        )

    patched_function = function_source.replace(
        anchor,
        ALIAS_LINE + anchor,
        1,
    )

    all_lines = source.splitlines(keepends=True)
    all_lines[function.lineno - 1:function.end_lineno] = [patched_function]
    patched = "".join(all_lines)

    # Validate the result before touching the user's model file.
    ast.parse(patched)
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / target.name
        candidate.write_text(patched, encoding="utf-8")
        py_compile.compile(str(candidate), doraise=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(
        f"{target.stem}.before_exog_path_alias_fix_{timestamp}{target.suffix}"
    )
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)

    return target, backup, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TARGET,
        help=(
            "Path to energy_bvar_model.py. "
            "Default: src/model/energy_bvar_model.py"
        ),
    )
    args = parser.parse_args()

    target, backup, changed = repair(args.path)

    if changed:
        print(f"Updated: {target}")
        print(f"Backup : {backup}")
        print('Added alias: ("exog_path", False)')
        print("Next: restart the notebook kernel and rerun all cells.")
    else:
        print(f"No change required: {target}")
        print('Alias ("exog_path", False) is already present.')


if __name__ == "__main__":
    main()
