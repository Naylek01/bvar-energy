from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

KERNELSPEC = {
    "display_name": "Python (bvar-energy)",
    "language": "python",
    "name": "bvar-energy",
}

LANGUAGE_INFO = {
    "name": "python",
    "version": "3.11",
}


def main() -> None:
    if not NOTEBOOK_ROOT.exists():
        print(f"No notebooks directory found at {NOTEBOOK_ROOT}; nothing to update.")
        return

    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    if not notebooks:
        print(f"No .ipynb files found below {NOTEBOOK_ROOT}; nothing to update.")
        return

    updated = 0
    for path in notebooks:
        with path.open("r", encoding="utf-8") as fh:
            notebook = json.load(fh)

        metadata = notebook.setdefault("metadata", {})
        old_kernel = metadata.get("kernelspec")
        old_language = metadata.get("language_info", {})

        metadata["kernelspec"] = dict(KERNELSPEC)
        language_info = dict(old_language) if isinstance(old_language, dict) else {}
        language_info.update(LANGUAGE_INFO)
        metadata["language_info"] = language_info

        if old_kernel != KERNELSPEC or old_language != language_info:
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                json.dump(notebook, fh, ensure_ascii=False, indent=1)
                fh.write("\n")
            updated += 1

    print(
        f"Notebook kernelspec: {updated} updated, "
        f"{len(notebooks) - updated} already configured."
    )


if __name__ == "__main__":
    main()
