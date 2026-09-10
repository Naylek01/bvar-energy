"""Portable saved-artifact reference helpers for ECB VAR.

New aggregate metadata uses logical ``forecast://`` references. Existing
absolute Windows/POSIX references remain readable after moving/cloning the
repository. This module is deliberately cycle-free and imports no project code.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

REPO_REFERENCE_PORTABILITY_V1 = True
PORTABILITY_CONTRACT_VERSION = "repo-reference-v1"
_REPO_ANCHORS = ("results", "data", "src", "notebooks", "configs", "tests", "assets")
_FORECAST_SCHEME = "forecast://"


def _normalise_text(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _module_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    env = str(os.getenv("ENERGY_BVAR_PROJECT_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _module_project_root()


def results_root(
    results_root: str | Path | None = None,
    *,
    project_root_value: str | Path | None = None,
) -> Path:
    if results_root is not None:
        return Path(results_root).expanduser().resolve()
    env = str(os.getenv("ENERGY_BVAR_RESULTS_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return project_root(project_root_value) / "results"


def _repo_suffix(value: Any) -> tuple[str, ...] | None:
    text = _normalise_text(value)
    if not text or text.startswith(_FORECAST_SCHEME):
        return None
    parts = tuple(piece for piece in text.split("/") if piece not in ("", "."))
    lower = tuple(piece.casefold() for piece in parts)
    matches: list[int] = []
    for anchor in _REPO_ANCHORS:
        matches.extend(i for i, piece in enumerate(lower) if piece == anchor.casefold())
    if not matches:
        return None
    return parts[max(matches):]


def forecast_store_identity(value: Any) -> dict[str, str] | None:
    """Parse model/vintage/run/forecast without native-OS Path semantics."""
    text = _normalise_text(value)
    if not text:
        return None
    if text.startswith(_FORECAST_SCHEME):
        pieces = [p for p in text[len(_FORECAST_SCHEME):].strip("/").split("/") if p]
        if len(pieces) != 4:
            return None
        model_id, vintage, run_id, forecast_name = pieces
        return {
            "model_id": model_id,
            "vintage": vintage,
            "run_id": run_id,
            "forecast_name": forecast_name,
        }

    pieces = [p for p in text.rstrip("/").split("/") if p]
    lower = [p.casefold() for p in pieces]
    positions = [i for i, p in enumerate(lower) if p == "forecasts"]
    if not positions:
        return None
    i = positions[-1]
    if i < 3 or i + 1 >= len(pieces):
        return None
    return {
        "model_id": pieces[i - 3],
        "vintage": pieces[i - 2],
        "run_id": pieces[i - 1],
        "forecast_name": pieces[i + 1],
    }


def forecast_store_reference(value: Any) -> str:
    ident = forecast_store_identity(value)
    if ident is None:
        raise ValueError(f"Cannot derive forecast-store identity from {value!r}.")
    return (
        _FORECAST_SCHEME
        + ident["model_id"] + "/" + ident["vintage"] + "/"
        + ident["run_id"] + "/" + ident["forecast_name"]
    )


def run_id_from_forecast_store(value: Any) -> str | None:
    ident = forecast_store_identity(value)
    return None if ident is None else ident["run_id"]


def resolve_repo_reference(
    value: Any,
    *,
    project_root_value: str | Path | None = None,
    base: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve native, repo-relative, or stale legacy absolute reference."""
    raw = str(value or "").strip()
    if not raw:
        raise FileNotFoundError("Empty path reference.")
    if _normalise_text(raw).startswith(_FORECAST_SCHEME):
        return resolve_forecast_store_reference(
            raw,
            project_root_value=project_root_value,
            must_exist=must_exist,
        )

    root = project_root(project_root_value)
    native = Path(raw).expanduser()
    if native.exists():
        return native.resolve()

    normal = _normalise_text(raw)
    windows_abs = bool(re.match(r"^[A-Za-z]:/", normal)) or normal.startswith("//")
    posix_abs = normal.startswith("/") and not normal.startswith("//")
    candidates: list[Path] = []

    if not windows_abs and not posix_abs and not raw.startswith("~"):
        rel_parts = [p for p in normal.split("/") if p]
        candidates.append(root.joinpath(*rel_parts))
        if base is not None:
            candidates.append(Path(base).expanduser().resolve().joinpath(*rel_parts))

    suffix = _repo_suffix(raw)
    if suffix:
        candidates.append(root.joinpath(*suffix))

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    for candidate in unique:
        if candidate.exists():
            return candidate.resolve()

    if must_exist:
        shown = ", ".join(str(path) for path in unique) or "<none>"
        raise FileNotFoundError(
            f"Could not resolve repository reference {raw!r}; candidates: {shown}"
        )
    return unique[0] if unique else native


def resolve_forecast_store_reference(
    value: Any,
    *,
    project_root_value: str | Path | None = None,
    results_root_value: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    ident = forecast_store_identity(value)
    if ident is not None:
        results = results_root(results_root_value, project_root_value=project_root_value)
        logical = (
            results / ident["model_id"] / ident["vintage"] / ident["run_id"]
            / "forecasts" / ident["forecast_name"]
        )
        if logical.exists():
            return logical.resolve()
        native = Path(str(value)).expanduser()
        if native.exists():
            return native.resolve()
        if must_exist:
            raise FileNotFoundError(
                f"Forecast store {forecast_store_reference(value)!r} is absent "
                f"below active results root {results}."
            )
        return logical
    return resolve_repo_reference(
        value,
        project_root_value=project_root_value,
        must_exist=must_exist,
    )


def portable_repo_reference(
    value: Any,
    *,
    project_root_value: str | Path | None = None,
) -> str:
    root = project_root(project_root_value)
    resolved = resolve_repo_reference(
        value,
        project_root_value=root,
        must_exist=False,
    )
    try:
        return resolved.resolve().relative_to(root).as_posix()
    except Exception:
        suffix = _repo_suffix(value)
        return "/".join(suffix) if suffix else _normalise_text(value)


def portable_forecast_store_map(stores: Mapping[Any, Any]) -> dict[str, str]:
    return {
        str(key): forecast_store_reference(value)
        for key, value in dict(stores or {}).items()
    }


__all__ = [
    "PORTABILITY_CONTRACT_VERSION",
    "project_root",
    "results_root",
    "forecast_store_identity",
    "forecast_store_reference",
    "run_id_from_forecast_store",
    "resolve_repo_reference",
    "resolve_forecast_store_reference",
    "portable_repo_reference",
    "portable_forecast_store_map",
]
