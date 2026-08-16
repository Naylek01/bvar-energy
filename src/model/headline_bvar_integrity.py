"""Headline run lineage helpers for the Delivery-1 regression harness."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from energy_bvar_model import hash_model_data
from headline_joint_bvar import MODEL_ID, native_to_state_levels

INTEGRITY_FILENAME = "data_integrity.json"
INTEGRITY_SCHEMA_VERSION = "headline-data-integrity-1.0"

FORECAST_DEFAULTS_BY_CODE_VERSION: dict[str, dict[str, Any]] = {
    "2026-08-12-headline-dashboard-v1": {
        "n_forecast_draws": 1000,
        "forecast_seed": 2026,
        "simulate_future_outliers": True,
        "provenance": "pipeline_default_at_vintage",
    },
    "2026-08-13-headline-dashboard-lineage-v2": {
        "n_forecast_draws": 1000,
        "forecast_seed": 2026,
        "simulate_future_outliers": True,
        "provenance": "pipeline_default_at_vintage",
    },
}


class HeadlineIntegrityError(RuntimeError):
    pass


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HeadlineIntegrityError(f"{path} must contain a JSON object.")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def resolve_headline_run(
    results_root: str | Path,
    *,
    vintage: str,
    run_id: str | None = None,
) -> Path:
    base = Path(results_root) / MODEL_ID / str(vintage)
    if run_id is not None:
        candidate = base / str(run_id)
        if not (candidate / "metadata.json").is_file():
            raise FileNotFoundError(candidate / "metadata.json")
        return candidate
    candidates = (
        sorted(
            p
            for p in base.iterdir()
            if p.is_dir()
            and (p / "metadata.json").is_file()
            and (p / "draws.npz").is_file()
        )
        if base.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(f"No saved Headline run found under {base}.")
    if len(candidates) != 1:
        raise HeadlineIntegrityError(
            f"{len(candidates)} Headline runs exist for vintage {vintage}. "
            "Pass --run-id explicitly; the harness never guesses between runs."
        )
    return candidates[0]


def validate_run_directory(run_directory: str | Path) -> dict:
    run_dir = Path(run_directory)
    metadata = _read_json(run_dir / "metadata.json")
    if str(metadata.get("model_id")) != MODEL_ID:
        raise HeadlineIntegrityError(
            f"Expected model_id={MODEL_ID!r}, found {metadata.get('model_id')!r}."
        )
    if run_dir.name != str(metadata.get("run_id")):
        raise HeadlineIntegrityError(
            "Run directory name does not equal metadata.run_id."
        )
    if run_dir.parent.name != str(metadata.get("vintage")):
        raise HeadlineIntegrityError(
            "Run directory vintage does not equal metadata.vintage."
        )
    return metadata


def current_headline_source_contract(
    vintage: str,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    from headline_bvar_pipeline import build_inputs, make_joint_seasonals

    inputs = build_inputs(vintage, project_root=project_root)
    state = native_to_state_levels(inputs.native_levels)
    exog = make_joint_seasonals(state.index)
    return {
        "source_data_hash": hash_model_data(state, exog),
        "exog_names": list(exog.columns),
        "sample_start": pd.Timestamp(state.index.min()).isoformat(),
        "sample_end": pd.Timestamp(state.index.max()).isoformat(),
        "n_calendar_rows": int(len(state)),
        "dataset_path": str(inputs.dataset_path),
        "weights_path": str(inputs.weights_path),
        "official_indices_path": str(inputs.official_indices_path),
    }


def _saved_forecast_metadata(run_dir: Path) -> dict:
    path = run_dir / "forecasts" / "unconditional" / "forecast_metadata.json"
    return _read_json(path) if path.is_file() else {}


def _forecast_draw_count(run_dir: Path) -> int | None:
    forecast_meta = _saved_forecast_metadata(run_dir)
    if forecast_meta.get("n_draws") is not None:
        return int(forecast_meta["n_draws"])
    path = run_dir / "forecasts" / "unconditional" / "headline_draws.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            for key in ("native_level_paths", "headline_level_paths"):
                if key in archive.files:
                    return int(np.asarray(archive[key]).shape[0])
    return None


def recover_forecast_reconstruction(
    run_directory: str | Path,
    *,
    forecast_seed: int | None = None,
    provenance: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_directory)
    metadata = validate_run_directory(run_dir)
    forecast_meta = _saved_forecast_metadata(run_dir)

    n_draws = metadata.get("n_forecast_draws")
    n_draws_origin = "metadata_at_estimation" if n_draws is not None else None
    if n_draws is None:
        n_draws = _forecast_draw_count(run_dir)
        if n_draws is not None:
            n_draws_origin = "recovered_from_saved_forecast"

    seed = metadata.get("forecast_seed")
    seed_origin = "metadata_at_estimation" if seed is not None else None
    if seed is None and forecast_seed is not None:
        seed = int(forecast_seed)
        seed_origin = str(provenance or "explicit_operator_input")

    defaults = FORECAST_DEFAULTS_BY_CODE_VERSION.get(
        str(metadata.get("code_version", ""))
    )
    if seed is None and defaults is not None:
        seed = int(defaults["forecast_seed"])
        seed_origin = str(defaults["provenance"])
    if n_draws is None and defaults is not None:
        n_draws = int(defaults["n_forecast_draws"])
        n_draws_origin = str(defaults["provenance"])

    simulate = metadata.get("simulate_future_outliers")
    simulate_origin = "metadata_at_estimation" if simulate is not None else None
    if simulate is None and forecast_meta.get("simulate_future_outliers") is not None:
        simulate = bool(forecast_meta["simulate_future_outliers"])
        simulate_origin = "recovered_from_saved_forecast"
    if simulate is None and defaults is not None:
        simulate = bool(defaults["simulate_future_outliers"])
        simulate_origin = str(defaults["provenance"])

    return {
        "n_forecast_draws": None if n_draws is None else int(n_draws),
        "forecast_seed": None if seed is None else int(seed),
        "simulate_future_outliers": None if simulate is None else bool(simulate),
        "provenance": {
            "n_forecast_draws": n_draws_origin,
            "forecast_seed": seed_origin,
            "simulate_future_outliers": simulate_origin,
        },
    }


def build_backfill_payload(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    forecast_seed: int | None = None,
    forecast_provenance: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_directory)
    metadata = validate_run_directory(run_dir)
    source = current_headline_source_contract(
        str(metadata["vintage"]), project_root=project_root
    )

    identity_hash = metadata.get("source_data_hash", metadata.get("data_hash"))
    matches_identity = None
    if identity_hash:
        matches_identity = str(identity_hash) == str(source["source_data_hash"])
        if not matches_identity:
            raise HeadlineIntegrityError(
                "Current processed Headline inputs do not match the data hash "
                "stored in the historical run identity. Refusing to backfill."
            )

    forecast = recover_forecast_reconstruction(
        run_dir,
        forecast_seed=forecast_seed,
        provenance=forecast_provenance,
    )

    if metadata.get("source_data_hash"):
        status = "verified_at_estimation"
    elif matches_identity is True:
        status = "backfilled_not_estimation_time"
    else:
        status = "unverified"

    return {
        "integrity_schema_version": INTEGRITY_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "run_id": str(metadata["run_id"]),
        "vintage": str(metadata["vintage"]),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": status,
        "source_data_hash": str(source["source_data_hash"]),
        "run_identity_data_hash": (
            None if identity_hash is None else str(identity_hash)
        ),
        "matches_run_identity_data_hash": matches_identity,
        "source_contract": source,
        "estimation_contract_recovered": {
            "exog_names": metadata.get("exog_names"),
            "balanced_end": metadata.get("balanced_end"),
            "last_calendar_date": metadata.get("last_calendar_date"),
        },
        "forecast_reconstruction": forecast,
        "note": (
            "Computed after estimation; this sidecar does not rewrite or "
            "pretend to be historical metadata.json."
        ),
    }


def write_backfill_sidecar(
    run_directory: str | Path,
    *,
    project_root: str | Path,
    forecast_seed: int | None = None,
    forecast_provenance: str | None = None,
    overwrite: bool = False,
) -> Path:
    run_dir = Path(run_directory)
    destination = run_dir / INTEGRITY_FILENAME
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    payload = build_backfill_payload(
        run_dir,
        project_root=project_root,
        forecast_seed=forecast_seed,
        forecast_provenance=forecast_provenance,
    )
    return _atomic_json(destination, payload)


def load_integrity_contract(run_directory: str | Path) -> dict[str, Any]:
    run_dir = Path(run_directory)
    metadata = validate_run_directory(run_dir)
    sidecar_path = run_dir / INTEGRITY_FILENAME
    sidecar = _read_json(sidecar_path) if sidecar_path.is_file() else {}
    forecast_backfill = dict(sidecar.get("forecast_reconstruction") or {})
    provenance = dict(forecast_backfill.get("provenance") or {})

    source_hash = metadata.get("source_data_hash")
    source_origin = "metadata_at_estimation" if source_hash else None
    if not source_hash:
        source_hash = sidecar.get("source_data_hash")
        if source_hash:
            source_origin = "backfilled_sidecar"

    def resolved(name: str):
        if metadata.get(name) is not None:
            return metadata[name], "metadata_at_estimation"
        return forecast_backfill.get(name), provenance.get(name)

    n_draws, n_draws_origin = resolved("n_forecast_draws")
    seed, seed_origin = resolved("forecast_seed")
    simulate, simulate_origin = resolved("simulate_future_outliers")

    status = (
        "verified_at_estimation"
        if metadata.get("source_data_hash")
        else str(sidecar.get("verification_status") or "unverified")
    )
    recovered = sidecar.get("estimation_contract_recovered") or {}
    return {
        "verification_status": status,
        "source_data_hash": source_hash,
        "source_data_hash_provenance": source_origin,
        "exog_names": metadata.get("exog_names", recovered.get("exog_names")),
        "balanced_end": metadata.get("balanced_end", recovered.get("balanced_end")),
        "last_calendar_date": metadata.get(
            "last_calendar_date", recovered.get("last_calendar_date")
        ),
        "n_forecast_draws": None if n_draws is None else int(n_draws),
        "forecast_seed": None if seed is None else int(seed),
        "simulate_future_outliers": None if simulate is None else bool(simulate),
        "provenance": {
            "n_forecast_draws": n_draws_origin,
            "forecast_seed": seed_origin,
            "simulate_future_outliers": simulate_origin,
        },
        "run_id": str(metadata["run_id"]),
        "vintage": str(metadata["vintage"]),
    }


__all__ = [
    "INTEGRITY_FILENAME",
    "INTEGRITY_SCHEMA_VERSION",
    "FORECAST_DEFAULTS_BY_CODE_VERSION",
    "HeadlineIntegrityError",
    "resolve_headline_run",
    "validate_run_directory",
    "current_headline_source_contract",
    "recover_forecast_reconstruction",
    "build_backfill_payload",
    "write_backfill_sidecar",
    "load_integrity_contract",
]
