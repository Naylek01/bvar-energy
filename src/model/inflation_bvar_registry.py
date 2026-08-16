"""Domain-neutral registry facade for the unified Inflation Dashboard.

The existing ``energy_bvar_registry`` SQLite schema is intentionally reused.
Headline joint runs live in the same ``runs`` / ``forecasts`` tables, while the
default Energy promotion/aggregation contract remains the canonical seven
Energy models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

import energy_bvar_registry as _energy_registry
from energy_bvar_pipeline import CANONICAL_MODEL_IDS, resolve_model_id

HEADLINE_MODEL_ID = "headline_joint"
HEADLINE_MODEL_IDS = (HEADLINE_MODEL_ID,)
ENERGY_MODEL_IDS = tuple(CANONICAL_MODEL_IDS)
ALL_COMPONENT_MODEL_IDS = ENERGY_MODEL_IDS + HEADLINE_MODEL_IDS

REGISTRY_SCHEMA_VERSION = _energy_registry.REGISTRY_SCHEMA_VERSION
RegistryError = _energy_registry.RegistryError
PromotionError = _energy_registry.PromotionError
default_registry_path = _energy_registry.default_registry_path
init_registry = _energy_registry.init_registry
list_aggregates = _energy_registry.list_aggregates
promote_aggregate = _energy_registry.promote_aggregate
get_promoted_aggregate = _energy_registry.get_promoted_aggregate


@dataclass(frozen=True)
class InflationScanReport:
    results_root: Path
    registry_path: Path
    component_runs_seen: int
    forecasts_seen: int
    aggregates_seen: int
    invalid_component_runs: int
    invalid_forecasts: int
    invalid_aggregates: int
    unexpected_directories: tuple[str, ...]
    energy_component_runs_seen: int
    headline_component_runs_seen: int
    headline_forecasts_seen: int

    def as_series(self) -> pd.Series:
        return pd.Series(
            {
                "results_root": str(self.results_root),
                "registry_path": str(self.registry_path),
                "component_runs_seen": self.component_runs_seen,
                "forecasts_seen": self.forecasts_seen,
                "aggregates_seen": self.aggregates_seen,
                "invalid_component_runs": self.invalid_component_runs,
                "invalid_forecasts": self.invalid_forecasts,
                "invalid_aggregates": self.invalid_aggregates,
                "energy_component_runs_seen": self.energy_component_runs_seen,
                "headline_component_runs_seen": self.headline_component_runs_seen,
                "headline_forecasts_seen": self.headline_forecasts_seen,
                "unexpected_directories": ", ".join(self.unexpected_directories),
            }
        )


def _canonical_model_id(model_id: str) -> str:
    value = str(model_id).strip()
    if value == HEADLINE_MODEL_ID:
        return HEADLINE_MODEL_ID
    return resolve_model_id(value)


def _frame(path: Path | str, sql: str, params: Sequence | Mapping = ()) -> pd.DataFrame:
    with _energy_registry._connect(path) as con:
        return pd.read_sql_query(sql, con, params=params)


def scan_results(
    *,
    results_root: Path | str,
    registry_path: Path | str | None = None,
) -> InflationScanReport:
    """Scan Energy and Headline stores idempotently into one SQLite registry."""
    results_root = Path(results_root).resolve()
    registry_path = (
        default_registry_path(results_root=results_root)
        if registry_path is None
        else Path(registry_path)
    )

    # Preserve the existing Energy scanner byte-for-byte as the first pass.
    energy_report = _energy_registry.scan_results(
        results_root=results_root,
        registry_path=registry_path,
    )

    now = _energy_registry._utcnow()
    headline_runs = 0
    headline_forecasts = 0
    invalid_runs = 0
    invalid_forecasts = 0

    headline_root = results_root / HEADLINE_MODEL_ID
    with _energy_registry._connect(registry_path) as con:
        if headline_root.is_dir():
            for vintage_dir in sorted(p for p in headline_root.iterdir() if p.is_dir()):
                vintage = vintage_dir.name
                for run_dir in sorted(p for p in vintage_dir.iterdir() if p.is_dir()):
                    headline_runs += 1
                    valid, _ = _energy_registry._upsert_run_from_disk(
                        con,
                        model_id=HEADLINE_MODEL_ID,
                        vintage=vintage,
                        run_id=run_dir.name,
                        run_dir=run_dir,
                        now=now,
                    )
                    invalid_runs += int(not valid)

                    seen, bad = _energy_registry._scan_forecasts_for_run(
                        con,
                        model_id=HEADLINE_MODEL_ID,
                        vintage=vintage,
                        run_id=run_dir.name,
                        run_dir=run_dir,
                        now=now,
                    )
                    headline_forecasts += int(seen)
                    invalid_forecasts += int(bad)

                    # The legacy scanner labels every display as Energy schema 1.0.
                    # Correct only the Headline rows; the SQLite schema itself stays unchanged.
                    forecasts_root = run_dir / "forecasts"
                    if forecasts_root.is_dir():
                        for forecast_dir in (p for p in forecasts_root.iterdir() if p.is_dir()):
                            display_path = forecast_dir / "display_v1.parquet"
                            if not display_path.is_file():
                                continue
                            schema = "headline-1.0"
                            headline_meta = forecast_dir / "headline_metadata.json"
                            if headline_meta.is_file():
                                try:
                                    import json
                                    payload = json.loads(
                                        headline_meta.read_text(encoding="utf-8")
                                    )
                                    schema = str(
                                        payload.get(
                                            "headline_display_schema_version",
                                            schema,
                                        )
                                    )
                                except Exception:
                                    pass
                            con.execute(
                                """
                                UPDATE forecasts
                                SET has_display=1, display_schema_version=?
                                WHERE model_id=? AND vintage=? AND run_id=?
                                  AND forecast_name=?
                                """,
                                (
                                    schema,
                                    HEADLINE_MODEL_ID,
                                    vintage,
                                    run_dir.name,
                                    forecast_dir.name,
                                ),
                            )

        con.execute(
            "INSERT INTO registry_meta(key, value) VALUES('inflation_last_scan_utc', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (now,),
        )

    unexpected = tuple(
        name
        for name in energy_report.unexpected_directories
        if name != HEADLINE_MODEL_ID
    )

    return InflationScanReport(
        results_root=results_root,
        registry_path=Path(registry_path),
        component_runs_seen=int(energy_report.component_runs_seen) + headline_runs,
        forecasts_seen=int(energy_report.forecasts_seen) + headline_forecasts,
        aggregates_seen=int(energy_report.aggregates_seen),
        invalid_component_runs=int(energy_report.invalid_component_runs) + invalid_runs,
        invalid_forecasts=int(energy_report.invalid_forecasts) + invalid_forecasts,
        invalid_aggregates=int(energy_report.invalid_aggregates),
        unexpected_directories=unexpected,
        energy_component_runs_seen=int(energy_report.component_runs_seen),
        headline_component_runs_seen=headline_runs,
        headline_forecasts_seen=headline_forecasts,
    )


def list_runs(
    registry_path: Path | str,
    *,
    model_id: str | None = None,
    vintage: str | None = None,
    status: str | None = None,
    present_only: bool = True,
) -> pd.DataFrame:
    clauses: list[str] = []
    params: list[object] = []
    if model_id is not None:
        clauses.append("model_id = ?")
        params.append(_canonical_model_id(model_id))
    if vintage is not None:
        clauses.append("vintage = ?")
        params.append(str(vintage))
    if status is not None:
        if status not in {"running", "complete", "failed"}:
            raise RegistryError(f"Unknown status {status!r}.")
        clauses.append("status = ?")
        params.append(status)
    if present_only:
        clauses.append("present_on_disk = 1")
    where = "" if not clauses else " WHERE " + " AND ".join(clauses)
    return _frame(
        registry_path,
        "SELECT * FROM runs"
        + where
        + " ORDER BY vintage DESC, model_id, created_at_utc DESC, run_id",
        params,
    )


def list_forecasts(
    registry_path: Path | str,
    *,
    model_id: str | None = None,
    vintage: str | None = None,
    forecast_name: str | None = None,
    valid_only: bool = False,
    present_only: bool = True,
) -> pd.DataFrame:
    clauses: list[str] = []
    params: list[object] = []
    if model_id is not None:
        clauses.append("model_id = ?")
        params.append(_canonical_model_id(model_id))
    if vintage is not None:
        clauses.append("vintage = ?")
        params.append(str(vintage))
    if forecast_name is not None:
        clauses.append("forecast_name = ?")
        params.append(str(forecast_name))
    if valid_only:
        clauses.append("valid = 1")
    if present_only:
        clauses.append("present_on_disk = 1")
    where = "" if not clauses else " WHERE " + " AND ".join(clauses)
    return _frame(
        registry_path,
        "SELECT * FROM forecasts"
        + where
        + " ORDER BY vintage DESC, model_id, run_id, forecast_name",
        params,
    )


def promote_run(
    registry_path: Path | str,
    model_id: str,
    vintage: str,
    run_id: str,
    *,
    required_forecast: str | None = "unconditional",
    require_draws: bool = False,
) -> None:
    canonical = _canonical_model_id(model_id)
    if canonical != HEADLINE_MODEL_ID:
        _energy_registry.promote_run(
            registry_path,
            canonical,
            vintage,
            run_id,
            required_forecast=required_forecast,
            require_draws=require_draws,
        )
        return

    vintage = str(vintage)
    run_id = str(run_id)
    with _energy_registry._connect(registry_path) as con:
        row = con.execute(
            "SELECT * FROM runs WHERE model_id=? AND vintage=? AND run_id=?",
            (canonical, vintage, run_id),
        ).fetchone()
        if row is None:
            raise PromotionError(f"Unknown run {canonical}/{vintage}/{run_id}.")
        if row["status"] != "complete" or not row["present_on_disk"]:
            raise PromotionError(
                f"Run {canonical}/{vintage}/{run_id[:12]} is not a complete on-disk run."
            )
        if require_draws and not row["has_draws"]:
            raise PromotionError(
                f"Run {canonical}/{vintage}/{run_id[:12]} has no draws.npz."
            )
        if required_forecast is not None:
            forecast = con.execute(
                """
                SELECT 1 FROM forecasts
                WHERE model_id=? AND vintage=? AND run_id=? AND forecast_name=?
                  AND valid=1 AND present_on_disk=1
                """,
                (canonical, vintage, run_id, str(required_forecast)),
            ).fetchone()
            if forecast is None:
                raise PromotionError(
                    f"Run {canonical}/{vintage}/{run_id[:12]} has no valid "
                    f"{required_forecast!r} forecast store."
                )
        con.execute(
            "UPDATE runs SET promoted=0 WHERE model_id=? AND vintage=?",
            (canonical, vintage),
        )
        con.execute(
            "UPDATE runs SET promoted=1 WHERE model_id=? AND vintage=? AND run_id=?",
            (canonical, vintage, run_id),
        )


def promoted_run_ids(
    registry_path: Path | str,
    vintage: str,
    *,
    forecast_name: str = "unconditional",
    require_all: bool = True,
    require_draws: bool = False,
    model_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return promoted runs; default remains the seven Energy models."""
    selected = (
        tuple(ENERGY_MODEL_IDS)
        if model_ids is None
        else tuple(_canonical_model_id(x) for x in model_ids)
    )
    vintage = str(vintage)
    out: dict[str, str] = {}
    missing: list[str] = []
    with _energy_registry._connect(registry_path) as con:
        for model_id in selected:
            row = con.execute(
                """
                SELECT r.run_id, r.has_draws
                FROM runs r
                JOIN forecasts f
                  ON f.model_id=r.model_id
                 AND f.vintage=r.vintage
                 AND f.run_id=r.run_id
                WHERE r.model_id=? AND r.vintage=? AND r.promoted=1
                  AND r.status='complete' AND r.present_on_disk=1
                  AND f.forecast_name=? AND f.valid=1 AND f.present_on_disk=1
                """,
                (model_id, vintage, str(forecast_name)),
            ).fetchone()
            if row is None or (require_draws and not row["has_draws"]):
                missing.append(model_id)
                continue
            out[model_id] = str(row["run_id"])
    if require_all and missing:
        raise PromotionError(
            f"Vintage {vintage!r} lacks promoted usable runs for {missing}."
        )
    return out


def set_run_status(
    registry_path: Path | str,
    *,
    model_id: str,
    vintage: str,
    run_id: str,
    status: str,
    error: str | None = None,
    directory: Path | str | None = None,
) -> None:
    canonical = _canonical_model_id(model_id)
    if canonical != HEADLINE_MODEL_ID:
        _energy_registry.set_run_status(
            registry_path,
            model_id=canonical,
            vintage=vintage,
            run_id=run_id,
            status=status,
            error=error,
            directory=directory,
        )
        return

    if status not in {"running", "complete", "failed"}:
        raise RegistryError(f"Unknown status {status!r}.")
    now = _energy_registry._utcnow()
    init_registry(registry_path)
    with _energy_registry._connect(registry_path) as con:
        con.execute(
            """
            INSERT INTO runs(
                model_id, vintage, run_id, directory, status, error,
                present_on_disk, first_seen_utc, last_seen_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id, vintage, run_id) DO UPDATE SET
                directory=COALESCE(excluded.directory, runs.directory),
                status=excluded.status,
                error=excluded.error,
                last_seen_utc=excluded.last_seen_utc
            """,
            (
                canonical,
                str(vintage),
                str(run_id),
                None if directory is None else str(Path(directory)),
                status,
                error,
                int(directory is not None and Path(directory).exists()),
                now,
                now,
            ),
        )


def run_coverage(
    registry_path: Path | str,
    vintage: str,
    *,
    forecast_name: str = "unconditional",
    include_headline: bool = True,
) -> pd.DataFrame:
    energy = _energy_registry.run_coverage(
        registry_path,
        vintage,
        forecast_name=forecast_name,
    )
    if not include_headline:
        return energy

    runs = list_runs(
        registry_path,
        model_id=HEADLINE_MODEL_ID,
        vintage=vintage,
        present_only=False,
    )
    forecasts = list_forecasts(
        registry_path,
        model_id=HEADLINE_MODEL_ID,
        vintage=vintage,
        forecast_name=forecast_name,
        valid_only=True,
        present_only=True,
    )
    complete = (
        runs.loc[(runs["status"] == "complete") & (runs["present_on_disk"] == 1)]
        if not runs.empty
        else runs
    )
    promoted = (
        complete.loc[complete["promoted"] == 1]
        if not complete.empty
        else complete
    )
    promoted_id = None if promoted.empty else str(promoted.iloc[0]["run_id"])
    promoted_forecast = (
        forecasts.loc[forecasts["run_id"].astype(str) == promoted_id]
        if promoted_id is not None and not forecasts.empty
        else pd.DataFrame()
    )

    row = pd.DataFrame(
        [
            {
                "model_id": HEADLINE_MODEL_ID,
                "label": "Headline HICP",
                "aggregate_key": "hicp_total",
                "n_runs": int(len(runs)),
                "n_complete": int(len(complete)),
                "n_with_draws": (
                    0 if complete.empty else int((complete["has_draws"] == 1).sum())
                ),
                f"n_{forecast_name}_forecasts": int(len(forecasts)),
                f"n_{forecast_name}_displays": (
                    0
                    if forecasts.empty
                    else int((forecasts["has_display"] == 1).sum())
                ),
                "promoted_run_id": promoted_id,
                "promoted_has_draws": (
                    False
                    if promoted.empty
                    else bool(promoted.iloc[0]["has_draws"])
                ),
                f"promoted_has_{forecast_name}": not promoted_forecast.empty,
                f"promoted_has_{forecast_name}_display": (
                    False
                    if promoted_forecast.empty
                    else bool(promoted_forecast.iloc[0]["has_display"])
                ),
            }
        ]
    ).set_index("model_id")
    return pd.concat([energy, row], axis=0)


__all__ = [
    "HEADLINE_MODEL_ID",
    "HEADLINE_MODEL_IDS",
    "ENERGY_MODEL_IDS",
    "ALL_COMPONENT_MODEL_IDS",
    "REGISTRY_SCHEMA_VERSION",
    "RegistryError",
    "PromotionError",
    "InflationScanReport",
    "default_registry_path",
    "init_registry",
    "scan_results",
    "list_runs",
    "list_forecasts",
    "list_aggregates",
    "run_coverage",
    "promote_run",
    "promoted_run_ids",
    "promote_aggregate",
    "get_promoted_aggregate",
    "set_run_status",
]
