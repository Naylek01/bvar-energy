"""SQLite registry for the energy-BVAR dashboard.

The registry is an index over the immutable stores already written below
``RESULTS_ROOT``.  It does not own model output and the scanner never mutates a
run directory.  SQLite is used because Dash background callbacks execute in a
separate process even in the intended single-user deployment.

Layout indexed by this module
-----------------------------
Component runs::

    results/<model_id>/<vintage>/<run_id>/
        metadata.json
        draws.npz                         # optional
        forecasts/<forecast_name>/
            forecast_metadata.json
            forecast_draws.npz
            display_v1.parquet               # optional compact dashboard cache

Aggregate runs::

    results/hicp_energy_aggregate/<vintage>/<aggregate_run_id>/
        metadata.json
        aggregate_draws.npz
        display_v1.parquet                   # optional compact dashboard cache

The registry deliberately has no ``jobs`` table.  ``runs.status`` and
``runs.error`` are sufficient for durable run state; live progress/cancellation
remain Dash background-callback concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence

import pandas as pd

from energy_bvar_pipeline import (
    CANONICAL_MODEL_IDS,
    PipelineError,
    find_project_root,
    model_spec,
    resolve_model_id,
)

__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "RegistryError",
    "PromotionError",
    "ScanReport",
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


REGISTRY_SCHEMA_VERSION = "1.1"
_AGGREGATE_MODEL_ID = "hicp_energy_aggregate"
_RUN_STATUSES = {"running", "complete", "failed"}


class RegistryError(PipelineError):
    """Base class for registry failures."""


class PromotionError(RegistryError):
    """Raised when a run cannot be made authoritative."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"Expected a JSON object in {path}.")
    return value


def default_registry_path(
    *,
    project_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> Path:
    """Return the default registry file beside the result tree."""
    if results_root is None:
        results_root = find_project_root(project_root) / "results"
    return Path(results_root) / "energy_bvar_registry.sqlite3"


def _connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def _ensure_column(
    con: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Add one nullable/defaulted column when upgrading an existing registry."""
    names = {str(row["name"]) for row in con.execute(f"PRAGMA table_info({table})")}
    if column not in names:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_registry(path: Path | str) -> Path:
    """Create or migrate the registry schema idempotently."""
    path = Path(path)
    with _connect(path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                model_id TEXT NOT NULL,
                vintage TEXT NOT NULL,
                run_id TEXT NOT NULL,
                directory TEXT,
                status TEXT NOT NULL DEFAULT 'complete'
                    CHECK (status IN ('running', 'complete', 'failed')),
                error TEXT,
                promoted INTEGER NOT NULL DEFAULT 0 CHECK (promoted IN (0, 1)),
                present_on_disk INTEGER NOT NULL DEFAULT 1
                    CHECK (present_on_disk IN (0, 1)),
                has_draws INTEGER NOT NULL DEFAULT 0 CHECK (has_draws IN (0, 1)),
                has_summary INTEGER NOT NULL DEFAULT 0 CHECK (has_summary IN (0, 1)),
                has_diagnostics INTEGER NOT NULL DEFAULT 0
                    CHECK (has_diagnostics IN (0, 1)),
                data_hash TEXT,
                config_hash TEXT,
                code_version TEXT,
                result_schema_version TEXT,
                frequency TEXT,
                missing_data_method TEXT,
                missing_treatment_exact INTEGER,
                p INTEGER,
                seed INTEGER,
                n_observations INTEGER,
                created_at_utc TEXT,
                metadata_json TEXT,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL,
                PRIMARY KEY (model_id, vintage, run_id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_promoted
            ON runs(model_id, vintage)
            WHERE promoted = 1;

            CREATE INDEX IF NOT EXISTS ix_runs_lookup
            ON runs(vintage, model_id, status, present_on_disk);

            CREATE TABLE IF NOT EXISTS forecasts (
                model_id TEXT NOT NULL,
                vintage TEXT NOT NULL,
                run_id TEXT NOT NULL,
                forecast_name TEXT NOT NULL,
                directory TEXT NOT NULL,
                valid INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0, 1)),
                error TEXT,
                present_on_disk INTEGER NOT NULL DEFAULT 1
                    CHECK (present_on_disk IN (0, 1)),
                has_display INTEGER NOT NULL DEFAULT 0
                    CHECK (has_display IN (0, 1)),
                display_schema_version TEXT,
                n_draws INTEGER,
                n_path_dates INTEGER,
                n_variables INTEGER,
                horizon INTEGER,
                tail_length INTEGER,
                frequency TEXT,
                simulate_future_outliers INTEGER,
                future_start TEXT,
                future_end TEXT,
                metadata_json TEXT,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL,
                PRIMARY KEY (model_id, vintage, run_id, forecast_name),
                FOREIGN KEY (model_id, vintage, run_id)
                    REFERENCES runs(model_id, vintage, run_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_forecasts_lookup
            ON forecasts(vintage, model_id, forecast_name, valid, present_on_disk);

            CREATE TABLE IF NOT EXISTS aggregates (
                vintage TEXT NOT NULL,
                aggregate_run_id TEXT NOT NULL,
                forecast_name TEXT NOT NULL DEFAULT 'unconditional',
                directory TEXT,
                status TEXT NOT NULL DEFAULT 'complete'
                    CHECK (status IN ('running', 'complete', 'failed')),
                error TEXT,
                promoted INTEGER NOT NULL DEFAULT 0 CHECK (promoted IN (0, 1)),
                present_on_disk INTEGER NOT NULL DEFAULT 1
                    CHECK (present_on_disk IN (0, 1)),
                has_draws INTEGER NOT NULL DEFAULT 0 CHECK (has_draws IN (0, 1)),
                has_display INTEGER NOT NULL DEFAULT 0
                    CHECK (has_display IN (0, 1)),
                display_schema_version TEXT,
                n_aggregate_draws INTEGER,
                scenario_active INTEGER,
                weekly_tax_mode TEXT,
                aggregate_module_version TEXT,
                created_at_utc TEXT,
                component_forecast_stores_json TEXT,
                metadata_json TEXT,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL,
                PRIMARY KEY (vintage, aggregate_run_id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_aggregates_promoted
            ON aggregates(vintage, forecast_name)
            WHERE promoted = 1;

            CREATE INDEX IF NOT EXISTS ix_aggregates_lookup
            ON aggregates(vintage, forecast_name, status, present_on_disk);
            """
        )
        _ensure_column(
            con, "forecasts", "has_display",
            "INTEGER NOT NULL DEFAULT 0 CHECK (has_display IN (0, 1))",
        )
        _ensure_column(con, "forecasts", "display_schema_version", "TEXT")
        _ensure_column(
            con, "aggregates", "has_display",
            "INTEGER NOT NULL DEFAULT 0 CHECK (has_display IN (0, 1))",
        )
        _ensure_column(con, "aggregates", "display_schema_version", "TEXT")

        con.execute(
            "INSERT INTO registry_meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (REGISTRY_SCHEMA_VERSION,),
        )
    return path


@dataclass(frozen=True)
class ScanReport:
    """Summary of one idempotent pass over ``RESULTS_ROOT``."""

    results_root: Path
    registry_path: Path
    component_runs_seen: int
    forecasts_seen: int
    aggregates_seen: int
    invalid_component_runs: int
    invalid_forecasts: int
    invalid_aggregates: int
    unexpected_directories: tuple[str, ...]

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
                "unexpected_directories": ", ".join(self.unexpected_directories),
            }
        )


def _as_int_bool(value) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _upsert_run_from_disk(
    con: sqlite3.Connection,
    *,
    model_id: str,
    vintage: str,
    run_id: str,
    run_dir: Path,
    now: str,
) -> tuple[bool, dict | None]:
    metadata_path = run_dir / "metadata.json"
    metadata: dict | None = None
    error: str | None = None
    status = "complete"

    if not metadata_path.is_file():
        status = "failed"
        error = "metadata.json is missing"
    else:
        try:
            metadata = _read_json(metadata_path)
            mismatches = []
            for key, expected in (
                ("model_id", model_id),
                ("vintage", vintage),
                ("run_id", run_id),
            ):
                if str(metadata.get(key)) != str(expected):
                    mismatches.append(
                        f"{key}: metadata={metadata.get(key)!r}, path={expected!r}"
                    )
            if mismatches:
                status = "failed"
                error = "Path/metadata identity mismatch: " + "; ".join(mismatches)
        except RegistryError as exc:
            status = "failed"
            error = str(exc)

    values = {
        "directory": str(run_dir.resolve()),
        "status": status,
        "error": error,
        "present_on_disk": 1,
        "has_draws": int((run_dir / "draws.npz").is_file()),
        "has_summary": int(
            (run_dir / "summary.parquet").is_file()
            or (run_dir / "summary.csv").is_file()
        ),
        "has_diagnostics": int(
            (run_dir / "diagnostics.parquet").is_file()
            or (run_dir / "diagnostics.csv").is_file()
        ),
        "data_hash": None if metadata is None else metadata.get("data_hash"),
        "config_hash": None if metadata is None else metadata.get("config_hash"),
        "code_version": None if metadata is None else metadata.get("code_version"),
        "result_schema_version": None
        if metadata is None
        else metadata.get("result_schema_version"),
        "frequency": None if metadata is None else metadata.get("frequency"),
        "missing_data_method": None
        if metadata is None
        else metadata.get("missing_data_method"),
        "missing_treatment_exact": None
        if metadata is None
        else _as_int_bool(metadata.get("missing_treatment_exact")),
        "p": None if metadata is None else metadata.get("p"),
        "seed": None if metadata is None else metadata.get("seed"),
        "n_observations": None if metadata is None else metadata.get("n_observations"),
        "created_at_utc": None if metadata is None else metadata.get("created_at_utc"),
        "metadata_json": None if metadata is None else _json(metadata),
    }

    con.execute(
        """
        INSERT INTO runs(
            model_id, vintage, run_id, directory, status, error, present_on_disk,
            has_draws, has_summary, has_diagnostics, data_hash, config_hash,
            code_version, result_schema_version, frequency, missing_data_method,
            missing_treatment_exact, p, seed, n_observations, created_at_utc,
            metadata_json, first_seen_utc, last_seen_utc
        ) VALUES(
            :model_id, :vintage, :run_id, :directory, :status, :error,
            :present_on_disk, :has_draws, :has_summary, :has_diagnostics,
            :data_hash, :config_hash, :code_version, :result_schema_version,
            :frequency, :missing_data_method, :missing_treatment_exact, :p,
            :seed, :n_observations, :created_at_utc, :metadata_json,
            :first_seen_utc, :last_seen_utc
        )
        ON CONFLICT(model_id, vintage, run_id) DO UPDATE SET
            directory=excluded.directory,
            status=excluded.status,
            error=excluded.error,
            present_on_disk=1,
            has_draws=excluded.has_draws,
            has_summary=excluded.has_summary,
            has_diagnostics=excluded.has_diagnostics,
            data_hash=excluded.data_hash,
            config_hash=excluded.config_hash,
            code_version=excluded.code_version,
            result_schema_version=excluded.result_schema_version,
            frequency=excluded.frequency,
            missing_data_method=excluded.missing_data_method,
            missing_treatment_exact=excluded.missing_treatment_exact,
            p=excluded.p,
            seed=excluded.seed,
            n_observations=excluded.n_observations,
            created_at_utc=excluded.created_at_utc,
            metadata_json=excluded.metadata_json,
            last_seen_utc=excluded.last_seen_utc
        """,
        {
            "model_id": model_id,
            "vintage": vintage,
            "run_id": run_id,
            **values,
            "first_seen_utc": now,
            "last_seen_utc": now,
        },
    )
    return status == "complete", metadata


def _scan_forecasts_for_run(
    con: sqlite3.Connection,
    *,
    model_id: str,
    vintage: str,
    run_id: str,
    run_dir: Path,
    now: str,
) -> tuple[int, int]:
    root = run_dir / "forecasts"
    if not root.is_dir():
        return 0, 0
    seen = invalid = 0
    for forecast_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        seen += 1
        name = forecast_dir.name
        metadata_path = forecast_dir / "forecast_metadata.json"
        draws_path = forecast_dir / "forecast_draws.npz"
        metadata: dict | None = None
        valid = True
        errors: list[str] = []
        if not metadata_path.is_file():
            valid = False
            errors.append("forecast_metadata.json is missing")
        else:
            try:
                metadata = _read_json(metadata_path)
            except RegistryError as exc:
                valid = False
                errors.append(str(exc))
        if not draws_path.is_file():
            valid = False
            errors.append("forecast_draws.npz is missing")
        if metadata is not None and str(metadata.get("forecast_name", name)) != name:
            valid = False
            errors.append(
                f"forecast_name mismatch: metadata={metadata.get('forecast_name')!r}, "
                f"directory={name!r}"
            )

        future_dates = [] if metadata is None else list(metadata.get("future_dates", []))
        row = {
            "model_id": model_id,
            "vintage": vintage,
            "run_id": run_id,
            "forecast_name": name,
            "directory": str(forecast_dir.resolve()),
            "valid": int(valid),
            "error": None if valid else "; ".join(errors),
            "present_on_disk": 1,
            "has_display": int((forecast_dir / "display_v1.parquet").is_file()),
            "display_schema_version": (
                "1.0" if (forecast_dir / "display_v1.parquet").is_file() else None
            ),
            "n_draws": None if metadata is None else metadata.get("n_draws"),
            "n_path_dates": None if metadata is None else metadata.get("n_path_dates"),
            "n_variables": None if metadata is None else metadata.get("n_variables"),
            "horizon": None if metadata is None else metadata.get("H"),
            "tail_length": None if metadata is None else metadata.get("tail_length"),
            "frequency": None if metadata is None else metadata.get("frequency"),
            "simulate_future_outliers": None
            if metadata is None
            else _as_int_bool(metadata.get("simulate_future_outliers")),
            "future_start": future_dates[0] if future_dates else None,
            "future_end": future_dates[-1] if future_dates else None,
            "metadata_json": None if metadata is None else _json(metadata),
            "first_seen_utc": now,
            "last_seen_utc": now,
        }
        con.execute(
            """
            INSERT INTO forecasts(
                model_id, vintage, run_id, forecast_name, directory, valid,
                error, present_on_disk, has_display, display_schema_version,
                n_draws, n_path_dates, n_variables, horizon, tail_length,
                frequency, simulate_future_outliers,
                future_start, future_end, metadata_json, first_seen_utc,
                last_seen_utc
            ) VALUES(
                :model_id, :vintage, :run_id, :forecast_name, :directory,
                :valid, :error, :present_on_disk, :has_display,
                :display_schema_version, :n_draws, :n_path_dates,
                :n_variables, :horizon, :tail_length, :frequency,
                :simulate_future_outliers, :future_start, :future_end,
                :metadata_json, :first_seen_utc, :last_seen_utc
            )
            ON CONFLICT(model_id, vintage, run_id, forecast_name) DO UPDATE SET
                directory=excluded.directory,
                valid=excluded.valid,
                error=excluded.error,
                present_on_disk=1,
                has_display=excluded.has_display,
                display_schema_version=excluded.display_schema_version,
                n_draws=excluded.n_draws,
                n_path_dates=excluded.n_path_dates,
                n_variables=excluded.n_variables,
                horizon=excluded.horizon,
                tail_length=excluded.tail_length,
                frequency=excluded.frequency,
                simulate_future_outliers=excluded.simulate_future_outliers,
                future_start=excluded.future_start,
                future_end=excluded.future_end,
                metadata_json=excluded.metadata_json,
                last_seen_utc=excluded.last_seen_utc
            """,
            row,
        )
        invalid += int(not valid)
    return seen, invalid


def _upsert_aggregate_from_disk(
    con: sqlite3.Connection,
    *,
    vintage: str,
    aggregate_run_id: str,
    run_dir: Path,
    now: str,
) -> bool:
    metadata_path = run_dir / "metadata.json"
    fallback_path = run_dir / "aggregate_config.json"
    metadata: dict | None = None
    errors: list[str] = []
    source = metadata_path if metadata_path.is_file() else fallback_path
    if source.is_file():
        try:
            metadata = _read_json(source)
        except RegistryError as exc:
            errors.append(str(exc))
    else:
        errors.append("metadata.json and aggregate_config.json are missing")

    if metadata is not None:
        meta_id = metadata.get("aggregate_run_id")
        if meta_id is not None and str(meta_id) != aggregate_run_id:
            errors.append(
                f"aggregate_run_id mismatch: metadata={meta_id!r}, path={aggregate_run_id!r}"
            )
        meta_vintage = metadata.get("vintage")
        if meta_vintage is not None and str(meta_vintage) != vintage:
            errors.append(
                f"vintage mismatch: metadata={meta_vintage!r}, path={vintage!r}"
            )

    has_draws = (run_dir / "aggregate_draws.npz").is_file()
    if not has_draws:
        errors.append("aggregate_draws.npz is missing")

    valid = not errors
    forecast_name = (
        "unconditional" if metadata is None else str(metadata.get("forecast_name", "unconditional"))
    )
    row = {
        "vintage": vintage,
        "aggregate_run_id": aggregate_run_id,
        "forecast_name": forecast_name,
        "directory": str(run_dir.resolve()),
        "status": "complete" if valid else "failed",
        "error": None if valid else "; ".join(errors),
        "present_on_disk": 1,
        "has_draws": int(has_draws),
        "has_display": int((run_dir / "display_v1.parquet").is_file()),
        "display_schema_version": (
            "1.0" if (run_dir / "display_v1.parquet").is_file() else None
        ),
        "n_aggregate_draws": None
        if metadata is None
        else metadata.get(
            "n_aggregate_draws_effective", metadata.get("aggregate_posterior_draws")
        ),
        "scenario_active": None
        if metadata is None
        else _as_int_bool(metadata.get("scenario_active")),
        "weekly_tax_mode": None
        if metadata is None
        else metadata.get("weekly_tax_mode_effective"),
        "aggregate_module_version": None
        if metadata is None
        else metadata.get("aggregate_module_version"),
        "created_at_utc": None if metadata is None else metadata.get("created_at_utc"),
        "component_forecast_stores_json": None
        if metadata is None
        else _json(metadata.get("component_forecast_stores", {})),
        "metadata_json": None if metadata is None else _json(metadata),
        "first_seen_utc": now,
        "last_seen_utc": now,
    }
    con.execute(
        """
        INSERT INTO aggregates(
            vintage, aggregate_run_id, forecast_name, directory, status, error,
            present_on_disk, has_draws, has_display, display_schema_version,
            n_aggregate_draws, scenario_active, weekly_tax_mode,
            aggregate_module_version, created_at_utc,
            component_forecast_stores_json, metadata_json, first_seen_utc,
            last_seen_utc
        ) VALUES(
            :vintage, :aggregate_run_id, :forecast_name, :directory, :status,
            :error, :present_on_disk, :has_draws, :has_display,
            :display_schema_version, :n_aggregate_draws, :scenario_active,
            :weekly_tax_mode, :aggregate_module_version,
            :created_at_utc, :component_forecast_stores_json, :metadata_json,
            :first_seen_utc, :last_seen_utc
        )
        ON CONFLICT(vintage, aggregate_run_id) DO UPDATE SET
            forecast_name=excluded.forecast_name,
            directory=excluded.directory,
            status=excluded.status,
            error=excluded.error,
            present_on_disk=1,
            has_draws=excluded.has_draws,
            has_display=excluded.has_display,
            display_schema_version=excluded.display_schema_version,
            n_aggregate_draws=excluded.n_aggregate_draws,
            scenario_active=excluded.scenario_active,
            weekly_tax_mode=excluded.weekly_tax_mode,
            aggregate_module_version=excluded.aggregate_module_version,
            created_at_utc=excluded.created_at_utc,
            component_forecast_stores_json=excluded.component_forecast_stores_json,
            metadata_json=excluded.metadata_json,
            last_seen_utc=excluded.last_seen_utc
        """,
        row,
    )
    return valid


def scan_results(
    *,
    results_root: Path | str,
    registry_path: Path | str | None = None,
) -> ScanReport:
    """Index existing component and aggregate stores without modifying them.

    The operation is idempotent. Rows not seen in the current filesystem pass
    are retained for audit history but marked ``present_on_disk=0``.
    """
    results_root = Path(results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    registry_path = (
        default_registry_path(results_root=results_root)
        if registry_path is None
        else Path(registry_path)
    )
    init_registry(registry_path)
    now = _utcnow()

    known_dirs = set(CANONICAL_MODEL_IDS) | {_AGGREGATE_MODEL_ID}
    unexpected = tuple(
        sorted(
            path.name
            for path in results_root.iterdir()
            if path.is_dir() and path.name not in known_dirs
        )
    )

    run_count = forecast_count = aggregate_count = 0
    invalid_runs = invalid_forecasts = invalid_aggregates = 0

    with _connect(registry_path) as con:
        # Do not delete history: a restored/moved result can become present again
        # on the next scan. Running rows without a store remain valid registry rows.
        con.execute("UPDATE runs SET present_on_disk=0")
        con.execute("UPDATE forecasts SET present_on_disk=0")
        con.execute("UPDATE aggregates SET present_on_disk=0")

        for model_id in CANONICAL_MODEL_IDS:
            model_root = results_root / model_id
            if not model_root.is_dir():
                continue
            for vintage_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
                vintage = vintage_dir.name
                for run_dir in sorted(path for path in vintage_dir.iterdir() if path.is_dir()):
                    run_count += 1
                    valid, _ = _upsert_run_from_disk(
                        con,
                        model_id=model_id,
                        vintage=vintage,
                        run_id=run_dir.name,
                        run_dir=run_dir,
                        now=now,
                    )
                    invalid_runs += int(not valid)
                    seen, bad = _scan_forecasts_for_run(
                        con,
                        model_id=model_id,
                        vintage=vintage,
                        run_id=run_dir.name,
                        run_dir=run_dir,
                        now=now,
                    )
                    forecast_count += seen
                    invalid_forecasts += bad

        aggregate_root = results_root / _AGGREGATE_MODEL_ID
        if aggregate_root.is_dir():
            for vintage_dir in sorted(path for path in aggregate_root.iterdir() if path.is_dir()):
                for run_dir in sorted(path for path in vintage_dir.iterdir() if path.is_dir()):
                    aggregate_count += 1
                    valid = _upsert_aggregate_from_disk(
                        con,
                        vintage=vintage_dir.name,
                        aggregate_run_id=run_dir.name,
                        run_dir=run_dir,
                        now=now,
                    )
                    invalid_aggregates += int(not valid)

        con.execute(
            "INSERT INTO registry_meta(key, value) VALUES('last_scan_utc', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (now,),
        )
        con.execute(
            "INSERT INTO registry_meta(key, value) VALUES('results_root', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(results_root),),
        )

    return ScanReport(
        results_root=results_root,
        registry_path=Path(registry_path),
        component_runs_seen=run_count,
        forecasts_seen=forecast_count,
        aggregates_seen=aggregate_count,
        invalid_component_runs=invalid_runs,
        invalid_forecasts=invalid_forecasts,
        invalid_aggregates=invalid_aggregates,
        unexpected_directories=unexpected,
    )


def _frame(path: Path | str, sql: str, params: Sequence | Mapping = ()) -> pd.DataFrame:
    with _connect(path) as con:
        return pd.read_sql_query(sql, con, params=params)


def list_runs(
    registry_path: Path | str,
    *,
    model_id: str | None = None,
    vintage: str | None = None,
    status: str | None = None,
    present_only: bool = True,
) -> pd.DataFrame:
    """Return component runs in dashboard-friendly tabular form."""
    clauses: list[str] = []
    params: list[object] = []
    if model_id is not None:
        clauses.append("model_id = ?")
        params.append(resolve_model_id(model_id))
    if vintage is not None:
        clauses.append("vintage = ?")
        params.append(str(vintage))
    if status is not None:
        if status not in _RUN_STATUSES:
            raise RegistryError(f"Unknown status {status!r}.")
        clauses.append("status = ?")
        params.append(status)
    if present_only:
        clauses.append("present_on_disk = 1")
    where = "" if not clauses else " WHERE " + " AND ".join(clauses)
    return _frame(
        registry_path,
        "SELECT * FROM runs" + where + " ORDER BY vintage DESC, model_id, created_at_utc DESC, run_id",
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
        params.append(resolve_model_id(model_id))
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
        "SELECT * FROM forecasts" + where + " ORDER BY vintage DESC, model_id, run_id, forecast_name",
        params,
    )


def list_aggregates(
    registry_path: Path | str,
    *,
    vintage: str | None = None,
    forecast_name: str | None = None,
    status: str | None = None,
    present_only: bool = True,
) -> pd.DataFrame:
    clauses: list[str] = []
    params: list[object] = []
    if vintage is not None:
        clauses.append("vintage = ?")
        params.append(str(vintage))
    if forecast_name is not None:
        clauses.append("forecast_name = ?")
        params.append(str(forecast_name))
    if status is not None:
        if status not in _RUN_STATUSES:
            raise RegistryError(f"Unknown status {status!r}.")
        clauses.append("status = ?")
        params.append(status)
    if present_only:
        clauses.append("present_on_disk = 1")
    where = "" if not clauses else " WHERE " + " AND ".join(clauses)
    return _frame(
        registry_path,
        "SELECT * FROM aggregates" + where + " ORDER BY vintage DESC, created_at_utc DESC, aggregate_run_id",
        params,
    )


def run_coverage(
    registry_path: Path | str,
    vintage: str,
    *,
    forecast_name: str = "unconditional",
) -> pd.DataFrame:
    """One row per canonical model showing what is actually usable."""
    rows = []
    with _connect(registry_path) as con:
        for model_id in CANONICAL_MODEL_IDS:
            counts = con.execute(
                """
                SELECT
                    COUNT(*) AS n_runs,
                    SUM(CASE WHEN status='complete' AND present_on_disk=1 THEN 1 ELSE 0 END) AS n_complete,
                    SUM(CASE WHEN has_draws=1 AND status='complete' AND present_on_disk=1 THEN 1 ELSE 0 END) AS n_with_draws
                FROM runs WHERE model_id=? AND vintage=?
                """,
                (model_id, str(vintage)),
            ).fetchone()
            promoted = con.execute(
                """
                SELECT run_id, has_draws FROM runs
                WHERE model_id=? AND vintage=? AND promoted=1
                  AND status='complete' AND present_on_disk=1
                """,
                (model_id, str(vintage)),
            ).fetchone()
            forecast_count = con.execute(
                """
                SELECT COUNT(*) FROM forecasts
                WHERE model_id=? AND vintage=? AND forecast_name=?
                  AND valid=1 AND present_on_disk=1
                """,
                (model_id, str(vintage), str(forecast_name)),
            ).fetchone()[0]
            display_count = con.execute(
                """
                SELECT COUNT(*) FROM forecasts
                WHERE model_id=? AND vintage=? AND forecast_name=?
                  AND valid=1 AND present_on_disk=1 AND has_display=1
                """,
                (model_id, str(vintage), str(forecast_name)),
            ).fetchone()[0]
            promoted_has_forecast = False
            promoted_has_display = False
            if promoted is not None:
                promoted_forecast = con.execute(
                    """
                    SELECT has_display FROM forecasts
                    WHERE model_id=? AND vintage=? AND run_id=?
                      AND forecast_name=? AND valid=1 AND present_on_disk=1
                    """,
                    (model_id, str(vintage), promoted["run_id"], str(forecast_name)),
                ).fetchone()
                promoted_has_forecast = promoted_forecast is not None
                promoted_has_display = bool(
                    promoted_forecast is not None and promoted_forecast["has_display"]
                )
            spec = model_spec(model_id)
            rows.append(
                {
                    "model_id": model_id,
                    "label": spec.label,
                    "aggregate_key": spec.aggregate_key,
                    "n_runs": int(counts["n_runs"] or 0),
                    "n_complete": int(counts["n_complete"] or 0),
                    "n_with_draws": int(counts["n_with_draws"] or 0),
                    f"n_{forecast_name}_forecasts": int(forecast_count or 0),
                    f"n_{forecast_name}_displays": int(display_count or 0),
                    "promoted_run_id": None if promoted is None else promoted["run_id"],
                    "promoted_has_draws": False if promoted is None else bool(promoted["has_draws"]),
                    f"promoted_has_{forecast_name}": promoted_has_forecast,
                    f"promoted_has_{forecast_name}_display": promoted_has_display,
                }
            )
    return pd.DataFrame(rows).set_index("model_id")


def promote_run(
    registry_path: Path | str,
    model_id: str,
    vintage: str,
    run_id: str,
    *,
    required_forecast: str | None = "unconditional",
    require_draws: bool = False,
) -> None:
    """Atomically mark one component run authoritative for a vintage."""
    canonical = resolve_model_id(model_id)
    vintage = str(vintage)
    run_id = str(run_id)
    with _connect(registry_path) as con:
        row = con.execute(
            """
            SELECT * FROM runs
            WHERE model_id=? AND vintage=? AND run_id=?
            """,
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
) -> dict[str, str]:
    """Return canonical ``model_id -> run_id`` mapping for ``run_aggregate``."""
    vintage = str(vintage)
    out: dict[str, str] = {}
    missing: list[str] = []
    with _connect(registry_path) as con:
        for model_id in CANONICAL_MODEL_IDS:
            row = con.execute(
                """
                SELECT r.run_id, r.has_draws
                FROM runs r
                JOIN forecasts f
                  ON f.model_id=r.model_id AND f.vintage=r.vintage AND f.run_id=r.run_id
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


def promote_aggregate(
    registry_path: Path | str,
    vintage: str,
    aggregate_run_id: str,
) -> None:
    """Atomically promote one aggregate within its forecast contract."""
    vintage = str(vintage)
    aggregate_run_id = str(aggregate_run_id)
    with _connect(registry_path) as con:
        row = con.execute(
            "SELECT * FROM aggregates WHERE vintage=? AND aggregate_run_id=?",
            (vintage, aggregate_run_id),
        ).fetchone()
        if row is None:
            raise PromotionError(
                f"Unknown aggregate {vintage}/{aggregate_run_id}."
            )
        if row["status"] != "complete" or not row["present_on_disk"]:
            raise PromotionError("Only complete on-disk aggregate runs can be promoted.")
        forecast_name = row["forecast_name"]
        con.execute(
            "UPDATE aggregates SET promoted=0 WHERE vintage=? AND forecast_name=?",
            (vintage, forecast_name),
        )
        con.execute(
            "UPDATE aggregates SET promoted=1 WHERE vintage=? AND aggregate_run_id=?",
            (vintage, aggregate_run_id),
        )


def get_promoted_aggregate(
    registry_path: Path | str,
    vintage: str,
    *,
    forecast_name: str = "unconditional",
) -> pd.Series | None:
    """Return the promoted aggregate row, or ``None`` when none exists."""
    with _connect(registry_path) as con:
        row = con.execute(
            """
            SELECT * FROM aggregates
            WHERE vintage=? AND forecast_name=? AND promoted=1
              AND status='complete' AND present_on_disk=1
            """,
            (str(vintage), str(forecast_name)),
        ).fetchone()
    return None if row is None else pd.Series(dict(row))


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
    """Create/update durable status for a background estimation run.

    Live progress is intentionally not stored here.  This function is for the
    three durable states needed after process restarts: running, complete,
    failed. A later filesystem scan enriches the same row with metadata/files.
    """
    if status not in _RUN_STATUSES:
        raise RegistryError(f"Unknown status {status!r}.")
    canonical = resolve_model_id(model_id)
    now = _utcnow()
    init_registry(registry_path)
    with _connect(registry_path) as con:
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
