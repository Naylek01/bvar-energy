"""General inspection helpers for the bvar-energy result tree.

The module is intentionally read-only.  It never modifies model runs, forecast
stores, aggregate stores, display caches, or the SQLite registry.

Supported objects
-----------------
Component run
    results/<model_id>/<vintage>/<run_id>/

Forecast store
    results/<model_id>/<vintage>/<run_id>/forecasts/<forecast_name>/

HICP Energy aggregate
    results/hicp_energy_aggregate/<vintage>/<aggregate_run_id>/

The resolver can work from an explicit path or automatically select a run.
Automatic selection prefers the promoted SQLite run when the registry exists.
If no promotion is available, it falls back to the most recent ``created_at``
recorded in metadata; filesystem modification time is deliberately not used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd


CANONICAL_MODEL_IDS = (
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
    "car_fuels_petrol",
    "car_fuels_diesel",
    "liquid_fuels",
)
AGGREGATE_MODEL_ID = "hicp_energy_aggregate"

MODEL_ALIASES = {
    "petrol": "car_fuels_petrol",
    "car_fuels_petrol": "car_fuels_petrol",
    "diesel": "car_fuels_diesel",
    "car_fuels_diesel": "car_fuels_diesel",
    "gas": "gas",
    "electricity": "electricity",
    "heat": "heat_energy",
    "heat_energy": "heat_energy",
    "solid": "solid_fuels",
    "solid_fuels": "solid_fuels",
    "liquid": "liquid_fuels",
    "liquid_fuels": "liquid_fuels",
    "aggregate": AGGREGATE_MODEL_ID,
    "hicp_energy": AGGREGATE_MODEL_ID,
    AGGREGATE_MODEL_ID: AGGREGATE_MODEL_ID,
}


class InspectionError(RuntimeError):
    """Raised when a result object cannot be resolved or inspected safely."""


@dataclass(frozen=True)
class ResultContext:
    kind: str
    project_root: Path
    results_root: Path
    directory: Path
    run_directory: Path
    model_id: str
    vintage: str
    run_id: str
    forecast_name: str | None = None
    selection_policy: str = "explicit path"

    def as_frame(self) -> pd.DataFrame:
        rows = [
            ("kind", self.kind),
            ("model_id", self.model_id),
            ("vintage", self.vintage),
            ("run_id", self.run_id),
            ("forecast_name", self.forecast_name),
            ("selection_policy", self.selection_policy),
            ("directory", str(self.directory)),
            ("run_directory", str(self.run_directory)),
            ("results_root", str(self.results_root)),
            ("project_root", str(self.project_root)),
        ]
        return pd.DataFrame(rows, columns=["field", "value"]).set_index("field")


# ---------------------------------------------------------------------------
# Project and path resolution
# ---------------------------------------------------------------------------

def find_project_root(start: str | Path | None = None) -> Path:
    """Find a bvar-energy-style project root from ``start`` or the CWD."""
    base = Path.cwd() if start is None else Path(start)
    base = base.expanduser().resolve()
    candidates = [base, *base.parents]
    for candidate in candidates:
        if (candidate / "results").exists() and (candidate / "src").exists():
            return candidate
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise InspectionError(
        f"Could not find project root from {base}. "
        "Pass project_root explicitly or run the notebook inside the project."
    )


def canonical_model_id(model: str) -> str:
    key = str(model).strip().lower()
    try:
        return MODEL_ALIASES[key]
    except KeyError as exc:
        raise InspectionError(
            f"Unknown model {model!r}. Expected one of "
            f"{sorted(set(MODEL_ALIASES))}."
        ) from exc


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InspectionError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InspectionError(f"Expected a JSON object in {path}.")
    return value


def _created_at_value(metadata: Mapping) -> pd.Timestamp:
    for key in ("created_at_utc", "created_at", "timestamp_utc", "timestamp"):
        value = metadata.get(key)
        if value:
            stamp = pd.to_datetime(value, utc=True, errors="coerce")
            if not pd.isna(stamp):
                return stamp
    return pd.Timestamp.min.tz_localize("UTC")


def _registry_path(results_root: Path) -> Path:
    return results_root / "energy_bvar_registry.sqlite3"


def _registry_has_table(path: Path, table: str) -> bool:
    if not path.is_file():
        return False
    try:
        with sqlite3.connect(path) as con:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _registry_promoted_component(
    registry_path: Path,
    model_id: str,
    vintage: str | None,
) -> tuple[str, str] | None:
    if not _registry_has_table(registry_path, "runs"):
        return None
    clauses = [
        "model_id=?",
        "promoted=1",
        "status='complete'",
        "present_on_disk=1",
    ]
    params: list[object] = [model_id]
    if vintage is not None:
        clauses.append("vintage=?")
        params.append(str(vintage))
    sql = (
        "SELECT vintage, run_id FROM runs WHERE "
        + " AND ".join(clauses)
        + " ORDER BY vintage DESC LIMIT 1"
    )
    try:
        with sqlite3.connect(registry_path) as con:
            row = con.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else (str(row[0]), str(row[1]))


def _registry_promoted_aggregate(
    registry_path: Path,
    vintage: str | None,
    forecast_name: str | None,
) -> tuple[str, str] | None:
    if not _registry_has_table(registry_path, "aggregates"):
        return None
    clauses = [
        "promoted=1",
        "status='complete'",
        "present_on_disk=1",
    ]
    params: list[object] = []
    if vintage is not None:
        clauses.append("vintage=?")
        params.append(str(vintage))
    if forecast_name is not None:
        clauses.append("forecast_name=?")
        params.append(str(forecast_name))
    sql = (
        "SELECT vintage, aggregate_run_id FROM aggregates WHERE "
        + " AND ".join(clauses)
        + " ORDER BY vintage DESC LIMIT 1"
    )
    try:
        with sqlite3.connect(registry_path) as con:
            row = con.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else (str(row[0]), str(row[1]))


def _filesystem_component_choice(
    results_root: Path,
    model_id: str,
    vintage: str | None,
) -> tuple[str, str, str]:
    model_root = results_root / model_id
    if not model_root.is_dir():
        raise InspectionError(f"No result directory exists for {model_id!r}: {model_root}")

    vintage_dirs = sorted(
        (p for p in model_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if vintage is not None:
        vintage_dirs = [p for p in vintage_dirs if p.name == str(vintage)]
    if not vintage_dirs:
        raise InspectionError(
            f"No result vintage found for model={model_id!r}, vintage={vintage!r}."
        )

    candidates: list[tuple[pd.Timestamp, str, str]] = []
    for vdir in vintage_dirs:
        for rdir in vdir.iterdir():
            if not rdir.is_dir():
                continue
            metadata_path = rdir / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = _read_json(metadata_path)
            candidates.append(
                (_created_at_value(metadata), vdir.name, rdir.name)
            )
        if candidates and vintage is None:
            # Prefer the newest vintage containing at least one valid metadata store.
            break
    if not candidates:
        raise InspectionError(f"No component runs with metadata.json found below {model_root}.")
    candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    _, selected_vintage, run_id = candidates[0]
    return selected_vintage, run_id, "metadata created_at fallback (no promoted registry row)"


def _filesystem_aggregate_choice(
    results_root: Path,
    vintage: str | None,
) -> tuple[str, str, str]:
    root = results_root / AGGREGATE_MODEL_ID
    if not root.is_dir():
        raise InspectionError(f"No aggregate result directory exists: {root}")
    vintage_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if vintage is not None:
        vintage_dirs = [p for p in vintage_dirs if p.name == str(vintage)]
    candidates: list[tuple[pd.Timestamp, str, str]] = []
    for vdir in vintage_dirs:
        for rdir in vdir.iterdir():
            if not rdir.is_dir():
                continue
            meta_path = rdir / "metadata.json"
            config_path = rdir / "aggregate_config.json"
            if meta_path.is_file():
                meta = _read_json(meta_path)
            elif config_path.is_file():
                meta = _read_json(config_path)
            else:
                continue
            candidates.append((_created_at_value(meta), vdir.name, rdir.name))
        if candidates and vintage is None:
            break
    if not candidates:
        raise InspectionError(f"No aggregate stores found below {root}.")
    candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    _, selected_vintage, run_id = candidates[0]
    return selected_vintage, run_id, "aggregate metadata created_at fallback (no promoted registry row)"


def detect_context(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ResultContext:
    """Detect component run, forecast store, or aggregate run from any child path."""
    raw = Path(path).expanduser().resolve()
    if not raw.exists():
        raise InspectionError(f"Path does not exist: {raw}")
    directory = raw.parent if raw.is_file() else raw

    root = find_project_root(project_root or directory)
    results_root = root / "results"
    try:
        rel = directory.relative_to(results_root)
    except ValueError as exc:
        raise InspectionError(
            f"{directory} is not inside the project results tree {results_root}."
        ) from exc

    parts = rel.parts
    if len(parts) < 3:
        raise InspectionError(
            "Expected path below results/<model>/<vintage>/<run_id>/..."
        )

    model_id, vintage, run_id = parts[:3]
    run_dir = results_root / model_id / vintage / run_id

    if model_id == AGGREGATE_MODEL_ID:
        return ResultContext(
            kind="aggregate_run",
            project_root=root,
            results_root=results_root,
            directory=directory if directory == run_dir else run_dir,
            run_directory=run_dir,
            model_id=model_id,
            vintage=vintage,
            run_id=run_id,
            forecast_name=None,
            selection_policy="explicit path",
        )

    canonical = canonical_model_id(model_id)
    forecast_name = None
    kind = "component_run"
    focus_dir = run_dir
    if len(parts) >= 5 and parts[3] == "forecasts":
        forecast_name = parts[4]
        focus_dir = run_dir / "forecasts" / forecast_name
        kind = "forecast_store"

    return ResultContext(
        kind=kind,
        project_root=root,
        results_root=results_root,
        directory=focus_dir,
        run_directory=run_dir,
        model_id=canonical,
        vintage=vintage,
        run_id=run_id,
        forecast_name=forecast_name,
        selection_policy="explicit path",
    )


def resolve_result(
    *,
    project_root: str | Path | None = None,
    path: str | Path | None = None,
    model: str = "gas",
    vintage: str | None = None,
    run_id: str | None = None,
    forecast_name: str | None = None,
    prefer_promoted: bool = True,
) -> ResultContext:
    """Resolve a result object by path or by model/vintage/run selectors."""
    if path is not None:
        context = detect_context(path, project_root=project_root)
        if forecast_name is not None and context.kind == "component_run":
            forecast_dir = context.run_directory / "forecasts" / str(forecast_name)
            if not forecast_dir.is_dir():
                raise InspectionError(f"Forecast store not found: {forecast_dir}")
            return ResultContext(
                **{
                    **asdict(context),
                    "kind": "forecast_store",
                    "directory": forecast_dir,
                    "forecast_name": str(forecast_name),
                }
            )
        return context

    root = find_project_root(project_root)
    results_root = root / "results"
    model_id = canonical_model_id(model)
    registry = _registry_path(results_root)

    if model_id == AGGREGATE_MODEL_ID:
        if run_id is not None:
            if vintage is None:
                matches = list((results_root / model_id).glob(f"*/{run_id}"))
                if len(matches) != 1:
                    raise InspectionError(
                        "With aggregate run_id supplied and vintage omitted, "
                        f"expected exactly one match; found {len(matches)}."
                    )
                vintage = matches[0].parent.name
            selected_run = str(run_id)
            policy = "explicit aggregate run_id"
        else:
            promoted = (
                _registry_promoted_aggregate(registry, vintage, forecast_name)
                if prefer_promoted
                else None
            )
            if promoted is not None:
                vintage, selected_run = promoted
                policy = "promoted aggregate from SQLite registry"
            else:
                vintage, selected_run, policy = _filesystem_aggregate_choice(
                    results_root, vintage
                )
        run_dir = results_root / model_id / str(vintage) / selected_run
        if not run_dir.is_dir():
            raise InspectionError(f"Resolved aggregate directory does not exist: {run_dir}")
        return ResultContext(
            kind="aggregate_run",
            project_root=root,
            results_root=results_root,
            directory=run_dir,
            run_directory=run_dir,
            model_id=model_id,
            vintage=str(vintage),
            run_id=selected_run,
            forecast_name=None,
            selection_policy=policy,
        )

    if run_id is not None:
        if vintage is None:
            matches = list((results_root / model_id).glob(f"*/{run_id}"))
            if len(matches) != 1:
                raise InspectionError(
                    "With run_id supplied and vintage omitted, expected exactly "
                    f"one match; found {len(matches)}."
                )
            vintage = matches[0].parent.name
        selected_run = str(run_id)
        policy = "explicit component run_id"
    else:
        promoted = (
            _registry_promoted_component(registry, model_id, vintage)
            if prefer_promoted
            else None
        )
        if promoted is not None:
            vintage, selected_run = promoted
            policy = "promoted component run from SQLite registry"
        else:
            vintage, selected_run, policy = _filesystem_component_choice(
                results_root, model_id, vintage
            )

    run_dir = results_root / model_id / str(vintage) / selected_run
    if not run_dir.is_dir():
        raise InspectionError(f"Resolved run directory does not exist: {run_dir}")

    if forecast_name is not None:
        forecast_dir = run_dir / "forecasts" / str(forecast_name)
        if not forecast_dir.is_dir():
            raise InspectionError(f"Forecast store not found: {forecast_dir}")
        kind = "forecast_store"
        directory = forecast_dir
    else:
        kind = "component_run"
        directory = run_dir

    return ResultContext(
        kind=kind,
        project_root=root,
        results_root=results_root,
        directory=directory,
        run_directory=run_dir,
        model_id=model_id,
        vintage=str(vintage),
        run_id=selected_run,
        forecast_name=None if forecast_name is None else str(forecast_name),
        selection_policy=policy,
    )


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------

def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _file_role(path: Path) -> str:
    name = path.name.lower()
    if name == "metadata.json":
        return "run metadata"
    if name == "aggregate_config.json":
        return "aggregate configuration / provenance"
    if name == "summary.parquet" or name == "summary.csv":
        return "posterior summary"
    if name == "diagnostics.parquet" or name == "diagnostics.csv":
        return "MCMC / stability diagnostics"
    if name == "history.parquet" or name == "history.csv":
        return "model input history"
    if name == "draws.npz":
        return "full posterior (Structural-ready)"
    if name == "fit_draws.npz":
        return "compact posterior fitted-path cache"
    if name == "forecast_metadata.json":
        return "forecast metadata"
    if name == "forecast_draws.npz":
        return "forecast predictive draws"
    if name == "hicp_metadata.json":
        return "component HICP metadata"
    if name == "hicp_draws.npz":
        return "component HICP predictive draws"
    if name == "hicp_history.csv":
        return "published component HICP history"
    if name.startswith("display_v") and path.suffix == ".parquet":
        return "dashboard display contract"
    if name == "aggregate_draws.npz":
        return "HICP Energy aggregate draws"
    if name.endswith(".json"):
        return "JSON metadata/configuration"
    if name.endswith(".npz"):
        return "NumPy compressed arrays"
    if name.endswith(".parquet"):
        return "Parquet table"
    if name.endswith(".csv"):
        return "CSV table"
    return "file"


def file_inventory(
    context_or_path: ResultContext | str | Path,
    *,
    recursive: bool = True,
) -> pd.DataFrame:
    """Inventory files with stable relative paths and human-readable sizes."""
    if isinstance(context_or_path, ResultContext):
        base = context_or_path.run_directory
    else:
        base = Path(context_or_path).expanduser().resolve()
        if base.is_file():
            base = base.parent
    iterator = base.rglob("*") if recursive else base.glob("*")
    rows = []
    for path in sorted(p for p in iterator if p.is_file()):
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(path.relative_to(base)),
                "type": path.suffix.lower().lstrip(".") or "file",
                "role": _file_role(path),
                "size_bytes": int(stat.st_size),
                "size": _human_bytes(int(stat.st_size)),
            }
        )
    return pd.DataFrame(rows)


def discover_forecasts(run_directory: str | Path) -> pd.DataFrame:
    """Discover every forecast contract below a component run."""
    run_dir = Path(run_directory)
    root = run_dir / "forecasts"
    rows = []
    if not root.is_dir():
        return pd.DataFrame(
            columns=[
                "forecast_name", "valid", "n_draws", "horizon", "tail_length",
                "frequency", "future_start", "future_end", "has_hicp",
                "has_display", "directory",
            ]
        )
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = directory / "forecast_metadata.json"
        valid = meta_path.is_file() and (directory / "forecast_draws.npz").is_file()
        meta = _read_json(meta_path) if meta_path.is_file() else {}
        future_dates = pd.to_datetime(meta.get("future_dates", []), errors="coerce")
        rows.append(
            {
                "forecast_name": directory.name,
                "valid": bool(valid),
                "n_draws": meta.get("n_draws"),
                "horizon": meta.get("horizon"),
                "tail_length": meta.get("tail_length"),
                "frequency": meta.get("frequency"),
                "future_start": (
                    pd.Timestamp(future_dates[0]) if len(future_dates) else pd.NaT
                ),
                "future_end": (
                    pd.Timestamp(future_dates[-1]) if len(future_dates) else pd.NaT
                ),
                "has_hicp": (
                    (directory / "hicp_metadata.json").is_file()
                    and (directory / "hicp_draws.npz").is_file()
                ),
                "has_display": any(directory.glob("display_v*.parquet")),
                "directory": str(directory),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# JSON / Parquet / NPZ readers
# ---------------------------------------------------------------------------

def flatten_mapping(
    value: Mapping,
    *,
    prefix: str = "",
    separator: str = ".",
) -> pd.DataFrame:
    """Flatten nested JSON-like metadata into a two-column audit table."""
    rows: list[tuple[str, object]] = []

    def walk(obj, key: str) -> None:
        if isinstance(obj, Mapping):
            if not obj:
                rows.append((key, {}))
            for child_key, child_value in obj.items():
                new_key = str(child_key) if not key else f"{key}{separator}{child_key}"
                walk(child_value, new_key)
        elif isinstance(obj, (list, tuple)):
            if len(obj) <= 12 and all(
                not isinstance(item, (Mapping, list, tuple)) for item in obj
            ):
                rows.append((key, list(obj)))
            else:
                rows.append((key, f"<{type(obj).__name__}: {len(obj)} items>"))
        else:
            rows.append((key, obj))

    walk(value, prefix)
    frame = pd.DataFrame(rows, columns=["field", "value"])
    if len(frame):
        frame = frame.set_index("field")
    return frame


def inspect_json(path: str | Path, *, flatten: bool = True):
    """Read a JSON object; optionally return a flattened DataFrame."""
    data = _read_json(Path(path))
    return flatten_mapping(data) if flatten else data


def load_table(path: str | Path) -> pd.DataFrame:
    """Load Parquet or CSV using the natural pandas representation."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise InspectionError(f"Unsupported table format: {path}")


def table_manifest(path: str | Path) -> pd.DataFrame:
    """Return table shape/schema without reading all Parquet values when possible."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            rows = int(pf.metadata.num_rows)
            columns = list(pf.schema.names)
            dtypes = [str(pf.schema_arrow.field(name).type) for name in columns]
            return pd.DataFrame(
                {
                    "column": columns,
                    "dtype": dtypes,
                    "rows": rows,
                }
            ).set_index("column")
        except Exception:
            frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, nrows=50)
        return pd.DataFrame(
            {
                "column": list(frame.columns),
                "dtype_sample": [str(dtype) for dtype in frame.dtypes],
            }
        ).set_index("column")
    else:
        raise InspectionError(f"Unsupported table format: {path}")

    return pd.DataFrame(
        {
            "column": list(frame.columns),
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "rows": len(frame),
        }
    ).set_index("column")


def _npz_member_header(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[tuple, str, bool]:
    with archive.open(info, "r") as fh:
        version = np.lib.format.read_magic(fh)
        shape, fortran_order, dtype = np.lib.format._read_array_header(fh, version)
    return tuple(shape), str(dtype), bool(fortran_order)


def npz_manifest(path: str | Path) -> pd.DataFrame:
    """Inspect NPZ array names, shapes and dtypes without materialising arrays."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise InspectionError(f"Expected .npz file: {path}")
    rows = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".npy"):
                    continue
                key = Path(info.filename).stem
                try:
                    shape, dtype, fortran = _npz_member_header(archive, info)
                except Exception:
                    # Extremely old or unusual NPY payload: safe fallback.
                    with np.load(path, allow_pickle=False) as npz:
                        arr = npz[key]
                        shape, dtype, fortran = arr.shape, str(arr.dtype), False
                rows.append(
                    {
                        "array": key,
                        "shape": str(tuple(shape)),
                        "dtype": dtype,
                        "fortran_order": fortran,
                        "storage": _human_bytes(info.compress_size),
                        "uncompressed": _human_bytes(info.file_size),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise InspectionError(f"Invalid NPZ archive {path}: {exc}") from exc
    return pd.DataFrame(rows).set_index("array") if rows else pd.DataFrame()


def summarize_npz_array(
    path: str | Path,
    key: str,
    *,
    quantiles: Sequence[float] = (0.05, 0.16, 0.50, 0.84, 0.95),
    max_values: int = 2_000_000,
) -> pd.Series:
    """Numerically summarize one NPZ member with bounded-memory sampling."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        if key not in npz.files:
            raise InspectionError(f"{key!r} not found in {path.name}. Available: {npz.files}")
        arr = np.asarray(npz[key])
    flat = arr.reshape(-1)
    original_n = int(flat.size)
    if original_n > max_values:
        indices = np.linspace(0, original_n - 1, max_values, dtype=np.int64)
        sample = flat[indices]
        sampled = True
    else:
        sample = flat
        sampled = False

    numeric = np.issubdtype(sample.dtype, np.number) or np.issubdtype(
        sample.dtype, np.datetime64
    )
    result = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "values": original_n,
        "sampled_for_stats": sampled,
        "sample_values": int(sample.size),
        "memory_if_uncompressed": _human_bytes(int(arr.nbytes)),
    }
    if np.issubdtype(sample.dtype, np.datetime64):
        finite = ~pd.isna(sample)
        result.update(
            {
                "missing": int((~finite).sum()),
                "min": sample[finite].min() if finite.any() else pd.NaT,
                "max": sample[finite].max() if finite.any() else pd.NaT,
            }
        )
        return pd.Series(result, name=key)

    if numeric:
        numeric_sample = sample.astype(float, copy=False)
        finite = np.isfinite(numeric_sample)
        values = numeric_sample[finite]
        result["non_finite"] = int((~finite).sum())
        if values.size:
            result.update(
                {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            )
            for q, value in zip(quantiles, np.quantile(values, quantiles)):
                result[f"q{int(round(100*q)):02d}"] = float(value)
    return pd.Series(result, name=key)


# ---------------------------------------------------------------------------
# Forecast comparison helpers
# ---------------------------------------------------------------------------

def _central_interval(interval: float) -> tuple[float, float]:
    interval = float(interval)
    if not 0 < interval < 1:
        raise InspectionError("interval must lie in (0, 1).")
    alpha = (1.0 - interval) / 2.0
    return alpha, 1.0 - alpha


def _load_forecast_product(
    forecast_dir: Path,
    product: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, str]:
    product = str(product)
    if product in {"hicp_yoy", "hicp_level"}:
        meta = _read_json(forecast_dir / "hicp_metadata.json")
        key = "hicp_yoy_paths" if product == "hicp_yoy" else "hicp_level_paths"
        with np.load(forecast_dir / "hicp_draws.npz", allow_pickle=False) as npz:
            paths = np.asarray(npz[key], dtype=float)
        dates = pd.DatetimeIndex(pd.to_datetime(meta["path_dates"]), name="date")
        label = "HICP YoY (%)" if product == "hicp_yoy" else "HICP index"
        return dates, paths, label

    if product.startswith("level:"):
        variable = product.split(":", 1)[1]
        meta = _read_json(forecast_dir / "forecast_metadata.json")
        variables = list(meta.get("variables", []))
        if variable not in variables:
            raise InspectionError(
                f"Variable {variable!r} not in forecast variables {variables}."
            )
        with np.load(forecast_dir / "forecast_draws.npz", allow_pickle=False) as npz:
            paths3 = np.asarray(npz["level_paths"], dtype=float)
        dates = pd.DatetimeIndex(pd.to_datetime(meta["path_dates"]), name="date")
        paths = paths3[:, :, variables.index(variable)]
        return dates, paths, f"{variable} level"

    raise InspectionError(
        "product must be 'hicp_yoy', 'hicp_level', or 'level:<variable>'."
    )


def compare_forecasts(
    run_directory: str | Path,
    *,
    forecast_names: Sequence[str] | None = None,
    product: str = "hicp_yoy",
    interval: float = 0.68,
) -> pd.DataFrame:
    """Compare conditional/unconditional/etc. forecast contracts on one metric."""
    run_dir = Path(run_directory)
    available = discover_forecasts(run_dir)
    if forecast_names is None:
        names = available.loc[available["valid"], "forecast_name"].tolist()
    else:
        names = [str(name) for name in forecast_names]
    if not names:
        raise InspectionError("No valid forecast stores were found.")

    lo, hi = _central_interval(interval)
    pieces = []
    for name in names:
        fdir = run_dir / "forecasts" / name
        if not fdir.is_dir():
            raise InspectionError(f"Forecast store does not exist: {fdir}")
        dates, paths, label = _load_forecast_product(fdir, product)
        quant = np.nanquantile(paths, [lo, 0.50, hi], axis=0)
        pieces.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "forecast_name": name,
                    "lower": quant[0],
                    "median": quant[1],
                    "upper": quant[2],
                    "n_draws": paths.shape[0],
                    "product": label,
                    "interval": interval,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def plot_forecast_comparison(
    comparison: pd.DataFrame,
    *,
    title: str | None = None,
):
    """Plot medians and credible bands using matplotlib default color cycle."""
    import matplotlib.pyplot as plt

    required = {"date", "forecast_name", "lower", "median", "upper", "product"}
    missing = required.difference(comparison.columns)
    if missing:
        raise InspectionError(f"Comparison table is missing {sorted(missing)}.")

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, group in comparison.groupby("forecast_name", sort=False):
        group = group.sort_values("date")
        line = ax.plot(group["date"], group["median"], label=name)[0]
        ax.fill_between(
            group["date"],
            group["lower"],
            group["upper"],
            alpha=0.16,
            color=line.get_color(),
        )
    ax.axhline(0.0, linewidth=0.8, alpha=0.35)
    ax.set_title(title or str(comparison["product"].iloc[0]))
    ax.set_xlabel("Date")
    ax.set_ylabel(str(comparison["product"].iloc[0]))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# High-level summaries
# ---------------------------------------------------------------------------

def run_overview(context: ResultContext) -> pd.DataFrame:
    """Compact human-readable overview for a component or aggregate result."""
    run_dir = context.run_directory
    metadata = {}
    for candidate in ("metadata.json", "aggregate_config.json"):
        path = run_dir / candidate
        if path.is_file():
            metadata.update(_read_json(path))

    rows = {
        "type": context.kind,
        "model": context.model_id,
        "vintage": context.vintage,
        "run_id": context.run_id,
        "forecast_focus": context.forecast_name,
        "selection": context.selection_policy,
        "frequency": metadata.get("frequency"),
        "lags": metadata.get("p"),
        "seed": metadata.get("seed"),
        "missing_data_method": metadata.get("missing_data_method"),
        "code_version": metadata.get("code_version"),
        "data_hash": metadata.get("data_hash"),
        "config_hash": metadata.get("config_hash"),
        "full_posterior": (run_dir / "draws.npz").is_file(),
        "fit_cache": (run_dir / "fit_draws.npz").is_file(),
        "forecast_count": (
            len(discover_forecasts(run_dir))
            if context.model_id != AGGREGATE_MODEL_ID
            else np.nan
        ),
    }

    if context.model_id == AGGREGATE_MODEL_ID:
        config_path = run_dir / "aggregate_config.json"
        if config_path.is_file():
            config = _read_json(config_path)
            rows.update(
                {
                    "forecast_name": config.get("forecast_name"),
                    "aggregate_draws_effective": config.get("n_aggregate_draws_effective"),
                    "weekly_tax_mode": config.get("weekly_tax_mode_effective"),
                    "component_forecast_stores": len(
                        config.get("component_forecast_stores", {})
                    ),
                }
            )

    return pd.DataFrame(
        [(key, value) for key, value in rows.items()],
        columns=["field", "value"],
    ).set_index("field")


def result_health(context: ResultContext) -> pd.DataFrame:
    """Presence checks for the minimum files expected by each result type."""
    run_dir = context.run_directory
    if context.model_id == AGGREGATE_MODEL_ID:
        checks = [
            ("aggregate_draws", run_dir / "aggregate_draws.npz", True),
            ("aggregate_config", run_dir / "aggregate_config.json", True),
            ("display_cache", next(iter(run_dir.glob("display_v*.parquet")), None), False),
        ]
    else:
        checks = [
            ("metadata", run_dir / "metadata.json", True),
            ("history", _first_existing(run_dir, ("history.parquet", "history.csv")), True),
            ("summary", _first_existing(run_dir, ("summary.parquet", "summary.csv")), True),
            ("diagnostics", _first_existing(run_dir, ("diagnostics.parquet", "diagnostics.csv")), True),
            ("full_posterior", run_dir / "draws.npz", True),
            ("fit_cache", run_dir / "fit_draws.npz", False),
            ("forecasts_dir", run_dir / "forecasts", True),
        ]
    rows = []
    for name, path, required in checks:
        exists = path is not None and Path(path).exists()
        rows.append(
            {
                "check": name,
                "required": required,
                "present": bool(exists),
                "path": None if path is None else str(path),
            }
        )
    return pd.DataFrame(rows).set_index("check")


def _first_existing(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


__all__ = [
    "AGGREGATE_MODEL_ID",
    "CANONICAL_MODEL_IDS",
    "InspectionError",
    "ResultContext",
    "canonical_model_id",
    "compare_forecasts",
    "detect_context",
    "discover_forecasts",
    "file_inventory",
    "find_project_root",
    "flatten_mapping",
    "inspect_json",
    "load_table",
    "npz_manifest",
    "plot_forecast_comparison",
    "resolve_result",
    "result_health",
    "run_overview",
    "summarize_npz_array",
    "table_manifest",
]
