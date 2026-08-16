"""Compact stored regression signatures for dashboard refactors.

Large Plotly vectors are hashed. Only +1/+2/+3 forecast points are retained in
clear text so reviewed diffs stay small and readable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

BASELINE_SCHEMA_VERSION = "inflation-dashboard-baseline-1.0"


def _canonical_bytes(values: Iterable[Any]) -> bytes:
    normalised: list[Any] = []
    for value in values:
        if value is None:
            normalised.append(None)
        elif isinstance(value, (pd.Timestamp, np.datetime64)):
            normalised.append(pd.Timestamp(value).isoformat())
        elif isinstance(value, (float, np.floating)):
            number = float(value)
            normalised.append(None if not np.isfinite(number) else number)
        elif isinstance(value, (int, np.integer, bool, np.bool_)):
            normalised.append(value.item() if hasattr(value, "item") else value)
        else:
            try:
                if pd.isna(value):
                    normalised.append(None)
                    continue
            except (TypeError, ValueError):
                pass
            normalised.append(str(value))
    return json.dumps(
        normalised,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def vector_hash(values: Iterable[Any]) -> str:
    return hashlib.sha256(_canonical_bytes(values)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def figure_signature(fig) -> dict[str, Any]:
    traces = []
    for index, trace in enumerate(fig.data):
        x = list(trace.x) if getattr(trace, "x", None) is not None else []
        y = list(trace.y) if getattr(trace, "y", None) is not None else []
        traces.append(
            {
                "index": index,
                "type": getattr(trace, "type", None),
                "name": getattr(trace, "name", None),
                "mode": getattr(trace, "mode", None),
                "n_x": len(x),
                "n_y": len(y),
                "x_hash": vector_hash(x),
                "y_hash": vector_hash(y),
                "x_first": None if not x else str(x[0]),
                "x_last": None if not x else str(x[-1]),
            }
        )
    title = None
    if getattr(fig.layout, "title", None) is not None:
        title = getattr(fig.layout.title, "text", None)
    return {
        "title": title,
        "trace_count": len(traces),
        "barmode": getattr(fig.layout, "barmode", None),
        "xaxis_type": getattr(fig.layout.xaxis, "type", None),
        "yaxis_title": getattr(getattr(fig.layout.yaxis, "title", None), "text", None),
        "traces": traces,
    }


def display_horizon_snapshot(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = frame.loc[
        (frame["record_type"].astype(str) == "fan")
        & (frame["segment"].astype(str) == "forecast")
    ].copy()
    if rows.empty:
        rows = frame.loc[
            (frame["record_type"].astype(str) == "fan")
            & frame["is_future"].fillna(False).astype(bool)
        ].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")

    output: list[dict[str, Any]] = []
    for (metric, series), block in rows.groupby(["metric", "series"], dropna=False):
        block = block.sort_values("date").head(3)
        for horizon, (_, row) in enumerate(block.iterrows(), start=1):
            output.append(
                {
                    "metric": None if pd.isna(metric) else str(metric),
                    "series": None if pd.isna(series) else str(series),
                    "horizon": horizon,
                    "date": (
                        None if pd.isna(row.get("date"))
                        else pd.Timestamp(row["date"]).isoformat()
                    ),
                    "mean": None if pd.isna(row.get("value")) else float(row["value"]),
                    "q50": None if pd.isna(row.get("q50")) else float(row["q50"]),
                    "q16": None if pd.isna(row.get("q16")) else float(row["q16"]),
                    "q84": None if pd.isna(row.get("q84")) else float(row["q84"]),
                }
            )
    return output


def display_signature(frame: pd.DataFrame, source_path: str | Path) -> dict[str, Any]:
    versions = []
    if "display_schema_version" in frame:
        versions = sorted(set(frame["display_schema_version"].dropna().astype(str)))
    return {
        "file_sha256": file_sha256(source_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "schema_versions": versions,
        "horizon_points_1_2_3": display_horizon_snapshot(frame),
    }


def newest_display(
    results_root: str | Path,
    *,
    include_parts: tuple[str, ...] = (),
    exclude_parts: tuple[str, ...] = (),
) -> Path | None:
    candidates = []
    for path in Path(results_root).rglob("display_v1.parquet"):
        text = str(path)
        if include_parts and not all(part in text for part in include_parts):
            continue
        if any(part in text for part in exclude_parts):
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def capture_display(
    display_path: str | Path,
    destination_dir: str | Path,
    *,
    label: str,
    copy_artifact: bool,
) -> dict[str, Any]:
    source = Path(display_path)
    frame = pd.read_parquet(source)
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if copy_artifact:
        shutil.copy2(source, destination / f"{label}_display_v1.parquet")
    signature = display_signature(frame, source)
    (destination / f"{label}_display_signature.json").write_text(
        json.dumps(signature, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return {"frame": frame, "signature": signature}


def write_figure_signature(fig, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(figure_signature(fig), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "vector_hash",
    "file_sha256",
    "figure_signature",
    "display_horizon_snapshot",
    "display_signature",
    "newest_display",
    "capture_display",
    "write_figure_signature",
]
