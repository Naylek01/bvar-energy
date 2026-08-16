"""Compatibility helpers for model-side source context.

The project has two processed-vintage manifest layouts:

* legacy/interim manifests record one CSV path per source under ``source_files``;
* the v9 single-workbook builder records ``raw_workbook`` + ``sheet_mapping``.

These helpers let tax-re-attribution adapters read the exact source context from
both layouts without changing notebook calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd




def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_processed_manifest(
    dataset_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[dict, Path]:
    dataset_path = Path(dataset_path)
    path = dataset_path.parent / "manifest.json" if manifest_path is None else Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read processed-vintage manifest: {path}") from exc


def _read_dated_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    date_candidates = [
        column for column in frame.columns
        if str(column).strip().lower() in {"date", "dates", "time", "period"}
    ]
    if not date_candidates:
        raise ValueError(f"No date column found in {path}.")
    date_col = date_candidates[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame = frame.set_index(date_col).sort_index()
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).last()
    return frame.apply(pd.to_numeric, errors="coerce")


def _parse_excel_date(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (pd.Timestamp,)):
        return pd.Timestamp(value).tz_localize(None).normalize()
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and 1 <= numeric <= 100000:
            return (pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric, unit="D")).normalize()
        return pd.NaT
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.startswith("#"):
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce")
    return pd.NaT if pd.isna(parsed) else pd.Timestamp(parsed).tz_localize(None).normalize()


def _resolve_single_workbook(manifest: Mapping, manifest_path: Path) -> tuple[Path, Mapping]:
    raw_value = manifest.get("raw_workbook")
    if not raw_value:
        raise FileNotFoundError(
            "The processed manifest contains neither usable legacy source_files nor "
            "a v9 raw_workbook entry. Rebuild the processed vintage with the current builder."
        )
    raw = Path(str(raw_value))
    if not raw.is_absolute():
        # Prefer the path exactly as recorded; relative manifests are resolved
        # from the project-style processed directory as a conservative fallback.
        candidate = (manifest_path.parent / raw).resolve()
        if candidate.is_file():
            raw = candidate
    if not raw.is_file():
        raise FileNotFoundError(
            f"Raw single-workbook snapshot not found: {raw}. This legacy processed "
            "vintage has no immutable source_context sidecars. Restore the exact "
            "workbook recorded by its manifest, or rebuild the vintage with the "
            "current builder."
        )

    # A living Excel workbook must never silently supply tax/HICP context to an
    # older processed vintage. Current builders write immutable source_files
    # sidecars, so this hash gate applies only to legacy workbook fallbacks.
    expected_hash = str(manifest.get("raw_workbook_sha256") or "").strip().lower()
    if expected_hash:
        actual_hash = _sha256(raw).lower()
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Processed-vintage source context is not immutable: the workbook "
                f"currently found at {raw} has SHA256 {actual_hash}, while the "
                f"manifest records {expected_hash}. Refusing to mix vintages. "
                "Restore the exact workbook or rebuild this processed vintage "
                "with the current builder so source_context sidecars are persisted."
            )

    sheets = manifest.get("sheet_mapping", {})
    if not isinstance(sheets, Mapping):
        raise ValueError("The v9 manifest has an invalid sheet_mapping entry.")
    return raw, sheets


def _read_simple_excel_sheet(
    workbook: Path,
    sheet_name: str,
    *,
    rename: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    raw = pd.read_excel(workbook, sheet_name=sheet_name, dtype=object)
    if raw.empty:
        raise ValueError(f"Workbook sheet {sheet_name!r} is empty.")
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = next(
        (c for c in raw.columns if c.strip().lower() in {"date", "dates", "period", "time"}),
        raw.columns[0],
    )
    dates = pd.DatetimeIndex([_parse_excel_date(v) for v in raw[date_col]])
    valid = ~dates.isna()
    frame = raw.loc[np.asarray(valid)].drop(columns=[date_col]).copy()
    if rename:
        frame = frame.rename(columns=dict(rename))
    frame.columns = [str(c).strip() for c in frame.columns]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.index = pd.DatetimeIndex(dates[valid], name="date")
    if frame.index.has_duplicates:
        conflicts = frame.groupby(level=0).nunique(dropna=True).gt(1).any(axis=1)
        if conflicts.any():
            first = conflicts.index[conflicts][0]
            raise ValueError(f"Conflicting duplicate date in sheet {sheet_name!r}: {first.date()}.")
        frame = frame.groupby(level=0).last()
    return frame.sort_index().dropna(how="all")


def _read_haver_excel_sheet(
    workbook: Path,
    sheet_name: str,
    ticker_map: Mapping[str, str],
) -> pd.DataFrame:
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None, dtype=object)
    if raw.empty:
        raise ValueError(f"Workbook Haver sheet {sheet_name!r} is empty.")

    wanted = {str(ticker).strip().casefold(): str(ticker) for ticker in ticker_map}
    locations: dict[str, tuple[int, int]] = {}
    for row in range(min(len(raw), 60)):
        for col in range(raw.shape[1]):
            value = raw.iat[row, col]
            if pd.isna(value):
                continue
            key = re.sub(r"\s+", " ", str(value)).strip().casefold()
            if key in wanted:
                locations[wanted[key]] = (row, col)

    missing = [ticker for ticker in ticker_map if ticker not in locations]
    if missing:
        raise KeyError(
            f"Haver sheet {sheet_name!r} is missing required ticker(s) {missing}."
        )

    first_ticker_row = min(row for row, _ in locations.values())
    best_dates: list[tuple[int, pd.Timestamp]] = []
    for row in range(max(0, first_ticker_row - 8), first_ticker_row):
        found: list[tuple[int, pd.Timestamp]] = []
        for col, value in enumerate(raw.iloc[row].tolist()):
            text = "" if pd.isna(value) else str(value).strip()
            if re.fullmatch(r"\d{6}", text):
                year, month = int(text[:4]), int(text[4:])
                if 1900 <= year <= 2200 and 1 <= month <= 12:
                    found.append((col, pd.Timestamp(year, month, 1)))
        if len(found) > len(best_dates):
            best_dates = found
    if not best_dates:
        raise ValueError(
            f"Could not locate YYYYMM date headers above the Haver tickers in {sheet_name!r}."
        )

    index = pd.DatetimeIndex([stamp for _, stamp in best_dates], name="date")
    frame = pd.DataFrame(index=index)
    for ticker, output_name in ticker_map.items():
        row, _ = locations[ticker]
        values = [raw.iat[row, col] for col, _stamp in best_dates]
        frame[output_name] = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    return frame.sort_index().dropna(how="all")


def load_manifest_source_frame(
    dataset_path: str | Path,
    source: str,
    *,
    manifest_path: str | Path | None = None,
    haver_ticker_map: Mapping[str, str] | None = None,
    simple_rename: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load one immutable source context, with a guarded workbook fallback.

    Current processed vintages should expose ``manifest['source_files']`` and
    are therefore independent of subsequent edits to the living Excel workbook.
    Legacy vintages may fall back to the workbook only when its SHA256 still
    matches the hash frozen in the processed manifest.
    """
    manifest, resolved_manifest_path = read_processed_manifest(dataset_path, manifest_path)

    legacy_value = manifest.get("source_files", {}).get(source)
    if legacy_value:
        legacy_path = Path(str(legacy_value))
        if not legacy_path.is_absolute():
            legacy_path = (resolved_manifest_path.parent / legacy_path).resolve()
        if legacy_path.is_file():
            contract = manifest.get("source_context_contract", {})
            mode = (
                "processed_vintage_sidecar"
                if isinstance(contract, Mapping)
                and contract.get("mode") == "immutable_processed_vintage_sidecars"
                else "legacy_source_csv"
            )
            return _read_dated_csv(legacy_path), {
                "mode": mode,
                "path": legacy_path,
                "manifest_path": resolved_manifest_path,
            }

    workbook, sheets = _resolve_single_workbook(manifest, resolved_manifest_path)
    sheet_name = sheets.get(source)
    if not sheet_name:
        raise KeyError(
            f"The v9 manifest does not record a sheet for source {source!r}. "
            f"Available mappings: {dict(sheets)}"
        )

    if source == "haver":
        if not haver_ticker_map:
            raise ValueError("haver_ticker_map is required when reading Haver from the v9 workbook.")
        frame = _read_haver_excel_sheet(workbook, str(sheet_name), haver_ticker_map)
    else:
        frame = _read_simple_excel_sheet(
            workbook,
            str(sheet_name),
            rename=simple_rename,
        )
    return frame, {
        "mode": "single_excel_workbook",
        "path": workbook,
        "sheet": str(sheet_name),
        "manifest_path": resolved_manifest_path,
    }


__all__ = [
    "read_processed_manifest",
    "load_manifest_source_frame",
]
