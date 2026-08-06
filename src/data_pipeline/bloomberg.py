"""Build a reproducible Bloomberg interim dataset from the values-only Excel sheet.

Expected project layout
-----------------------
bvar-energy/
├─ src/data_pipeline/bloomberg.py
└─ data/raw/bloomberg/<vintage>/*.xlsx

The script reads Sheet2 only by default. Sheet2 should be a pasted-values copy
of the Bloomberg-linked Sheet1, so the pipeline keeps working when Bloomberg is
not connected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "bloomberg"
RAW_DIR_NAMES = ("bloomberg", "bloomberg-data", "bloomberg_data")
DEFAULT_SHEET = "Sheet2"

SECURITY_SUFFIXES = {
    "index", "comdty", "curncy", "equity", "govt", "corp",
    "mtge", "m-mkt", "pfd", "fund",
}
MISSING_TEXT = {
    "", "#n/a", "#n/a n/a", "#n/a field not applicable",
    "#value!", "#ref!", "#name?", "n/a", "na", "nan", "none",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_vintage(name: str) -> pd.Timestamp | None:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(name, fmt))
        except ValueError:
            pass
    return None


def default_raw_root() -> Path:
    base = PROJECT_ROOT / "data" / "raw"
    for name in RAW_DIR_NAMES:
        path = base / name
        if path.exists():
            return path
    return base / RAW_DIR_NAMES[0]


def find_latest_workbook(raw_root: str | Path) -> Path:
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Bloomberg raw-data folder does not exist: {root}")

    files = [p for p in root.rglob("*.xlsx") if not p.name.startswith("~$")]
    if not files:
        raise FileNotFoundError(f"No Bloomberg .xlsx workbook found below {root}")

    def rank(path: Path) -> tuple[pd.Timestamp, int, int]:
        vintage = parse_vintage(path.parent.name) or pd.Timestamp.min
        preferred_name = int(
            "ticker" in path.name.lower() or "bloomberg" in path.name.lower()
        )
        return vintage, preferred_name, path.stat().st_mtime_ns

    return max(files, key=rank)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def looks_like_ticker(value: object) -> bool:
    text = clean_text(value)
    if not text:
        return False
    parts = text.lower().split()
    return bool(parts and parts[-1] in SECURITY_SUFFIXES)


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "series"


def detect_layout(raw: pd.DataFrame) -> tuple[int, int, int, list[int]]:
    scan_rows = min(50, len(raw))

    field_candidates: list[tuple[int, int]] = []
    for row in range(scan_rows):
        first = clean_text(raw.iat[row, 0]).lower()
        values = [clean_text(v).upper() for v in raw.iloc[row, 1:]]
        field_count = sum(
            bool(v) and (v.startswith("PX_") or v in {"LAST_PRICE", "YLD_YTM_MID"})
            for v in values
        )
        if first in {"date", "dates"} and field_count:
            field_candidates.append((field_count, row))

    if not field_candidates:
        raise ValueError(
            "Could not find the Bloomberg field row. Expected 'Dates' in the "
            "first column and fields such as PX_LAST in the remaining columns."
        )
    field_row = max(field_candidates)[1]

    ticker_candidates: list[tuple[int, int]] = []
    for row in range(max(0, field_row - 12), field_row):
        score = sum(looks_like_ticker(v) for v in raw.iloc[row, 1:])
        ticker_candidates.append((score, row))
    ticker_score, ticker_row = max(ticker_candidates)
    if ticker_score == 0:
        raise ValueError("Could not find Bloomberg tickers above the field row.")

    included = [
        col for col in range(1, raw.shape[1])
        if clean_text(raw.iat[ticker_row, col])
        and clean_text(raw.iat[field_row, col])
    ]
    if not included:
        raise ValueError("No populated Bloomberg series columns were found.")

    description_row = ticker_row - 1
    for row in range(ticker_row - 1, max(-1, ticker_row - 8), -1):
        nonempty = sum(bool(clean_text(raw.iat[row, col])) for col in included)
        if nonempty:
            description_row = row
            break

    return ticker_row, field_row, description_row, included


def to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.map(
        lambda x: np.nan
        if clean_text(x).lower() in MISSING_TEXT
        else x
    )
    return pd.to_numeric(cleaned, errors="coerce")


def unique_series_id(ticker: str, field: str, used: set[str]) -> str:
    base = f"bbg_{slug(ticker)}"
    candidate = base
    if candidate in used or field.upper() != "PX_LAST":
        candidate = f"{base}_{slug(field)}"
    number = 2
    while candidate in used:
        candidate = f"{base}_{slug(field)}_{number}"
        number += 1
    used.add(candidate)
    return candidate


def infer_update_frequency(series: pd.Series) -> tuple[int, float | None, str]:
    valid = series.dropna()
    if valid.empty:
        return 0, None, "missing"

    changes = valid[valid.ne(valid.shift())]
    gaps = changes.index.to_series().diff().dt.days.dropna()
    median_gap = float(gaps.median()) if not gaps.empty else None

    if median_gap is None:
        label = "single_value"
    elif median_gap <= 3:
        label = "daily"
    elif median_gap <= 10:
        label = "weekly"
    elif median_gap <= 45:
        label = "monthly"
    elif median_gap <= 120:
        label = "quarterly"
    elif median_gap <= 250:
        label = "semiannual"
    else:
        label = "annual"

    return int(len(changes)), median_gap, label


def parse_workbook(
    path: str | Path,
    sheet_name: str = DEFAULT_SHEET,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = Path(path)
    workbook = pd.ExcelFile(path, engine="openpyxl")
    if sheet_name not in workbook.sheet_names:
        raise ValueError(
            f"Required values-only sheet {sheet_name!r} not found. "
            f"Available sheets: {workbook.sheet_names}"
        )

    raw = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        header=None,
        dtype=object,
    )
    raw = raw.dropna(axis=1, how="all")
    ticker_row, field_row, description_row, included = detect_layout(raw)

    dates = pd.to_datetime(raw.iloc[field_row + 1 :, 0], errors="coerce")
    valid_dates = dates.notna()
    dates = dates.loc[valid_dates].dt.normalize()

    output = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    metadata_rows: list[dict] = []
    used_ids: set[str] = set()

    for col in included:
        ticker = clean_text(raw.iat[ticker_row, col])
        field = clean_text(raw.iat[field_row, col])
        description = clean_text(raw.iat[description_row, col])
        series_id = unique_series_id(ticker, field, used_ids)

        values = to_numeric(raw.iloc[field_row + 1 :, col])
        values = values.loc[valid_dates]
        values.index = output.index
        output[series_id] = values.to_numpy()

        n_changes, median_gap, inferred_frequency = infer_update_frequency(
            output[series_id]
        )
        valid = output[series_id].dropna()
        metadata_rows.append(
            {
                "series_id": series_id,
                "ticker": ticker,
                "field": field,
                "excel_column": col + 1,
                "description": description,
                "n_observations": int(valid.size),
                "first_date": (
                    valid.index.min().date().isoformat() if not valid.empty else None
                ),
                "last_date": (
                    valid.index.max().date().isoformat() if not valid.empty else None
                ),
                "missing_share": float(output[series_id].isna().mean()),
                "n_value_changes": n_changes,
                "median_days_between_changes": median_gap,
                "inferred_update_frequency": inferred_frequency,
            }
        )

    duplicate_dates = int(output.index.duplicated(keep=False).sum())
    if duplicate_dates:
        output = output.groupby(level=0).last()
    output = output.sort_index()

    metadata = pd.DataFrame(metadata_rows)
    layout = {
        "sheet": sheet_name,
        "ticker_row_excel": ticker_row + 1,
        "description_row_excel": description_row + 1,
        "field_row_excel": field_row + 1,
        "data_start_row_excel": field_row + 2,
        "date_column_excel": 1,
        "series_count": len(metadata),
        "duplicate_date_rows_resolved": duplicate_dates,
    }
    return output, metadata, layout


def validate(panel: pd.DataFrame, metadata: pd.DataFrame) -> None:
    if panel.empty:
        raise ValueError("The Bloomberg output panel is empty.")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("Bloomberg dates are not sorted.")
    if panel.index.has_duplicates:
        raise ValueError("Bloomberg dates still contain duplicates.")
    if not panel.columns.is_unique:
        raise ValueError("Bloomberg output column names are not unique.")
    if metadata["n_observations"].sum() == 0:
        raise ValueError("All Bloomberg series are empty after numeric conversion.")


def write_outputs(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    raw_path: Path,
    output_root: Path,
    layout: dict,
    file_format: str,
) -> dict[str, Path]:
    vintage = raw_path.parent.name
    output_dir = output_root / vintage
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "parquet" if file_format == "parquet" else "csv"
    panel_path = output_dir / f"bloomberg_panel.{suffix}"
    metadata_path = output_dir / "bloomberg_series_metadata.csv"
    diagnostics_path = output_dir / "bloomberg_diagnostics.csv"
    manifest_path = output_dir / "manifest.json"

    export = panel.reset_index()
    if file_format == "parquet":
        export.to_parquet(panel_path, index=False)
    else:
        export.to_csv(panel_path, index=False, date_format="%Y-%m-%d")

    metadata.to_csv(metadata_path, index=False)

    diagnostics = metadata[
        [
            "series_id", "ticker", "field", "n_observations",
            "first_date", "last_date", "missing_share",
            "n_value_changes", "median_days_between_changes",
            "inferred_update_frequency",
        ]
    ].copy()
    diagnostics.to_csv(diagnostics_path, index=False)

    manifest = {
        "source": "Bloomberg",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(raw_path.resolve()),
        "raw_sha256": sha256(raw_path),
        "vintage": vintage,
        "source_sheet": layout["sheet"],
        "policy": (
            "Read the values-only Sheet2 copy; never depend on live Bloomberg "
            "formulas in Sheet1."
        ),
        "layout": layout,
        "date_span": {
            "start": panel.index.min().date().isoformat(),
            "end": panel.index.max().date().isoformat(),
            "rows": int(len(panel)),
        },
        "series": metadata[
            ["series_id", "ticker", "field", "description"]
        ].to_dict(orient="records"),
        "outputs": {
            "panel": panel_path.name,
            "metadata": metadata_path.name,
            "diagnostics": diagnostics_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "panel": panel_path,
        "metadata": metadata_path,
        "diagnostics": diagnostics_path,
        "manifest": manifest_path,
    }


def build(
    raw_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    sheet_name: str = DEFAULT_SHEET,
    file_format: str = "csv",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    raw_path = Path(raw_path)
    panel, metadata, layout = parse_workbook(raw_path, sheet_name=sheet_name)
    validate(panel, metadata)
    outputs = write_outputs(
        panel=panel,
        metadata=metadata,
        raw_path=raw_path,
        output_root=Path(output_root),
        layout=layout,
        file_format=file_format,
    )
    return panel, metadata, outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Bloomberg interim panel from values-only Sheet2."
    )
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    args = parser.parse_args()

    raw_root = args.raw_root or default_raw_root()
    raw_path = args.raw or find_latest_workbook(raw_root)
    panel, metadata, outputs = build(
        raw_path=raw_path,
        output_root=args.output_root,
        sheet_name=args.sheet,
        file_format=args.format,
    )

    print(f"Raw file: {raw_path}")
    print(f"Sheet: {args.sheet}")
    print(f"Rows: {len(panel):,}")
    print(
        f"Span: {panel.index.min().date()} -> {panel.index.max().date()}"
    )
    print(f"Series: {len(metadata)}")
    for row in metadata.itertuples(index=False):
        print(
            f"  {row.series_id}: {row.n_observations:,} obs, "
            f"{row.first_date} -> {row.last_date}, "
            f"inferred {row.inferred_update_frequency}"
        )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
