"""Build the World Bank Pink Sheet input used by the energy BVAR.

Expected raw layout, for example::

    data/raw/world_bank/2026-08-04/CMO-Historical-Data-Monthly.xlsx

The source-stage output remains in nominal USD per MMBtu. Conversion to EUR
belongs in the later dataset-merging step, using the chosen EUR/USD series.
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
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "world_bank"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "world_bank"


SERIES = {
    "Natural gas, Europe": "wb_natural_gas_europe_usd_mmbtu",
}
EXPECTED_UNITS = {"Natural gas, Europe": "($/mmbtu)"}
DESCRIPTION_LABELS = {"Natural gas, Europe": "Natural Gas (Europe)"}
MISSING_MARKERS = {"", "-", "..", "...", "…"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_latest_raw_file(raw_root: str | Path) -> Path:
    """Find a Pink Sheet workbook in the latest YYYY-MM-DD directory."""
    candidates = []
    for path in Path(raw_root).glob("*/*.xlsx"):
        try:
            vintage = pd.Timestamp(path.parent.name)
        except ValueError:
            continue

        name = path.name.lower()
        if "cmo-historical-data-monthly" in name or "pink_sheet" in name:
            candidates.append((vintage, path))

    if not candidates:
        raise FileNotFoundError(
            f"No World Bank monthly Pink Sheet workbook found below {raw_root}."
        )
    return max(candidates, key=lambda item: item[0])[1]


def _parse_month(value: object) -> pd.Timestamp:
    if isinstance(value, (pd.Timestamp, datetime)):
        value = pd.Timestamp(value)
        return pd.Timestamp(value.year, value.month, 1)

    match = re.fullmatch(r"(\d{4})M(\d{2})", str(value).strip())
    if not match:
        return pd.NaT
    year, month = map(int, match.groups())
    return pd.Timestamp(year, month, 1)


def _workbook_update(raw: pd.DataFrame) -> str | None:
    for value in raw.iloc[:5].to_numpy().ravel():
        if isinstance(value, str) and value.lower().startswith("updated on"):
            return re.sub(r"^updated on\s+", "", value, flags=re.I).strip()
    return None


def _series_description(path: Path, source_name: str) -> tuple[str | None, str | None]:
    raw = pd.read_excel(path, sheet_name="Description", header=None)
    target = DESCRIPTION_LABELS.get(source_name, source_name).lower()

    for _, row in raw.iterrows():
        values = [str(value).strip() for value in row if pd.notna(value)]
        if any(target in value.lower() for value in values):
            description = next(value for value in values if target in value.lower())
            source = values[-1] if len(values) > 1 else None
            return description, source
    return None, None


def _read_prices(path: Path) -> tuple[pd.DataFrame, dict[str, str], str | None]:
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=None)

    header_row = next(
        (
            i
            for i, row in raw.iterrows()
            if set(SERIES).issubset(
                {str(value).strip() for value in row if pd.notna(value)}
            )
        ),
        None,
    )
    if header_row is None:
        raise ValueError("Could not locate the Pink Sheet series header row.")

    headers = [
        str(value).strip() if pd.notna(value) else "date"
        for value in raw.iloc[header_row]
    ]
    units = [
        str(value).strip() if pd.notna(value) else ""
        for value in raw.iloc[header_row + 1]
    ]

    source = raw.iloc[header_row + 2 :].copy()
    source.columns = headers
    source = source.rename(columns={source.columns[0]: "date"})
    source["date"] = source["date"].map(_parse_month)
    source = source[source["date"].notna()].copy()

    data = pd.DataFrame(index=pd.DatetimeIndex(source["date"], name="date"))
    unit_map = {}

    for source_name, output_name in SERIES.items():
        if source_name not in source.columns:
            raise ValueError(f"Series {source_name!r} is absent from Monthly Prices.")

        unit = units[headers.index(source_name)]
        expected = EXPECTED_UNITS[source_name]
        if unit.lower() != expected.lower():
            raise ValueError(
                f"Unexpected unit for {source_name!r}: {unit!r}; expected {expected!r}."
            )

        values = source[source_name].map(
            lambda value: np.nan
            if str(value).strip() in MISSING_MARKERS
            else value
        )
        data[output_name] = pd.to_numeric(values, errors="coerce").to_numpy()
        unit_map[output_name] = unit

    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    return data, unit_map, _workbook_update(raw)


def validate_world_bank_dataset(data: pd.DataFrame) -> None:
    missing = [column for column in SERIES.values() if column not in data]
    if missing:
        raise ValueError(f"Missing output columns: {missing}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("The index must be a DatetimeIndex.")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Monthly dates must be unique and sorted.")
    if not (data.index.day == 1).all():
        raise ValueError("Monthly dates must be stored at month start.")
    if data.isna().any().any():
        raise ValueError(f"Missing observations: {data.isna().sum().to_dict()}")
    if (data <= 0).any().any():
        raise ValueError("Commodity prices must be strictly positive.")

    expected = pd.date_range(data.index.min(), data.index.max(), freq="MS")
    if not data.index.equals(expected):
        gaps = expected.difference(data.index).strftime("%Y-%m").tolist()
        raise ValueError(f"Missing calendar months: {gaps}")


def build_world_bank_dataset(
    raw_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Extract the configured monthly World Bank series and metadata."""
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(path)

    data, units, workbook_updated_on = _read_prices(path)
    validate_world_bank_dataset(data)

    diagnostics = []
    series_metadata = {}
    for source_name, output_name in SERIES.items():
        description, source = _series_description(path, source_name)
        values = data[output_name]
        diagnostics.append(
            {
                "series": output_name,
                "source_label": source_name,
                "unit": units[output_name],
                "first_observation": values.index.min().date().isoformat(),
                "last_observation": values.index.max().date().isoformat(),
                "observations": int(values.notna().sum()),
                "missing_values": int(values.isna().sum()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
        series_metadata[output_name] = {
            "source_label": source_name,
            "unit": units[output_name],
            "description": description,
            "source": source,
        }

    metadata = {
        "workbook_updated_on": workbook_updated_on,
        "units": units,
        "series": series_metadata,
    }
    return data, pd.DataFrame(diagnostics).set_index("series"), metadata


def save_world_bank_dataset(
    data: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metadata: dict[str, object],
    raw_path: str | Path,
    output_root: str | Path,
    file_format: str = "csv",
) -> dict[str, Path]:
    """Save one dated processed vintage and its manifest."""
    raw = Path(raw_path)
    try:
        vintage = pd.Timestamp(raw.parent.name).date().isoformat()
    except ValueError:
        vintage = datetime.now(timezone.utc).date().isoformat()

    destination = Path(output_root) / vintage
    destination.mkdir(parents=True, exist_ok=True)

    if file_format not in {"csv", "parquet"}:
        raise ValueError("file_format must be 'csv' or 'parquet'.")

    suffix = ".csv" if file_format == "csv" else ".parquet"
    data_path = destination / f"world_bank_monthly{suffix}"
    diagnostics_path = destination / f"world_bank_diagnostics{suffix}"
    manifest_path = destination / "manifest.json"

    if file_format == "csv":
        data.to_csv(data_path, date_format="%Y-%m-%d")
        diagnostics.to_csv(diagnostics_path)
    else:
        try:
            data.to_parquet(data_path)
            diagnostics.to_parquet(diagnostics_path)
        except ImportError as error:
            raise ImportError(
                "Parquet output requires pyarrow or fastparquet."
            ) from error

    manifest = {
        "source": "World Bank Commodity Price Data (The Pink Sheet)",
        "raw_file": str(raw),
        "raw_sha256": _sha256(raw),
        "vintage": vintage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_observation": data.index.min().date().isoformat(),
        "last_observation": data.index.max().date().isoformat(),
        "observations": int(len(data)),
        "columns": list(data.columns),
        "construction": {
            "sheet": "Monthly Prices",
            "frequency": "monthly",
            "date_convention": "month start",
            "currency": "nominal USD",
            "fx_conversion": "Deferred to the dataset-merging step",
            "model_role": "Public proxy for the paper's European border-gas price",
        },
        **metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "data": data_path,
        "diagnostics": diagnostics_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the World Bank Pink Sheet monthly dataset."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Optional path to one raw workbook. If omitted, the latest vintage is detected automatically.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Root containing YYYY-MM-DD Pink Sheet workbook vintages (default: {DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    args = parser.parse_args()

    raw_path = args.raw or find_latest_raw_file(args.raw_root)
    data, diagnostics, metadata = build_world_bank_dataset(raw_path)
    paths = save_world_bank_dataset(
        data,
        diagnostics,
        metadata,
        raw_path,
        args.output_root,
        file_format=args.format,
    )

    print(f"Raw file: {raw_path}")
    print(f"Observations: {len(data):,}")
    print(f"Span: {data.index.min().date()} -> {data.index.max().date()}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
