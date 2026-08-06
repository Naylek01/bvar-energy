"""Extract the HICP block and PPI Energy from the local Haver databases.

Install the Haver package in the active environment:

    python -m pip install haver

Run from the project root:

    python src/data_pipeline/haver_offline_v6.py --refresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import Haver
    Haver.path("ini")  # first Haver instruction, before every data call
except ImportError:  # pragma: no cover
    Haver = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "haver"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "haver"
SCRIPT_VERSION = "2026-08-04-v6-offline-cache"

HAVER_SERIES = {
    "hicp_total": "H023HICP@EUDATA",
    "hicp_car_fuels": "H023HN22@EUDATA",
    "hicp_electricity": "H023HW51@EUDATA",
    "hicp_gas": "H023HW52@EUDATA",
    "hicp_liquid_fuels": "H023HW53@EUDATA",
    "hicp_solid_fuels": "H023HW54@EUDATA",
    "hicp_heat_energy": "H023HW55@EUDATA",
    "ppi_energy": "H025PP@G10",
}

SERIES_DESCRIPTION = {
    "hicp_total": "HICP total",
    "hicp_car_fuels": (
        "HICP fuels and lubricants for personal transport equipment"
    ),
    "hicp_electricity": "HICP electricity",
    "hicp_gas": "HICP gas",
    "hicp_liquid_fuels": "HICP liquid fuels",
    "hicp_solid_fuels": "HICP solid fuels",
    "hicp_heat_energy": (
        "HICP other energy for heating and cooling"
    ),
    "ppi_energy": "Producer price index, energy",
}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vintage_key(path: Path) -> tuple[int, str]:
    digits = re.sub(r"\D", "", path.parent.name)
    return (int(digits), path.parent.name) if len(digits) == 8 else (-1, path.parent.name)


def _latest_existing_snapshot(raw_root: Path, filename: str) -> Path | None:
    """Return the newest valid snapshot across all dated raw vintages."""
    candidates = [
        path
        for path in Path(raw_root).glob(f"*/{filename}")
        if path.is_file() and path.stat().st_size > 0
    ]
    return max(candidates, key=_vintage_key) if candidates else None


def _monthly_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.PeriodIndex):
        dates = index.to_timestamp(how="start")
    else:
        dates = pd.to_datetime(index, errors="coerce")

    dates = pd.DatetimeIndex(dates)
    if dates.isna().any():
        bad = list(pd.Index(index)[dates.isna()][:5])
        raise ValueError(f"Haver returned unparseable dates: {bad}")

    return dates.to_period("M").to_timestamp(how="start")


def fetch_haver_series(
    series_name: str,
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Fetch one monthly Haver series using the known working call pattern."""
    if Haver is None:
        raise RuntimeError(
            "The Haver package is unavailable. Run "
            "'python -m pip install haver' in the active environment."
        )

    result = Haver.data(
        codes=[ticker],
        startdate=pd.Timestamp(start).strftime("%Y-%m-%d"),
        enddate=pd.Timestamp(end).strftime("%Y-%m-%d"),
        frequency="monthly",
        aggmode="relaxed",
    )

    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(
            f"Haver did not return a DataFrame for {series_name} ({ticker}). "
            f"Response: {result!r}"
        )
    if result.shape[1] != 1:
        raise RuntimeError(
            f"{series_name} ({ticker}): expected one returned column, "
            f"found {result.shape[1]}: {list(result.columns)}"
        )

    result = result.copy()
    result.columns = [series_name]
    result.index = _monthly_index(result.index)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result[series_name] = pd.to_numeric(result[series_name], errors="coerce")

    full_index = pd.date_range(
        start=pd.Timestamp(start).to_period("M").to_timestamp(how="start"),
        end=pd.Timestamp(end).to_period("M").to_timestamp(how="start"),
        freq="MS",
    )
    result = result.reindex(full_index)
    result.index.name = "date"

    observed = result[series_name].dropna()
    if observed.empty:
        raise ValueError(
            f"{series_name} ({ticker}): Haver returned no numeric observations."
        )

    print(
        f"{ticker:20s} "
        f"{observed.index.min().date()} -> "
        f"{observed.index.max().date()} "
        f"({observed.size:,} observations)"
    )
    return result


def download_haver_panel(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    frames = [
        fetch_haver_series(name, ticker, start, end)
        for name, ticker in HAVER_SERIES.items()
    ]
    panel = pd.concat(frames, axis=1).sort_index()
    panel.index.name = "date"
    return panel.dropna(how="all")


def build_haver_dataset(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    vintage: str | None = None,
    start: str = "1996-01-01",
    end: str | None = None,
    refresh: bool = False,
    offline: bool = False,
) -> dict[str, Path]:
    if refresh and offline:
        raise ValueError("--refresh and --offline cannot be used together.")

    vintage = vintage or datetime.now().strftime("%Y%m%d")
    end = end or datetime.now().strftime("%Y-%m-%d")

    raw_root = Path(raw_root)
    raw_dir = raw_root / vintage
    output_dir = Path(output_root) / vintage
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "haver_hicp_ppi_source.csv"
    requested_raw_path = raw_dir / filename
    raw_path = requested_raw_path

    if not refresh and not raw_path.exists():
        fallback = _latest_existing_snapshot(raw_root, filename)
        if fallback is not None:
            raw_path = fallback
            print(f"Reusing earlier Haver snapshot: {raw_path}")

    if raw_path.exists() and not refresh:
        panel = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        panel.index = _monthly_index(panel.index)
        panel.index.name = "date"
        access_mode = (
            "cache_current_vintage"
            if raw_path == requested_raw_path
            else "cache_previous_vintage"
        )
    else:
        if offline:
            raise FileNotFoundError(
                f"No Haver snapshot was found below {raw_root}. "
                "Run once without --offline or use --refresh on the desk machine."
            )
        raw_path = requested_raw_path
        panel = download_haver_panel(start, end)
        panel.to_csv(raw_path, date_format="%Y-%m-%d")
        access_mode = "Haver.data"

    missing = [name for name in HAVER_SERIES if name not in panel.columns]
    if missing:
        raise ValueError(
            f"The raw Haver snapshot is missing columns: {missing}"
        )

    panel = panel[list(HAVER_SERIES)].apply(pd.to_numeric, errors="coerce")
    panel = panel.sort_index().dropna(how="all")

    if panel.empty:
        raise ValueError(
            "The Haver panel is empty. Delete the cached raw snapshot and "
            "rerun with --refresh."
        )

    monthly_path = output_dir / "haver_monthly.csv"
    diagnostics_path = output_dir / "haver_diagnostics.csv"
    metadata_path = output_dir / "haver_series_metadata.csv"
    manifest_path = output_dir / "manifest.json"

    panel.to_csv(monthly_path, date_format="%Y-%m-%d")

    diagnostics = []
    metadata = []
    for name, ticker in HAVER_SERIES.items():
        observed = panel[name].dropna()
        diagnostics.append(
            {
                "series": name,
                "ticker": ticker,
                "observations": int(observed.size),
                "first_date": observed.index.min().date().isoformat(),
                "last_date": observed.index.max().date().isoformat(),
                "missing_values": int(panel[name].isna().sum()),
            }
        )
        metadata.append(
            {
                "series": name,
                "ticker": ticker,
                "description": SERIES_DESCRIPTION[name],
                "frequency": "monthly",
                "transformation": "none",
            }
        )

    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    pd.DataFrame(metadata).to_csv(metadata_path, index=False)

    manifest = {
        "source": "Haver",
        "script_version": SCRIPT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "vintage": vintage,
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "access_mode": access_mode,
        "raw_file": str(raw_path.resolve()),
        "raw_snapshot_vintage": raw_path.parent.name,
        "requested_raw_vintage": vintage,
        "raw_sha256": _sha256(raw_path),
        "series": HAVER_SERIES,
        "outputs": {
            "monthly": str(monthly_path.resolve()),
            "diagnostics": str(diagnostics_path.resolve()),
            "metadata": str(metadata_path.resolve()),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    outputs = {
        "monthly": monthly_path,
        "diagnostics": diagnostics_path,
        "metadata": metadata_path,
        "manifest": manifest_path,
    }

    print(f"\nHaver vintage: {vintage}")
    print(f"Python:         {sys.executable}")
    print(f"Raw snapshot:   {raw_path}")
    print(
        f"Monthly panel:  {panel.index.min().date()} -> "
        f"{panel.index.max().date()} "
        f"({len(panel):,} rows, {panel.shape[1]} series)"
    )
    for name, path in outputs.items():
        print(f"{name:12s} {path}")

    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vintage", default=None)
    parser.add_argument("--start", default="1996-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> None:
    print(f"Running {Path(__file__).name} [{SCRIPT_VERSION}]")
    args = _parser().parse_args()
    build_haver_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        vintage=args.vintage,
        start=args.start,
        end=args.end,
        refresh=args.refresh,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
