"""Download and prepare Eurostat household gas and electricity prices.

Eurostat is now used only for:
- household natural-gas prices, consumption band D2;
- household electricity prices, consumption band DC.

HICP and PPI Energy are supplied by Haver and are intentionally absent here.

Run directly from VS Code or from the project root:

    python src/data_pipeline/eurostat_energy_offline_v2.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "eurostat"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "eurostat"

SDMX_TSV_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/"
    "sdmx/2.1/data/{dataset}?format=tsv&compressed=true"
)
SCRIPT_VERSION = "2026-08-04-energy-only-v2-offline-cache"

DATASETS = (
    "nrg_pc_202_h",
    "nrg_pc_202",
    "nrg_pc_204_h",
    "nrg_pc_204",
)

# EA is the chain-linked aggregate whose membership changes over time.  The
# numbered aggregates are fallbacks.  Selection is made only among real TSV
# rows that contain observations.
GEO_PREFERENCE = (
    "EA",
    "EA21",
    "EA20",
    "EA19",
    "EA18",
    "EA17",
    "EA16",
    "EA15",
    "EA13",
    "EA12",
    "EA11",
)

PRICE_UNIT_PREFERENCE = ("KWH", "MWH", "MWH_GCV", "GJ_GCV", "GJ")

ENERGY_SPECS = {
    "nrg_pc_202_h": {
        "prefix": "estat_gas_household",
        "product_preference": ("4100",),
        "consumption_dimensions": ("consom", "nrg_cons"),
        "consumption_codes": ("4141100", "GJ20-199", "D2"),
    },
    "nrg_pc_202": {
        "prefix": "estat_gas_household",
        "product_preference": ("4100",),
        "consumption_dimensions": ("nrg_cons", "consom"),
        "consumption_codes": ("GJ20-199", "4141100", "D2"),
    },
    "nrg_pc_204_h": {
        "prefix": "estat_electricity_household",
        "product_preference": ("6000",),
        "consumption_dimensions": ("consom", "nrg_cons"),
        "consumption_codes": ("4161150", "KWH2500-4999", "DC"),
    },
    "nrg_pc_204": {
        "prefix": "estat_electricity_household",
        "product_preference": ("6000",),
        "consumption_dimensions": ("nrg_cons", "consom"),
        "consumption_codes": ("KWH2500-4999", "4161150", "DC"),
    },
}

NUMBER_RE = re.compile(
    r"^[\s\u00a0]*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


# ---------------------------------------------------------------------------
# Files and downloads
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_eurostat_tsv(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            compressed = file.read(2) == b"\x1f\x8b"
        opener = gzip.open if compressed else open
        with opener(path, "rt", encoding="utf-8-sig", errors="replace") as file:
            header = file.readline()
    except (OSError, UnicodeError):
        return False
    return "\\TIME_PERIOD" in header and "\t" in header


def _find_dataset_file(folder: Path, dataset: str) -> Path | None:
    exact = (
        f"estat_{dataset}.tsv.gz",
        f"{dataset}.tsv.gz",
        f"estat_{dataset}.tsv",
        f"{dataset}.tsv",
    )
    for name in exact:
        matches = [path for path in folder.rglob(name) if path.is_file()]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime_ns)

    candidates = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and dataset.lower() in path.name.lower()
        and path.name.lower().endswith((".tsv", ".tsv.gz"))
        and _is_eurostat_tsv(path)
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def _download_dataset(
    dataset: str,
    destination: Path,
    session: requests.Session,
    *,
    timeout: int = 300,
) -> str:
    url = SDMX_TSV_URL.format(dataset=dataset)
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            content = response.content
            if not content:
                raise RuntimeError("empty response")

            # Depending on the HTTP headers, requests may already have removed
            # the transport compression.  Store one real gzip file either way.
            if content[:2] != b"\x1f\x8b":
                if b"\\TIME_PERIOD" not in content[:1000] or b"\t" not in content[:1000]:
                    preview = content[:300].decode("utf-8", errors="replace")
                    raise RuntimeError(f"response is not a Eurostat TSV: {preview!r}")
                content = gzip.compress(content)

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(content)
            if not _is_eurostat_tsv(temporary):
                temporary.unlink(missing_ok=True)
                raise RuntimeError("downloaded gzip does not contain a valid Eurostat TSV")
            temporary.replace(destination)
            return response.url

        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)

    raise RuntimeError(
        f"Could not download {dataset} from Eurostat after three attempts. "
        f"URL: {url}. Last error: {last_error}"
    )


def _prepare_raw_files(
    raw_root: Path,
    vintage: str,
    *,
    refresh: bool,
    offline: bool,
) -> tuple[Path, dict[str, Path], dict[str, str]]:
    if refresh and offline:
        raise ValueError("--refresh and --offline cannot be used together.")

    raw_root = Path(raw_root)
    raw_dir = raw_root / vintage
    raw_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    requests_used: dict[str, str] = {}

    session = requests.Session()
    session.headers.update({"User-Agent": "bvar-energy-eurostat-pipeline/1.0"})

    for dataset in DATASETS:
        existing = None if refresh else _find_dataset_file(raw_dir, dataset)

        if existing is None and not refresh:
            # Search all earlier vintages. This makes --offline reusable on a
            # later day without copying the raw files into today's folder.
            existing = _find_dataset_file(raw_root, dataset)
            if existing is not None and existing.parent != raw_dir:
                print(f"Reusing earlier Eurostat snapshot for {dataset}: {existing}")

        if existing is None:
            if offline:
                raise FileNotFoundError(
                    f"Missing {dataset} below {raw_root}. "
                    "Offline mode does not download missing files."
                )
            target = raw_dir / f"estat_{dataset}.tsv.gz"
            requests_used[dataset] = _download_dataset(dataset, target, session)
            existing = target

        files[dataset] = existing

    return raw_dir, files, requests_used


# ---------------------------------------------------------------------------
# TSV parsing and real-row selection
# ---------------------------------------------------------------------------


def _read_tsv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        na_filter=False,
        compression="infer",
        low_memory=False,
    )
    raw.columns = [str(column).strip().lstrip("\ufeff") for column in raw.columns]
    key_column = raw.columns[0]
    dimensions = key_column.split("\\", 1)[0].split(",")

    dims = raw[key_column].astype(str).str.split(",", expand=True)
    if dims.shape[1] != len(dimensions):
        raise ValueError(
            f"Invalid Eurostat series key in {path.name}: expected "
            f"{len(dimensions)} dimensions, found {dims.shape[1]}."
        )
    dims.columns = [dimension.strip() for dimension in dimensions]
    dims = dims.apply(lambda column: column.astype(str).str.strip())

    values = raw.drop(columns=key_column)
    values.columns = [str(column).strip() for column in values.columns]
    return dims, values


def _find_dimension_and_code(
    dims: pd.DataFrame,
    dimension_candidates: Iterable[str],
    code_candidates: Iterable[str],
    *,
    dataset: str,
    role: str,
) -> tuple[str, str]:
    """Find a contract code without assuming one fixed dimension name."""
    dimension_names = [
        name for name in dimension_candidates if name in dims.columns
    ]

    if not dimension_names:
        wanted = set(code_candidates)
        dimension_names = [
            column
            for column in dims.columns
            if wanted.intersection(set(dims[column].unique()))
        ]

    matches: list[tuple[str, str]] = []
    for dimension in dimension_names:
        available = set(dims[dimension].unique())
        for code in code_candidates:
            if code in available:
                matches.append((dimension, code))
                break

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        dimension_order = tuple(dimension_candidates)
        code_order = tuple(code_candidates)
        return min(
            matches,
            key=lambda item: (
                dimension_order.index(item[0])
                if item[0] in dimension_order else len(dimension_order),
                code_order.index(item[1]),
            ),
        )

    available = {
        column: sorted(dims[column].unique())[:30]
        for column in dims.columns
    }
    raise ValueError(
        f"{dataset}: could not locate the {role}. "
        f"Expected one of {tuple(code_candidates)} in dimensions "
        f"{tuple(dimension_candidates)}. Available dimensions/codes: {available}"
    )


def _parse_period(value: object) -> pd.Timestamp | None:
    text = str(value).strip()
    patterns = (
        (r"^(\d{4})[-]?M(\d{2})$", lambda y, p: pd.Timestamp(int(y), int(p), 1)),
        (r"^(\d{4})-(\d{2})$", lambda y, p: pd.Timestamp(int(y), int(p), 1)),
        (r"^(\d{4})[-]?[SH]([12])$", lambda y, p: pd.Timestamp(int(y), 1 if p == "1" else 7, 1)),
        (r"^(\d{4})[-]?Q([1-4])$", lambda y, p: pd.Timestamp(int(y), 3 * (int(p) - 1) + 1, 1)),
        (r"^(\d{4})$", lambda y, _: pd.Timestamp(int(y), 1, 1)),
    )
    for pattern, builder in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            groups = match.groups()
            return builder(groups[0], groups[1] if len(groups) > 1 else "")
    return None


def _parse_cell(value: object) -> tuple[float, str]:
    text = str(value).strip().replace("\u00a0", " ")
    if not text or text.startswith(":"):
        return np.nan, text[1:].strip() if text.startswith(":") else ""
    match = NUMBER_RE.match(text)
    if not match:
        return np.nan, text
    return float(match.group(1)), text[match.end() :].strip()


def _row_to_series(
    row: pd.Series,
    name: str,
) -> tuple[pd.Series, dict[str, int | str | None]]:
    observations: dict[pd.Timestamp, float] = {}
    flags = 0
    source_missing = 0

    for period, cell in row.items():
        date = _parse_period(period)
        if date is None:
            continue
        value, flag = _parse_cell(cell)
        observations[date] = value
        flags += bool(flag)
        source_missing += pd.isna(value)

    series = pd.Series(observations, name=name, dtype=float).sort_index()
    series.index = pd.DatetimeIndex(series.index, name="date")
    observed = series.dropna()
    stats: dict[str, int | str | None] = {
        "observations": int(observed.size),
        "flags": int(flags),
        "source_missing": int(source_missing),
        "first_date": observed.index.min().date().isoformat() if not observed.empty else None,
        "last_date": observed.index.max().date().isoformat() if not observed.empty else None,
    }
    return series, stats


def _preference_rank(value: str, preference: Iterable[str]) -> int:
    ordered = tuple(preference)
    try:
        return ordered.index(value)
    except ValueError:
        return len(ordered) + 100


def _ea_mask(dims: pd.DataFrame, requested_geo: str | None) -> pd.Series:
    if "geo" not in dims:
        raise KeyError(f"The dataset has no geo dimension: {list(dims.columns)}")
    if requested_geo:
        mask = dims["geo"].eq(requested_geo)
        if not mask.any():
            raise ValueError(
                f"Requested geography {requested_geo!r} is unavailable. "
                f"Available euro-area codes: "
                f"{sorted(code for code in dims['geo'].unique() if code.startswith('EA'))}"
            )
        return mask
    return dims["geo"].str.startswith("EA")


def _convert_price_to_eur_kwh(series: pd.Series, unit: str) -> pd.Series:
    unit = unit.upper()
    if unit == "KWH":
        return series
    if unit in {"MWH", "MWH_GCV"}:
        return series / 1000.0
    if unit in {"GJ", "GJ_GCV", "GJ_NCV"}:
        return series / 277.7777777778
    raise ValueError(f"Unsupported household-price unit {unit!r}.")


# ---------------------------------------------------------------------------
# Dataset-specific extractors
# ---------------------------------------------------------------------------


def _select_energy_contract(
    dataset: str,
    dims: pd.DataFrame,
    values: pd.DataFrame,
    *,
    requested_geo: str | None,
) -> tuple[dict[str, str], str]:
    spec = ENERGY_SPECS[dataset]
    consumption_dimension, consumption_code = _find_dimension_and_code(
        dims,
        spec["consumption_dimensions"],
        spec["consumption_codes"],
        dataset=dataset,
        role="household consumption band",
    )

    # Product is not mandatory. Some historical files omit the dimension
    # because the dataset already identifies gas or electricity.
    mask = dims[consumption_dimension].eq(consumption_code)
    if "freq" in dims:
        mask &= dims["freq"].eq("S")
    if "currency" in dims and "EUR" in set(dims["currency"].unique()):
        mask &= dims["currency"].eq("EUR")
    mask &= _ea_mask(dims, requested_geo)

    required_taxes = {"I_TAX", "X_TAX"}
    group_columns = [column for column in dims.columns if column != "tax"]
    contracts: list[tuple[tuple[int, int, int, int, int], dict[str, str], str]] = []

    for _, group in dims.loc[mask].groupby(group_columns, dropna=False, sort=False):
        if not required_taxes.issubset(set(group["tax"])):
            continue
        contract = group.iloc[0].drop(labels="tax").to_dict()
        unit = contract.get("unit", "")

        observation_counts: list[int] = []
        last_dates: list[int] = []
        for tax in required_taxes:
            index = group.index[group["tax"].eq(tax)][0]
            _, stats = _row_to_series(values.loc[index], "candidate")
            observation_counts.append(int(stats["observations"]))
            last_dates.append(
                pd.Timestamp(stats["last_date"]).toordinal() if stats["last_date"] else 0
            )

        if min(observation_counts) == 0:
            continue
        score = (
            0 if requested_geo else _preference_rank(
                contract.get("geo", ""), GEO_PREFERENCE
            ),
            _preference_rank(
                contract.get("product", ""),
                spec.get("product_preference", ()),
            ) if "product" in dims.columns else 0,
            _preference_rank(unit, PRICE_UNIT_PREFERENCE),
            -min(last_dates),
            -min(observation_counts),
        )
        contracts.append((score, contract, consumption_dimension))

    if not contracts:
        available = dims.loc[mask].drop_duplicates().head(50).to_dict("records")
        raise ValueError(
            f"{dataset}: no real TSV contract contains both I_TAX and X_TAX. "
            f"Available matching rows: {available}"
        )

    _, contract, consumption_dimension = min(contracts, key=lambda item: item[0])
    return contract, consumption_dimension


def _extract_energy_dataset(
    dataset: str,
    path: Path,
    *,
    requested_geo: str | None,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, str]]:
    dims, values = _read_tsv(path)
    contract, _ = _select_energy_contract(
        dataset,
        dims,
        values,
        requested_geo=requested_geo,
    )
    spec = ENERGY_SPECS[dataset]
    prefix = str(spec["prefix"])
    unit = contract["unit"]

    raw_series: dict[str, pd.Series] = {}
    raw_stats: dict[str, dict[str, int | str | None]] = {}
    diagnostics: list[dict[str, object]] = []

    for tax, suffix in (("I_TAX", "wtax"), ("X_TAX", "ntax"), ("X_VAT", "xvat")):
        mask = pd.Series(True, index=dims.index)
        for dimension, code in contract.items():
            mask &= dims[dimension].eq(code)
        mask &= dims["tax"].eq(tax)
        matches = dims.index[mask]
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            raise ValueError(f"{dataset}: duplicate TSV rows for {contract} and tax={tax}.")

        source_key = {**contract, "tax": tax}
        name = f"{prefix}_{suffix}"
        series, stats = _row_to_series(values.loc[matches[0]], name)
        series = _convert_price_to_eur_kwh(series, unit)
        raw_series[suffix] = series
        raw_stats[suffix] = stats
        diagnostics.append(
            _diagnostic(
                output=name,
                dataset=dataset,
                path=path,
                series=series,
                source_key=source_key,
                source_unit=f"EUR/{unit}",
                final_unit="EUR/kWh",
                stats=stats,
            )
        )

    if not {"wtax", "ntax"}.issubset(raw_series):
        raise ValueError(f"{dataset}: I_TAX and X_TAX are required.")

    aligned = pd.concat(raw_series, axis=1)
    result = pd.DataFrame(index=aligned.index)
    result[f"{prefix}_cpr_wtax"] = aligned["wtax"]
    result[f"{prefix}_cpr_ntax"] = aligned["ntax"]

    if "xvat" in aligned:
        result[f"{prefix}_vat"] = 100.0 * (aligned["wtax"] / aligned["xvat"] - 1.0)
        result[f"{prefix}_exc"] = aligned["xvat"] - aligned["ntax"]
        note = "VAT = 100*(I_TAX/X_VAT - 1); EXC = X_VAT - X_TAX."
    else:
        result[f"{prefix}_vat"] = np.nan
        result[f"{prefix}_exc"] = np.nan
        note = "X_VAT is absent; VAT and excise remain missing."

    for column, final_unit in ((f"{prefix}_vat", "%"), (f"{prefix}_exc", "EUR/kWh")):
        diagnostics.append(
            _diagnostic(
                output=column,
                dataset=dataset,
                path=path,
                series=result[column],
                source_key=contract,
                source_unit="derived",
                final_unit=final_unit,
                stats={
                    "flags": sum(int(item["flags"]) for item in raw_stats.values()),
                    "source_missing": int(result[column].isna().sum()),
                },
                note=note,
            )
        )

    result.index.name = "date"
    return result.sort_index(), diagnostics, contract


def _diagnostic(
    *,
    output: str,
    dataset: str,
    path: Path,
    series: pd.Series,
    source_key: dict[str, str],
    source_unit: str,
    final_unit: str,
    stats: dict[str, int | str | None],
    note: str = "",
) -> dict[str, object]:
    observed = series.dropna()
    return {
        "output": output,
        "dataset": dataset,
        "source_file": str(path.resolve()),
        "source_key": json.dumps(source_key, sort_keys=True),
        "source_unit": source_unit,
        "final_unit": final_unit,
        "observations": int(observed.size),
        "first_date": observed.index.min().date().isoformat() if not observed.empty else None,
        "last_date": observed.index.max().date().isoformat() if not observed.empty else None,
        "missing_values": int(series.isna().sum()),
        "flagged_values": int(stats.get("flags", 0) or 0),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Build and export
# ---------------------------------------------------------------------------


def _combine_old_new(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([old, new]).sort_index()
    return combined.loc[~combined.index.duplicated(keep="last")]


def _validate(semiannual: pd.DataFrame) -> None:
    if not isinstance(semiannual.index, pd.DatetimeIndex):
        raise TypeError("semiannual: index must be a DatetimeIndex.")
    if semiannual.empty:
        raise ValueError("The Eurostat household-price panel is empty.")
    if semiannual.index.has_duplicates or not semiannual.index.is_monotonic_increasing:
        raise ValueError("semiannual: dates must be unique and sorted.")
    if not semiannual.index.month.isin([1, 7]).all():
        raise ValueError("Semiannual observations must be dated in January or July.")
    if np.isinf(semiannual.to_numpy(dtype=float)).any():
        raise ValueError("semiannual: infinite values found.")

    price_columns = [column for column in semiannual if "cpr_" in column]
    if (semiannual[price_columns].dropna() <= 0).any().any():
        raise ValueError("Household prices must be strictly positive.")

    for prefix in ("estat_gas_household", "estat_electricity_household"):
        columns = [
            f"{prefix}_cpr_wtax",
            f"{prefix}_cpr_ntax",
            f"{prefix}_vat",
            f"{prefix}_exc",
        ]
        complete = semiannual[columns].dropna()
        reconstructed = (
            complete[f"{prefix}_cpr_ntax"]
            + complete[f"{prefix}_exc"]
        ) * (1.0 + complete[f"{prefix}_vat"] / 100.0)

        if not np.allclose(
            complete[f"{prefix}_cpr_wtax"],
            reconstructed,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"{prefix}: tax-price identity failed.")


def _write_frame(frame: pd.DataFrame, base: Path, output_format: str) -> Path:
    if output_format == "parquet":
        path = base.with_suffix(".parquet")
        frame.to_parquet(path)
    else:
        path = base.with_suffix(".csv")
        frame.to_csv(path, date_format="%Y-%m-%d")
    return path


def build_eurostat_dataset(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    vintage: str | None = None,
    geo: str | None = None,
    refresh: bool = False,
    offline: bool = False,
    output_format: str = "csv",
) -> dict[str, Path]:
    vintage = vintage or datetime.now().strftime("%Y%m%d")
    raw_dir, files, requests_used = _prepare_raw_files(
        Path(raw_root),
        vintage,
        refresh=refresh,
        offline=offline,
    )

    selections: dict[str, object] = {}
    energy: dict[str, pd.DataFrame] = {}

    for dataset in DATASETS:
        table, _, selection = _extract_energy_dataset(
            dataset,
            files[dataset],
            requested_geo=geo,
        )
        energy[dataset] = table
        selections[dataset] = selection

    gas = _combine_old_new(
        energy["nrg_pc_202_h"],
        energy["nrg_pc_202"],
    )
    electricity = _combine_old_new(
        energy["nrg_pc_204_h"],
        energy["nrg_pc_204"],
    )
    semiannual = (
        gas.join(electricity, how="outer")
        .sort_index()
        .dropna(how="all")
    )
    semiannual.index.name = "date"
    _validate(semiannual)

    destination = Path(output_root) / vintage
    destination.mkdir(parents=True, exist_ok=True)

    semiannual_path = _write_frame(
        semiannual,
        destination / "eurostat_semiannual",
        output_format,
    )
    all_path = _write_frame(
        semiannual,
        destination / "eurostat_all",
        output_format,
    )

    diagnostics = []
    for column in semiannual.columns:
        observed = semiannual[column].dropna()
        diagnostics.append(
            {
                "series": column,
                "frequency": "semiannual",
                "observations": int(observed.size),
                "first_date": (
                    observed.index.min().date().isoformat()
                    if not observed.empty else None
                ),
                "last_date": (
                    observed.index.max().date().isoformat()
                    if not observed.empty else None
                ),
                "missing_values": int(semiannual[column].isna().sum()),
                "unit": "%" if column.endswith("_vat") else "EUR/kWh",
            }
        )

    diagnostics_path = destination / "eurostat_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)

    outputs = {
        "semiannual": semiannual_path,
        "all": all_path,
        "diagnostics": diagnostics_path,
    }

    manifest = {
        "source": "Eurostat SDMX 2.1 TSV",
        "script_version": SCRIPT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "vintage": vintage,
        "requested_geo": geo,
        "raw_directory": str(raw_dir.resolve()),
        "download_requests": requests_used,
        "cache_reuse": {
            dataset: {
                "reused_from_previous_vintage": path.parent != raw_dir,
                "snapshot_vintage": path.parent.name,
            }
            for dataset, path in files.items()
        },
        "raw_files": {
            dataset: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for dataset, path in files.items()
        },
        "selected_real_series": selections,
        "outputs": {
            name: str(path.resolve())
            for name, path in outputs.items()
        },
        "columns": list(semiannual.columns),
        "notes": [
            "Eurostat supplies only household gas and electricity prices.",
            "HICP and PPI Energy are supplied by Haver.",
            "Current tables overwrite historical tables on overlapping dates.",
            "VAT = 100*(I_TAX/X_VAT - 1).",
            "EXC = X_VAT - X_TAX.",
        ],
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    outputs["manifest"] = manifest_path

    print(f"Eurostat vintage: {vintage}")
    print(f"Raw directory:    {raw_dir}")
    print(
        f"Semiannual:       {semiannual.index.min().date()} -> "
        f"{semiannual.index.max().date()} "
        f"({len(semiannual):,} rows, {semiannual.shape[1]} series)"
    )
    for name, path in outputs.items():
        print(f"{name:14s} {path}")

    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vintage", default=None, help="Example: 20260804. Default: today.")
    parser.add_argument("--geo", default=None, help="Optional exact code such as EA20.")
    parser.add_argument("--refresh", action="store_true", help="Redownload the four TSV files.")
    parser.add_argument("--offline", action="store_true", help="Use existing raw files only.")
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    return parser


def main() -> None:
    print(f"Running {Path(__file__).name} [{SCRIPT_VERSION}]")
    args = _parser().parse_args()
    build_eurostat_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        vintage=args.vintage,
        geo=args.geo,
        refresh=args.refresh,
        offline=args.offline,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
