"""This version replaces the former raw -> five source scripts -> interim -> builder
architecture with one auditable workbook input.

Expected workbook sheets
------------------------
The normal names are:

    Haver
    Bloomberg_copy
    European_Commission
    World_Bank
    Eurostat
    Eurostat_HICP
    Eurostat_Headline_HICP   (optional; headline suite)
    Metadata                 (optional; audit only)

``Bloomberg_copy`` must be the VALUES-ONLY snapshot used by Python. A separate
``Bloomberg_Live`` sheet may contain Bloomberg formulas; this builder ignores it.
The source sheets may be displayed newest-first or oldest-first: every reader
parses the dates and sorts chronologically in memory before any time-series
calculation.

Recommended raw layout
----------------------

    data/raw/20260815/bvar_energy_raw_data.xlsm

or simply

    data/raw/bvar_energy_raw_data.xlsm

Legacy ``raw_energy_bvar.xlsx`` / ``raw_energy_bvar.xlsm`` names remain supported.

Run from the project root:

    python src/data_pipeline/build_dataset_headline_joint_v14_FINAL.py

or point explicitly to the workbook:

    python src/data_pipeline/build_dataset_headline_joint_v14_FINAL.py \
        --raw data/raw/20260815/bvar_energy_raw_data.xlsm

Outputs are written directly to ``data/processed/<build_vintage>/``. No source
pipeline is launched and no ``data/interim`` directory is read.

Important Bloomberg convention
------------------------------
* BCSL0018 Index      : USD/barrel -> EUR/barrel
* GNEBM1 PVMO Index   : USD/metric tonne -> EUR/metric tonne
                         (NO invented tonne-to-barrel conversion)
* QS1 Comdty          : USD/metric tonne -> EUR/barrel using 7.44 bbl/t,
                         the gasoil conversion used in the paper
* TTFGDAHD BCFV Index : EUR/MWh, no FX conversion
* EURUSD Curncy       : USD per EUR

Final workbook/freshness contract
---------------------------------
The production workbook may be either ``.xlsm`` (preferred, because it retains
the Config/VBA refresh controls) or ``.xlsx``. Python does **not** execute Excel
VBA, Power Query, Haver, or Bloomberg formulas. Before running this builder,
refresh the workbook in Excel, refresh the Haver block, refresh Bloomberg_Live,
copy Bloomberg values into ``Bloomberg_copy``, wait for Power Query refreshes to
finish, and save the workbook. The builder then reads only the saved/cached
worksheet values.

The processed model CSVs contain LEVELS, exactly as before. Absolute
differences remain a model-layer transformation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ===========================================================================
# Paths and version
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
SCRIPT_VERSION = "2026-08-15-v14.1-final-dashboard-contract"

SUPPORTED_WORKBOOK_SUFFIXES = {".xlsx", ".xlsm"}
RAW_WORKBOOK_PATTERNS = (
    "bvar_energy_raw_data*.xlsm",
    "bvar_energy_raw_data*.xlsx",
    "raw_energy_bvar*.xlsm",
    "raw_energy_bvar*.xlsx",
)
PREFERRED_RAW_WORKBOOK_NAMES = (
    "bvar_energy_raw_data.xlsm",
    "raw_energy_bvar.xlsm",
    "bvar_energy_raw_data.xlsx",
    "raw_energy_bvar.xlsx",
)

# Immutable processed-vintage source context used by tax/HICP adapters.  These
# compact sidecars sever future scenario/fitted calculations from subsequent
# edits to the living raw workbook.
SOURCE_CONTEXT_FILES = {
    "haver": "source_context_haver.csv",
    "european_commission": "source_context_european_commission.csv",
    "eurostat": "source_context_eurostat_energy.csv",
}

# Files produced by the retired WP374 decomposition branch.  They are removed
# only after a successful explicit overwrite so a v14 build cannot leave stale
# objects that look current.
DEPRECATED_HEADLINE_DECOMPOSITION_OUTPUTS = (
    "unprocessed_food_monthly.csv",
    "processed_food_monthly.csv",
    "neig_monthly.csv",
    "services_monthly.csv",
    "compensation_per_employee_quarterly.csv",
    "compensation_per_employee_monthly_linear.csv",
    "vat_standard_ea.csv",
    "vat_standard_ea_diagnostics.csv",
)


# ===========================================================================
# Workbook contract
# ===========================================================================

SHEET_ALIASES = {
    "haver": ("Haver",),
    "bloomberg": (
        "Bloomberg_copy",
        "Bloomberg_snapshot",
        "Bloomberg",
    ),
    "european_commission": (
        "European_Commission",
        "EC_WOB_VIEW",
    ),
    "world_bank": (
        "World_Bank",
        "WORLD_BANK_GAS_RAW",
    ),
    "eurostat": (
        "Eurostat",
        "EUROSTAT_ENERGY_VIEW",
    ),
    "eurostat_hicp": (
        "Eurostat_HICP",
        "Eurostat_hicp",
        "EUROSTAT_HICP_VIEW",
    ),
    "eurostat_headline_hicp": (
        "Eurostat_Headline_HICP",
        "EUROSTAT_HEADLINE_HICP_VIEW",
        # Backward-compatible typo in the workbook used during development.
        "EUROSTAT_HEDLINE_HICP_VIEW",
    ),
    "metadata": ("Metadata",),
}

LIVE_SHEET_MARKERS = ("live", "formula", "bdh")

# Haver is now intentionally minimal in the production workbook. Only series
# actually consumed by the six Energy model panels are hard requirements.
# Total HICP, car fuels and liquid fuels are supplied by the Eurostat HICP
# lineage and therefore remain optional/backward-compatible Haver inputs.
ENERGY_HAVER_REQUIRED_TICKERS = {
    "H023HW51@EUDATA": "hicp_electricity",
    "H023HW52@EUDATA": "hicp_gas",
    "H023HW54@EUDATA": "hicp_solid_fuels",
    "H023HW55@EUDATA": "hicp_heat_energy",
    "H025PP@G10": "ppi_energy",
}
ENERGY_HAVER_OPTIONAL_TICKERS = {
    "H023HICP@EUDATA": "hicp_total",
    "H023HN22@EUDATA": "hicp_car_fuels",
    "H023HW53@EUDATA": "hicp_liquid_fuels",
}
ENERGY_HAVER_TICKERS = {
    **ENERGY_HAVER_REQUIRED_TICKERS,
    **ENERGY_HAVER_OPTIONAL_TICKERS,
}

# Backward-compatible name used by older project code: all Energy Haver tickers.
HAVER_TICKERS = ENERGY_HAVER_TICKERS

# Exact Bloomberg snapshot tickers. There is deliberately NO PDS1 fallback:
# GNEBM1 PVMO is now the project's refined-petroleum / Eurobob series.
BLOOMBERG_TICKERS = {
    "crude_oil_usd": "BCSL0018 Index",
    "ttf_eur_mwh": "TTFGDAHD BCFV Index",
    "eurusd": "EURUSD Curncy",
    "refined_diesel_usd": "QS1 Comdty",
    "refined_petroleum_usd": "GNEBM1 PVMO Index",
}

BLOOMBERG_UNITS = {
    "crude_oil_usd": {
        "ticker": BLOOMBERG_TICKERS["crude_oil_usd"],
        "input_unit": "USD/barrel",
        "output_unit": "EUR/barrel",
        "barrels_per_metric_tonne": None,
    },
    "refined_petroleum_usd": {
        "ticker": BLOOMBERG_TICKERS["refined_petroleum_usd"],
        "input_unit": "USD/metric tonne",
        "output_unit": "EUR/metric tonne",
        "barrels_per_metric_tonne": None,
        "note": (
            "Eurobob oxygenated gasoline FOB Rotterdam barges. The project "
            "retains the source tonne unit after FX conversion; no density-"
            "based tonne-to-barrel conversion is invented."
        ),
    },
    "refined_diesel_usd": {
        "ticker": BLOOMBERG_TICKERS["refined_diesel_usd"],
        "input_unit": "USD/metric tonne",
        "output_unit": "EUR/barrel",
        "barrels_per_metric_tonne": 7.44,
        "note": "Gasoil/diesel conversion used in the paper: 7.44 barrels/tonne.",
    },
    "ttf_eur_mwh": {
        "ticker": BLOOMBERG_TICKERS["ttf_eur_mwh"],
        "input_unit": "EUR/MWh",
        "output_unit": "EUR/MWh",
        "barrels_per_metric_tonne": None,
    },
    "eurusd": {
        "ticker": BLOOMBERG_TICKERS["eurusd"],
        "input_unit": "USD per EUR",
        "output_unit": "USD per EUR",
        "barrels_per_metric_tonne": None,
    },
}

EC_COLUMN_MAP = {
    "EUR_price_with_tax_euro95": "wob_petroleum_cpr_wtax",
    "EUR_price_wo_tax_euro95": "wob_petroleum_cpr_ntax",
    "EUR_price_with_tax_diesel": "wob_diesel_cpr_wtax",
    "EUR_price_wo_tax_diesel": "wob_diesel_cpr_ntax",
    "EUR_price_with_tax_heating_oil": "wob_gas_cpr_wtax",
    "EUR_price_wo_tax_heating_oil": "wob_gas_cpr_ntax",
}

WORLD_BANK_REQUIRED = ("wb_natural_gas_europe_usd_mmbtu",)

EUROSTAT_REQUIRED = (
    "estat_gas_household_cpr_wtax",
    "estat_gas_household_cpr_ntax",
    "estat_gas_household_vat",
    "estat_gas_household_exc",
    "estat_electricity_household_cpr_wtax",
    "estat_electricity_household_cpr_ntax",
    "estat_electricity_household_vat",
    "estat_electricity_household_exc",
)


# Eurostat ECOICOP v2 aggregation inputs. These are validation/aggregation
# series only; they do not enter the six BVAR estimation datasets.
EUROSTAT_HICP_SERIES = {
    "TOTAL": "hicp_total",
    "NRG": "hicp_energy",
    "ELC_GAS": "hicp_electricity_gas_solid_heat",
    "FUEL": "hicp_fuel_special_aggregate",
    "CP045": "hicp_household_energy",
    "CP0451": "hicp_electricity",
    "CP0452": "hicp_gas",
    "CP0453": "hicp_liquid_fuels",
    "CP0454": "hicp_solid_fuels",
    "CP0455": "hicp_heat_cooling_energy",
    "CP0722": "hicp_fuels_lubricants_personal_transport",
    "CP07221": "hicp_diesel",
    "CP07222": "hicp_petrol",
    "CP07223": "hicp_other_transport_fuels",
    "CP07224": "hicp_lubricants",
}

EUROSTAT_HICP_OUTPUT_ORDER = [
    "hicp_total",
    "hicp_energy",
    "hicp_household_energy",
    "hicp_electricity",
    "hicp_gas",
    "hicp_liquid_fuels",
    "hicp_solid_fuels",
    "hicp_heat_cooling_energy",
    "hicp_fuels_lubricants_personal_transport",
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
    "hicp_lubricants",
    "hicp_electricity_gas_solid_heat",
    "hicp_fuel_special_aggregate",
]

EUROSTAT_HICP_REQUIRED_COLUMNS = (
    "dataset",
    "measure",
    "series",
    "date",
    "year",
    "coicop18",
    "geo",
    "statinfo",
    "freq",
    "unit",
    "value",
    "index_2025",
    "weight_per_thousand",
    "flag",
)

EUROSTAT_HICP_INDEX_DATASET = "prc_hicp_minr"
EUROSTAT_HICP_WEIGHT_DATASET = "prc_hicp_iw"
EUROSTAT_HICP_INDEX_UNIT = "I25"
EUROSTAT_HICP_GEO = "EA"
EUROSTAT_HICP_WEIGHT_TOLERANCE_PER_THOUSAND = 0.05

# The ECOICOP-v2 back-series is deliberately not forced to have a common
# start. Most series are backcast to 1996-01; the detailed petrol/diesel
# split begins in 2016-12. A later start is an error; an earlier backfill is
# surfaced as a warning and recorded in the manifest.
EUROSTAT_HICP_EXPECTED_INDEX_START = {
    series: pd.Timestamp("2016-12-01")
    if code in {"CP07221", "CP07222", "CP07223", "CP07224"}
    else pd.Timestamp("1996-01-01")
    for code, series in EUROSTAT_HICP_SERIES.items()
}

EUROSTAT_HICP_EXPECTED_WEIGHT_START_YEAR = {
    series: 2016
    if code in {"CP07221", "CP07222", "CP07223", "CP07224"}
    else 1996
    for code, series in EUROSTAT_HICP_SERIES.items()
}

# Known ECOICOP-v2 break flag in the Eurostat back-series. Any additional
# 'b' flag is treated as a contract change requiring review.
EUROSTAT_HICP_EXPECTED_BREAKS = {
    "hicp_energy": {pd.Timestamp("2017-01-01")},
    "hicp_electricity_gas_solid_heat": {pd.Timestamp("2017-01-01")},
    "hicp_fuel_special_aggregate": {pd.Timestamp("2017-01-01")},
}


# Headline-suite HICP contract (ECOICOP v2 special aggregates).
#
# Monthly production Food is the official Eurostat FOOD aggregate.
# Annual Food aggregation weight remains FOOD_NP + FOOD_P because the
# Headline four-component weight identity is defined from those official
# item-weight blocks.
HEADLINE_HICP_INDEX_SERIES = {
    "TOTAL": "hicp_total",
    "NRG": "hicp_energy",
    "FOOD": "hicp_food",
    "FOOD_NP": "hicp_unprocessed_food",
    "FOOD_P": "hicp_processed_food",
    "IGD_NNRG": "hicp_neig",
    "SERV": "hicp_services",
}
HEADLINE_HICP_WEIGHT_SERIES = {
    "TOTAL": "hicp_total",
    "NRG": "hicp_energy",
    "FOOD_NP": "hicp_unprocessed_food",
    "FOOD_P": "hicp_processed_food",
    "IGD_NNRG": "hicp_neig",
    "SERV": "hicp_services",
}
HEADLINE_HICP_INDEX_OUTPUT_ORDER = [
    "hicp_total",
    "hicp_energy",
    "hicp_food",
    "hicp_unprocessed_food",
    "hicp_processed_food",
    "hicp_neig",
    "hicp_services",
]
HEADLINE_HICP_WEIGHT_OUTPUT_ORDER = [
    "hicp_total",
    "hicp_energy",
    "hicp_unprocessed_food",
    "hicp_processed_food",
    "hicp_neig",
    "hicp_services",
]
HEADLINE_COMPONENTS_FOR_TOTAL = [
    "hicp_energy",
    "hicp_unprocessed_food",
    "hicp_processed_food",
    "hicp_neig",
    "hicp_services",
]
HEADLINE_WEIGHT_TOLERANCE_PER_THOUSAND = 0.10

# Final production Headline contract: ONE joint monthly BVAR over the four
# aggregate HICP components. No alternative component model datasets are built.
HEADLINE_JOINT_SPEC = {
    "file": "headline_joint_monthly.csv",
    "frequency": "monthly",
    "lags": 12,
    "comparison_lags": [6, 12],
    "columns": [
        "hicp_energy",
        "hicp_food",
        "hicp_neig",
        "hicp_services",
    ],
    "equations": {
        "headline_joint": [
            "hicp_energy",
            "hicp_food",
            "hicp_neig",
            "hicp_services",
        ]
    },
    "target_columns": [
        "hicp_energy",
        "hicp_food",
        "hicp_neig",
        "hicp_services",
    ],
    "transformations": {
        "hicp_energy": "logdiff",
        "hicp_food": "logdiff",
        "hicp_neig": "logdiff",
        "hicp_services": "logdiff",
    },
    "baseline": "dense BVAR(12)",
    "sensitivity": "dense BVAR(6) only",
    "source_contract": {
        "hicp_energy": "Eurostat NRG",
        "hicp_food": "Eurostat FOOD",
        "hicp_neig": "Eurostat IGD_NNRG",
        "hicp_services": "Eurostat SERV",
        "food_weight": "Eurostat FOOD_NP + FOOD_P",
    },
}



# ===========================================================================
# Model specification
# ===========================================================================

DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS = 1
PRE_TAX_EUR_KWH_TO_EUR_MWH = 1_000.0
MMBTU_PER_MWH = 3.412
COMPLETE_PERIOD_COVERAGE_RATIO = 0.80
COVERAGE_REFERENCE_PERIODS = 24
DEFAULT_TAX_CARRY_WARN_MONTHS = 6
DEFAULT_TAX_CARRY_ERROR_MONTHS = 12
SEMIANNUAL_MONTHS = (1, 7)

COVERAGE_TO_PANEL_COLUMN = {
    "crude_oil_eur": "crude_oil_eur",
    "refined_petroleum_eur": "refined_petroleum_eur",
    "refined_diesel_eur": "refined_diesel_eur",
    "ttf_eur_mwh": "natural_gas_eur_mwh",
    "eurusd": None,
}

MODEL_SPECS = {
    "car_fuels": {
        "file": "car_fuels_weekly.csv",
        "frequency": "weekly",
        "lags": 24,
        "columns": [
            "wob_petrol_pre_tax",
            "wob_diesel_pre_tax",
            "crude_oil_eur",
            "refined_petroleum_eur",
            "refined_diesel_eur",
        ],
        "equations": {
            "petrol": [
                "wob_petrol_pre_tax",
                "crude_oil_eur",
                "refined_petroleum_eur",
            ],
            "diesel": [
                "wob_diesel_pre_tax",
                "crude_oil_eur",
                "refined_diesel_eur",
            ],
        },
        "target_columns": ["wob_petrol_pre_tax", "wob_diesel_pre_tax"],
        "units": {
            "wob_petrol_pre_tax": "EUR/1000L",
            "wob_diesel_pre_tax": "EUR/1000L",
            "crude_oil_eur": "EUR/barrel",
            "refined_petroleum_eur": "EUR/metric tonne",
            "refined_diesel_eur": "EUR/barrel",
        },
    },
    "liquid_fuels": {
        "file": "liquid_fuels_weekly.csv",
        "frequency": "weekly",
        "lags": 24,
        "columns": [
            "wob_heating_oil_pre_tax",
            "crude_oil_eur",
            "refined_diesel_eur",
        ],
        "equations": {
            "liquid_fuels": [
                "wob_heating_oil_pre_tax",
                "crude_oil_eur",
                "refined_diesel_eur",
            ]
        },
        "target_columns": ["wob_heating_oil_pre_tax"],
        "units": {
            "wob_heating_oil_pre_tax": "EUR/1000L",
            "crude_oil_eur": "EUR/barrel",
            "refined_diesel_eur": "EUR/barrel",
        },
    },
    "gas": {
        "file": "gas_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["gas_pre_tax", "natural_gas_eur_mwh"],
        "equations": {"gas": ["gas_pre_tax", "natural_gas_eur_mwh"]},
        "target_columns": ["gas_pre_tax"],
        "units": {
            "gas_pre_tax": "EUR/MWh",
            "natural_gas_eur_mwh": "EUR/MWh",
        },
    },
    "electricity": {
        "file": "electricity_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["electricity_pre_tax", "natural_gas_eur_mwh"],
        "equations": {
            "electricity": ["electricity_pre_tax", "natural_gas_eur_mwh"]
        },
        "seasonal_dummies": True,
        "target_columns": ["electricity_pre_tax"],
        "units": {
            "electricity_pre_tax": "EUR/kWh",
            "natural_gas_eur_mwh": "EUR/MWh",
        },
    },
    "heat_energy": {
        "file": "heat_energy_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["hicp_heat_energy", "natural_gas_eur_mwh", "ppi_energy"],
        "equations": {
            "heat_energy": ["hicp_heat_energy", "natural_gas_eur_mwh", "ppi_energy"]
        },
        "target_columns": ["hicp_heat_energy"],
    },
    "solid_fuels": {
        "file": "solid_fuels_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["hicp_solid_fuels", "natural_gas_eur_mwh", "ppi_energy"],
        "equations": {
            "solid_fuels": ["hicp_solid_fuels", "natural_gas_eur_mwh", "ppi_energy"]
        },
        "target_columns": ["hicp_solid_fuels"],
    },
}


# ===========================================================================
# Generic utilities
# ===========================================================================


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_vintage_name(name: str) -> date | None:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(name, fmt).date()
        except ValueError:
            pass
    return None


def find_raw_workbook(raw_root: str | Path = DEFAULT_RAW_ROOT) -> Path:
    """Find the latest production workbook (.xlsm preferred; .xlsx supported).

    The final production filename is ``bvar_energy_raw_data.xlsm``. Legacy
    ``raw_energy_bvar`` names remain supported so older snapshots stay readable.
    Old source-specific XLSX files are never selected merely because they exist
    below ``data/raw``.
    """
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw-data folder does not exist: {root}")

    # Final production contract: the living top-level workbook wins over any
    # dated/archive copies below data/raw.
    for name in PREFERRED_RAW_WORKBOOK_NAMES:
        canonical = root / name
        if canonical.is_file() and not canonical.name.startswith("~$"):
            return canonical.resolve()

    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in RAW_WORKBOOK_PATTERNS:
        for path in root.rglob(pattern):
            if not path.is_file() or path.name.startswith("~$"):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)

    if not candidates:
        top_level = [
            p
            for suffix in sorted(SUPPORTED_WORKBOOK_SUFFIXES)
            for p in root.glob(f"*{suffix}")
            if p.is_file() and not p.name.startswith("~$")
        ]
        if len(top_level) == 1:
            candidates = top_level
        elif len(top_level) > 1:
            listing = "\n  ".join(str(p) for p in top_level)
            raise FileNotFoundError(
                "No canonical bvar_energy_raw_data/raw_energy_bvar workbook was "
                "found and several top-level Excel workbooks exist below "
                f"{root}:\n  {listing}\nPass --raw explicitly."
            )

    if not candidates:
        raise FileNotFoundError(
            f"No production .xlsm/.xlsx workbook found below {root}. "
            "Expected bvar_energy_raw_data* or raw_energy_bvar*. "
            "Save the refreshed workbook there or pass --raw explicitly."
        )

    preference = {
        name.casefold(): len(PREFERRED_RAW_WORKBOOK_NAMES) - rank
        for rank, name in enumerate(PREFERRED_RAW_WORKBOOK_NAMES)
    }

    def rank(path: Path) -> tuple[int, int, int, int]:
        vintage = _parse_vintage_name(path.parent.name)
        vintage_ord = vintage.toordinal() if vintage else -1
        exact_preference = preference.get(path.name.casefold(), 0)
        macro_enabled = int(path.suffix.casefold() == ".xlsm")
        return vintage_ord, exact_preference, macro_enabled, path.stat().st_mtime_ns

    return max(candidates, key=rank)


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _norm_text(value: object) -> str:
    return _clean_text(value).casefold()


def _parse_excel_date(value: object) -> pd.Timestamp:
    """Parse real Excel dates, date strings, or Excel serial numbers."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).tz_localize(None).normalize()
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and 1 <= numeric <= 100000:
            return (pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric, unit="D")).normalize()
        return pd.NaT

    text = _clean_text(value)
    if not text or text.startswith("#"):
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_localize(None).normalize()


def _parse_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([_parse_excel_date(v) for v in values])


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _input_order(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        return "single_or_empty"
    if index.is_monotonic_increasing:
        return "ascending"
    if index.is_monotonic_decreasing:
        return "descending"
    return "unsorted"


def _deduplicate_frame(frame: pd.DataFrame, *, source: str) -> tuple[pd.DataFrame, int, int]:
    duplicate_mask = frame.index.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_dates = int(frame.index[duplicate_mask].nunique())

    if duplicate_rows:
        for stamp, block in frame.loc[duplicate_mask].groupby(level=0):
            if (block.nunique(dropna=True) > 1).any():
                raise ValueError(
                    f"{source}: duplicate date {pd.Timestamp(stamp).date()} carries "
                    "conflicting numeric values. Fix the workbook snapshot."
                )
        frame = frame.groupby(level=0).last()

    return frame, duplicate_rows, duplicate_dates


def _sheet_name(
    xls: pd.ExcelFile,
    logical: str,
    *,
    required: bool = True,
) -> str | None:

    lookup = {
        name.casefold(): name
        for name in xls.sheet_names
    }

    for alias in SHEET_ALIASES[logical]:
        resolved = lookup.get(alias.casefold())

        if resolved is None:
            continue

        if (
            logical == "bloomberg"
            and any(
                marker in resolved.casefold()
                for marker in LIVE_SHEET_MARKERS
            )
        ):
            continue

        return resolved

    if not required:
        return None

    raise ValueError(
        f"Workbook is missing the {logical!r} sheet. "
        f"Expected one of {SHEET_ALIASES[logical]} "
        f"(case-insensitive). "
        f"Available sheets: {xls.sheet_names}"
    )

def _finalize_source_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    raw_rows: int,
    invalid_date_rows: int,
    error_markers: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = frame.copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    input_idx = pd.DatetimeIndex(frame.index)
    order = _input_order(input_idx)
    frame, duplicate_rows, duplicate_dates = _deduplicate_frame(frame, source=source)
    all_missing_rows = int(frame.isna().all(axis=1).sum())
    frame = frame.dropna(how="all").sort_index()
    frame.index = pd.DatetimeIndex(frame.index, name="date")

    if frame.empty:
        raise ValueError(f"{source}: no usable numeric observations were read from the workbook.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{source}: internal date normalization failed.")

    stats = {
        "raw_rows": int(raw_rows),
        "retained_rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "input_order": order,
        "invalid_date_rows": int(invalid_date_rows),
        "duplicate_date_rows": duplicate_rows,
        "duplicate_dates": duplicate_dates,
        "all_missing_rows_dropped": all_missing_rows,
        "error_markers_coerced_to_nan": int(error_markers),
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
    }
    return frame, stats


# ===========================================================================
# Excel readers
# ===========================================================================


def _haver_period_columns(
    raw: pd.DataFrame,
    *,
    ticker_row: int,
) -> list[tuple[int, pd.Timestamp]]:
    """Find the monthly date header belonging to a Haver horizontal block."""
    best: list[tuple[int, pd.Timestamp]] = []
    for r in range(0, ticker_row):
        found: list[tuple[int, pd.Timestamp]] = []
        for c, value in enumerate(raw.iloc[r].tolist()):
            token = _clean_text(value).replace(" ", "")
            match = re.fullmatch(r"(\d{4})(\d{2})(?:\*M)?", token, flags=re.I)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                if 1900 <= year <= 2200 and 1 <= month <= 12:
                    found.append((c, pd.Timestamp(year, month, 1)))
        if len(found) > len(best):
            best = found

    if len(best) < 12:
        raise ValueError(
            "Could not locate a monthly Haver date header above Excel row "
            f"{ticker_row + 1}; found only {len(best)} period columns."
        )
    return best


def _read_haver_sheet(xls: pd.ExcelFile, sheet: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read only the monthly Haver series used by the Energy production models.

    Headline production is sourced entirely from the Eurostat Headline HICP VIEW.
    """
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
    raw_rows = len(raw)

    ticker_contract: dict[str, tuple[str, bool]] = {}
    for ticker, output in ENERGY_HAVER_REQUIRED_TICKERS.items():
        ticker_contract[ticker] = (output, True)
    for ticker, output in ENERGY_HAVER_OPTIONAL_TICKERS.items():
        ticker_contract[ticker] = (output, False)

    wanted = {ticker.casefold(): ticker for ticker in ticker_contract}
    ticker_locations: dict[str, tuple[int, int]] = {}
    scan_rows = min(len(raw), 120)
    for r in range(scan_rows):
        for c in range(raw.shape[1]):
            token = _norm_text(raw.iat[r, c])
            if token in wanted:
                ticker_locations[wanted[token]] = (r, c)

    missing_energy = [
        ticker for ticker in ENERGY_HAVER_REQUIRED_TICKERS
        if ticker not in ticker_locations
    ]
    if missing_energy:
        raise ValueError(
            f"Haver sheet is missing required Energy ticker(s): {missing_energy}. "
            "Refresh/save the Energy Haver block before running the builder."
        )

    series: dict[str, pd.Series] = {}
    error_markers = 0
    for ticker, (output_name, _required) in ticker_contract.items():
        location = ticker_locations.get(ticker)
        if location is None:
            continue
        row, _col = location
        period_columns = _haver_period_columns(raw, ticker_row=row)
        values: list[object] = []
        dates: list[pd.Timestamp] = []
        for col, stamp in period_columns:
            value = raw.iat[row, col]
            if isinstance(value, str) and value.strip().startswith("#"):
                error_markers += 1
            values.append(value)
            dates.append(stamp)
        s = pd.Series(
            pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(dates, name="date"),
            name=output_name,
        )
        if s.index.has_duplicates:
            grouped = s.groupby(level=0)
            for stamp, block in grouped:
                if block.dropna().nunique() > 1:
                    raise ValueError(
                        f"haver {ticker}: conflicting duplicate period {stamp.date()}."
                    )
            s = grouped.last()
        series[output_name] = s.sort_index()

    panel = pd.concat(series, axis=1).sort_index()
    frame, stats = _finalize_source_frame(
        panel,
        source="haver",
        raw_rows=raw_rows,
        invalid_date_rows=0,
        error_markers=error_markers,
    )
    stats["energy_tickers_required"] = ENERGY_HAVER_REQUIRED_TICKERS
    stats["energy_tickers_optional"] = ENERGY_HAVER_OPTIONAL_TICKERS
    stats["energy_optional_tickers_found"] = sorted(
        t for t in ENERGY_HAVER_OPTIONAL_TICKERS if t in ticker_locations
    )
    stats["energy_optional_tickers_missing"] = sorted(
        t for t in ENERGY_HAVER_OPTIONAL_TICKERS if t not in ticker_locations
    )
    stats["haver_contract"] = "monthly Energy inputs only"
    return frame, stats

def _read_bloomberg_sheet(xls: pd.ExcelFile, sheet: str) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
    raw_rows = len(raw)
    if raw.empty:
        raise ValueError("Bloomberg sheet is empty.")

    wanted_norm = {logical: _norm_text(ticker) for logical, ticker in BLOOMBERG_TICKERS.items()}
    ticker_row = None
    ticker_columns: dict[str, int] = {}

    for r in range(min(80, len(raw))):
        row_lookup = {_norm_text(value): c for c, value in enumerate(raw.iloc[r].tolist()) if _clean_text(value)}
        found = {
            logical: row_lookup[norm]
            for logical, norm in wanted_norm.items()
            if norm in row_lookup
        }
        if len(found) == len(wanted_norm):
            ticker_row = r
            ticker_columns = found
            break

    if ticker_row is None:
        available = []
        for r in range(min(20, len(raw))):
            available.extend(_clean_text(v) for v in raw.iloc[r].tolist() if _clean_text(v))
        raise ValueError(
            "Bloomberg snapshot does not contain all required tickers. Expected: "
            f"{list(BLOOMBERG_TICKERS.values())}. Ensure Python reads the "
            "VALUES-ONLY 'Bloomberg' sheet, not Bloomberg_Live."
        )

    field_row = None
    for r in range(ticker_row + 1, min(ticker_row + 15, len(raw))):
        first = _norm_text(raw.iat[r, 0])
        fields = [_norm_text(raw.iat[r, c]) for c in ticker_columns.values()]
        if first in {"date", "dates"} and all(field in {"px_last", "last_price"} for field in fields):
            field_row = r
            break

    if field_row is None:
        raise ValueError(
            "Could not locate the Bloomberg 'Dates | PX_LAST ...' field row below the ticker row."
        )

    date_values = raw.iloc[field_row + 1 :, 0]
    dates = _parse_dates(date_values)
    valid_dates = ~dates.isna()
    invalid_date_rows = int((~valid_dates).sum())

    panel = pd.DataFrame(index=pd.DatetimeIndex(dates[valid_dates], name="date"))
    error_markers = 0
    for logical, col in ticker_columns.items():
        values = raw.iloc[field_row + 1 :, col].reset_index(drop=True)
        error_markers += int(
            values.map(lambda v: isinstance(v, str) and v.strip().startswith("#")).sum()
        )
        numeric = pd.to_numeric(values, errors="coerce")
        panel[logical] = numeric.loc[np.asarray(valid_dates)].to_numpy()

    frame, stats = _finalize_source_frame(
        panel,
        source="bloomberg",
        raw_rows=raw_rows,
        invalid_date_rows=invalid_date_rows,
        error_markers=error_markers,
    )
    stats["ticker_row_excel"] = int(ticker_row + 1)
    stats["field_row_excel"] = int(field_row + 1)
    stats["tickers"] = BLOOMBERG_TICKERS
    return frame, stats


def _read_simple_sheet(
    xls: pd.ExcelFile,
    sheet: str,
    *,
    source: str,
    rename: dict[str, str] | None = None,
    required_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(xls, sheet_name=sheet, dtype=object)
    raw_rows = len(raw)
    if raw.empty:
        raise ValueError(f"{source}: sheet {sheet!r} is empty.")

    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = next(
        (c for c in raw.columns if c.strip().lower() in {"date", "dates", "period", "time"}),
        raw.columns[0],
    )

    dates = _parse_dates(raw[date_col])
    valid_dates = ~dates.isna()
    invalid_date_rows = int((~valid_dates).sum())

    data = raw.loc[np.asarray(valid_dates)].drop(columns=[date_col]).copy()
    if rename:
        data = data.rename(columns=rename)
    data.columns = [str(c).strip() for c in data.columns]

    error_markers = 0
    for column in data.columns:
        error_markers += int(
            data[column].map(lambda v: isinstance(v, str) and v.strip().startswith("#")).sum()
        )
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.index = pd.DatetimeIndex(dates[valid_dates], name="date")

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required column(s) {missing} in sheet {sheet!r}. "
            f"Available columns: {list(data.columns)}"
        )

    frame, stats = _finalize_source_frame(
        data,
        source=source,
        raw_rows=raw_rows,
        invalid_date_rows=invalid_date_rows,
        error_markers=error_markers,
    )
    return frame, stats



def _flag_tokens(value: object) -> set[str]:
    text = _clean_text(value).casefold()
    if not text:
        return set()
    return set(re.findall(r"[a-z]+", text))


def _read_eurostat_hicp_sheet(
    xls: pd.ExcelFile,
    sheet: str,
    *,
    discovery: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
]:
    """Read the standalone Eurostat ECOICOP-v2 HICP view.

    Structural contracts are always hard requirements. Empirical contracts
    (starts, break dates, weight identities) are collected into diagnostics.
    In strict mode the caller may reject them only after those diagnostics have
    been written. ``discovery=True`` reports them without blocking the HICP
    sidecar build.
    """
    raw = pd.read_excel(xls, sheet_name=sheet, dtype=object)
    raw_rows = len(raw)
    if raw.empty:
        raise ValueError(f"eurostat_hicp: sheet {sheet!r} is empty.")

    raw.columns = [str(c).strip() for c in raw.columns]
    missing_columns = [
        column for column in EUROSTAT_HICP_REQUIRED_COLUMNS
        if column not in raw.columns
    ]
    if missing_columns:
        raise ValueError(
            "eurostat_hicp: missing required column(s) "
            f"{missing_columns} in sheet {sheet!r}. "
            f"Available columns: {list(raw.columns)}"
        )

    data = raw.loc[:, EUROSTAT_HICP_REQUIRED_COLUMNS].copy()
    for column in (
        "dataset", "measure", "series", "coicop18", "geo",
        "statinfo", "freq", "unit", "flag",
    ):
        data[column] = data[column].map(_clean_text)

    data["date"] = _parse_dates(data["date"])
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    for column in ("value", "index_2025", "weight_per_thousand"):
        data[column] = (
            pd.to_numeric(data[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

    allowed_measures = {"hicp_index", "hicp_weight"}
    measures = set(data["measure"])
    if measures != allowed_measures:
        raise ValueError(
            "eurostat_hicp: expected exactly measures "
            f"{sorted(allowed_measures)}, found {sorted(measures)}."
        )

    index_rows = data.loc[data["measure"] == "hicp_index"].copy()
    weight_rows = data.loc[data["measure"] == "hicp_weight"].copy()
    if index_rows.empty or weight_rows.empty:
        raise ValueError("eurostat_hicp: both index and weight blocks must be non-empty.")

    index_contract = {
        "dataset": set(index_rows["dataset"]),
        "freq": set(index_rows["freq"]),
        "unit": set(index_rows["unit"]),
        "geo": set(index_rows["geo"]),
    }
    expected_index_contract = {
        "dataset": {EUROSTAT_HICP_INDEX_DATASET},
        "freq": {"M"},
        "unit": {EUROSTAT_HICP_INDEX_UNIT},
        "geo": {EUROSTAT_HICP_GEO},
    }
    if index_contract != expected_index_contract:
        raise ValueError(
            "eurostat_hicp: monthly index contract changed. "
            f"Expected {expected_index_contract}, found {index_contract}."
        )

    weight_contract = {
        "dataset": set(weight_rows["dataset"]),
        "freq": set(weight_rows["freq"]),
        "geo": set(weight_rows["geo"]),
    }
    expected_weight_contract = {
        "dataset": {EUROSTAT_HICP_WEIGHT_DATASET},
        "freq": {"A"},
        "geo": {EUROSTAT_HICP_GEO},
    }
    if weight_contract != expected_weight_contract:
        raise ValueError(
            "eurostat_hicp: annual weight contract changed. "
            f"Expected {expected_weight_contract}, found {weight_contract}."
        )

    statinfo_values = sorted(v for v in set(weight_rows["statinfo"]) if v)
    if len(statinfo_values) != 1:
        raise ValueError(
            "eurostat_hicp: expected exactly one non-empty statinfo value "
            f"for item weights, found {statinfo_values}."
        )
    weight_statinfo = statinfo_values[0]

    expected_codes = set(EUROSTAT_HICP_SERIES)
    for label, block in (("indices", index_rows), ("weights", weight_rows)):
        returned_codes = set(block["coicop18"])
        missing_codes = sorted(expected_codes - returned_codes)
        unexpected_codes = sorted(returned_codes - expected_codes)
        if missing_codes or unexpected_codes:
            raise ValueError(
                f"eurostat_hicp {label}: ECOICOP coverage changed. "
                f"Missing={missing_codes}; unexpected={unexpected_codes}."
            )
        for code, expected_series in EUROSTAT_HICP_SERIES.items():
            observed_series = set(block.loc[block["coicop18"] == code, "series"])
            if observed_series != {expected_series}:
                raise ValueError(
                    f"eurostat_hicp {label}: code {code} expected series "
                    f"{expected_series!r}, found {sorted(observed_series)}."
                )

    if index_rows["date"].isna().any() or index_rows["year"].isna().any():
        raise ValueError("eurostat_hicp: every hicp_index row must have date and year.")
    index_year = index_rows["date"].dt.year.astype("Int64")
    if not index_year.equals(index_rows["year"].astype("Int64")):
        bad = index_rows.loc[
            index_year != index_rows["year"].astype("Int64"),
            ["series", "date", "year"],
        ].head()
        raise ValueError(f"eurostat_hicp: index date/year mismatch:\n{bad}")

    if weight_rows["date"].notna().any() or weight_rows["year"].isna().any():
        raise ValueError("eurostat_hicp: hicp_weight rows must have date=null and non-null year.")

    bad_index_values = (
        index_rows["value"].isna()
        | index_rows["index_2025"].isna()
        | index_rows["weight_per_thousand"].notna()
        | ((index_rows["value"] - index_rows["index_2025"]).abs() > 1e-12)
    )
    if bad_index_values.any():
        raise ValueError(
            "eurostat_hicp: invalid hicp_index value mapping in "
            f"{int(bad_index_values.sum())} row(s)."
        )

    bad_weight_values = (
        weight_rows["value"].isna()
        | weight_rows["weight_per_thousand"].isna()
        | weight_rows["index_2025"].notna()
        | ((weight_rows["value"] - weight_rows["weight_per_thousand"]).abs() > 1e-12)
    )
    if bad_weight_values.any():
        raise ValueError(
            "eurostat_hicp: invalid hicp_weight value mapping in "
            f"{int(bad_weight_values.sum())} row(s)."
        )

    if index_rows.duplicated(["series", "date", "measure"]).any():
        raise ValueError("eurostat_hicp: duplicate (series, date, measure) index key.")
    if weight_rows.duplicated(["series", "year", "measure"]).any():
        raise ValueError("eurostat_hicp: duplicate (series, year, measure) weight key.")

    empirical_failures: list[dict[str, object]] = []

    def record_failure(contract, series, observed, expected, message) -> None:
        empirical_failures.append({
            "contract": contract,
            "series": series,
            "observed": str(observed),
            "expected": str(expected),
            "message": str(message),
        })

    start_status: dict[str, dict[str, object]] = {}
    for series in EUROSTAT_HICP_OUTPUT_ORDER:
        idx = index_rows.loc[index_rows["series"] == series].sort_values("date")
        years = weight_rows.loc[weight_rows["series"] == series].sort_values("year")
        dates = pd.DatetimeIndex(idx["date"])
        expected_dates = pd.date_range(dates.min(), dates.max(), freq="MS")
        missing_dates = expected_dates.difference(dates)
        if len(missing_dates):
            raise ValueError(
                f"eurostat_hicp {series}: monthly calendar gap; first missing {missing_dates[0].date()}."
            )

        year_values = years["year"].astype(int).to_numpy()
        expected_years = np.arange(year_values.min(), year_values.max() + 1)
        missing_years = sorted(set(expected_years) - set(year_values))
        if missing_years:
            raise ValueError(
                f"eurostat_hicp {series}: annual weight gap; first missing year {missing_years[0]}."
            )

        actual_start = dates.min()
        expected_start = EUROSTAT_HICP_EXPECTED_INDEX_START[series]
        actual_weight_start = int(year_values.min())
        expected_weight_start = EUROSTAT_HICP_EXPECTED_WEIGHT_START_YEAR[series]

        index_status = "match" if actual_start == expected_start else (
            "earlier_than_frozen_contract" if actual_start < expected_start else "later_than_frozen_contract"
        )
        weight_status = "match" if actual_weight_start == expected_weight_start else (
            "earlier_than_frozen_contract" if actual_weight_start < expected_weight_start else "later_than_frozen_contract"
        )

        if actual_start != expected_start:
            record_failure(
                "index_start", series,
                actual_start.date().isoformat(), expected_start.date().isoformat(),
                f"index starts {actual_start.date()}, frozen contract is {expected_start.date()}",
            )
        if actual_weight_start != expected_weight_start:
            record_failure(
                "weight_start", series,
                actual_weight_start, expected_weight_start,
                f"weights start {actual_weight_start}, frozen contract is {expected_weight_start}",
            )

        start_status[series] = {
            "expected_index_start": expected_start.date().isoformat(),
            "actual_index_start": actual_start.date().isoformat(),
            "index_start_status": index_status,
            "expected_weight_start_year": expected_weight_start,
            "actual_weight_start_year": actual_weight_start,
            "weight_start_status": weight_status,
        }

    latest_weight_year = int(weight_rows["year"].max())
    latest_weight_series = set(weight_rows.loc[weight_rows["year"] == latest_weight_year, "series"])
    missing_latest_weights = sorted(set(EUROSTAT_HICP_OUTPUT_ORDER) - latest_weight_series)
    if missing_latest_weights:
        raise ValueError(
            f"eurostat_hicp: latest weight year {latest_weight_year} is missing series {missing_latest_weights}."
        )

    actual_breaks: dict[str, set[pd.Timestamp]] = {}
    for row in index_rows.itertuples(index=False):
        if "b" in _flag_tokens(row.flag):
            actual_breaks.setdefault(row.series, set()).add(pd.Timestamp(row.date))
    expected_breaks = {series: set(dates) for series, dates in EUROSTAT_HICP_EXPECTED_BREAKS.items()}
    all_break_series = set(actual_breaks) | set(expected_breaks)
    break_status: dict[str, dict[str, object]] = {}
    for series in sorted(all_break_series):
        actual = actual_breaks.get(series, set())
        expected = expected_breaks.get(series, set())
        actual_text = "|".join(d.date().isoformat() for d in sorted(actual))
        expected_text = "|".join(d.date().isoformat() for d in sorted(expected))
        status = "match" if actual == expected else "contract_changed"
        break_status[series] = {
            "actual_break_dates": actual_text,
            "expected_break_dates": expected_text,
            "break_status": status,
        }
        if actual != expected:
            record_failure(
                "break_dates", series, actual_text or "<none>", expected_text or "<none>",
                "break-flag contract changed",
            )

    indices = (
        index_rows.pivot(index="date", columns="series", values="index_2025")
        .sort_index().reindex(columns=EUROSTAT_HICP_OUTPUT_ORDER)
    )
    indices.index = pd.DatetimeIndex(indices.index, name="date")

    weights = (
        weight_rows.assign(year_int=weight_rows["year"].astype(int))
        .pivot(index="year_int", columns="series", values="weight_per_thousand")
        .sort_index().reindex(columns=EUROSTAT_HICP_OUTPUT_ORDER)
    )
    weights.index.name = "year"

    by_code = (
        weight_rows.assign(year_int=weight_rows["year"].astype(int))
        .pivot(index="year_int", columns="coicop18", values="weight_per_thousand")
        .sort_index()
    )
    identity_records: list[dict[str, object]] = []

    def add_identity(name, required_codes, lhs, rhs, *, guard_start_year=None) -> None:
        available = by_code[required_codes].dropna()
        if available.empty:
            record_failure(
                "weight_identity", name, "<no complete year>", "at least one complete year",
                f"no complete years for weight identity {name!r}",
            )
            return
        for year, row in available.iterrows():
            lhs_value = float(lhs(row))
            rhs_value = float(rhs(row))
            error = lhs_value - rhs_value
            guarded = guard_start_year is None or int(year) >= guard_start_year
            status = (
                "pass" if guarded and abs(error) <= EUROSTAT_HICP_WEIGHT_TOLERANCE_PER_THOUSAND
                else "fail" if guarded else "transition_not_guarded"
            )
            identity_records.append({
                "year": int(year),
                "identity": name,
                "lhs_per_thousand": lhs_value,
                "rhs_per_thousand": rhs_value,
                "error_per_thousand": error,
                "abs_error_per_thousand": abs(error),
                "tolerance_per_thousand": EUROSTAT_HICP_WEIGHT_TOLERANCE_PER_THOUSAND,
                "guarded": bool(guarded),
                "status": status,
            })
            if guarded and abs(error) > EUROSTAT_HICP_WEIGHT_TOLERANCE_PER_THOUSAND:
                record_failure(
                    "weight_identity", name,
                    f"{int(year)}: lhs={lhs_value:.6f}, rhs={rhs_value:.6f}, error={error:.6f}",
                    f"|error| <= {EUROSTAT_HICP_WEIGHT_TOLERANCE_PER_THOUSAND:.6f}",
                    f"weight identity fails in {int(year)} by {error:.6f} per thousand",
                )

    add_identity(
        "CP045 = CP0451 + CP0452 + CP0453 + CP0454 + CP0455",
        ["CP045", "CP0451", "CP0452", "CP0453", "CP0454", "CP0455"],
        lambda r: r["CP045"],
        lambda r: r["CP0451"] + r["CP0452"] + r["CP0453"] + r["CP0454"] + r["CP0455"],
    )
    add_identity(
        "NRG = ELC_GAS + FUEL",
        ["NRG", "ELC_GAS", "FUEL"],
        lambda r: r["NRG"],
        lambda r: r["ELC_GAS"] + r["FUEL"],
    )
    add_identity(
        "CP0722 = CP07221 + CP07222 + CP07223 + CP07224",
        ["CP0722", "CP07221", "CP07222", "CP07223", "CP07224"],
        lambda r: r["CP0722"],
        lambda r: r["CP07221"] + r["CP07222"] + r["CP07223"] + r["CP07224"],
        guard_start_year=2016,
    )
    add_identity(
        "NRG = CP045 + CP0722 - CP07224",
        ["NRG", "CP045", "CP0722", "CP07224"],
        lambda r: r["NRG"],
        lambda r: r["CP045"] + r["CP0722"] - r["CP07224"],
        guard_start_year=2017,
    )
    add_identity(
        "FUEL = CP0453 + CP0722 - CP07224",
        ["FUEL", "CP0453", "CP0722", "CP07224"],
        lambda r: r["FUEL"],
        lambda r: r["CP0453"] + r["CP0722"] - r["CP07224"],
        guard_start_year=2017,
    )
    identity_diagnostics = pd.DataFrame(identity_records).sort_values(["identity", "year"]).reset_index(drop=True)

    flagged = data.loc[data["flag"].map(bool)].copy()
    flags = flagged[["measure", "series", "coicop18", "date", "year", "flag"]].sort_values(
        ["series", "measure", "date", "year"], na_position="last"
    )

    metadata_rows = []
    for code, series in EUROSTAT_HICP_SERIES.items():
        idx = index_rows.loc[index_rows["series"] == series].sort_values("date")
        wt = weight_rows.loc[weight_rows["series"] == series].sort_values("year")
        index_flag_codes = sorted(set().union(*(_flag_tokens(v) for v in idx["flag"])))
        weight_flag_codes = sorted(set().union(*(_flag_tokens(v) for v in wt["flag"])))
        break_dates = sorted(
            pd.Timestamp(r.date).date().isoformat()
            for r in idx.itertuples(index=False)
            if "b" in _flag_tokens(r.flag)
        )
        row = {
            "series": series,
            "coicop18": code,
            "geo": EUROSTAT_HICP_GEO,
            "index_dataset": EUROSTAT_HICP_INDEX_DATASET,
            "index_unit": EUROSTAT_HICP_INDEX_UNIT,
            "index_first_date": idx["date"].min().date().isoformat(),
            "index_last_date": idx["date"].max().date().isoformat(),
            "index_observations": int(len(idx)),
            "weight_dataset": EUROSTAT_HICP_WEIGHT_DATASET,
            "weight_statinfo": weight_statinfo,
            "weight_first_year": int(wt["year"].min()),
            "weight_last_year": int(wt["year"].max()),
            "weight_observations": int(len(wt)),
            "index_flag_codes": "|".join(index_flag_codes),
            "weight_flag_codes": "|".join(weight_flag_codes),
            "break_dates": "|".join(break_dates),
            **start_status[series],
        }
        row.update(break_status.get(series, {
            "actual_break_dates": "",
            "expected_break_dates": "",
            "break_status": "no_break_contract",
        }))
        metadata_rows.append(row)
    series_metadata = pd.DataFrame(metadata_rows)

    stats = {
        "raw_rows": int(raw_rows),
        "retained_rows": int(len(data)),
        "columns": int(raw.shape[1]),
        "input_order": "long_by_series",
        "invalid_date_rows": 0,
        "duplicate_date_rows": 0,
        "duplicate_dates": 0,
        "all_missing_rows_dropped": 0,
        "error_markers_coerced_to_nan": 0,
        "first_date": indices.index.min().date().isoformat(),
        "last_date": indices.index.max().date().isoformat(),
        "first_weight_year": int(weights.index.min()),
        "last_weight_year": int(weights.index.max()),
        "geo": EUROSTAT_HICP_GEO,
        "index_unit": EUROSTAT_HICP_INDEX_UNIT,
        "weight_statinfo": weight_statinfo,
        "index_rows": int(len(index_rows)),
        "weight_rows": int(len(weight_rows)),
        "flagged_rows": int(len(flags)),
        "discovery_mode": bool(discovery),
        "empirical_contract_failures": int(len(empirical_failures)),
    }
    audit = {
        "series_metadata": series_metadata,
        "flags": flags,
        "weight_identities": identity_diagnostics,
        "validation_failures": pd.DataFrame(
            empirical_failures,
            columns=["contract", "series", "observed", "expected", "message"],
        ),
        "weight_statinfo": weight_statinfo,
        "latest_weight_year": latest_weight_year,
        "discovery_mode": bool(discovery),
    }
    return indices, weights, stats, audit


def _read_headline_hicp_sheet(
    xls: pd.ExcelFile,
    sheet: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Read the Headline HICP Power Query view.

    The monthly index block and annual weight block deliberately have different
    ECOICOP coverage:

    * indices: TOTAL, NRG, FOOD, FOOD_NP, FOOD_P, IGD_NNRG, SERV;
    * weights: TOTAL, NRG, FOOD_NP, FOOD_P, IGD_NNRG, SERV.

    ``FOOD`` is the official Eurostat aggregate used directly by the joint
    Headline BVAR. Its four-component annual aggregation weight is still built
    from ``FOOD_NP + FOOD_P``.
    """
    raw = pd.read_excel(xls, sheet_name=sheet, dtype=object)
    if raw.empty:
        raise ValueError(f"headline_hicp: sheet {sheet!r} is empty.")

    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in EUROSTAT_HICP_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"headline_hicp: missing required column(s) {missing}.")

    data = raw.loc[:, EUROSTAT_HICP_REQUIRED_COLUMNS].copy()
    for c in (
        "dataset", "measure", "series", "coicop18",
        "geo", "statinfo", "freq", "unit", "flag",
    ):
        data[c] = data[c].map(_clean_text)

    data["date"] = _parse_dates(data["date"])
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    for c in ("value", "index_2025", "weight_per_thousand"):
        data[c] = (
            pd.to_numeric(data[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

    idx = data.loc[data["measure"] == "hicp_index"].copy()
    wgt = data.loc[data["measure"] == "hicp_weight"].copy()
    if idx.empty or wgt.empty:
        raise ValueError("headline_hicp: both index and weight blocks are required.")

    bad_idx = idx.loc[
        (idx["dataset"] != EUROSTAT_HICP_INDEX_DATASET)
        | (idx["freq"] != "M")
        | (idx["unit"] != EUROSTAT_HICP_INDEX_UNIT)
        | (idx["geo"] != EUROSTAT_HICP_GEO)
    ]
    if not bad_idx.empty:
        raise ValueError(
            "headline_hicp: monthly index contract changed. "
            "Expected prc_hicp_minr / M / I25 / EA."
        )

    bad_wgt = wgt.loc[
        (wgt["dataset"] != EUROSTAT_HICP_WEIGHT_DATASET)
        | (wgt["freq"] != "A")
        | (wgt["geo"] != EUROSTAT_HICP_GEO)
    ]
    if not bad_wgt.empty:
        raise ValueError(
            "headline_hicp: annual weight contract changed. "
            "Expected prc_hicp_iw / A / EA."
        )

    expected_index_codes = set(HEADLINE_HICP_INDEX_SERIES)
    observed_index_codes = set(idx["coicop18"])
    if observed_index_codes != expected_index_codes:
        raise ValueError(
            "headline_hicp indices: ECOICOP coverage changed. "
            f"Missing={sorted(expected_index_codes-observed_index_codes)}; "
            f"unexpected={sorted(observed_index_codes-expected_index_codes)}."
        )
    for code, expected_series in HEADLINE_HICP_INDEX_SERIES.items():
        observed = set(idx.loc[idx["coicop18"] == code, "series"])
        if observed != {expected_series}:
            raise ValueError(
                f"headline_hicp indices: {code} expected "
                f"{expected_series!r}, found {sorted(observed)}."
            )

    expected_weight_codes = set(HEADLINE_HICP_WEIGHT_SERIES)
    observed_weight_codes = set(wgt["coicop18"])
    if observed_weight_codes != expected_weight_codes:
        raise ValueError(
            "headline_hicp weights: ECOICOP coverage changed. "
            f"Missing={sorted(expected_weight_codes-observed_weight_codes)}; "
            f"unexpected={sorted(observed_weight_codes-expected_weight_codes)}."
        )
    for code, expected_series in HEADLINE_HICP_WEIGHT_SERIES.items():
        observed = set(wgt.loc[wgt["coicop18"] == code, "series"])
        if observed != {expected_series}:
            raise ValueError(
                f"headline_hicp weights: {code} expected "
                f"{expected_series!r}, found {sorted(observed)}."
            )

    if idx["date"].isna().any() or idx["year"].isna().any():
        raise ValueError("headline_hicp: index rows require date and year.")
    if wgt["date"].notna().any() or wgt["year"].isna().any():
        raise ValueError("headline_hicp: weight rows require date=null and year.")

    index_year = idx["date"].dt.year.astype("Int64")
    if not index_year.equals(idx["year"].astype("Int64")):
        bad = idx.loc[
            index_year != idx["year"].astype("Int64"),
            ["series", "date", "year"],
        ].head()
        raise ValueError(f"headline_hicp: index date/year mismatch:\n{bad}")

    bad_index_values = (
        idx["value"].isna()
        | idx["index_2025"].isna()
        | idx["weight_per_thousand"].notna()
        | ((idx["value"] - idx["index_2025"]).abs() > 1e-12)
    )
    if bad_index_values.any():
        raise ValueError(
            "headline_hicp: invalid hicp_index value mapping in "
            f"{int(bad_index_values.sum())} row(s)."
        )

    bad_weight_values = (
        wgt["value"].isna()
        | wgt["weight_per_thousand"].isna()
        | wgt["index_2025"].notna()
        | ((wgt["value"] - wgt["weight_per_thousand"]).abs() > 1e-12)
    )
    if bad_weight_values.any():
        raise ValueError(
            "headline_hicp: invalid hicp_weight value mapping in "
            f"{int(bad_weight_values.sum())} row(s)."
        )

    if idx.duplicated(["series", "date", "measure"]).any():
        raise ValueError("headline_hicp: duplicate index keys.")
    if wgt.duplicated(["series", "year", "measure"]).any():
        raise ValueError("headline_hicp: duplicate weight keys.")

    for series in HEADLINE_HICP_INDEX_OUTPUT_ORDER:
        dates = pd.DatetimeIndex(
            idx.loc[idx["series"] == series, "date"].sort_values()
        )
        if len(dates) == 0:
            raise ValueError(f"headline_hicp {series}: no monthly observations.")
        missing_dates = pd.date_range(
            dates.min(), dates.max(), freq="MS"
        ).difference(dates)
        if len(missing_dates):
            raise ValueError(
                f"headline_hicp {series}: monthly gap at "
                f"{missing_dates[0].date()}."
            )

    for series in HEADLINE_HICP_WEIGHT_OUTPUT_ORDER:
        years = sorted(
            wgt.loc[wgt["series"] == series, "year"]
            .dropna().astype(int).unique()
        )
        if not years:
            raise ValueError(f"headline_hicp {series}: no annual weights.")
        missing_years = sorted(
            set(range(years[0], years[-1] + 1)) - set(years)
        )
        if missing_years:
            raise ValueError(
                f"headline_hicp {series}: annual weight gap at "
                f"{missing_years[0]}."
            )

    indices = (
        idx.pivot(index="date", columns="series", values="index_2025")
        .sort_index()
        .reindex(columns=HEADLINE_HICP_INDEX_OUTPUT_ORDER)
    )
    indices.index = pd.DatetimeIndex(indices.index, name="date")

    weights = (
        wgt.assign(year_int=wgt["year"].astype(int))
        .pivot(index="year_int", columns="series", values="weight_per_thousand")
        .sort_index()
        .reindex(columns=HEADLINE_HICP_WEIGHT_OUTPUT_ORDER)
    )
    weights.index.name = "year"

    identities = []
    for year, row in weights.iterrows():
        components = row[HEADLINE_COMPONENTS_FOR_TOTAL]
        if components.notna().all():
            component_sum = float(components.sum())
            error = component_sum - 1000.0
            status = (
                "pass"
                if abs(error) <= HEADLINE_WEIGHT_TOLERANCE_PER_THOUSAND
                else "fail"
            )
        else:
            component_sum, error, status = np.nan, np.nan, "not_testable"

        total_weight = row.get("hicp_total", np.nan)
        identities.append(
            {
                "year": int(year),
                "component_weight_sum": component_sum,
                "component_sum_minus_1000": error,
                "total_weight": total_weight,
                "status": status,
            }
        )

    identities_df = pd.DataFrame(identities)
    failures = identities_df.loc[identities_df["status"] == "fail"]
    if not failures.empty:
        r = failures.iloc[0]
        raise ValueError(
            "headline_hicp: component weights fail the 1000 identity in "
            f"{int(r['year'])}: sum={r['component_weight_sum']:.6f}."
        )

    starts: dict[str, dict[str, object]] = {}
    all_series = list(dict.fromkeys(
        HEADLINE_HICP_INDEX_OUTPUT_ORDER + HEADLINE_HICP_WEIGHT_OUTPUT_ORDER
    ))
    for series in all_series:
        index_start = indices[series].first_valid_index() if series in indices.columns else None
        index_end = indices[series].last_valid_index() if series in indices.columns else None
        weight_start = weights[series].first_valid_index() if series in weights.columns else None
        weight_end = weights[series].last_valid_index() if series in weights.columns else None
        starts[series] = {
            "index_start": index_start.date().isoformat() if index_start is not None else None,
            "index_end": index_end.date().isoformat() if index_end is not None else None,
            "weight_start": int(weight_start) if weight_start is not None else None,
            "weight_end": int(weight_end) if weight_end is not None else None,
        }

    stats = {
        "raw_rows": int(len(raw)),
        "retained_rows": int(len(data)),
        "columns": int(data.shape[1]),
        "input_order": "mixed_index_and_weight_blocks",
        "invalid_date_rows": 0,
        "duplicate_date_rows": 0,
        "duplicate_dates": 0,
        "all_missing_rows_dropped": 0,
        "error_markers_coerced_to_nan": 0,
        "first_date": indices.index.min().date().isoformat(),
        "last_date": indices.index.max().date().isoformat(),
        "first_weight_year": int(weights.index.min()),
        "last_weight_year": int(weights.index.max()),
        "geo": EUROSTAT_HICP_GEO,
        "index_unit": EUROSTAT_HICP_INDEX_UNIT,
        "food_index_source": "Eurostat prc_hicp_minr / FOOD / I25 / EA",
        "food_weight_source": "Eurostat prc_hicp_iw / FOOD_NP + FOOD_P",
    }
    audit = {"weight_identities": identities_df, "series_starts": starts}
    return indices, weights, stats, audit


def _build_headline_joint_weights(
    headline_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual Energy/Food/NEIG/Services weights for the joint Headline BVAR.

    The monthly Food index comes from the official Eurostat FOOD aggregate.
    Its annual HICP aggregation weight is constructed from the official
    subcomponent weights:

        Food = Processed Food + Unprocessed Food.

    Whenever all four component weights are published, their sum must equal
    1000 within HEADLINE_WEIGHT_TOLERANCE_PER_THOUSAND.
    """
    required = [
        "hicp_energy",
        "hicp_unprocessed_food",
        "hicp_processed_food",
        "hicp_neig",
        "hicp_services",
    ]
    missing = [c for c in required if c not in headline_weights.columns]
    if missing:
        raise KeyError(
            f"headline_joint_weights: missing source weight columns {missing}."
        )

    src = headline_weights.copy().astype(float).sort_index()
    food_complete = src[
        ["hicp_unprocessed_food", "hicp_processed_food"]
    ].notna().all(axis=1)
    food = (
        src["hicp_unprocessed_food"] + src["hicp_processed_food"]
    ).where(food_complete)

    joint = pd.DataFrame(
        {
            "hicp_energy": src["hicp_energy"],
            "hicp_food": food,
            "hicp_neig": src["hicp_neig"],
            "hicp_services": src["hicp_services"],
        },
        index=src.index,
    )
    joint.index.name = "year"

    rows = []
    for year, row in joint.iterrows():
        if row.notna().all():
            component_sum = float(row.sum())
            error = component_sum - 1000.0
            status = (
                "pass"
                if abs(error) <= HEADLINE_WEIGHT_TOLERANCE_PER_THOUSAND
                else "fail"
            )
        else:
            component_sum, error, status = np.nan, np.nan, "not_testable"
        rows.append(
            {
                "year": int(year),
                "energy_weight": row["hicp_energy"],
                "food_weight": row["hicp_food"],
                "food_weight_source": (
                    "hicp_processed_food + hicp_unprocessed_food"
                ),
                "neig_weight": row["hicp_neig"],
                "services_weight": row["hicp_services"],
                "component_weight_sum": component_sum,
                "component_sum_minus_1000": error,
                "status": status,
            }
        )

    diagnostics = pd.DataFrame(rows)
    failures = diagnostics.loc[diagnostics["status"] == "fail"]
    if not failures.empty:
        first = failures.iloc[0]
        raise ValueError(
            "headline_joint_weights: Energy+Food+NEIG+Services fail the "
            f"1000 identity in {int(first['year'])}: "
            f"sum={float(first['component_weight_sum']):.6f}."
        )
    return joint, diagnostics


def _validate_headline_joint_panel(panel: pd.DataFrame) -> None:
    """Validate the four positive HICP indices used by the joint BVAR."""
    _validate_model_panel("headline_joint", panel, HEADLINE_JOINT_SPEC)
    for column in HEADLINE_JOINT_SPEC["columns"]:
        observed = panel[column].dropna()
        if observed.empty:
            raise ValueError(f"headline_joint: {column} has no observations.")
        bad = observed <= 0
        if bad.any():
            first = bad[bad].index[0]
            raise ValueError(
                f"headline_joint: {column} must be strictly positive for "
                f"log differences; first invalid observation is "
                f"{pd.Timestamp(first).date()}."
            )



def _read_workbook_sources(
    raw_path: Path,
    *,
    hicp_discovery: bool = False,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, object]],
    dict[str, str],
    list[dict[str, object]],
    dict[str, object] | None,
]:
    """Read six-model sources and attempt the independent HICP lineage.

    HICP is optional here. Missing or structurally broken HICP never prevents
    the five core source blocks from being returned; the caller decides whether
    HICP is required for the requested run.
    """
    try:
        xls = pd.ExcelFile(raw_path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "Reading the raw Excel workbook (.xlsm/.xlsx) requires openpyxl. Install it with "
            "'python -m pip install openpyxl'."
        ) from exc

    resolved = {
        logical: _sheet_name(xls, logical)
        for logical in (
            "haver", "bloomberg", "european_commission", "world_bank", "eurostat"
        )
    }
    hicp_sheet = _sheet_name(xls, "eurostat_hicp", required=False)
    if hicp_sheet is not None:
        resolved["eurostat_hicp"] = hicp_sheet
    headline_hicp_sheet = _sheet_name(xls, "eurostat_headline_hicp", required=False)
    if headline_hicp_sheet is not None:
        resolved["headline_hicp"] = headline_hicp_sheet

    frames: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict[str, object]] = {}
    frames["haver"], stats["haver"] = _read_haver_sheet(xls, resolved["haver"])
    frames["bloomberg"], stats["bloomberg"] = _read_bloomberg_sheet(xls, resolved["bloomberg"])
    frames["european_commission"], stats["european_commission"] = _read_simple_sheet(
        xls,
        resolved["european_commission"],
        source="european_commission",
        rename=EC_COLUMN_MAP,
        required_columns=("wob_petroleum_cpr_ntax", "wob_diesel_cpr_ntax", "wob_gas_cpr_ntax"),
    )
    frames["world_bank"], stats["world_bank"] = _read_simple_sheet(
        xls, resolved["world_bank"], source="world_bank", required_columns=WORLD_BANK_REQUIRED,
    )
    frames["eurostat"], stats["eurostat"] = _read_simple_sheet(
        xls, resolved["eurostat"], source="eurostat", required_columns=EUROSTAT_REQUIRED,
    )

    hicp_audit: dict[str, object] | None = None
    if hicp_sheet is None:
        hicp_audit = {
            "status": "missing",
            "error": "Eurostat_HICP sheet is absent.",
            "discovery_mode": bool(hicp_discovery),
        }
    else:
        try:
            (
                frames["eurostat_hicp_indices"],
                frames["eurostat_hicp_weights"],
                stats["eurostat_hicp"],
                parsed_audit,
            ) = _read_eurostat_hicp_sheet(xls, hicp_sheet, discovery=hicp_discovery)
            hicp_audit = {"status": "parsed", **parsed_audit}
        except (ValueError, KeyError) as exc:
            hicp_audit = {
                "status": "error",
                "error": str(exc),
                "discovery_mode": bool(hicp_discovery),
            }

    # Optional Headline HICP lineage. Failures are recorded, never allowed to
    # invalidate the independent six-model Energy build.
    if headline_hicp_sheet is not None:
        try:
            (
                frames["headline_hicp_indices"],
                frames["headline_hicp_weights"],
                stats["headline_hicp"],
                frames["headline_hicp_audit"],
            ) = _read_headline_hicp_sheet(xls, headline_hicp_sheet)
        except (ValueError, KeyError) as exc:
            frames["headline_hicp_error"] = pd.DataFrame({"error": [str(exc)]})


    metadata_records: list[dict[str, object]] = []
    metadata_sheet = _sheet_name(xls, "metadata", required=False)
    if metadata_sheet is not None:
        metadata_raw = pd.read_excel(xls, sheet_name=metadata_sheet, header=None, dtype=object)
        for row in metadata_raw.itertuples(index=False, name=None):
            values = [_clean_text(v) for v in row]
            if any(values):
                metadata_records.append({f"column_{i+1}": value for i, value in enumerate(values)})
        resolved["metadata"] = metadata_sheet

    return frames, stats, resolved, metadata_records, hicp_audit

# ===========================================================================
# Frequency helpers and partial-period diagnostics
# ===========================================================================


def _required_series(
    frame: pd.DataFrame,
    candidates: str | tuple[str, ...],
    *,
    source: str,
    logical_name: str,
) -> pd.Series:
    options = (candidates,) if isinstance(candidates, str) else candidates
    for column in options:
        if column in frame.columns:
            series = frame[column].astype(float).rename(logical_name)
            if series.notna().sum() == 0:
                raise ValueError(f"{source}.{column} is entirely missing.")
            return series
    raise KeyError(
        f"Missing {logical_name!r} in {source}. Expected one of {options}. "
        f"Available columns: {list(frame.columns)}"
    )


def _monthly_mean(series: pd.Series) -> pd.Series:
    out = series.groupby(series.index.to_period("M")).mean()
    out.index = out.index.to_timestamp(how="start")
    out.index.name = "date"
    return out.sort_index()


def _wob_hicp_mapping_diagnostics(
    wob: pd.DataFrame,
    hicp_indices: pd.DataFrame,
) -> pd.DataFrame:
    """Check CP07221=diesel and CP07222=petrol against WOB consumer prices.

    Weight identities are symmetric in petrol/diesel and cannot detect an
    inversion. Monthly log-change correlations can: own-series correlation must
    exceed cross-series correlation for each fuel.
    """
    price_columns = {
        "petrol": "wob_petroleum_cpr_wtax" if "wob_petroleum_cpr_wtax" in wob.columns else "wob_petroleum_cpr_ntax",
        "diesel": "wob_diesel_cpr_wtax" if "wob_diesel_cpr_wtax" in wob.columns else "wob_diesel_cpr_ntax",
    }
    hicp_columns = {"petrol": "hicp_petrol", "diesel": "hicp_diesel"}
    records = []

    for wob_fuel in ("petrol", "diesel"):
        price = _required_series(
            wob, price_columns[wob_fuel], source="european_commission",
            logical_name=f"wob_{wob_fuel}_mapping_check",
        )
        monthly_price = _monthly_mean(price)
        wob_growth = np.log(monthly_price).diff()
        own_name = hicp_columns[wob_fuel]
        cross_name = hicp_columns["diesel" if wob_fuel == "petrol" else "petrol"]
        own = pd.concat([wob_growth.rename("wob"), np.log(hicp_indices[own_name]).diff().rename("hicp")], axis=1).dropna()
        cross = pd.concat([wob_growth.rename("wob"), np.log(hicp_indices[cross_name]).diff().rename("hicp")], axis=1).dropna()
        if len(own) < 24 or len(cross) < 24:
            raise ValueError(f"WOB/HICP mapping check for {wob_fuel} has too few overlapping monthly changes.")
        own_corr = float(own["wob"].corr(own["hicp"]))
        cross_corr = float(cross["wob"].corr(cross["hicp"]))
        records.append({
            "wob_series": wob_fuel,
            "wob_column": price_columns[wob_fuel],
            "expected_hicp_series": own_name,
            "cross_hicp_series": cross_name,
            "observations_own": int(len(own)),
            "observations_cross": int(len(cross)),
            "own_log_change_correlation": own_corr,
            "cross_log_change_correlation": cross_corr,
            "own_minus_cross_margin": own_corr - cross_corr,
            "status": "pass" if own_corr > cross_corr else "fail",
        })
    return pd.DataFrame(records)
def _monthly_last(series: pd.Series) -> pd.Series:
    out = series.groupby(series.index.to_period("M")).last()
    out.index = out.index.to_timestamp(how="start")
    out.index.name = "date"
    return out.sort_index()


def _weekly_mean(series: pd.Series) -> pd.Series:
    """Monday-Sunday mean, labelled by the Monday starting the week."""
    out = series.groupby(series.index.to_period("W-SUN")).mean()
    out.index = out.index.start_time
    out.index.name = "date"
    return out.sort_index()


def _weekly_last(series: pd.Series) -> pd.Series:
    out = series.groupby(series.index.to_period("W-SUN")).last()
    out.index = out.index.start_time
    out.index.name = "date"
    return out.sort_index()


def _aggregation_coverage(
    raw: pd.Series,
    *,
    frequency: str,
    label: str,
    method: str,
    label_shift_weeks: int = 0,
) -> dict[str, object]:
    observed = raw.dropna().sort_index()
    if observed.empty:
        raise ValueError(f"{label}: no observations to aggregate.")

    if frequency == "monthly":
        periods = observed.index.to_period("M")
    elif frequency == "weekly":
        periods = observed.index.to_period("W-SUN")
    else:
        raise ValueError(f"Unsupported aggregation frequency: {frequency}")

    counts = observed.groupby(periods).size().sort_index()
    last_period = counts.index[-1]
    last_count = int(counts.iloc[-1])
    reference = counts.iloc[:-1].tail(COVERAGE_REFERENCE_PERIODS)
    expected = max(int(round(float(reference.median()))), 1) if len(reference) else last_count
    ratio = float(last_count) / float(expected) if expected else float("nan")
    complete = bool(ratio >= COMPLETE_PERIOD_COVERAGE_RATIO)
    panel_label = last_period.start_time + pd.DateOffset(weeks=label_shift_weeks)

    period_code = "M" if frequency == "monthly" else "W-SUN"
    final_days = observed[observed.index.to_period(period_code) == last_period]
    partial_mean = float(final_days.mean())
    last_input_value = float(final_days.iloc[-1])

    recent = observed.tail(COVERAGE_REFERENCE_PERIODS * max(expected, 1))
    level_diff = recent.diff().dropna()
    daily_innovation_var = float(level_diff.var(ddof=1)) if len(level_diff) > 1 else None
    positive = recent[recent > 0]
    logret = np.log(positive).diff().dropna() if len(positive) > 1 else positive.iloc[:0]
    daily_logreturn_var = float(logret.var(ddof=1)) if len(logret) > 1 else None

    return {
        "series": label,
        "frequency": frequency,
        "aggregation": method,
        "final_period_start": last_period.start_time.date().isoformat(),
        "final_period_end": last_period.end_time.date().isoformat(),
        "panel_label_date": panel_label.date().isoformat(),
        "label_shift_weeks": int(label_shift_weeks),
        "final_period_input_observations": last_count,
        "expected_input_observations": expected,
        "coverage_ratio": round(ratio, 4),
        "final_period_complete": complete,
        "first_input_date": final_days.index.min().date().isoformat(),
        "last_input_date": observed.index.max().date().isoformat(),
        "partial_mean": partial_mean,
        "last_input_value": last_input_value,
        "daily_innovation_var": daily_innovation_var,
        "daily_logreturn_var": daily_logreturn_var,
        "reference_periods_used": int(len(reference)),
    }


def partial_period_conditional_estimate(
    partial_mean: float,
    last_input_value: float,
    k: int,
    n: int,
    daily_var: float | None = None,
) -> dict[str, float | None]:
    if n <= 0 or k <= 0 or k > n:
        raise ValueError(f"Require 0 < k <= n; got k={k}, n={n}.")
    cond_mean = (k * partial_mean + (n - k) * last_input_value) / n
    if daily_var is None or n == k:
        cond_var = 0.0 if n == k else None
    else:
        cond_var = daily_var / n**2 * sum(r * r for r in range(1, n - k + 1))
    return {
        "conditional_mean": float(cond_mean),
        "conditional_variance": None if cond_var is None else float(cond_var),
        "partial_mean_bias": float(cond_mean - partial_mean),
    }


def _final_period_input_records(
    raw: pd.Series,
    *,
    frequency: str,
    panel_label_date: str,
    panel_column: str | None,
    series_label: str,
) -> list[dict[str, object]]:
    observed = raw.dropna().sort_index()
    period = "M" if frequency == "monthly" else "W-SUN"
    last_period = observed.index.to_period(period)[-1]
    final = observed[observed.index.to_period(period) == last_period]
    return [
        {
            "series": series_label,
            "panel_column": panel_column if panel_column is not None else "",
            "panel_label_date": panel_label_date,
            "frequency": frequency,
            "input_date": stamp.date().isoformat(),
            "input_value": float(value),
        }
        for stamp, value in final.items()
    ]


def _bind_coverage_to_panels(
    coverage: list[dict[str, object]],
    panels: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    bound: list[dict[str, object]] = []
    for original in coverage:
        row = dict(original)
        series = str(row["series"])
        if series not in COVERAGE_TO_PANEL_COLUMN:
            raise KeyError(f"{series} has no COVERAGE_TO_PANEL_COLUMN entry.")

        panel_column = COVERAGE_TO_PANEL_COLUMN[series]
        row["panel_column"] = panel_column
        label = pd.Timestamp(row["panel_label_date"])
        models: list[str] = []
        files: list[str] = []
        present: list[bool] = []

        if panel_column is not None:
            for name, panel in panels.items():
                if panel_column not in panel.columns:
                    continue
                models.append(name)
                files.append(MODEL_SPECS[name]["file"])
                present.append(bool(label in panel.index and pd.notna(panel.at[label, panel_column])))

        row["affected_models"] = "|".join(models)
        row["affected_files"] = "|".join(files)
        row["present_in_panels"] = int(sum(present))
        row["panel_targets"] = len(models)
        bound.append(row)
    return bound


def _warn_on_partial_periods(coverage: list[dict[str, object]]) -> None:
    for row in coverage:
        if row["final_period_complete"]:
            continue
        warnings.warn(
            f"{row['series']}: {row['panel_label_date']} is based on "
            f"{row['final_period_input_observations']}/"
            f"{row['expected_input_observations']} expected input observations "
            f"({row['coverage_ratio']:.0%}); it is a partial aggregate.",
            stacklevel=2,
        )


def _regular_panel(
    columns: dict[str, pd.Series],
    *,
    frequency: str,
    target_columns: list[str],
) -> pd.DataFrame:
    panel = pd.concat(columns, axis=1).sort_index()
    target_first_dates = [
        panel[column].first_valid_index()
        for column in target_columns
        if panel[column].first_valid_index() is not None
    ]
    if len(target_first_dates) != len(target_columns):
        missing = [column for column in target_columns if panel[column].notna().sum() == 0]
        raise ValueError(f"Target series are entirely missing: {missing}")

    start = min(target_first_dates)
    end = panel.last_valid_index()
    if end is None:
        raise ValueError("Cannot build a panel with no observed values.")

    if frequency == "weekly":
        start = pd.Timestamp(start).to_period("W-SUN").start_time
        end = pd.Timestamp(end).to_period("W-SUN").start_time
        index = pd.date_range(start, end, freq="W-MON", name="date")
    elif frequency == "monthly":
        start = pd.Timestamp(start).to_period("M").to_timestamp(how="start")
        end = pd.Timestamp(end).to_period("M").to_timestamp(how="start")
        index = pd.date_range(start, end, freq="MS", name="date")
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    regular = panel.reindex(index)
    observed_any = regular.notna().any(axis=1)
    if not observed_any.any():
        raise ValueError("Cannot build a panel with no observed values.")
    first = observed_any[observed_any].index[0]
    last = observed_any[observed_any].index[-1]
    return regular.loc[first:last]


# ===========================================================================
# Paper-specific transformations
# ===========================================================================


def _commodity_prices_in_euro(bloomberg: pd.DataFrame) -> dict[str, pd.Series]:
    fx = _required_series(
        bloomberg, "eurusd", source="bloomberg", logical_name="eurusd"
    )
    if (fx.dropna() <= 0).any():
        raise ValueError("EURUSD must be strictly positive.")

    output: dict[str, pd.Series] = {}
    for logical in ("crude_oil_usd", "refined_petroleum_usd", "refined_diesel_usd"):
        price = _required_series(
            bloomberg, logical, source="bloomberg", logical_name=logical
        )
        aligned = pd.concat([price, fx], axis=1)
        eur_name = logical.replace("_usd", "_eur")
        converted = aligned[logical] / aligned["eurusd"]
        barrels_per_tonne = BLOOMBERG_UNITS[logical]["barrels_per_metric_tonne"]
        if barrels_per_tonne is not None:
            converted = converted / float(barrels_per_tonne)
        output[eur_name] = converted.rename(eur_name)

    output["eurusd"] = fx
    output["ttf_eur_mwh"] = _required_series(
        bloomberg, "ttf_eur_mwh", source="bloomberg", logical_name="ttf_eur_mwh"
    )
    return output


def _ols_scale_no_intercept(x: pd.Series, y: pd.Series, *, label: str) -> float:
    common = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(common) < 6:
        raise ValueError(f"{label}: at least 6 overlapping observations required; found {len(common)}.")
    denominator = float(np.dot(common["x"], common["x"]))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"{label}: invalid OLS denominator.")
    scale = float(np.dot(common["x"], common["y"]) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label}: estimated scale must be positive; got {scale}.")
    return scale


def _build_natural_gas_series(
    bloomberg_prices: dict[str, pd.Series],
    world_bank: pd.DataFrame,
) -> tuple[pd.Series, dict[str, object]]:
    ttf = _monthly_mean(bloomberg_prices["ttf_eur_mwh"]).rename("ttf")
    fx_monthly = _monthly_mean(bloomberg_prices["eurusd"]).rename("eurusd")
    wb_usd = _monthly_last(
        _required_series(
            world_bank,
            "wb_natural_gas_europe_usd_mmbtu",
            source="world_bank",
            logical_name="wb_usd_mmbtu",
        )
    )

    aligned = pd.concat([wb_usd, fx_monthly], axis=1)
    invalid_fx = aligned["eurusd"].notna() & (aligned["eurusd"] <= 0)
    if invalid_fx.any():
        bad_date = invalid_fx[invalid_fx].index[0]
        raise ValueError(f"EURUSD invalid at {bad_date.date().isoformat()}.")
    proxy_eur = (aligned["wb_usd_mmbtu"] / aligned["eurusd"]).rename("proxy_eur")

    first_ttf = ttf.first_valid_index()
    first_proxy = proxy_eur.first_valid_index()
    if first_ttf is None:
        raise ValueError("Bloomberg TTF is entirely missing.")
    if first_proxy is None:
        raise ValueError("World Bank natural-gas proxy is entirely missing.")

    proxy_at_first_ttf = proxy_eur.get(first_ttf, np.nan)
    if np.isfinite(proxy_at_first_ttf):
        anchor_date = first_ttf
        anchor_fallback_used = False
    else:
        overlap0 = pd.concat([proxy_eur, ttf], axis=1).dropna().sort_index()
        if overlap0.empty:
            raise ValueError("No overlap between TTF and World Bank gas proxy.")
        anchor_date = overlap0.index.min()
        anchor_fallback_used = True

    ttf_anchor = float(ttf.loc[anchor_date])
    proxy_anchor = float(proxy_eur.loc[anchor_date])
    if not np.isfinite(ttf_anchor) or ttf_anchor <= 0:
        raise ValueError("TTF chain-link anchor must be positive.")
    if not np.isfinite(proxy_anchor) or proxy_anchor <= 0:
        raise ValueError("World Bank gas proxy chain-link anchor must be positive.")

    anchor_ratio = ttf_anchor / proxy_anchor
    backcast = (anchor_ratio * proxy_eur).rename("natural_gas_eur_mwh")
    full_index = pd.date_range(
        min(first_proxy, first_ttf),
        max(proxy_eur.last_valid_index(), ttf.last_valid_index()),
        freq="MS",
        name="date",
    )
    combined = ttf.reindex(full_index).rename("natural_gas_eur_mwh")
    backcast = backcast.reindex(full_index)
    backcast_mask = combined.isna() & (combined.index < first_ttf)
    combined.loc[backcast_mask] = backcast.loc[backcast_mask]

    unresolved_pre_ttf = (combined.index < first_ttf) & combined.isna().to_numpy()
    implied_level_ratio = float(anchor_ratio / MMBTU_PER_MWH)
    if not 0.6 <= implied_level_ratio <= 1.6:
        warnings.warn(
            "natural_gas_eur_mwh: chain-link anchor implies TTF/proxy level "
            f"ratio {implied_level_ratio:.2f} after MMBtu-to-MWh adjustment.",
            stacklevel=2,
        )

    common = ttf.index.intersection(proxy_eur.index).sort_values()
    overlap = pd.DataFrame({"ttf": ttf.reindex(common), "proxy": proxy_eur.reindex(common)}).dropna()
    growth = np.log(overlap).diff().dropna() if len(overlap) > 1 else overlap.iloc[:0]
    if len(growth) >= 12:
        growth_correlation = float(growth["ttf"].corr(growth["proxy"]))
        growth_rmse = float(np.sqrt(np.mean((growth["ttf"] - growth["proxy"]) ** 2)))
    else:
        growth_correlation = None
        growth_rmse = None

    info = {
        "method": (
            "Bloomberg TTF monthly mean; pre-TTF months are backcast with "
            "World Bank Natural gas, Europe via backward chain-linking."
        ),
        "formula": "backcast_t = TTF_anchor * proxy_EUR_t / proxy_EUR_anchor",
        "proxy_input_unit": "USD/MMBtu",
        "proxy_currency_conversion": "USD/MMBtu divided by USD-per-EUR",
        "output_unit": "EUR/MWh (set by TTF anchor)",
        "anchor_date": anchor_date.date().isoformat(),
        "anchor_ratio": float(anchor_ratio),
        "anchor_fallback_used": bool(anchor_fallback_used),
        "first_ttf_date": first_ttf.date().isoformat(),
        "first_proxy_date": first_proxy.date().isoformat(),
        "mmbtu_per_mwh_reference": MMBTU_PER_MWH,
        "implied_ttf_over_proxy_level_ratio": implied_level_ratio,
        "overlap_months": int(len(overlap)),
        "overlap_log_growth_correlation": growth_correlation,
        "overlap_log_growth_rmse": growth_rmse,
        "backcast_observations": int(backcast_mask.sum()),
        "unresolved_pre_ttf_missing": int(unresolved_pre_ttf.sum()),
    }
    return combined, info


def _expand_semesters(
    series: pd.Series,
    monthly_index: pd.DatetimeIndex,
    *,
    carry_forward_edge: bool = True,
    warn_after_months: int = DEFAULT_TAX_CARRY_WARN_MONTHS,
    error_after_months: int = DEFAULT_TAX_CARRY_ERROR_MONTHS,
) -> tuple[pd.Series, int]:
    sparse = _monthly_last(series.dropna())
    if sparse.empty:
        raise ValueError(f"{series.name}: no semiannual observations available.")

    unexpected = sorted(set(sparse.index.month) - set(SEMIANNUAL_MONTHS))
    if unexpected:
        raise ValueError(
            f"{series.name}: semiannual observations must be January/July; found months {unexpected}."
        )

    expanded = sparse.reindex(monthly_index).ffill(limit=5)
    carried = 0
    last_observation = sparse.last_valid_index()
    valid_until = last_observation + pd.DateOffset(months=5)

    if carry_forward_edge:
        tail = monthly_index > valid_until
        carried = int((tail & expanded.isna()).sum())
        expanded.loc[tail] = expanded.loc[tail].fillna(sparse.loc[last_observation])
        if error_after_months is not None and carried > error_after_months:
            raise ValueError(
                f"{series.name}: last semester {last_observation.date()} would be "
                f"carried {carried} months, above the {error_after_months}-month limit."
            )
        if carried > warn_after_months:
            warnings.warn(
                f"{series.name}: holding last published semester constant for {carried} months "
                "beyond its six-month validity window.",
                stacklevel=2,
            )

    expanded.name = series.name
    return expanded, carried


def _build_pre_tax_price(
    *,
    hicp: pd.Series,
    wtax: pd.Series,
    vat: pd.Series,
    exc: pd.Series,
    name: str,
    price_unit_multiplier: float = 1.0,
    source_price_unit: str = "source price unit",
    output_price_unit: str = "source price unit",
    tax_carry_warn_months: int = DEFAULT_TAX_CARRY_WARN_MONTHS,
    tax_carry_error_months: int = DEFAULT_TAX_CARRY_ERROR_MONTHS,
) -> tuple[pd.Series, dict[str, object]]:
    if not np.isfinite(price_unit_multiplier) or price_unit_multiplier <= 0:
        raise ValueError("price_unit_multiplier must be strictly positive.")

    hicp = _monthly_last(hicp).rename("hicp")
    wtax_sparse = (price_unit_multiplier * _monthly_last(wtax.dropna())).rename("wtax")
    vat_sparse = _monthly_last(vat.dropna()).rename("vat")
    exc_sparse = (price_unit_multiplier * _monthly_last(exc.dropna())).rename("exc")

    scale = _ols_scale_no_intercept(
        hicp.reindex(wtax_sparse.index),
        wtax_sparse,
        label=f"{name}: HICP -> Eurostat after-tax price ({output_price_unit})",
    )

    start = min(
        stamp
        for stamp in (
            hicp.first_valid_index(),
            vat_sparse.first_valid_index(),
            exc_sparse.first_valid_index(),
        )
        if stamp is not None
    )
    end = hicp.last_valid_index()
    if end is None:
        raise ValueError(f"{name}: HICP target is entirely missing.")
    index = pd.date_range(start, end, freq="MS", name="date")

    hicp_m = hicp.reindex(index)
    vat_m, vat_carried = _expand_semesters(
        vat_sparse,
        index,
        warn_after_months=tax_carry_warn_months,
        error_after_months=tax_carry_error_months,
    )
    exc_m, exc_carried = _expand_semesters(
        exc_sparse,
        index,
        warn_after_months=tax_carry_warn_months,
        error_after_months=tax_carry_error_months,
    )
    after_tax_implied = scale * hicp_m
    pre_tax = (after_tax_implied / (1.0 + vat_m / 100.0) - exc_m).rename(name)

    if (pre_tax.dropna() <= 0).any():
        bad = pre_tax[pre_tax <= 0].head().to_dict()
        raise ValueError(f"{name}: non-positive constructed pre-tax prices: {bad}")

    fit = pd.concat(
        [wtax_sparse, (scale * hicp).reindex(wtax_sparse.index).rename("fitted")], axis=1
    ).dropna()
    rmse = float(np.sqrt(np.mean((fit["wtax"] - fit["fitted"]) ** 2)))
    unit_tag = output_price_unit.replace("/", "_").replace(" ", "_")

    info = {
        "formula": f"pre_tax_{unit_tag} = gamma_{unit_tag} * HICP / (1 + VAT/100) - EXC_{unit_tag}",
        "source_price_unit": source_price_unit,
        "output_price_unit": output_price_unit,
        "price_unit_multiplier": float(price_unit_multiplier),
        "gamma": scale,
        "gamma_unit": f"{output_price_unit} per HICP index point",
        "ols_observations": int(len(fit)),
        "after_tax_fit_rmse": rmse,
        "after_tax_fit_rmse_unit": output_price_unit,
        "tax_months_carried_forward": {"vat": vat_carried, "excise": exc_carried},
        "first_valid": pre_tax.first_valid_index().date().isoformat() if pre_tax.first_valid_index() is not None else None,
        "last_valid": pre_tax.last_valid_index().date().isoformat() if pre_tax.last_valid_index() is not None else None,
    }
    return pre_tax, info


# ===========================================================================
# Panel validation and diagnostics
# ===========================================================================


def _validate_model_panel(name: str, panel: pd.DataFrame, spec: dict) -> None:
    if list(panel.columns) != spec["columns"]:
        raise ValueError(
            f"{name}: expected columns {spec['columns']}, found {list(panel.columns)}"
        )
    if panel.empty:
        raise ValueError(f"{name}: output panel is empty.")
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates must be unique and sorted ascending internally.")
    if np.isinf(panel.to_numpy(dtype=float)).any():
        raise ValueError(f"{name}: infinite values found.")
    if panel.isna().all().any():
        bad = panel.columns[panel.isna().all()].tolist()
        raise ValueError(f"{name}: entirely missing columns: {bad}")

    if spec["frequency"] == "weekly":
        if not (panel.index.weekday == 0).all():
            raise ValueError(f"{name}: weekly observations must be Monday-labelled.")
        expected = pd.date_range(panel.index.min(), panel.index.max(), freq="W-MON")
    else:
        if not (panel.index.day == 1).all():
            raise ValueError(f"{name}: monthly dates must be month-starts.")
        expected = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    if not panel.index.equals(expected):
        raise ValueError(f"{name}: calendar index is not regular.")


def _usable_design_rows(equation_panel: pd.DataFrame, lags: int) -> int:
    transformed = equation_panel.diff()
    blocks = [transformed]
    blocks.extend(transformed.shift(lag) for lag in range(1, lags + 1))
    return int(len(pd.concat(blocks, axis=1).dropna()))


def _diagnostic_rows(model: str, panel: pd.DataFrame, spec: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in panel.columns:
        observed = panel[column].dropna()
        if observed.empty:
            internal_gaps = 0
            leading_missing = len(panel)
            trailing_missing = len(panel)
        else:
            inside = panel[column].loc[observed.index.min(): observed.index.max()]
            internal_gaps = int(inside.isna().sum())
            leading_missing = int(panel.index.get_loc(observed.index.min()))
            trailing_missing = int(len(panel) - 1 - panel.index.get_loc(observed.index.max()))

        rows.append({
            "diagnostic_type": "series",
            "model": model,
            "equation": None,
            "output_file": spec["file"],
            "frequency": spec["frequency"],
            "lags_in_paper": spec["lags"],
            "series": column,
            "observations": int(observed.size),
            "first_date": observed.index.min().date().isoformat() if not observed.empty else None,
            "last_date": observed.index.max().date().isoformat() if not observed.empty else None,
            "missing_values": int(panel[column].isna().sum()),
            "internal_gaps": internal_gaps,
            "leading_missing": leading_missing,
            "trailing_missing": trailing_missing,
            "coverage_percent": round(100.0 * observed.size / len(panel), 2),
            "balanced_level_observations": None,
            "balanced_difference_observations": None,
            "usable_after_difference_and_lags": None,
            "usable_edge_only_upper_bound": None,
            "rows_lost_to_interior_gaps": None,
        })

    for equation, columns in spec["equations"].items():
        equation_panel = panel[columns]
        balanced_levels = equation_panel.dropna()
        balanced_differences = equation_panel.diff().dropna()
        usable = _usable_design_rows(equation_panel, int(spec["lags"]))
        naive_usable = max(0, len(balanced_differences) - int(spec["lags"]))
        any_obs = equation_panel.notna().any(axis=1)
        if any_obs.any():
            first = any_obs[any_obs].index[0]
            last = any_obs[any_obs].index[-1]
            internal = int(equation_panel.isna().any(axis=1).loc[first:last].sum())
        else:
            internal = 0

        rows.append({
            "diagnostic_type": "equation_sample",
            "model": model,
            "equation": equation,
            "output_file": spec["file"],
            "frequency": spec["frequency"],
            "lags_in_paper": spec["lags"],
            "series": " | ".join(columns),
            "observations": int(len(balanced_levels)),
            "first_date": balanced_levels.index.min().date().isoformat() if not balanced_levels.empty else None,
            "last_date": balanced_levels.index.max().date().isoformat() if not balanced_levels.empty else None,
            "missing_values": int(len(panel) - len(balanced_levels)),
            "internal_gaps": internal,
            "leading_missing": None,
            "trailing_missing": None,
            "coverage_percent": round(100.0 * len(balanced_levels) / len(panel), 2),
            "balanced_level_observations": int(len(balanced_levels)),
            "balanced_difference_observations": int(len(balanced_differences)),
            "usable_after_difference_and_lags": int(usable),
            "usable_edge_only_upper_bound": int(naive_usable),
            "rows_lost_to_interior_gaps": int(naive_usable - usable),
        })
    return rows


# ===========================================================================
# Main build
# ===========================================================================


def build_model_datasets(
    *,
    raw_path: str | Path | None = None,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    build_vintage: str | None = None,
    weekly_commodity_shift_weeks: int = DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS,
    overwrite: bool = False,
    max_tax_carry_months: int = DEFAULT_TAX_CARRY_ERROR_MONTHS,
    hicp_discovery: bool = False,
    require_hicp: bool = False,
    require_headline: bool = False,
) -> dict[str, Path]:
    """Read one Excel snapshot; always build Energy and opportunistically build Headline inputs.

    ``require_headline`` enforces only the current joint BVAR contract:
    Energy/Food/NEIG/Services plus annual four-component weights.

    """
    if weekly_commodity_shift_weeks not in (0, 1):
        raise ValueError("weekly_commodity_shift_weeks must be 0 or 1.")
    if max_tax_carry_months < 0:
        raise ValueError("max_tax_carry_months must be non-negative.")

    raw = Path(raw_path) if raw_path is not None else find_raw_workbook(raw_root)
    if not raw.is_file():
        raise FileNotFoundError(raw)
    if raw.suffix.lower() not in SUPPORTED_WORKBOOK_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_WORKBOOK_SUFFIXES))
        raise ValueError(
            f"The single raw input must be an Excel workbook ({allowed}): {raw}"
        )

    snapshot_vintage = _parse_vintage_name(raw.parent.name)
    build_vintage = build_vintage or (
        snapshot_vintage.strftime("%Y%m%d") if snapshot_vintage else datetime.now().strftime("%Y%m%d")
    )
    output_dir = Path(output_root) / build_vintage
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains a completed build. Pass --overwrite "
            "or choose another --build-vintage."
        )

    frames, read_stats, sheets, metadata_records, hicp_audit = _read_workbook_sources(
        raw,
        hicp_discovery=hicp_discovery,
    )
    wob = frames["european_commission"]
    world_bank = frames["world_bank"]
    bloomberg = frames["bloomberg"]
    haver = frames["haver"]
    eurostat = frames["eurostat"]
    hicp_available = "eurostat_hicp_indices" in frames and "eurostat_hicp_weights" in frames
    hicp_indices = frames.get("eurostat_hicp_indices")
    hicp_weights = frames.get("eurostat_hicp_weights")
    headline_hicp_available = "headline_hicp_indices" in frames and "headline_hicp_weights" in frames
    headline_indices = frames.get("headline_hicp_indices")
    headline_weights = frames.get("headline_hicp_weights")

    # ------------------------------------------------------------------
    # Bloomberg: daily inputs -> EUR units -> weekly/monthly aggregation
    # ------------------------------------------------------------------
    bbg = _commodity_prices_in_euro(bloomberg)

    def _weekly_commodity(series: pd.Series) -> pd.Series:
        weekly = _weekly_mean(series)
        if weekly_commodity_shift_weeks:
            weekly.index = weekly.index + pd.DateOffset(weeks=weekly_commodity_shift_weeks)
            weekly.index.name = "date"
        return weekly

    crude_weekly = _weekly_commodity(bbg["crude_oil_eur"])
    refined_petroleum_weekly = _weekly_commodity(bbg["refined_petroleum_eur"])
    refined_diesel_weekly = _weekly_commodity(bbg["refined_diesel_eur"])

    coverage_raw_series: dict[str, tuple[pd.Series, str]] = {
        "crude_oil_eur": (bbg["crude_oil_eur"], "weekly"),
        "refined_petroleum_eur": (bbg["refined_petroleum_eur"], "weekly"),
        "refined_diesel_eur": (bbg["refined_diesel_eur"], "weekly"),
        "ttf_eur_mwh": (bbg["ttf_eur_mwh"], "monthly"),
        "eurusd": (bbg["eurusd"], "monthly"),
    }

    aggregation_coverage = [
        _aggregation_coverage(
            bbg[logical],
            frequency="weekly",
            label=logical,
            method="Monday-Sunday mean of daily Bloomberg snapshot values",
            label_shift_weeks=int(weekly_commodity_shift_weeks),
        )
        for logical in ("crude_oil_eur", "refined_petroleum_eur", "refined_diesel_eur")
    ]
    aggregation_coverage.extend(
        _aggregation_coverage(
            bbg[logical],
            frequency="monthly",
            label=logical,
            method="calendar-month mean of daily Bloomberg snapshot values",
        )
        for logical in ("ttf_eur_mwh", "eurusd")
    )

    # ------------------------------------------------------------------
    # WOB weekly consumer prices, already supplied as euro-area aggregates
    # ------------------------------------------------------------------
    wob_petrol = _weekly_last(
        _required_series(
            wob, "wob_petroleum_cpr_ntax",
            source="european_commission", logical_name="wob_petrol_pre_tax",
        )
    )
    wob_diesel = _weekly_last(
        _required_series(
            wob, "wob_diesel_cpr_ntax",
            source="european_commission", logical_name="wob_diesel_pre_tax",
        )
    )
    wob_heating = _weekly_last(
        _required_series(
            wob, "wob_gas_cpr_ntax",
            source="european_commission", logical_name="wob_heating_oil_pre_tax",
        )
    )

    _warn_on_partial_periods(aggregation_coverage)

    car_fuels = _regular_panel(
        {
            "wob_petrol_pre_tax": wob_petrol,
            "wob_diesel_pre_tax": wob_diesel,
            "crude_oil_eur": crude_weekly,
            "refined_petroleum_eur": refined_petroleum_weekly,
            "refined_diesel_eur": refined_diesel_weekly,
        },
        frequency="weekly",
        target_columns=MODEL_SPECS["car_fuels"]["target_columns"],
    )
    liquid_fuels = _regular_panel(
        {
            "wob_heating_oil_pre_tax": wob_heating,
            "crude_oil_eur": crude_weekly,
            "refined_diesel_eur": refined_diesel_weekly,
        },
        frequency="weekly",
        target_columns=MODEL_SPECS["liquid_fuels"]["target_columns"],
    )

    # ------------------------------------------------------------------
    # Monthly gas / electricity / heat / solid fuels
    # ------------------------------------------------------------------
    natural_gas, gas_backcast_info = _build_natural_gas_series(bbg, world_bank)

    hicp_gas = _required_series(haver, "hicp_gas", source="haver", logical_name="hicp_gas")
    hicp_electricity = _required_series(
        haver, "hicp_electricity", source="haver", logical_name="hicp_electricity"
    )

    gas_pre_tax, gas_pre_tax_info = _build_pre_tax_price(
        hicp=hicp_gas,
        wtax=_required_series(eurostat, "estat_gas_household_cpr_wtax", source="eurostat", logical_name="gas_wtax"),
        vat=_required_series(eurostat, "estat_gas_household_vat", source="eurostat", logical_name="gas_vat"),
        exc=_required_series(eurostat, "estat_gas_household_exc", source="eurostat", logical_name="gas_exc"),
        name="gas_pre_tax",
        price_unit_multiplier=PRE_TAX_EUR_KWH_TO_EUR_MWH,
        source_price_unit="EUR/kWh",
        output_price_unit="EUR/MWh",
        tax_carry_error_months=max_tax_carry_months,
    )
    electricity_pre_tax, electricity_pre_tax_info = _build_pre_tax_price(
        hicp=hicp_electricity,
        wtax=_required_series(eurostat, "estat_electricity_household_cpr_wtax", source="eurostat", logical_name="electricity_wtax"),
        vat=_required_series(eurostat, "estat_electricity_household_vat", source="eurostat", logical_name="electricity_vat"),
        exc=_required_series(eurostat, "estat_electricity_household_exc", source="eurostat", logical_name="electricity_exc"),
        name="electricity_pre_tax",
        price_unit_multiplier=1.0,
        source_price_unit="EUR/kWh",
        output_price_unit="EUR/kWh",
        tax_carry_error_months=max_tax_carry_months,
    )

    hicp_heat = _monthly_last(
        _required_series(haver, "hicp_heat_energy", source="haver", logical_name="hicp_heat_energy")
    )
    hicp_solid = _monthly_last(
        _required_series(haver, "hicp_solid_fuels", source="haver", logical_name="hicp_solid_fuels")
    )
    ppi_energy = _monthly_last(
        _required_series(haver, "ppi_energy", source="haver", logical_name="ppi_energy")
    )

    gas = _regular_panel(
        {"gas_pre_tax": gas_pre_tax, "natural_gas_eur_mwh": natural_gas},
        frequency="monthly",
        target_columns=MODEL_SPECS["gas"]["target_columns"],
    )
    electricity = _regular_panel(
        {"electricity_pre_tax": electricity_pre_tax, "natural_gas_eur_mwh": natural_gas},
        frequency="monthly",
        target_columns=MODEL_SPECS["electricity"]["target_columns"],
    )
    heat_energy = _regular_panel(
        {"hicp_heat_energy": hicp_heat, "natural_gas_eur_mwh": natural_gas, "ppi_energy": ppi_energy},
        frequency="monthly",
        target_columns=MODEL_SPECS["heat_energy"]["target_columns"],
    )
    solid_fuels = _regular_panel(
        {"hicp_solid_fuels": hicp_solid, "natural_gas_eur_mwh": natural_gas, "ppi_energy": ppi_energy},
        frequency="monthly",
        target_columns=MODEL_SPECS["solid_fuels"]["target_columns"],
    )

    panels = {
        "car_fuels": car_fuels,
        "liquid_fuels": liquid_fuels,
        "gas": gas,
        "electricity": electricity,
        "heat_energy": heat_energy,
        "solid_fuels": solid_fuels,
    }
    for name, panel in panels.items():
        _validate_model_panel(name, panel, MODEL_SPECS[name])

    # ------------------------------------------------------------------
    # Final Headline production input: one joint monthly BVAR only.
    # ------------------------------------------------------------------
    headline_joint_blockers: list[str] = []
    headline_joint_panel: pd.DataFrame | None = None
    headline_joint_weights: pd.DataFrame | None = None
    headline_joint_weight_diagnostics = pd.DataFrame()

    if not headline_hicp_available:
        headline_joint_blockers.append(
            "Eurostat Headline HICP VIEW unavailable or invalid"
        )
    else:
        assert headline_indices is not None and headline_weights is not None
        energy_target = headline_indices["hicp_energy"].rename("hicp_energy")
        food_target = headline_indices["hicp_food"].rename("hicp_food")
        neig_target = headline_indices["hicp_neig"].rename("hicp_neig")
        services_target = headline_indices["hicp_services"].rename("hicp_services")

        if not food_target.notna().any():
            headline_joint_blockers.append(
                "Eurostat FOOD aggregate HICP missing from Eurostat Headline HICP VIEW"
            )
        else:
            headline_joint_panel = _regular_panel(
                {
                    "hicp_energy": energy_target,
                    "hicp_food": food_target,
                    "hicp_neig": neig_target,
                    "hicp_services": services_target,
                },
                frequency="monthly",
                target_columns=HEADLINE_JOINT_SPEC["target_columns"],
            )
            common_start = max(
                headline_joint_panel[column].first_valid_index()
                for column in HEADLINE_JOINT_SPEC["target_columns"]
            )
            headline_joint_panel = headline_joint_panel.loc[common_start:].copy()
            _validate_headline_joint_panel(headline_joint_panel)
            (
                headline_joint_weights,
                headline_joint_weight_diagnostics,
            ) = _build_headline_joint_weights(headline_weights)

    if require_headline and (
        headline_joint_panel is None or headline_joint_weights is None
    ):
        raise ValueError(
            "Joint Headline BVAR inputs are required but incomplete. "
            f"Blockers: {headline_joint_blockers}"
        )

    aggregation_coverage = _bind_coverage_to_panels(aggregation_coverage, panels)

    partial_period_inputs: list[dict[str, object]] = []
    for row in aggregation_coverage:
        if row["final_period_complete"]:
            continue
        raw_series, freq = coverage_raw_series[str(row["series"])]
        panel_column = row["panel_column"]
        partial_period_inputs.extend(
            _final_period_input_records(
                raw_series,
                frequency=freq,
                panel_label_date=str(row["panel_label_date"]),
                panel_column=None if pd.isna(panel_column) else str(panel_column),
                series_label=str(row["series"]),
            )
        )

    # ------------------------------------------------------------------
    # Write processed outputs. Keep model files chronologically ascending;
    # the model layer expects and validates this order.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    diagnostic_rows: list[dict[str, object]] = []

    for name, panel in panels.items():
        path = output_dir / MODEL_SPECS[name]["file"]
        panel.to_csv(path, date_format="%Y-%m-%d")
        outputs[name] = path
        diagnostic_rows.extend(_diagnostic_rows(name, panel, MODEL_SPECS[name]))

    if headline_joint_panel is not None:
        path = output_dir / HEADLINE_JOINT_SPEC["file"]
        headline_joint_panel.to_csv(path, date_format="%Y-%m-%d")
        outputs["headline_joint"] = path
        diagnostic_rows.extend(
            _diagnostic_rows(
                "headline_joint",
                headline_joint_panel,
                HEADLINE_JOINT_SPEC,
            )
        )


    diagnostics_path = output_dir / "model_datasets_diagnostics.csv"
    sources_path = output_dir / "source_vintages.csv"  # kept for compatibility
    coverage_path = output_dir / "aggregation_coverage.csv"
    partial_inputs_path = output_dir / "partial_period_inputs.csv"

    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_path, index=False)
    pd.DataFrame(aggregation_coverage).to_csv(coverage_path, index=False)
    partial_inputs_frame = pd.DataFrame(
        partial_period_inputs,
        columns=["series", "panel_column", "panel_label_date", "frequency", "input_date", "input_value"],
    )
    partial_inputs_frame.to_csv(partial_inputs_path, index=False)

    # Headline audit sidecars. They are written whenever their source exists,
    # independently of whether all four model panels are ready.
    headline_indices_path = output_dir / "headline_hicp_indices_monthly.csv"
    headline_weights_path = output_dir / "headline_hicp_weights_annual.csv"
    headline_weight_identity_path = output_dir / "headline_hicp_weight_identity_diagnostics.csv"
    headline_joint_weights_path = output_dir / "headline_joint_weights_annual.csv"
    headline_joint_weight_identity_path = output_dir / "headline_joint_weight_identity_diagnostics.csv"

    if headline_hicp_available:
        assert headline_indices is not None and headline_weights is not None
        headline_indices.to_csv(headline_indices_path, date_format="%Y-%m-%d")
        headline_weights.to_csv(headline_weights_path, index_label="year")
        audit_obj = frames.get("headline_hicp_audit")
        if isinstance(audit_obj, dict):
            audit_obj["weight_identities"].to_csv(headline_weight_identity_path, index=False)
    if headline_joint_weights is not None:
        headline_joint_weights.to_csv(
            headline_joint_weights_path,
            index_label="year",
        )
        headline_joint_weight_diagnostics.to_csv(
            headline_joint_weight_identity_path,
            index=False,
        )

    # ------------------------------------------------------------------
    # Optional HICP aggregation lineage. It never changes the six BVAR panels.
    # Diagnostics are written before any --require-hicp failure is raised.
    # ------------------------------------------------------------------
    hicp_indices_path = output_dir / "hicp_indices_monthly.csv"
    hicp_weights_path = output_dir / "hicp_weights_annual.csv"
    hicp_series_metadata_path = output_dir / "hicp_series_metadata.csv"
    hicp_flags_path = output_dir / "hicp_flags.csv"
    hicp_weight_identities_path = output_dir / "hicp_weight_identity_diagnostics.csv"
    hicp_validation_failures_path = output_dir / "hicp_validation_failures.csv"
    wob_hicp_mapping_path = output_dir / "wob_hicp_mapping_diagnostics.csv"

    wob_hicp_mapping = pd.DataFrame()
    hicp_empirical_failures = pd.DataFrame()
    hicp_error = None

    if hicp_available:
        assert hicp_indices is not None and hicp_weights is not None
        assert hicp_audit is not None and hicp_audit.get("status") == "parsed"
        hicp_indices.to_csv(hicp_indices_path, date_format="%Y-%m-%d")
        hicp_weights.to_csv(hicp_weights_path, index_label="year")
        hicp_audit["series_metadata"].to_csv(hicp_series_metadata_path, index=False)
        hicp_audit["flags"].to_csv(hicp_flags_path, index=False, date_format="%Y-%m-%d")
        hicp_audit["weight_identities"].to_csv(hicp_weight_identities_path, index=False)
        hicp_empirical_failures = hicp_audit["validation_failures"].copy()
        hicp_empirical_failures.to_csv(hicp_validation_failures_path, index=False)

        wob_hicp_mapping = _wob_hicp_mapping_diagnostics(wob, hicp_indices)
        wob_hicp_mapping.to_csv(wob_hicp_mapping_path, index=False)
    else:
        hicp_error = None if hicp_audit is None else hicp_audit.get("error")

    mapping_failures = (
        wob_hicp_mapping.loc[wob_hicp_mapping["status"] != "pass"]
        if not wob_hicp_mapping.empty else pd.DataFrame()
    )
    strict_hicp_messages = []
    if hicp_error:
        strict_hicp_messages.append(str(hicp_error))
    if not hicp_empirical_failures.empty:
        strict_hicp_messages.extend(hicp_empirical_failures["message"].astype(str).tolist())
    if not mapping_failures.empty:
        for row in mapping_failures.itertuples(index=False):
            strict_hicp_messages.append(
                f"WOB/HICP mapping {row.wob_series}: own correlation "
                f"{row.own_log_change_correlation:.4f} <= cross correlation "
                f"{row.cross_log_change_correlation:.4f}"
            )

    if require_hicp and not hicp_available:
        raise ValueError(
            "HICP aggregation lineage is required but unavailable. "
            f"Reason: {hicp_error or 'Eurostat_HICP sheet not found.'}"
        )
    if require_hicp and strict_hicp_messages and not hicp_discovery:
        preview = "; ".join(strict_hicp_messages[:12])
        extra = f" (+{len(strict_hicp_messages)-12} more)" if len(strict_hicp_messages) > 12 else ""
        raise ValueError(
            "Eurostat HICP empirical validation failed after diagnostics were written. "
            f"{preview}{extra}. Use --hicp-discovery to calibrate a new snapshot contract."
        )
    workbook_hash = _sha256(raw)
    snapshot_label = raw.parent.name if snapshot_vintage else "unversioned_workbook"

    # Persist the exact source context required later by tax re-attribution and
    # fitted/scenario materialisation.  These are parsed, numeric, dated copies
    # of the workbook state used for THIS processed vintage; the living workbook
    # may subsequently be refreshed without altering historical provenance.
    source_context_paths: dict[str, Path] = {}
    for source, filename in SOURCE_CONTEXT_FILES.items():
        frame = frames[source].copy().sort_index()
        path = output_dir / filename
        frame.to_csv(path, index_label="date", date_format="%Y-%m-%d")
        source_context_paths[source] = path

    # Explicit overwrite is opt-in.  Once the new production contract has been
    # successfully materialised, remove only outputs owned by the retired
    # Headline decomposition branch; never touch arbitrary user files.
    removed_deprecated_outputs: list[str] = []
    if overwrite:
        for filename in DEPRECATED_HEADLINE_DECOMPOSITION_OUTPUTS:
            stale = output_dir / filename
            if stale.is_file():
                stale.unlink()
                removed_deprecated_outputs.append(filename)

    source_table = pd.DataFrame([
        {
            "source": source,
            "sheet": sheets[source],
            "snapshot_vintage": snapshot_label,
            "raw_workbook": str(raw.resolve()),
            "raw_workbook_sha256": workbook_hash,
            "input_rows": read_stats[source]["retained_rows"],
            "input_columns": read_stats[source]["columns"],
            "first_observation": read_stats[source]["first_date"],
            "last_observation": read_stats[source]["last_date"],
            "first_weight_year": read_stats[source].get("first_weight_year"),
            "last_weight_year": read_stats[source].get("last_weight_year"),
            "geo_contract": read_stats[source].get("geo"),
            "statinfo_contract": read_stats[source].get("weight_statinfo"),
            "unit_contract": read_stats[source].get("index_unit"),
            "input_order_in_excel": read_stats[source]["input_order"],
            "invalid_date_rows": read_stats[source]["invalid_date_rows"],
            "duplicate_date_rows": read_stats[source]["duplicate_date_rows"],
            "all_missing_rows_dropped": read_stats[source]["all_missing_rows_dropped"],
            "error_markers_coerced_to_nan": read_stats[source]["error_markers_coerced_to_nan"],
        }
        for source in (
            "haver",
            "bloomberg",
            "european_commission",
            "world_bank",
            "eurostat",
            *(("eurostat_hicp",) if hicp_available else ()),
            *(("headline_hicp",) if headline_hicp_available else ()),
        )
    ])
    source_table.to_csv(sources_path, index=False)

    auxiliary_outputs = {
        "diagnostics": str(diagnostics_path.resolve()),
        "source_vintages": str(sources_path.resolve()),
        "aggregation_coverage": str(coverage_path.resolve()),
        "partial_period_inputs": str(partial_inputs_path.resolve()),
        **{
            f"source_context_{source}": str(path.resolve())
            for source, path in source_context_paths.items()
        },
    }

    if hicp_available:
        auxiliary_outputs.update({
            "hicp_indices_monthly": str(hicp_indices_path.resolve()),
            "hicp_weights_annual": str(hicp_weights_path.resolve()),
            "hicp_series_metadata": str(hicp_series_metadata_path.resolve()),
            "hicp_flags": str(hicp_flags_path.resolve()),
            "hicp_weight_identity_diagnostics": str(hicp_weight_identities_path.resolve()),
            "hicp_validation_failures": str(hicp_validation_failures_path.resolve()),
            "wob_hicp_mapping_diagnostics": str(wob_hicp_mapping_path.resolve()),
        })
        hicp_manifest_section = {
            "status": "discovery" if hicp_discovery else "available",
            "source_sheet": sheets["eurostat_hicp"],
            "index_dataset": EUROSTAT_HICP_INDEX_DATASET,
            "weight_dataset": EUROSTAT_HICP_WEIGHT_DATASET,
            "geo": EUROSTAT_HICP_GEO,
            "index_unit": EUROSTAT_HICP_INDEX_UNIT,
            "weight_statinfo": hicp_audit["weight_statinfo"],
            "weight_unit": "per thousand of total HICP; raw published weights, not renormalized",
            "latest_weight_year": int(hicp_audit["latest_weight_year"]),
            "weight_identity_tolerance_per_thousand": EUROSTAT_HICP_WEIGHT_TOLERANCE_PER_THOUSAND,
            "empirical_contract_failures": int(len(hicp_empirical_failures)),
            "wob_hicp_mapping_pass": bool(wob_hicp_mapping["status"].eq("pass").all()),
            "known_break_contract": {
                series: [stamp.date().isoformat() for stamp in sorted(dates)]
                for series, dates in EUROSTAT_HICP_EXPECTED_BREAKS.items()
            },
            "future_weight_policy_for_aggregation_module": (
                "If a forecast crosses into a calendar year whose weights are not yet published, "
                "carry forward the latest published annual weights."
            ),
            "refresh_timestamp_evidence": {
                "status": "unavailable_in_current_metadata_sheet",
                "note": (
                    "The current Metadata sheet contains refresh methods but no per-query UTC timestamps. "
                    "A common workbook hash proves a common saved file, not identical Power Query refresh times."
                ),
            },
            "files": {
                key: value for key, value in auxiliary_outputs.items() if key.startswith("hicp_") or key == "wob_hicp_mapping_diagnostics"
            },
        }
    else:
        hicp_manifest_section = None if not hicp_error else {
            "status": "error",
            "error": str(hicp_error),
            "source_sheet": sheets.get("eurostat_hicp"),
        }

    headline_auxiliary_outputs: dict[str, str] = {}
    for key, path in (
        ("headline_hicp_indices_monthly", headline_indices_path),
        ("headline_hicp_weights_annual", headline_weights_path),
        ("headline_hicp_weight_identity_diagnostics", headline_weight_identity_path),
        ("headline_joint_weights_annual", headline_joint_weights_path),
        ("headline_joint_weight_identity_diagnostics", headline_joint_weight_identity_path),
    ):
        if path.exists():
            headline_auxiliary_outputs[key] = str(path.resolve())
    auxiliary_outputs.update(headline_auxiliary_outputs)

    joint_ready = (
        headline_joint_panel is not None
        and headline_joint_weights is not None
    )
    headline_manifest_section = {
        "production_model": "headline_joint",
        "status": "complete" if joint_ready else "unavailable",
        "source_sheet": sheets.get("headline_hicp"),
        "joint_model": {
            "ready": bool(joint_ready),
            "spec": HEADLINE_JOINT_SPEC,
            "food_index_source": {
                "provider": "Eurostat",
                "dataset": EUROSTAT_HICP_INDEX_DATASET,
                "code": "FOOD",
                "unit": EUROSTAT_HICP_INDEX_UNIT,
                "geo": EUROSTAT_HICP_GEO,
                "series": "hicp_food",
            },
            "food_weight_construction": (
                "Eurostat FOOD_NP + FOOD_P annual item weights"
            ),
            "blockers": headline_joint_blockers,
            "path": (
                str(outputs["headline_joint"].resolve())
                if "headline_joint" in outputs
                else None
            ),
            "annual_weights_path": (
                str(headline_joint_weights_path.resolve())
                if headline_joint_weights_path.exists()
                else None
            ),
        },
        "headline_hicp_start_policy": (
            "No forced common start. The joint BVAR model layer determines "
            "the first jointly usable transformed observation."
        ),
        "files": headline_auxiliary_outputs,
    }

    manifest = {
        "project": "ECB energy STIP six-model dataset build + final joint Headline-HICP BVAR input",
        "script_version": SCRIPT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "build_vintage": build_vintage,
        "input_architecture": "single_excel_workbook_xlsm_or_xlsx",
        "raw_workbook": str(raw.resolve()),
        "raw_workbook_sha256": workbook_hash,
        "raw_workbook_bytes": raw.stat().st_size,
        "snapshot_vintage": snapshot_label,
        "sheet_mapping": sheets,
        "source_files": {
            source: path.name
            for source, path in source_context_paths.items()
        },
        "source_context_contract": {
            "mode": "immutable_processed_vintage_sidecars",
            "sources": sorted(source_context_paths),
            "note": (
                "Tax/HICP/fitted/scenario adapters must prefer these files over "
                "the living Excel workbook. Legacy vintages without sidecars may "
                "use the workbook only when its SHA256 matches the manifest."
            ),
        },
        "deprecated_outputs_removed_on_overwrite": removed_deprecated_outputs,
        "ignored_live_sheet": "Bloomberg_Live",
        "workbook_requirement": (
            "Production workbook may be .xlsm or .xlsx; .xlsm is preferred because "
            "it preserves Config/VBA. Python never executes VBA, Power Query, Haver "
            "or Bloomberg formulas. Refresh and save the workbook in Excel first. "
            "Bloomberg_copy is the values-only snapshot consumed by Python; "
            "Bloomberg_Live is intentionally ignored."
        ),
        "workbook_freshness_contract": {
            "haver": (
                "Refresh the Haver block in Excel and save before the build. "
                "Only H023HW51, H023HW52, H023HW54, H023HW55 and H025PP@G10 "
                "are hard Energy requirements; total/car-fuels/liquid-fuels Haver "
                "series are optional because the Eurostat HICP lineage owns them."
            ),
            "bloomberg": (
                "Refresh Bloomberg_Live, copy values/number formats to Bloomberg_copy, "
                "then save before the build."
            ),
            "power_query": (
                "Apply Config and refresh Power Query, wait for completion, then save. "
                "Python reads cached worksheet values and does not execute queries."
            ),
        },
        "source_read_statistics": read_stats,
        "metadata_sheet_records": metadata_records,
        "hicp_aggregation_inputs": hicp_manifest_section,
        "headline_inputs": headline_manifest_section,
        "bloomberg_mapping": BLOOMBERG_TICKERS,
        "bloomberg_units": BLOOMBERG_UNITS,
        "refined_petroleum_policy": (
            "GNEBM1 PVMO is converted only from USD/metric tonne to EUR/metric "
            "tonne using EURUSD. No tonne-to-barrel factor is applied."
        ),
        "known_approximations_vs_paper": [
            "World Bank Natural gas, Europe substitutes for the paid Energy Intelligence European border-gas series; the pre-TTF history is chain-linked to TTF.",
            "GNEBM1 PVMO (Eurobob oxygenated gasoline FOB Rotterdam barges) is used for refined petroleum; its EUR/metric-tonne unit is retained rather than inventing a gasoline tonne-to-barrel conversion.",
            "QS1 is a Low Sulphur Gas Oil future rather than the paper's Refinitiv spot assessment; 7.44 barrels/tonne is applied only to this gasoil/diesel series.",
            "The available WOB workbook starts in 2005, later than the paper's full historical sample.",
            "Eurostat household bands are the gas GJ20-199/D2 and electricity KWH2500-4999/DC contracts selected in the workbook Power Query.",
        ],
        "construction": {
            "excel_date_order": (
                "Source sheets may be newest-first. Every source is sorted "
                "ascending in memory before differencing, lags or aggregation."
            ),
            "weekly_calendar": {
                "week_definition": "Monday-Sunday, labelled by Monday",
                "commodity_shift_weeks": int(weekly_commodity_shift_weeks),
                "alignment": (
                    "shift=1 aligns WOB Monday t with the preceding completed "
                    "Monday-Sunday commodity week; shift=0 uses the same week."
                ),
            },
            "commodity_units": {
                "crude_oil_eur": "EUR/barrel",
                "refined_petroleum_eur": "EUR/metric tonne",
                "refined_diesel_eur": "EUR/barrel",
            },
            "pre_tax": {
                "gas": gas_pre_tax_info,
                "electricity": electricity_pre_tax_info,
            },
            "natural_gas": gas_backcast_info,
            "transformations": (
                "Outputs contain levels. Apply ordinary absolute differences "
                "in the BVAR model layer."
            ),
            "ragged_edge": (
                "Regular calendars are retained with NaN for unreleased cells. "
                "The builder never drops the ragged edge merely because one series is missing."
            ),
            "partial_final_period": (
                "aggregation_coverage.csv records the final weekly/monthly "
                "Bloomberg aggregation coverage; partial_period_inputs.csv "
                "stores the raw daily observations backing incomplete periods."
            ),
            "aggregation_coverage": aggregation_coverage,
        },
        "models": {
            name: {
                **spec,
                "path": str(outputs[name].resolve()),
                "rows": int(len(panels[name])),
                "first_date": panels[name].index.min().date().isoformat(),
                "last_date": panels[name].index.max().date().isoformat(),
            }
            for name, spec in MODEL_SPECS.items()
        },
        "headline_joint_model": (
            None
            if headline_joint_panel is None
            else {
                **HEADLINE_JOINT_SPEC,
                "path": str(outputs["headline_joint"].resolve()),
                "rows": int(len(headline_joint_panel)),
                "first_date": headline_joint_panel.index.min().date().isoformat(),
                "last_date": headline_joint_panel.index.max().date().isoformat(),
                "annual_weights_path": (
                    str(headline_joint_weights_path.resolve())
                    if headline_joint_weights_path.exists()
                    else None
                ),
            }
        ),
        "auxiliary_outputs": auxiliary_outputs,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Hash after every file exists.
    manifest["output_sha256"] = {
        **{name: _sha256(path) for name, path in outputs.items()},
        "diagnostics": _sha256(diagnostics_path),
        "source_vintages": _sha256(sources_path),
        "aggregation_coverage": _sha256(coverage_path),
        "partial_period_inputs": _sha256(partial_inputs_path),
        **{
            f"source_context_{source}": _sha256(path)
            for source, path in source_context_paths.items()
        },
    }
    if hicp_available:
        manifest["output_sha256"].update({
            "hicp_indices_monthly": _sha256(hicp_indices_path),
            "hicp_weights_annual": _sha256(hicp_weights_path),
            "hicp_series_metadata": _sha256(hicp_series_metadata_path),
            "hicp_flags": _sha256(hicp_flags_path),
            "hicp_weight_identity_diagnostics": _sha256(hicp_weight_identities_path),
            "hicp_validation_failures": _sha256(hicp_validation_failures_path),
            "wob_hicp_mapping_diagnostics": _sha256(wob_hicp_mapping_path),
        })
    for key, path_text in headline_auxiliary_outputs.items():
        manifest["output_sha256"][key] = _sha256(Path(path_text))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    outputs.update({
        "diagnostics": diagnostics_path,
        "source_vintages": sources_path,
        "aggregation_coverage": coverage_path,
        "partial_period_inputs": partial_inputs_path,
        "manifest": manifest_path,
        **{
            f"source_context_{source}": path
            for source, path in source_context_paths.items()
        },
    })
    if hicp_available:
        outputs.update({
            "hicp_indices_monthly": hicp_indices_path,
            "hicp_weights_annual": hicp_weights_path,
            "hicp_series_metadata": hicp_series_metadata_path,
            "hicp_flags": hicp_flags_path,
            "hicp_weight_identity_diagnostics": hicp_weight_identities_path,
            "hicp_validation_failures": hicp_validation_failures_path,
            "wob_hicp_mapping_diagnostics": wob_hicp_mapping_path,
        })

    for key, path_text in headline_auxiliary_outputs.items():
        outputs[key] = Path(path_text)

    print(f"Builder:       {SCRIPT_VERSION}")
    print(f"Raw workbook:  {raw}")
    print(f"Build vintage: {build_vintage}")
    print("Source sheets:")
    for source in (
        "haver",
        "bloomberg",
        "european_commission",
        "world_bank",
        "eurostat",
        *(("eurostat_hicp",) if hicp_available else ()),
        *(("headline_hicp",) if headline_hicp_available else ()),
    ):
        stat = read_stats[source]
        suffix = (
            f"; weights {stat['first_weight_year']} -> {stat['last_weight_year']}"
            if source in {"eurostat_hicp", "headline_hicp"}
            and stat.get("first_weight_year") is not None
            else ""
        )
        print(
            f"  {source:22s} {sheets[source]:22s} "
            f"{stat['first_date']} -> {stat['last_date']} "
            f"({stat['retained_rows']:,} rows; Excel order {stat['input_order']}{suffix})"
        )

    partial = [row for row in aggregation_coverage if not row["final_period_complete"]]
    if partial:
        print("\nPartial final Bloomberg aggregation periods:")
        for row in partial:
            print(
                f"  {row['series']:24s} {row['panel_label_date']}  "
                f"{row['final_period_input_observations']}/"
                f"{row['expected_input_observations']} inputs "
                f"({row['coverage_ratio']:.0%})"
            )

    print("\nProcessed model datasets:")
    for name, panel in panels.items():
        spec = MODEL_SPECS[name]
        print(
            f"  {spec['file']:30s} {panel.index.min().date()} -> "
            f"{panel.index.max().date()} | {len(panel):,} rows | {panel.shape[1]} series"
        )
    if hicp_available:
        assert hicp_indices is not None and hicp_weights is not None
        print("\nHICP aggregation inputs:")
        print(
            f"  {hicp_indices_path.name:30s} {hicp_indices.index.min().date()} -> "
            f"{hicp_indices.index.max().date()} | {len(hicp_indices):,} months | {hicp_indices.shape[1]} series"
        )
        print(
            f"  {hicp_weights_path.name:30s} {int(hicp_weights.index.min())} -> "
            f"{int(hicp_weights.index.max())} | {len(hicp_weights):,} years | {hicp_weights.shape[1]} series"
        )
        print(
            f"  mode={'DISCOVERY' if hicp_discovery else 'NORMAL'} | "
            f"WOB mapping={'PASS' if wob_hicp_mapping['status'].eq('pass').all() else 'FAIL'}"
        )
    else:
        print("\nHICP aggregation inputs: unavailable; six BVAR datasets remain valid and independent.")

    print("\nHeadline joint production input:")
    if headline_joint_panel is not None:
        print(
            f"  {HEADLINE_JOINT_SPEC['file']:30s} "
            f"{headline_joint_panel.index.min().date()} -> "
            f"{headline_joint_panel.index.max().date()} | "
            f"{len(headline_joint_panel):,} rows | "
            f"{headline_joint_panel.shape[1]} series"
        )
        if headline_joint_weights is not None:
            complete_weights = headline_joint_weights.dropna(how="any")
            if not complete_weights.empty:
                print(
                    f"  {headline_joint_weights_path.name:30s} "
                    f"{int(complete_weights.index.min())} -> "
                    f"{int(complete_weights.index.max())} | "
                    f"{len(complete_weights):,} complete years"
                )
    else:
        print("  not ready")
    if headline_joint_blockers:
        print("  blockers:")
        for blocker in headline_joint_blockers:
            print(f"    - {blocker}")


    print(f"\nOutput directory: {output_dir}")
    return outputs


# Backward-compatible call used elsewhere in the project.
def build_dataset(**kwargs) -> dict[str, Path]:
    return build_model_datasets(**kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help=("Path to the single production workbook (.xlsm preferred; .xlsx supported). "
              "Default: auto-detect bvar_energy_raw_data*/raw_energy_bvar* below --raw-root."),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=(f"Folder searched for bvar_energy_raw_data*/raw_energy_bvar* .xlsm/.xlsx "
              f"(default: {DEFAULT_RAW_ROOT})."),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Processed output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--build-vintage",
        default=None,
        help="Output folder name. Default: raw parent vintage if dated, otherwise today YYYYMMDD.",
    )
    parser.add_argument(
        "--weekly-commodity-shift-weeks",
        type=int,
        choices=(0, 1),
        default=DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS,
        help="1 (default) aligns WOB Monday t with the preceding completed commodity week.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-tax-carry-months",
        type=int,
        default=DEFAULT_TAX_CARRY_ERROR_MONTHS,
        help="Maximum months taxes may be carried beyond the final six-month Eurostat semester.",
    )
    parser.add_argument(
        "--hicp-discovery",
        action="store_true",
        help=(
            "Report empirical HICP starts/breaks/weight identities without "
            "blocking the HICP sidecar build. Structural contracts remain hard."
        ),
    )
    parser.add_argument(
        "--require-hicp",
        action="store_true",
        help=(
            "Require a valid Eurostat_HICP lineage. Without this flag, HICP "
            "absence/errors never block the six BVAR datasets."
        ),
    )
    parser.add_argument(
        "--require-headline",
        action="store_true",
        help=(
            "Require the current joint Headline BVAR input: HICP Energy/Food/"
            "NEIG/Services plus complete annual four-component weights."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    build_model_datasets(
        raw_path=args.raw,
        raw_root=args.raw_root,
        output_root=args.output_root,
        build_vintage=args.build_vintage,
        weekly_commodity_shift_weeks=args.weekly_commodity_shift_weeks,
        overwrite=args.overwrite,
        max_tax_carry_months=args.max_tax_carry_months,
        hicp_discovery=args.hicp_discovery,
        require_hicp=args.require_hicp,
        require_headline=args.require_headline,
    )


if __name__ == "__main__":
    main()
