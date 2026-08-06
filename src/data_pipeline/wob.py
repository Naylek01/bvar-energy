"""Build the Weekly Oil Bulletin dataset used by the energy BVAR.

The raw workbook is expected at, for example::

    data/raw/european_commission/2026-08-04/weekly_oil_bulletin.xlsx

The public workbook starts in 2005. It directly provides euro-area prices with
and without taxes. VAT rates and indirect taxes are reconstructed from the
country sheets using annual product-consumption weights and the changing euro-
area composition.
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
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "european_commission"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "european_commission"
SCRIPT_VERSION = "2026-08-04-run-ready-v2"


PRODUCTS = {
    "petroleum": "euro95",
    "diesel": "diesel",
    "gas": "heating_oil",
}

OUTPUT_COLUMNS = [
    "wob_petroleum_cpr_wtax",
    "wob_petroleum_cpr_ntax",
    "wob_petroleum_idt",
    "wob_petroleum_vat",
    "wob_diesel_cpr_wtax",
    "wob_diesel_cpr_ntax",
    "wob_diesel_idt",
    "wob_diesel_vat",
    "wob_gas_cpr_wtax",
    "wob_gas_cpr_ntax",
    "wob_gas_idt",
    "wob_gas_vat",
]

# Entry dates reproduce the changing composition of the euro area.
EA_ENTRY = {
    "AT": "1999-01-01",
    "BE": "1999-01-01",
    "DE": "1999-01-01",
    "ES": "1999-01-01",
    "FI": "1999-01-01",
    "FR": "1999-01-01",
    "IE": "1999-01-01",
    "IT": "1999-01-01",
    "LU": "1999-01-01",
    "NL": "1999-01-01",
    "PT": "1999-01-01",
    "GR": "2001-01-01",
    "SI": "2007-01-01",
    "CY": "2008-01-01",
    "MT": "2008-01-01",
    "SK": "2009-01-01",
    "EE": "2011-01-01",
    "LV": "2014-01-01",
    "LT": "2015-01-01",
    "HR": "2023-01-01",
}
EA_ENTRY = {country: pd.Timestamp(date) for country, date in EA_ENTRY.items()}

TAX_COLUMNS = [
    "country",
    "since",
    "euro95",
    "diesel",
    "heating_oil",
    "fuel_oil_1",
    "fuel_oil_2",
    "lpg",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_latest_raw_file(
    raw_root: str | Path,
    filename: str | None = None,
) -> Path:
    """Return the historical WOB workbook in the latest vintage directory."""
    root = Path(raw_root)
    candidates = []

    paths = root.glob(f"*/{filename}") if filename else root.glob("*/*.xlsx")
    for path in paths:
        try:
            vintage = pd.Timestamp(path.parent.name)
        except ValueError:
            continue

        name = path.name.lower().replace("-", "_").replace(" ", "_")
        if filename is None and not all(token in name for token in ("weekly", "oil", "bulletin")):
            continue

        # Prefer the historical workbook if several WOB files share a vintage.
        priority = 1 if "history" in name else 0
        candidates.append((vintage, priority, path))

    if not candidates:
        expected = filename or "a Weekly Oil Bulletin historical workbook"
        raise FileNotFoundError(f"No {expected} found below {root}.")

    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _read_price_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    columns = raw.iloc[0].tolist()
    data = raw.iloc[3:].copy()
    data.columns = columns
    data = data.rename(columns={data.columns[0]: "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[data["date"].notna()].copy()

    keep = ["date"] + [
        column
        for column in data.columns
        if isinstance(column, str)
        and (
            column.startswith("EUR_price_")
            or column.endswith("_exchange_rate")
        )
    ]

    data = data.loc[:, keep]
    for column in keep[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return (
        data.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )


def _read_tax_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    data = raw.iloc[4:, :8].copy()
    data.columns = TAX_COLUMNS

    data["country"] = (
        data["country"]
        .ffill()
        .astype(str)
        .str.strip()
        .str.rstrip("_")
    )
    data["since"] = pd.to_datetime(data["since"], errors="coerce")
    data = data[
        data["since"].notna()
        & data["country"].str.fullmatch(r"[A-Z]{2}")
    ].copy()

    value_columns = TAX_COLUMNS[2:]
    data[value_columns] = data[value_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # A blank cell on a new effective-date row means that this product did not
    # change on that date. Carry the last published value forward.
    data = data.sort_values(["country", "since"])
    data[value_columns] = data.groupby("country", sort=False)[
        value_columns
    ].ffill()

    return data


def _read_consumption(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Consumption", header=None)
    columns = [str(value) if pd.notna(value) else "" for value in raw.iloc[0]]

    # The source workbook contains labels such as heATing/heBEing. Normalize
    # all of them to the intended ``heating_oil`` suffix.
    columns = [
        re.sub(
            r"_consumption_he[A-Z]{2}ing_oil$",
            "_consumption_heating_oil",
            column,
        )
        for column in columns
    ]

    data = raw.iloc[2:].copy()
    data.columns = columns
    data = data.rename(columns={columns[0]: "year"})
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data[data["year"].notna()].copy()
    data["year"] = data["year"].astype(int)

    records = []
    pattern = re.compile(
        r"^([A-Z]{2})_consumption_(euro95|diesel|heating_oil)$"
    )

    for column in data.columns:
        match = pattern.fullmatch(column)
        if not match:
            continue

        country, product = match.groups()
        block = data[["year", column]].rename(columns={column: "consumption"})
        block["country"] = country
        block["product"] = product
        records.append(block)

    consumption = pd.concat(records, ignore_index=True)
    consumption["consumption"] = pd.to_numeric(
        consumption["consumption"],
        errors="coerce",
    )
    return consumption.sort_values(["country", "product", "year"])


def _expand_effective_dates(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    parts = []
    date_frame = pd.DataFrame({"date": dates.sort_values()})

    for country, group in events.groupby("country", sort=False):
        right = group.rename(columns={"since": "date"}).sort_values("date")
        expanded = pd.merge_asof(
            date_frame,
            right,
            on="date",
            direction="backward",
        )
        expanded["country"] = country
        parts.append(expanded)

    return pd.concat(parts, ignore_index=True)


def _to_product_long(data: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return data.melt(
        id_vars=["date", "country"],
        value_vars=list(PRODUCTS.values()),
        var_name="product",
        value_name=value_name,
    )


def _build_country_panel(
    dates: pd.DatetimeIndex,
    prices_with_tax: pd.DataFrame,
    consumption: pd.DataFrame,
) -> pd.DataFrame:
    countries = sorted(EA_ENTRY)
    index = pd.MultiIndex.from_product(
        [dates.sort_values(), countries],
        names=["date", "country"],
    )
    panel = index.to_frame(index=False)
    panel = panel[
        panel.apply(
            lambda row: row["date"] >= EA_ENTRY[row["country"]],
            axis=1,
        )
    ].copy()

    exchange_parts = []
    for country in countries:
        column = f"{country}_exchange_rate"
        if column in prices_with_tax.columns:
            exchange = prices_with_tax[column]
        else:
            exchange = pd.Series(1.0, index=prices_with_tax.index)

        block = exchange.rename("eur_per_currency").reset_index()
        block["country"] = country
        exchange_parts.append(block)

    exchange_rates = pd.concat(exchange_parts, ignore_index=True)
    panel = panel.merge(
        exchange_rates,
        on=["date", "country"],
        how="left",
        validate="one_to_one",
    )
    panel["eur_per_currency"] = pd.to_numeric(
        panel["eur_per_currency"],
        errors="coerce",
    ).fillna(1.0)

    products = pd.DataFrame({"product": list(PRODUCTS.values()), "key": 1})
    panel["key"] = 1
    panel = panel.merge(products, on="key").drop(columns="key")
    panel["year"] = panel["date"].dt.year.astype("int64")

    consumption_parts = []
    for (country, product), group in consumption.groupby(
        ["country", "product"],
        sort=False,
    ):
        left = panel[
            (panel["country"] == country)
            & (panel["product"] == product)
        ][["date", "country", "product", "year"]].sort_values("year")

        right = group[["year", "consumption"]].sort_values("year").copy()
        right["year"] = right["year"].astype("int64")

        block = pd.merge_asof(
            left,
            right,
            on="year",
            direction="backward",
        )
        consumption_parts.append(block)

    consumption_weekly = pd.concat(consumption_parts, ignore_index=True)
    return panel.merge(
        consumption_weekly,
        on=["date", "country", "product", "year"],
        how="left",
        validate="one_to_one",
    )


def _weighted_summary(group: pd.DataFrame, value_column: str) -> pd.Series:
    total = group["consumption"].notna() & (group["consumption"] > 0)
    valid = total & group[value_column].notna()

    total_weight = group.loc[total, "consumption"].sum()
    used_weight = group.loc[valid, "consumption"].sum()

    value = np.nan
    if used_weight > 0:
        value = np.average(
            group.loc[valid, value_column],
            weights=group.loc[valid, "consumption"],
        )

    return pd.Series(
        {
            "value": value,
            "coverage": used_weight / total_weight if total_weight > 0 else np.nan,
            "countries": int(valid.sum()),
        }
    )


def _aggregate_weighted(
    panel: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    records = []
    for (date, product), group in panel.groupby(["date", "product"], sort=True):
        summary = _weighted_summary(group, value_column)
        records.append({"date": date, "product": product, **summary.to_dict()})
    return pd.DataFrame(records)


def build_wob_dataset(
    raw_path: str | Path,
    minimum_coverage: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the twelve weekly WOB series and a diagnostics table.

    Returns
    -------
    data
        Weekly euro-area prices, VAT rates and indirect taxes.
    diagnostics
        Coverage and the comparison between direct national-tax aggregation
        and the accounting-identity estimate of indirect taxes.
    """
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(path)

    prices_with_tax = _read_price_sheet(path, "Prices with taxes")
    prices_without_tax = _read_price_sheet(path, "Prices wo taxes")

    if not prices_with_tax.index.equals(prices_without_tax.index):
        common_dates = prices_with_tax.index.intersection(prices_without_tax.index)
        prices_with_tax = prices_with_tax.loc[common_dates]
        prices_without_tax = prices_without_tax.loc[common_dates]

    dates = prices_with_tax.index
    consumption = _read_consumption(path)
    panel = _build_country_panel(dates, prices_with_tax, consumption)

    vat = _expand_effective_dates(_read_tax_sheet(path, "VAT"), dates)
    excise = _expand_effective_dates(
        _read_tax_sheet(path, "Excise duties"),
        dates,
    )
    other = _expand_effective_dates(
        _read_tax_sheet(path, "Other Indirect Taxes"),
        dates,
    )

    taxes = (
        _to_product_long(vat, "vat_pct")
        .merge(
            _to_product_long(excise, "excise_local"),
            on=["date", "country", "product"],
            how="outer",
        )
        .merge(
            _to_product_long(other, "other_tax_local"),
            on=["date", "country", "product"],
            how="outer",
        )
    )

    panel = panel.merge(
        taxes,
        on=["date", "country", "product"],
        how="left",
        validate="one_to_one",
    )

    numeric = [
        "consumption",
        "eur_per_currency",
        "vat_pct",
        "excise_local",
        "other_tax_local",
    ]
    panel[numeric] = panel[numeric].apply(pd.to_numeric, errors="coerce")

    panel["idt_local"] = panel[["excise_local", "other_tax_local"]].sum(
        axis=1,
        min_count=1,
    )
    panel["idt_eur"] = panel["idt_local"] * panel["eur_per_currency"]

    vat_summary = _aggregate_weighted(panel, "vat_pct")
    idt_summary = _aggregate_weighted(panel, "idt_eur")

    vat_value = vat_summary.pivot(index="date", columns="product", values="value")
    vat_coverage = vat_summary.pivot(
        index="date",
        columns="product",
        values="coverage",
    )
    idt_direct = idt_summary.pivot(index="date", columns="product", values="value")
    idt_coverage = idt_summary.pivot(
        index="date",
        columns="product",
        values="coverage",
    )

    output = pd.DataFrame(index=dates)
    diagnostics = pd.DataFrame(index=dates)

    for label, product in PRODUCTS.items():
        wtax = pd.to_numeric(
            prices_with_tax[f"EUR_price_with_tax_{product}"],
            errors="coerce",
        )
        ntax = pd.to_numeric(
            prices_without_tax[f"EUR_price_wo_tax_{product}"],
            errors="coerce",
        )
        vat_rate = vat_value[product].where(
            vat_coverage[product] >= minimum_coverage
        )

        # This accounting identity is also the robust repair used when WOB
        # indirect-tax observations are clearly inconsistent.
        idt_identity = wtax.div(1.0 + vat_rate.div(100.0)).sub(ntax)

        output[f"wob_{label}_cpr_wtax"] = wtax
        output[f"wob_{label}_cpr_ntax"] = ntax
        output[f"wob_{label}_idt"] = idt_identity
        output[f"wob_{label}_vat"] = vat_rate

        diagnostics[f"{label}_vat_coverage"] = vat_coverage[product]
        diagnostics[f"{label}_idt_direct_coverage"] = idt_coverage[product]
        diagnostics[f"{label}_idt_direct"] = idt_direct[product].where(
            idt_coverage[product] >= minimum_coverage
        )
        diagnostics[f"{label}_idt_identity"] = idt_identity
        diagnostics[f"{label}_idt_direct_gap"] = (
            diagnostics[f"{label}_idt_direct"] - idt_identity
        )

    output = output.loc[:, OUTPUT_COLUMNS].sort_index()
    output.index.name = "date"
    diagnostics.index.name = "date"

    validate_wob_dataset(output)
    return output, diagnostics.sort_index()


def validate_wob_dataset(data: pd.DataFrame) -> None:
    """Fail early when the processed dataset is structurally invalid."""
    missing = [column for column in OUTPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing output columns: {missing}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("The dataset index must be a DatetimeIndex.")
    if data.index.has_duplicates:
        raise ValueError("Duplicate weekly dates found.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted.")

    vat_columns = [column for column in data if column.endswith("_vat")]
    price_columns = [column for column in data if not column.endswith("_vat")]

    if (data[vat_columns] < 0).any().any() or (data[vat_columns] > 40).any().any():
        raise ValueError("A VAT rate lies outside the expected 0%-40% range.")
    if (data[price_columns] < 0).any().any():
        raise ValueError("A price or indirect-tax series contains negatives.")


def save_wob_dataset(
    data: pd.DataFrame,
    diagnostics: pd.DataFrame,
    raw_path: str | Path,
    output_root: str | Path,
    file_format: str = "csv",
) -> dict[str, Path]:
    """Save one dated processed vintage and its manifest."""
    raw = Path(raw_path)
    root = Path(output_root)
    try:
        vintage = pd.Timestamp(raw.parent.name).date().isoformat()
    except ValueError:
        vintage = datetime.now(timezone.utc).date().isoformat()
    destination = root / vintage
    destination.mkdir(parents=True, exist_ok=True)

    if file_format not in {"csv", "parquet"}:
        raise ValueError("file_format must be 'csv' or 'parquet'.")

    suffix = ".csv" if file_format == "csv" else ".parquet"
    data_path = destination / f"wob_weekly{suffix}"
    diagnostics_path = destination / f"wob_diagnostics{suffix}"
    manifest_path = destination / "manifest.json"

    if file_format == "csv":
        data.to_csv(data_path, date_format="%Y-%m-%d")
        diagnostics.to_csv(diagnostics_path, date_format="%Y-%m-%d")
    else:
        try:
            data.to_parquet(data_path)
            diagnostics.to_parquet(diagnostics_path)
        except ImportError as error:
            raise ImportError(
                "Parquet output requires pyarrow or fastparquet. "
                "Use --format csv or install pyarrow."
            ) from error

    manifest = {
        "source": "European Commission Weekly Oil Bulletin",
        "raw_file": str(raw),
        "raw_sha256": _sha256(raw),
        "vintage": vintage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_observation": data.index.min().date().isoformat(),
        "last_observation": data.index.max().date().isoformat(),
        "observations": int(len(data)),
        "columns": list(data.columns),
        "construction": {
            "prices": "Official EUR aggregate columns from the workbook",
            "vat": "Annual product-consumption weighted national VAT rates",
            "idt": "WTAX / (1 + VAT/100) - NTAX",
            "membership": "Changing euro-area composition",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "data": data_path,
        "diagnostics": diagnostics_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the European Commission Weekly Oil Bulletin dataset."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Optional path to one raw workbook. When omitted, the latest vintage is detected automatically.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Root containing YYYY-MM-DD WOB vintages (default: {DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Processed-file format (default: csv).",
    )
    args = parser.parse_args()

    raw_path = args.raw or find_latest_raw_file(args.raw_root)
    data, diagnostics = build_wob_dataset(
        raw_path,
        minimum_coverage=args.minimum_coverage,
    )
    paths = save_wob_dataset(
        data,
        diagnostics,
        raw_path,
        args.output_root,
        file_format=args.format,
    )

    print(f"wob.py version: {SCRIPT_VERSION}")
    print(f"Raw file: {raw_path}")
    print(f"Observations: {len(data):,}")
    print(f"Span: {data.index.min().date()} -> {data.index.max().date()}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
