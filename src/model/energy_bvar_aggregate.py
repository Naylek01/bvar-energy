
"""Aggregation layer for the ECB-style euro-area energy BVAR suite.

This module deliberately contains the aggregation mechanics outside notebooks
and outside the dashboard.  The notebook and dashboard should both call these
functions.

Core conventions
----------------
1. HICP subcomponents are aggregated with annual Laspeyres-type weights and
   December chain links.
2. Annual weights are kept in their published Eurostat scale (per thousand of
   total HICP). They are normalised only inside the relevant aggregate.
3. If a forecast crosses into a year whose weights are not yet available, the
   latest published weight vector is carried forward.
4. Weekly petrol, diesel and liquid-fuel model outputs must be converted to
   monthly post-tax price paths before they are mapped into HICP indices.
5. Petrol/diesel/liquid-fuel WOB prices and HICP indices are different objects.
   The bridge is estimated and diagnosed explicitly; no hidden unit conversion
   is allowed.
6. Cross-model posterior draws are not a joint posterior.  The helper
   ``independent_draw_pairing`` makes the maintained independence assumption
   explicit rather than pairing identical draw numbers across separate chains.

The processed-vintage directory is expected to contain the v10 builder outputs:
``hicp_indices_monthly.csv``, ``hicp_weights_annual.csv``,
``hicp_series_metadata.csv``, ``hicp_flags.csv`` and
``hicp_weight_identity_diagnostics.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Published HICP contracts from the v10 builder
# ---------------------------------------------------------------------------

HOUSEHOLD_COMPONENTS = (
    "hicp_electricity",
    "hicp_gas",
    "hicp_liquid_fuels",
    "hicp_solid_fuels",
    "hicp_heat_cooling_energy",
)

FUEL_COMPONENTS_POST_2017 = (
    "hicp_liquid_fuels",
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
)

CAR_FUELS_COMPONENTS_POST_2017 = (
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
    "hicp_lubricants",
)

ENERGY_COMPONENTS_POST_2017 = (
    "hicp_electricity",
    "hicp_gas",
    "hicp_liquid_fuels",
    "hicp_solid_fuels",
    "hicp_heat_cooling_energy",
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
)

ENERGY_SPECIAL_COMPONENTS = (
    "hicp_electricity_gas_solid_heat",
    "hicp_fuel_special_aggregate",
)

MODEL_AGGREGATE_COMPONENTS = (
    "car_fuels",
    "liquid_fuels",
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
)


DEFAULT_RECONSTRUCTION_TOLERANCE = 0.05  # HICP index points
DEFAULT_WEIGHT_TOLERANCE = 0.05          # per thousand of total HICP


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_monthly_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame.set_index("date").sort_index()
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    if frame.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate monthly dates.")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="MS", name="date")
    if not frame.index.equals(expected):
        raise ValueError(f"{path.name}: calendar is not a complete monthly MS index.")
    return frame.apply(pd.to_numeric, errors="coerce")


def _read_annual_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "year" not in frame.columns:
        raise ValueError(f"{path.name}: missing 'year' column.")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame = frame.set_index("year").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate years.")
    if len(frame) and not np.array_equal(
        frame.index.to_numpy(), np.arange(frame.index.min(), frame.index.max() + 1)
    ):
        raise ValueError(f"{path.name}: annual calendar has an interior gap.")
    return frame.apply(pd.to_numeric, errors="coerce")


def load_aggregation_inputs(processed_dir: str | Path) -> dict:
    """Load and validate the v10 aggregation sidecars from one processed vintage."""
    root = Path(processed_dir)
    required = {
        "indices": root / "hicp_indices_monthly.csv",
        "weights": root / "hicp_weights_annual.csv",
        "series_metadata": root / "hicp_series_metadata.csv",
        "flags": root / "hicp_flags.csv",
        "weight_diagnostics": root / "hicp_weight_identity_diagnostics.csv",
        "manifest": root / "manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Processed vintage is missing aggregation input(s):\n  "
            + "\n  ".join(missing)
        )

    indices = _read_monthly_csv(required["indices"])
    weights = _read_annual_csv(required["weights"])
    metadata = pd.read_csv(required["series_metadata"])
    flags = pd.read_csv(required["flags"])
    diagnostics = pd.read_csv(required["weight_diagnostics"])
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))

    needed_indices = {
        "hicp_energy",
        "hicp_household_energy",
        "hicp_fuel_special_aggregate",
        "hicp_electricity_gas_solid_heat",
        *HOUSEHOLD_COMPONENTS,
        *CAR_FUELS_COMPONENTS_POST_2017,
    }
    missing_columns = sorted(needed_indices.difference(indices.columns))
    if missing_columns:
        raise ValueError(
            "hicp_indices_monthly.csv is missing required aggregation columns: "
            f"{missing_columns}."
        )
    missing_weight_columns = sorted(needed_indices.difference(weights.columns))
    if missing_weight_columns:
        raise ValueError(
            "hicp_weights_annual.csv is missing required aggregation columns: "
            f"{missing_weight_columns}."
        )

    return {
        "processed_dir": root,
        "indices": indices,
        "weights": weights,
        "series_metadata": metadata,
        "flags": flags,
        "weight_diagnostics": diagnostics,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# Laspeyres-type chain linking
# ---------------------------------------------------------------------------

def _weight_row(
    weights: pd.DataFrame,
    year: int,
    columns: Sequence[str],
    *,
    carry_forward: bool = True,
) -> tuple[pd.Series, int]:
    columns = list(columns)
    available = weights.index[weights.index <= int(year)]
    if year in weights.index:
        weight_year = int(year)
    elif carry_forward and len(available):
        weight_year = int(available.max())
    else:
        raise KeyError(f"No HICP weights are available for year {year}.")

    row = weights.loc[weight_year, columns].astype(float)
    if row.isna().any():
        missing = row.index[row.isna()].tolist()
        raise ValueError(
            f"Weight year {weight_year} has missing component weights: {missing}."
        )
    if (row < 0).any() or float(row.sum()) <= 0:
        raise ValueError(f"Weight year {weight_year} contains invalid weights.")
    return row, weight_year


def _normalised_weight_row(
    weights: pd.DataFrame,
    year: int,
    columns: Sequence[str],
    *,
    carry_forward: bool = True,
) -> tuple[pd.Series, int]:
    row, weight_year = _weight_row(
        weights, year, columns, carry_forward=carry_forward
    )
    return row / float(row.sum()), weight_year


def chain_link_laspeyres(
    component_indices: pd.DataFrame,
    annual_weights: pd.DataFrame,
    *,
    components: Sequence[str],
    aggregate_anchor: float,
    anchor_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    carry_forward_weights: bool = True,
) -> dict:
    """Chain-link component indices from a December anchor.

    For calendar year y,

        A_t = A_Dec(y-1) * sum_i s_i,y * I_i,t / I_i,Dec(y-1),

    where ``s_i,y`` are the component weights normalised inside the aggregate.

    ``anchor_date`` must be a December month-start.  The first calculated month
    is the following January.  Trailing months with unavailable component data
    are omitted, but an interior missing component month raises.
    """
    components = list(components)
    missing = [name for name in components if name not in component_indices.columns]
    if missing:
        raise KeyError(f"Missing component index columns: {missing}.")

    data = component_indices[components].astype(float).sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("component_indices must have a DatetimeIndex.")
    if data.index.has_duplicates:
        raise ValueError("component_indices contains duplicate dates.")
    if not ((data.index.day == 1).all()):
        raise ValueError("component_indices dates must be month-starts.")

    anchor_date = pd.Timestamp(anchor_date).to_period("M").to_timestamp(how="start")
    if anchor_date.month != 12:
        raise ValueError("anchor_date must be December.")
    if anchor_date not in data.index:
        raise KeyError(f"Component anchor month {anchor_date.date()} is unavailable.")
    if not np.isfinite(float(aggregate_anchor)) or float(aggregate_anchor) <= 0:
        raise ValueError("aggregate_anchor must be finite and positive.")

    if end_date is None:
        end_date = data.index.max()
    end_date = pd.Timestamp(end_date).to_period("M").to_timestamp(how="start")
    if end_date <= anchor_date:
        raise ValueError("end_date must be after anchor_date.")

    out_values: dict[pd.Timestamp, float] = {}
    out_terms: dict[pd.Timestamp, np.ndarray] = {}
    weight_year_used: dict[pd.Timestamp, int] = {}

    previous_december = anchor_date
    previous_aggregate = float(aggregate_anchor)
    first_year = anchor_date.year + 1
    last_year = end_date.year
    stopped = False

    for year in range(first_year, last_year + 1):
        shares, source_weight_year = _normalised_weight_row(
            annual_weights,
            year,
            components,
            carry_forward=carry_forward_weights,
        )
        base_components = data.reindex([previous_december])[components].iloc[0]
        if base_components.isna().any() or (base_components <= 0).any():
            bad = base_components.index[
                base_components.isna() | (base_components <= 0)
            ].tolist()
            raise ValueError(
                f"Invalid component anchor values at {previous_december.date()}: {bad}."
            )

        year_end = min(pd.Timestamp(year, 12, 1), end_date)
        months = pd.date_range(pd.Timestamp(year, 1, 1), year_end, freq="MS")

        for position, month in enumerate(months):
            row = data.reindex([month])[components].iloc[0]
            if row.isna().any():
                # Only a genuinely trailing publication edge may be incomplete.
                remaining = pd.date_range(month, end_date, freq="MS")
                later = data.reindex(remaining)[components]
                if later.notna().all(axis=1).any():
                    missing_names = row.index[row.isna()].tolist()
                    raise ValueError(
                        f"Interior component gap at {month.date()}: {missing_names}."
                    )
                stopped = True
                break

            relatives = row.to_numpy(dtype=float) / base_components.to_numpy(dtype=float)
            terms = previous_aggregate * shares.to_numpy(dtype=float) * relatives
            out_values[month] = float(terms.sum())
            out_terms[month] = terms
            weight_year_used[month] = source_weight_year

        december = pd.Timestamp(year, 12, 1)
        if stopped or december > end_date:
            break
        if december not in out_values:
            break
        previous_aggregate = out_values[december]
        previous_december = december

    index = pd.DatetimeIndex(sorted(out_values), name="date")
    aggregate = pd.Series(
        [out_values[date] for date in index],
        index=index,
        name="aggregate_index",
        dtype=float,
    )
    terms = pd.DataFrame(
        [out_terms[date] for date in index],
        index=index,
        columns=components,
        dtype=float,
    )
    terms.index.name = "date"
    weight_year = pd.Series(
        [weight_year_used[date] for date in index],
        index=index,
        name="weight_year_used",
        dtype=int,
    )
    return {
        "index": aggregate,
        "terms": terms,
        "weight_year_used": weight_year,
        "components": components,
        "anchor_date": anchor_date,
        "anchor_value": float(aggregate_anchor),
        "future_weight_policy": (
            "latest published annual weights carried forward"
            if carry_forward_weights
            else "year-specific published weights required"
        ),
    }


def reconstruction_error_table(
    reconstructed: pd.Series,
    published: pd.Series,
    *,
    tolerance: float = DEFAULT_RECONSTRUCTION_TOLERANCE,
    label: str = "aggregate",
    hard_fail: bool = True,
) -> pd.DataFrame:
    """Compare a reconstructed index with its published counterpart."""
    joined = pd.concat(
        [
            reconstructed.rename("reconstructed"),
            published.astype(float).rename("published"),
        ],
        axis=1,
    ).dropna()
    if joined.empty:
        raise ValueError(f"{label}: no overlapping reconstructed/published months.")
    joined["error"] = joined["reconstructed"] - joined["published"]
    joined["absolute_error"] = joined["error"].abs()
    joined["tolerance"] = float(tolerance)
    joined["pass"] = joined["absolute_error"] <= float(tolerance)

    maximum = float(joined["absolute_error"].max())
    if hard_fail and maximum > float(tolerance):
        worst = joined["absolute_error"].idxmax()
        raise AssertionError(
            f"{label}: maximum reconstruction error {maximum:.6f} index points "
            f"at {pd.Timestamp(worst).date()} exceeds {tolerance:.6f}."
        )
    return joined



def annual_reanchored_reconstruction_errors(
    component_indices: pd.DataFrame,
    annual_weights: pd.DataFrame,
    published_aggregate: pd.Series,
    *,
    components: Sequence[str],
    first_year: int,
    last_year: int | None = None,
    tolerance: float = DEFAULT_RECONSTRUCTION_TOLERANCE,
) -> pd.DataFrame:
    """Validate each calendar year from its *published* previous-December anchor.

    This diagnostic separates local Laspeyres/weight errors from cumulative
    chain drift.  For every year y we reset the aggregate anchor to the
    published value at Dec(y-1), reconstruct that year's available months, and
    compare them with the published aggregate.

    A cumulative reconstruction can drift slowly while each annual block is
    locally correct. Conversely, failure of the annual re-anchored test points
    to the formula, component definitions or weights themselves.
    """
    components = list(components)
    data = component_indices[components].astype(float).sort_index()
    published = published_aggregate.astype(float).sort_index()

    if last_year is None:
        last_year = int(min(data.index.max().year, published.index.max().year))

    blocks = []
    for year in range(int(first_year), int(last_year) + 1):
        anchor_date = pd.Timestamp(year - 1, 12, 1)
        if anchor_date not in data.index or anchor_date not in published.index:
            continue
        if data.loc[anchor_date, components].isna().any():
            continue
        anchor_value = published.loc[anchor_date]
        if not np.isfinite(anchor_value) or float(anchor_value) <= 0:
            continue

        year_end = min(
            pd.Timestamp(year, 12, 1),
            data.index.max(),
            published.index.max(),
        )
        if year_end < pd.Timestamp(year, 1, 1):
            continue

        reconstructed = chain_link_laspeyres(
            data,
            annual_weights,
            components=components,
            aggregate_anchor=float(anchor_value),
            anchor_date=anchor_date,
            end_date=year_end,
        )["index"]

        error = reconstruction_error_table(
            reconstructed,
            published,
            tolerance=tolerance,
            label=f"annual_reanchor_{year}",
            hard_fail=False,
        )
        error = error.copy()
        error.insert(0, "year", year)
        error.insert(1, "anchor_date", anchor_date)
        error.insert(2, "anchor_value", float(anchor_value))
        blocks.append(error)

    if not blocks:
        raise ValueError("No annual re-anchored reconstruction block could be formed.")

    out = pd.concat(blocks).sort_index()
    out.index.name = "date"
    return out


def assert_reconstruction_suite(suite: Mapping) -> None:
    """Raise one compact error *after* every reconstruction test is available."""
    summary = pd.DataFrame(suite["summary"])
    if "pass" not in summary.columns:
        raise ValueError("Reconstruction suite summary has no 'pass' column.")
    failed = summary.loc[~summary["pass"].astype(bool)]
    if failed.empty:
        return
    detail = "; ".join(
        (
            f"{name}: cumulative={row['max_abs_error']:.6f}, "
            f"annual={row['annual_reanchored_max_abs_error']:.6f}, "
            f"tol={row['tolerance']:.6f}"
        )
        for name, row in failed.iterrows()
    )
    raise AssertionError(
        "Historical HICP reconstruction gate failed after all tests were "
        f"evaluated. {detail}"
    )



def historical_reconstruction_suite(
    processed_dir: str | Path,
    *,
    tolerance: float = DEFAULT_RECONSTRUCTION_TOLERANCE,
) -> dict:
    """Evaluate the complete historical aggregation validation suite.

    Both diagnostics are retained for every specification:

    * ``cumulative``: one chain starting from the first published December;
    * ``annual_reanchored``: every calendar year restarted from the published
      previous-December aggregate.

    The function itself does not stop at the first failure.  Use
    :func:`assert_reconstruction_suite` after displaying ``summary``.
    """
    inputs = load_aggregation_inputs(processed_dir)
    indices = inputs["indices"]
    weights = inputs["weights"]

    specs = {
        "household_energy": {
            "target": "hicp_household_energy",
            "components": HOUSEHOLD_COMPONENTS,
            "anchor_date": "1996-12-01",
        },
        "energy_special_split": {
            "target": "hicp_energy",
            "components": ENERGY_SPECIAL_COMPONENTS,
            "anchor_date": "1996-12-01",
        },
        "car_fuels_cp0722": {
            "target": "hicp_fuels_lubricants_personal_transport",
            "components": CAR_FUELS_COMPONENTS_POST_2017,
            "anchor_date": "2016-12-01",
        },
        "fuel_special_post_2017": {
            "target": "hicp_fuel_special_aggregate",
            "components": FUEL_COMPONENTS_POST_2017,
            "anchor_date": "2016-12-01",
        },
        "energy_detailed_post_2017": {
            "target": "hicp_energy",
            "components": ENERGY_COMPONENTS_POST_2017,
            "anchor_date": "2016-12-01",
        },
    }

    reconstructions = {}
    errors = {}
    annual_errors = {}
    summary_rows = []

    for label, spec in specs.items():
        anchor_date = pd.Timestamp(spec["anchor_date"])
        first_year = anchor_date.year + 1
        target = str(spec["target"])
        components = list(spec["components"])

        anchor_value = float(indices.at[anchor_date, target])
        result = chain_link_laspeyres(
            indices,
            weights,
            components=components,
            aggregate_anchor=anchor_value,
            anchor_date=anchor_date,
            end_date=indices.index.max(),
        )

        cumulative_error = reconstruction_error_table(
            result["index"],
            indices[target],
            tolerance=tolerance,
            label=label,
            hard_fail=False,
        )

        annual_error = annual_reanchored_reconstruction_errors(
            indices,
            weights,
            indices[target],
            components=components,
            first_year=first_year,
            last_year=int(indices.index.max().year),
            tolerance=tolerance,
        )

        cumulative_pass = bool(cumulative_error["pass"].all())
        annual_pass = bool(annual_error["pass"].all())

        reconstructions[label] = result
        errors[label] = cumulative_error
        annual_errors[label] = annual_error

        summary_rows.append(
            {
                "test": label,
                "target": target,
                "start": cumulative_error.index.min(),
                "end": cumulative_error.index.max(),
                "observations": int(len(cumulative_error)),
                "max_abs_error": float(cumulative_error["absolute_error"].max()),
                "mean_abs_error": float(cumulative_error["absolute_error"].mean()),
                "annual_reanchored_max_abs_error": float(
                    annual_error["absolute_error"].max()
                ),
                "annual_reanchored_mean_abs_error": float(
                    annual_error["absolute_error"].mean()
                ),
                "tolerance": float(tolerance),
                "cumulative_pass": cumulative_pass,
                "annual_reanchored_pass": annual_pass,
                "pass": cumulative_pass and annual_pass,
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("test")
    return {
        "summary": summary,
        "reconstructions": reconstructions,
        "errors": errors,
        "annual_reanchored_errors": annual_errors,
        "inputs": inputs,
    }


# ---------------------------------------------------------------------------
# Exact additive contributions
# ---------------------------------------------------------------------------

def yoy_from_index(index: pd.Series) -> pd.Series:
    """Year-on-year percentage change of a monthly price index."""
    series = index.astype(float).sort_index()
    return (100.0 * (series / series.shift(12) - 1.0)).rename(
        f"{series.name or 'index'}_yoy"
    )


def yoy_contributions_from_terms(
    aggregate_index: pd.Series,
    component_terms: pd.DataFrame,
) -> pd.DataFrame:
    """Exact additive contributions to aggregate YoY inflation.

    If A_t = sum_i T_i,t, define

        contribution_i,t = 100 * (T_i,t - T_i,t-12) / A_t-12.

    This remains exactly additive across annual re-weighting and January
    chain-link breaks because the decomposition is performed on the actual
    Laspeyres terms, not on a weighted average of component YoY rates.
    """
    aggregate = aggregate_index.astype(float).sort_index()
    terms = component_terms.astype(float).reindex(aggregate.index)
    denominator = aggregate.shift(12)
    contributions = 100.0 * terms.diff(12).div(denominator, axis=0)
    aggregate_yoy = 100.0 * (aggregate / denominator - 1.0)
    check = contributions.sum(axis=1, min_count=1) - aggregate_yoy
    valid = check.dropna()
    if len(valid) and float(valid.abs().max()) > 1e-10:
        raise AssertionError(
            "YoY component contributions are not additive to numerical precision: "
            f"max error={float(valid.abs().max()):.3e}."
        )
    contributions["aggregate_yoy"] = aggregate_yoy
    contributions["sum_contributions"] = contributions[
        [c for c in terms.columns]
    ].sum(axis=1, min_count=1)
    contributions["additivity_error"] = (
        contributions["sum_contributions"] - contributions["aggregate_yoy"]
    )
    return contributions


# ---------------------------------------------------------------------------
# WOB price -> HICP bridge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriceIndexBridge:
    """Empirical WOB-price to HICP-index bridge used only at the aggregation layer.

    The *applied* bridge is deliberately estimated through the origin in
    absolute monthly differences:

        ΔHICP_t = beta * ΔPrice_t + error_t.

    This mirrors the paper's absolute-change specification and guarantees that
    a constant consumer-price path implies a constant HICP path.
    """
    label: str
    beta: float
    observations: int
    correlation: float
    r_squared: float
    uncentered_r_squared: float
    rmse_index_change: float
    first_date: str
    last_date: str
    beta_first_half: float
    beta_second_half: float
    beta_half_relative_gap: float
    unrestricted_intercept: float
    unrestricted_intercept_se: float
    unrestricted_intercept_tstat: float
    method: str = "absolute_difference_no_intercept"

    @property
    def intercept(self) -> float:
        """Applied intercept, fixed by construction."""
        return 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "method": self.method,
            "intercept": 0.0,
            "beta": self.beta,
            "observations": self.observations,
            "correlation": self.correlation,
            "r_squared": self.r_squared,
            "uncentered_r_squared": self.uncentered_r_squared,
            "rmse_index_change": self.rmse_index_change,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "beta_first_half": self.beta_first_half,
            "beta_second_half": self.beta_second_half,
            "beta_half_relative_gap": self.beta_half_relative_gap,
            "unrestricted_intercept": self.unrestricted_intercept,
            "unrestricted_intercept_se": self.unrestricted_intercept_se,
            "unrestricted_intercept_tstat": self.unrestricted_intercept_tstat,
        }


def fit_price_index_bridge(
    monthly_consumer_price: pd.Series,
    hicp_index: pd.Series,
    *,
    label: str,
    minimum_observations: int = 24,
    max_unrestricted_intercept_tstat: float = 2.58,
    max_half_beta_relative_gap: float = 0.25,
) -> PriceIndexBridge:
    """Estimate a WOB-price -> HICP bridge in monthly absolute differences.

    Applied specification
    ---------------------
        ΔHICP_t = beta * ΔPrice_t + error_t

    There is intentionally **no applied intercept**.  The ECB STIP paper models
    these consumer-price equations in absolute changes because per-unit excise,
    refining and distribution margins make percentage-change relationships less
    stable.  A zero intercept also passes the required constant-price test:
    if the consumer-price path is flat, the mapped HICP path is flat.

    Diagnostics
    -----------
    An unrestricted intercept regression is still estimated diagnostically.
    A significant intercept signals that the simple bridge is misspecified.
    The slope is also estimated separately on the two half-samples.
    """
    price = monthly_consumer_price.astype(float).sort_index()
    hicp = hicp_index.astype(float).sort_index()

    if (price.dropna() <= 0).any() or (hicp.dropna() <= 0).any():
        raise ValueError(f"{label}: price and HICP levels must be positive.")

    data = pd.concat(
        [
            price.diff().rename("price_change"),
            hicp.diff().rename("hicp_change"),
        ],
        axis=1,
    ).dropna()

    if len(data) < int(minimum_observations):
        raise ValueError(
            f"{label}: only {len(data)} overlapping monthly changes; "
            f"need at least {minimum_observations}."
        )

    x = data["price_change"].to_numpy(dtype=float)
    y = data["hicp_change"].to_numpy(dtype=float)
    xx = float(x @ x)
    if not np.isfinite(xx) or xx <= 0:
        raise ValueError(f"{label}: consumer-price changes contain no variation.")

    beta = float((x @ y) / xx)
    fitted = beta * x
    residual = y - fitted

    sse = float(residual @ residual)
    centered_sst = float((y - y.mean()) @ (y - y.mean()))
    uncentered_sst = float(y @ y)
    r_squared = np.nan if centered_sst <= 0 else 1.0 - sse / centered_sst
    uncentered_r_squared = (
        np.nan if uncentered_sst <= 0 else 1.0 - sse / uncentered_sst
    )
    correlation = float(data["price_change"].corr(data["hicp_change"]))

    midpoint = len(data) // 2
    if midpoint < 2 or len(data) - midpoint < 2:
        raise ValueError(f"{label}: sample is too short for half-sample diagnostics.")

    def _slope(block: pd.DataFrame) -> float:
        xb = block["price_change"].to_numpy(dtype=float)
        yb = block["hicp_change"].to_numpy(dtype=float)
        denom = float(xb @ xb)
        if denom <= 0:
            raise ValueError(f"{label}: zero change variance in a bridge half-sample.")
        return float((xb @ yb) / denom)

    beta_first = _slope(data.iloc[:midpoint])
    beta_second = _slope(data.iloc[midpoint:])
    scale = max(abs(beta), 1e-12)
    beta_half_relative_gap = abs(beta_second - beta_first) / scale

    # Unrestricted intercept is diagnostic only; it is never propagated.
    X = np.column_stack([np.ones(len(x)), x])
    unrestricted = np.linalg.lstsq(X, y, rcond=None)[0]
    unrestricted_residual = y - X @ unrestricted
    dof = len(y) - 2
    sigma2 = float(unrestricted_residual @ unrestricted_residual) / max(dof, 1)
    covariance = sigma2 * np.linalg.inv(X.T @ X)
    intercept_se = float(np.sqrt(max(covariance[0, 0], 0.0)))
    intercept = float(unrestricted[0])
    intercept_tstat = (
        np.nan if intercept_se <= 0 else float(intercept / intercept_se)
    )

    if (
        np.isfinite(intercept_tstat)
        and abs(intercept_tstat) > float(max_unrestricted_intercept_tstat)
    ):
        raise AssertionError(
            f"{label}: unrestricted bridge intercept is statistically material "
            f"(t={intercept_tstat:.2f}); a zero-intercept bridge is not adequate."
        )

    if beta_half_relative_gap > float(max_half_beta_relative_gap):
        raise AssertionError(
            f"{label}: bridge slope is unstable across half-samples "
            f"(relative gap={beta_half_relative_gap:.1%}, "
            f"limit={max_half_beta_relative_gap:.1%})."
        )

    return PriceIndexBridge(
        label=str(label),
        beta=beta,
        observations=int(len(data)),
        correlation=correlation,
        r_squared=float(r_squared),
        uncentered_r_squared=float(uncentered_r_squared),
        rmse_index_change=float(np.sqrt(np.mean(residual**2))),
        first_date=data.index.min().date().isoformat(),
        last_date=data.index.max().date().isoformat(),
        beta_first_half=beta_first,
        beta_second_half=beta_second,
        beta_half_relative_gap=float(beta_half_relative_gap),
        unrestricted_intercept=intercept,
        unrestricted_intercept_se=intercept_se,
        unrestricted_intercept_tstat=float(intercept_tstat),
    )


def bridge_diagnostic_table(bridges: Mapping[str, PriceIndexBridge]) -> pd.DataFrame:
    """Compact bridge diagnostics for notebook display."""
    rows = []
    for name, bridge in bridges.items():
        row = bridge.as_dict()
        row["component"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("component")


def price_paths_to_hicp_paths(
    monthly_price_paths: np.ndarray,
    dates: Sequence[pd.Timestamp],
    *,
    bridge: PriceIndexBridge,
    anchor_price: float,
    anchor_hicp: float,
) -> np.ndarray:
    """Map monthly consumer-price paths into HICP paths draw by draw.

    Applied recursion:

        HICP_t = HICP_{t-1} + beta * (Price_t - Price_{t-1}).

    Therefore a flat price path produces a flat HICP path exactly.
    """
    values = np.asarray(monthly_price_paths, dtype=float)
    dates = pd.DatetimeIndex(dates)

    if values.ndim != 2 or values.shape[1] != len(dates):
        raise ValueError("monthly_price_paths must have shape (draws, len(dates)).")
    if len(dates) == 0:
        return values.copy()

    expected = pd.date_range(dates[0], dates[-1], freq="MS", name=dates.name)
    if not dates.equals(expected):
        raise ValueError("Bridge dates must be a complete monthly calendar.")
    if not np.isfinite(anchor_price) or anchor_price <= 0:
        raise ValueError("anchor_price must be positive.")
    if not np.isfinite(anchor_hicp) or anchor_hicp <= 0:
        raise ValueError("anchor_hicp must be positive.")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All monthly price paths must be finite and positive.")

    previous_price = np.full(values.shape[0], float(anchor_price), dtype=float)
    previous_index = np.full(values.shape[0], float(anchor_hicp), dtype=float)
    out = np.empty_like(values, dtype=float)

    for t in range(values.shape[1]):
        price_change = values[:, t] - previous_price
        current_index = previous_index + bridge.beta * price_change
        if np.any(~np.isfinite(current_index)) or np.any(current_index <= 0):
            raise ValueError(
                f"{bridge.label}: mapped HICP index became non-positive or non-finite."
            )
        out[:, t] = current_index
        previous_price = values[:, t]
        previous_index = current_index

    # Hard invariant requested by the bridge critique.
    flat = np.full((1, 2), float(anchor_price), dtype=float)
    flat_out = np.empty_like(flat)
    prev_p = float(anchor_price)
    prev_i = float(anchor_hicp)
    for t in range(flat.shape[1]):
        prev_i = prev_i + bridge.beta * (flat[0, t] - prev_p)
        flat_out[0, t] = prev_i
        prev_p = flat[0, t]
    if not np.allclose(flat_out, float(anchor_hicp), atol=1e-12, rtol=0.0):
        raise AssertionError("Constant-price bridge invariant failed.")

    return out


# ---------------------------------------------------------------------------
# Frequency conversion and draw alignment
# ---------------------------------------------------------------------------

def weekly_paths_with_history_to_monthly_mean(
    paths: np.ndarray,
    path_dates: Sequence[pd.Timestamp],
    history: pd.Series,
    *,
    return_coverage: bool = False,
):
    """Create *complete* calendar-month means from weekly paths plus history.

    Weeks before the first simulated week but inside its calendar month are
    filled from observed WOB history.  An incomplete interior month raises.
    An incomplete final month is dropped rather than averaged over a partial
    set of weeks.

    When ``return_coverage=True`` the function also returns a table containing
    the expected and available Monday counts for every considered month.
    """
    values = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(path_dates).to_period("W-SUN").start_time
    dates = pd.DatetimeIndex(dates, name="date")

    if values.ndim != 2 or values.shape[1] != len(dates):
        raise ValueError("paths must have shape (draws, len(path_dates)).")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("path_dates must be unique and sorted.")
    if len(dates) == 0:
        empty_dates = pd.DatetimeIndex([], name="date")
        empty_coverage = pd.DataFrame(
            columns=[
                "expected_mondays",
                "observed_history_mondays",
                "simulated_mondays",
                "available_mondays",
                "complete",
                "retained",
            ]
        )
        return (values.copy(), empty_dates, empty_coverage) if return_coverage else (
            values.copy(),
            empty_dates,
        )

    observed = history.astype(float).dropna().sort_index()
    observed.index = pd.DatetimeIndex(observed.index).to_period("W-SUN").start_time
    if observed.index.has_duplicates:
        observed = observed.groupby(level=0).last()

    first_month = dates[0].to_period("M")
    last_month = dates[-1].to_period("M")
    months = pd.period_range(first_month, last_month, freq="M")
    lookup = {pd.Timestamp(date): j for j, date in enumerate(dates)}

    monthly_values = []
    monthly_dates = []
    coverage_rows = []
    n_draws = values.shape[0]

    for month in months:
        start = month.to_timestamp(how="start")
        end = month.to_timestamp(how="end").normalize()
        mondays = pd.date_range(start, end, freq="W-MON")

        columns = []
        n_history = 0
        n_simulated = 0
        for monday in mondays:
            monday = pd.Timestamp(monday)
            if monday in lookup:
                columns.append(values[:, lookup[monday]])
                n_simulated += 1
            elif monday < dates[0] and monday in observed.index:
                columns.append(np.full(n_draws, float(observed.loc[monday])))
                n_history += 1

        available = n_history + n_simulated
        complete = available == len(mondays)
        is_last = month == last_month
        retained = complete

        coverage_rows.append(
            {
                "month": start,
                "expected_mondays": int(len(mondays)),
                "observed_history_mondays": int(n_history),
                "simulated_mondays": int(n_simulated),
                "available_mondays": int(available),
                "complete": bool(complete),
                "retained": bool(retained),
            }
        )

        if not complete:
            if is_last:
                continue
            raise ValueError(
                f"Weekly path/history cannot form a complete monthly mean "
                f"for {month}: {available}/{len(mondays)} Mondays available."
            )

        block = np.column_stack(columns)
        monthly_values.append(np.mean(block, axis=1))
        monthly_dates.append(start)

    if not monthly_values:
        raise ValueError("No complete monthly mean could be formed.")

    monthly = np.column_stack(monthly_values)
    monthly_index = pd.DatetimeIndex(monthly_dates, name="date")
    coverage = pd.DataFrame(coverage_rows).set_index("month")
    coverage.index = pd.DatetimeIndex(coverage.index, name="date")

    if return_coverage:
        return monthly, monthly_index, coverage
    return monthly, monthly_index


def independent_draw_pairing(
    component_paths: Mapping[str, np.ndarray],
    *,
    n_draws: int | None = None,
    seed: int = 2026,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Randomly pair draws from independently estimated component models.

    Identical draw numbers in separate Gibbs chains have no probabilistic
    relationship. Random re-pairing therefore serves only to make the maintained
    cross-model independence assumption explicit and auditable; under
    exchangeable independent draws it does **not** create a richer joint
    posterior and does not by itself widen or improve the aggregate fan.
    """
    if not component_paths:
        raise ValueError("component_paths is empty.")
    arrays = {name: np.asarray(value, dtype=float) for name, value in component_paths.items()}
    for name, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(f"{name}: expected a two-dimensional draw x date array.")
        if len(array) < 1:
            raise ValueError(f"{name}: no posterior paths.")

    if n_draws is None:
        n_draws = min(len(array) for array in arrays.values())
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError("n_draws must be positive.")

    rng = np.random.default_rng(seed)
    paired = {}
    indices = {}
    for name, array in arrays.items():
        replace = n_draws > len(array)
        chosen = rng.choice(len(array), size=n_draws, replace=replace)
        paired[name] = array[chosen]
        indices[name] = chosen
    return paired, indices


# ---------------------------------------------------------------------------
# Forecast aggregation
# ---------------------------------------------------------------------------

def model_energy_weights(annual_weights: pd.DataFrame) -> pd.DataFrame:
    """Return raw annual weights for the six model aggregates.

    The car-fuels weight is the energy-relevant transport basket:
    diesel + petrol + other transport fuels. Lubricants are excluded because
    Eurostat's NRG special aggregate excludes them.
    """
    required = {
        "hicp_energy",
        "hicp_liquid_fuels",
        "hicp_gas",
        "hicp_electricity",
        "hicp_heat_cooling_energy",
        "hicp_solid_fuels",
        "hicp_diesel",
        "hicp_petrol",
        "hicp_other_transport_fuels",
    }
    missing = sorted(required.difference(annual_weights.columns))
    if missing:
        raise KeyError(f"Missing annual HICP weights: {missing}.")

    out = pd.DataFrame(index=annual_weights.index)
    out["car_fuels"] = (
        annual_weights["hicp_diesel"]
        + annual_weights["hicp_petrol"]
        + annual_weights["hicp_other_transport_fuels"]
    )
    out["liquid_fuels"] = annual_weights["hicp_liquid_fuels"]
    out["gas"] = annual_weights["hicp_gas"]
    out["electricity"] = annual_weights["hicp_electricity"]
    out["heat_energy"] = annual_weights["hicp_heat_cooling_energy"]
    out["solid_fuels"] = annual_weights["hicp_solid_fuels"]
    out["hicp_energy"] = annual_weights["hicp_energy"]

    diff = out[list(MODEL_AGGREGATE_COMPONENTS)].sum(axis=1) - out["hicp_energy"]
    available = out[list(MODEL_AGGREGATE_COMPONENTS)].notna().all(axis=1)

    # 2016 is the documented ECOICOP-v2 transition year in the current
    # Eurostat back history. The detailed petrol/diesel/other-fuels split is
    # used as a hard current-definition identity only from 2017 onward.
    guarded = available & (out.index >= 2017)
    bad = guarded & (diff.abs() > DEFAULT_WEIGHT_TOLERANCE)
    if bad.any():
        year = int(diff.index[bad][0])
        raise AssertionError(
            f"Model energy weights do not sum to NRG in {year}: "
            f"error={float(diff.loc[year]):.4f} per thousand."
        )
    out["weight_identity_error"] = diff
    out["weight_identity_guarded"] = guarded
    return out



def historical_model_component_reconstruction(
    processed_dir: str | Path,
    *,
    tolerance: float = DEFAULT_RECONSTRUCTION_TOLERANCE,
) -> dict:
    """Build the six-component history used by the forecast aggregator.

    ``car_fuels`` is the energy-relevant transport-fuel sub-aggregate
    (diesel + petrol + other transport fuels), excluding lubricants.  It is
    chain-linked from December 2016 with an arbitrary level anchor of 100; its
    scale cancels in the higher-level Laspeyres relatives.

    The resulting six-component reconstruction is checked against published
    HICP Energy from January 2017 onward.  This is the deterministic historical
    decomposition that must be supplied to ``draw_yoy_and_contributions`` so
    forecast contributions remain exactly additive across the origin.
    """
    inputs = load_aggregation_inputs(processed_dir)
    indices = inputs["indices"]
    weights = inputs["weights"]

    anchor_date = pd.Timestamp("2016-12-01")
    transport = chain_link_laspeyres(
        indices,
        weights,
        components=(
            "hicp_diesel",
            "hicp_petrol",
            "hicp_other_transport_fuels",
        ),
        aggregate_anchor=100.0,
        anchor_date=anchor_date,
        end_date=indices.index.max(),
    )
    transport_index = pd.concat(
        [
            pd.Series(
                [100.0],
                index=pd.DatetimeIndex([anchor_date], name="date"),
                name="car_fuels",
            ),
            transport["index"].rename("car_fuels"),
        ]
    ).sort_index()

    component_history = pd.DataFrame(index=indices.index)
    component_history["car_fuels"] = transport_index.reindex(indices.index)
    component_history["liquid_fuels"] = indices["hicp_liquid_fuels"]
    component_history["gas"] = indices["hicp_gas"]
    component_history["electricity"] = indices["hicp_electricity"]
    component_history["heat_energy"] = indices["hicp_heat_cooling_energy"]
    component_history["solid_fuels"] = indices["hicp_solid_fuels"]

    model_weights = _model_weight_table_for_chain(weights)
    energy = chain_link_laspeyres(
        component_history,
        model_weights,
        components=MODEL_AGGREGATE_COMPONENTS,
        aggregate_anchor=float(indices.at[anchor_date, "hicp_energy"]),
        anchor_date=anchor_date,
        end_date=indices.index.max(),
    )
    error = reconstruction_error_table(
        energy["index"],
        indices["hicp_energy"],
        tolerance=tolerance,
        label="six_model_component_energy_history",
        hard_fail=False,
    )
    annual_error = annual_reanchored_reconstruction_errors(
        component_history,
        model_weights,
        indices["hicp_energy"],
        components=MODEL_AGGREGATE_COMPONENTS,
        first_year=2017,
        last_year=int(indices.index.max().year),
        tolerance=tolerance,
    )
    cumulative_pass = bool(error["pass"].all())
    annual_pass = bool(annual_error["pass"].all())

    return {
        "component_history": component_history,
        "model_weights": model_weights,
        "transport_energy": transport,
        "energy": energy,
        "error": error,
        "annual_reanchored_error": annual_error,
        "summary": pd.Series(
            {
                "start": error.index.min(),
                "end": error.index.max(),
                "observations": int(len(error)),
                "max_abs_error": float(error["absolute_error"].max()),
                "mean_abs_error": float(error["absolute_error"].mean()),
                "annual_reanchored_max_abs_error": float(
                    annual_error["absolute_error"].max()
                ),
                "annual_reanchored_mean_abs_error": float(
                    annual_error["absolute_error"].mean()
                ),
                "tolerance": float(tolerance),
                "cumulative_pass": cumulative_pass,
                "annual_reanchored_pass": annual_pass,
                "pass": cumulative_pass and annual_pass,
            },
            name="six_model_component_energy_history",
        ),
        "inputs": inputs,
    }



def _model_weight_table_for_chain(annual_weights: pd.DataFrame) -> pd.DataFrame:
    model_weights = model_energy_weights(annual_weights)
    return model_weights[list(MODEL_AGGREGATE_COMPONENTS)]


def aggregate_component_draw_paths_laspeyres(
    component_history: pd.DataFrame,
    component_paths: Mapping[str, np.ndarray],
    path_dates: Sequence[pd.Timestamp],
    annual_weights: pd.DataFrame,
    aggregate_history: pd.Series,
    *,
    components: Sequence[str],
    carry_forward_weights: bool = True,
) -> dict:
    """Aggregate matched monthly HICP component paths draw by draw.

    ``component_history`` must contain observed component-index history through
    the month immediately preceding ``path_dates[0]``. ``component_paths`` then
    supplies the nowcast/forecast index paths on ``path_dates``.

    The function reconstructs January-to-origin observations for the first
    forecast year so that a forecast beginning mid-year uses the correct
    December(y-1) chain base.
    """
    components = list(components)
    dates = pd.DatetimeIndex(path_dates, name="date")
    if len(dates) == 0:
        raise ValueError("path_dates is empty.")
    expected = pd.date_range(dates[0], dates[-1], freq="MS", name="date")
    if not dates.equals(expected):
        raise ValueError("path_dates must be a complete monthly MS calendar.")

    missing_history = [name for name in components if name not in component_history.columns]
    missing_paths = [name for name in components if name not in component_paths]
    if missing_history or missing_paths:
        raise KeyError(
            f"Missing component history={missing_history}, paths={missing_paths}."
        )

    arrays = {name: np.asarray(component_paths[name], dtype=float) for name in components}
    shapes = {name: array.shape for name, array in arrays.items()}
    n_draws = next(iter(arrays.values())).shape[0]
    for name, array in arrays.items():
        if array.shape != (n_draws, len(dates)):
            raise ValueError(
                f"{name}: expected shape {(n_draws, len(dates))}, found {array.shape}."
            )
        if np.any(~np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(f"{name}: component HICP paths must be finite and positive.")

    history = component_history[components].astype(float).sort_index()
    first_year = dates[0].year
    anchor_date = pd.Timestamp(first_year - 1, 12, 1)
    if anchor_date not in history.index:
        raise KeyError(
            f"Component history must include the previous December {anchor_date.date()}."
        )
    aggregate_history = aggregate_history.astype(float).sort_index()
    if anchor_date not in aggregate_history.index or pd.isna(aggregate_history.loc[anchor_date]):
        raise KeyError(
            f"Published aggregate history must include {anchor_date.date()}."
        )

    # We need every month from January of the first path year to the end of the
    # requested path. Months before path_dates[0] are observed history.
    calculation_dates = pd.date_range(
        pd.Timestamp(first_year, 1, 1), dates[-1], freq="MS", name="date"
    )
    missing_weight_columns = [
        name for name in components if name not in annual_weights.columns
    ]
    if missing_weight_columns:
        raise KeyError(
            "annual_weights is missing component columns "
            f"{missing_weight_columns}."
        )
    weights_for_chain = annual_weights[components].astype(float).sort_index()

    full_component = {}
    for name in components:
        matrix = np.empty((n_draws, len(calculation_dates)), dtype=float)
        for t, month in enumerate(calculation_dates):
            if month < dates[0]:
                value = history[name].get(month, np.nan)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"{name}: missing observed history at {month.date()}."
                    )
                matrix[:, t] = float(value)
            else:
                j = dates.get_loc(month)
                matrix[:, t] = arrays[name][:, j]
        full_component[name] = matrix

    term_paths = np.empty(
        (n_draws, len(calculation_dates), len(components)), dtype=float
    )
    aggregate_paths = np.empty((n_draws, len(calculation_dates)), dtype=float)
    weight_year_used = np.empty(len(calculation_dates), dtype=int)

    previous_december = anchor_date
    previous_aggregate = np.full(
        n_draws, float(aggregate_history.loc[anchor_date]), dtype=float
    )

    for year in range(first_year, calculation_dates[-1].year + 1):
        shares, source_weight_year = _normalised_weight_row(
            weights_for_chain,
            year,
            components,
            carry_forward=carry_forward_weights,
        )

        if previous_december < calculation_dates[0]:
            base = history.loc[previous_december, components].to_numpy(dtype=float)
            base = np.repeat(base[None, :], n_draws, axis=0)
        else:
            prev_pos = calculation_dates.get_loc(previous_december)
            base = np.column_stack(
                [full_component[name][:, prev_pos] for name in components]
            )

        year_months = calculation_dates[calculation_dates.year == year]
        for month in year_months:
            pos = calculation_dates.get_loc(month)
            current = np.column_stack(
                [full_component[name][:, pos] for name in components]
            )
            relatives = current / base
            terms = (
                previous_aggregate[:, None]
                * shares.to_numpy(dtype=float)[None, :]
                * relatives
            )
            term_paths[:, pos, :] = terms
            aggregate_paths[:, pos] = terms.sum(axis=1)
            weight_year_used[pos] = source_weight_year

        december = pd.Timestamp(year, 12, 1)
        if december in calculation_dates:
            previous_aggregate = aggregate_paths[:, calculation_dates.get_loc(december)]
            previous_december = december

    path_positions = calculation_dates.get_indexer(dates)
    if (path_positions < 0).any():
        raise RuntimeError("Internal path alignment failure.")

    return {
        "level_paths": aggregate_paths[:, path_positions],
        "term_paths": term_paths[:, path_positions, :],
        "path_dates": dates,
        "components": components,
        "weight_year_used": pd.Series(
            weight_year_used[path_positions], index=dates, name="weight_year_used"
        ),
        "calculation_level_paths": aggregate_paths,
        "calculation_term_paths": term_paths,
        "calculation_dates": calculation_dates,
        "anchor_date": anchor_date,
        "anchor_level": float(aggregate_history.loc[anchor_date]),
        "cross_component_dependence": "caller supplied matched draw paths",
        "future_weight_policy": (
            "latest published annual weights carried forward"
            if carry_forward_weights
            else "year-specific published weights required"
        ),
    }



def aggregate_draw_paths_laspeyres(
    component_history: pd.DataFrame,
    component_paths: Mapping[str, np.ndarray],
    path_dates: Sequence[pd.Timestamp],
    annual_weights: pd.DataFrame,
    aggregate_history: pd.Series,
    *,
    components: Sequence[str] = MODEL_AGGREGATE_COMPONENTS,
    carry_forward_weights: bool = True,
) -> dict:
    """Aggregate the six model components into HICP Energy draw by draw."""
    components = list(components)
    if components != list(MODEL_AGGREGATE_COMPONENTS):
        raise ValueError(
            "aggregate_draw_paths_laspeyres is the six-model HICP Energy "
            "wrapper. Use aggregate_component_draw_paths_laspeyres for another "
            "component set."
        )
    model_weights = _model_weight_table_for_chain(annual_weights)
    result = aggregate_component_draw_paths_laspeyres(
        component_history,
        component_paths,
        path_dates,
        model_weights,
        aggregate_history,
        components=components,
        carry_forward_weights=carry_forward_weights,
    )
    result["aggregate"] = "hicp_energy"
    return result



def draw_yoy_and_contributions(
    aggregate_result: Mapping,
    historical_reconstruction: Mapping,
) -> dict:
    """Compute draw-wise YoY inflation and exact additive component contributions.

    ``historical_reconstruction`` should be a deterministic Laspeyres result
    whose ``index`` and ``terms`` reach the month immediately before the first
    aggregate draw path.  This lets 12-month changes cross the forecast origin
    without switching decomposition conventions.
    """
    path_dates = pd.DatetimeIndex(aggregate_result["path_dates"], name="date")
    level_paths = np.asarray(aggregate_result["level_paths"], dtype=float)
    term_paths = np.asarray(aggregate_result["term_paths"], dtype=float)
    components = list(aggregate_result["components"])

    hist_index = pd.Series(historical_reconstruction["index"]).astype(float).sort_index()
    hist_terms = pd.DataFrame(historical_reconstruction["terms"]).astype(float).sort_index()
    if list(hist_terms.columns) != components:
        raise ValueError(
            "Historical and forecast term decompositions use different components."
        )

    n_draws = level_paths.shape[0]
    yoy = np.full_like(level_paths, np.nan, dtype=float)
    contributions = np.full(
        (n_draws, len(path_dates), len(components)), np.nan, dtype=float
    )

    date_to_pos = {pd.Timestamp(date): i for i, date in enumerate(path_dates)}
    for t, date in enumerate(path_dates):
        lag_date = pd.Timestamp(date) - pd.DateOffset(months=12)
        if lag_date in date_to_pos:
            lag_pos = date_to_pos[lag_date]
            lag_level = level_paths[:, lag_pos]
            lag_terms = term_paths[:, lag_pos, :]
        else:
            if lag_date not in hist_index.index or lag_date not in hist_terms.index:
                continue
            lag_level = np.full(n_draws, float(hist_index.loc[lag_date]))
            lag_terms = np.repeat(
                hist_terms.loc[lag_date].to_numpy(dtype=float)[None, :],
                n_draws,
                axis=0,
            )

        yoy[:, t] = 100.0 * (level_paths[:, t] / lag_level - 1.0)
        contributions[:, t, :] = (
            100.0 * (term_paths[:, t, :] - lag_terms) / lag_level[:, None]
        )

    additive = np.nansum(contributions, axis=2)
    mask = np.isfinite(yoy)
    if mask.any():
        error = np.abs(additive[mask] - yoy[mask])
        if len(error) and float(np.nanmax(error)) > 1e-10:
            raise AssertionError(
                "Draw-wise YoY contributions are not exactly additive: "
                f"max error={float(np.nanmax(error)):.3e}."
            )

    return {
        "yoy_paths": yoy,
        "contribution_paths": contributions,
        "path_dates": path_dates,
        "components": components,
        "additivity_error": additive - yoy,
    }


# ---------------------------------------------------------------------------
# Small helpers for unmodelled residual transport fuel
# ---------------------------------------------------------------------------

def carry_last_index_path(
    history: pd.Series,
    dates: Sequence[pd.Timestamp],
    n_draws: int,
) -> np.ndarray:
    """Deterministic zero-inflation path for a tiny unmodelled HICP residual."""
    series = history.astype(float).sort_index().dropna()
    dates = pd.DatetimeIndex(dates)
    if series.empty:
        raise ValueError("Residual HICP history is empty.")
    prior = series.loc[series.index < dates[0]]
    if prior.empty:
        raise ValueError("No residual HICP observation exists before the path.")
    value = float(prior.iloc[-1])
    return np.full((int(n_draws), len(dates)), value, dtype=float)


def summarise_draw_paths(
    paths: np.ndarray,
    dates: Sequence[pd.Timestamp],
    *,
    quantiles: Sequence[float] = (0.05, 0.16, 0.50, 0.84, 0.95),
) -> pd.DataFrame:
    """Return a compact posterior fan table."""
    values = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(dates, name="date")
    if values.ndim != 2 or values.shape[1] != len(dates):
        raise ValueError("paths must have shape (draws, len(dates)).")
    q = np.nanquantile(values, quantiles, axis=0)
    names = [f"q{int(round(100*x)):02d}" for x in quantiles]
    return pd.DataFrame(q.T, index=dates, columns=names)


__all__ = [
    "HOUSEHOLD_COMPONENTS",
    "FUEL_COMPONENTS_POST_2017",
    "CAR_FUELS_COMPONENTS_POST_2017",
    "ENERGY_COMPONENTS_POST_2017",
    "ENERGY_SPECIAL_COMPONENTS",
    "MODEL_AGGREGATE_COMPONENTS",
    "PriceIndexBridge",
    "load_aggregation_inputs",
    "chain_link_laspeyres",
    "reconstruction_error_table",
    "annual_reanchored_reconstruction_errors",
    "historical_reconstruction_suite",
    "assert_reconstruction_suite",
    "yoy_from_index",
    "yoy_contributions_from_terms",
    "fit_price_index_bridge",
    "bridge_diagnostic_table",
    "price_paths_to_hicp_paths",
    "weekly_paths_with_history_to_monthly_mean",
    "independent_draw_pairing",
    "model_energy_weights",
    "historical_model_component_reconstruction",
    "aggregate_component_draw_paths_laspeyres",
    "aggregate_draw_paths_laspeyres",
    "draw_yoy_and_contributions",
    "carry_last_index_path",
    "summarise_draw_paths",
]
