"""Plotting functions for the ECB-style energy BVAR-SV-outlier model.

Forecast figures distinguish observed data, ragged-edge nowcasts and future
forecasts with continuous, explicitly anchored paths.
"""

from __future__ import annotations

from html import escape
from math import ceil
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from IPython.display import HTML
except ImportError:  # Allows the plotting module to import outside Jupyter.
    HTML = None

from energy_bvar_model import (
    chain_acf,
    coefficient_posterior_table,
    key_parameter_draws,
    posterior_outlier_probability,
    posterior_volatility_summary,
    standardized_structural_residuals,
)


def _axes_array(axes):
    return np.atleast_1d(axes).ravel()


_SUPERSCRIPT_TRANSLATION = str.maketrans({
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "-": "⁻", "+": "⁺",
})


def format_number(
    value,
    decimals: int = 2,
    scientific_below: float = 1e-2,
    scientific_above: float = 1e4,
    missing: str = "—",
) -> str:
    """Format numbers compactly for plots and notebook tables.

    Ordinary values use a fixed number of decimals. Very small or very large
    non-zero values use a readable ``mantissa × 10ᵉ`` representation with
    Unicode superscripts, which renders consistently in matplotlib and pandas.
    """
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(number)
    if number == 0.0:
        return f"{0.0:.{decimals}f}"
    absolute = abs(number)
    if absolute < scientific_below or absolute >= scientific_above:
        exponent = int(np.floor(np.log10(absolute)))
        mantissa = number / (10.0 ** exponent)
        exponent_text = str(exponent).translate(_SUPERSCRIPT_TRANSLATION)
        return f"{mantissa:.{decimals}f} × 10{exponent_text}"
    return f"{number:.{decimals}f}"


def format_numeric_table(table, decimals: int = 2) -> pd.DataFrame:
    """Return a display copy with compact formatting for numeric columns."""
    frame = table.to_frame() if isinstance(table, pd.Series) else table.copy()
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: format_number(value, decimals=decimals))
    return frame


def style_numeric_table(
    table,
    decimals: int = 2,
    caption: str | None = None,
):
    """Return a dependency-free HTML table for notebook display.

    ``pandas.DataFrame.style`` requires the optional Jinja2 package. The
    project tables should remain usable in a minimal scientific environment,
    so this helper deliberately builds HTML with ``DataFrame.to_html`` instead.
    Numeric values keep the compact two-decimal/scientific formatting used
    throughout the notebook.

    When IPython is unavailable, the already formatted DataFrame is returned.
    """
    original = table.to_frame() if isinstance(table, pd.Series) else table.copy()
    frame = format_numeric_table(original, decimals=decimals)

    if HTML is None:
        return frame

    numeric_positions = [
        position + 1
        for position, column in enumerate(original.columns)
        if pd.api.types.is_numeric_dtype(original[column])
    ]
    text_positions = [
        position + 1
        for position, column in enumerate(original.columns)
        if not pd.api.types.is_numeric_dtype(original[column])
    ]

    alignment_rules = []
    for position in numeric_positions:
        alignment_rules.append(
            f".energy-table td:nth-child({position}) {{ text-align: right; "
            "font-variant-numeric: tabular-nums; white-space: nowrap; }}"
        )
    for position in text_positions:
        alignment_rules.append(
            f".energy-table td:nth-child({position}) {{ text-align: left; }}"
        )

    table_html = frame.to_html(
        border=0,
        classes=["energy-table"],
        justify="left",
        escape=True,
        na_rep="—",
    )
    caption_html = (
        f'<div class="energy-table-caption">{escape(str(caption))}</div>'
        if caption
        else ""
    )
    css = """
    <style>
      .energy-table-wrap {
        margin: 0.35rem 0 1.05rem 0;
        max-width: 100%;
        overflow-x: auto;
      }
      .energy-table-caption {
        font-weight: 650;
        font-size: 13px;
        margin: 0 0 7px 1px;
      }
      .energy-table {
        border-collapse: collapse;
        width: auto;
        min-width: 360px;
        font-size: 12px;
        line-height: 1.35;
        background: transparent;
      }
      .energy-table thead th {
        text-align: left;
        font-weight: 650;
        padding: 7px 10px;
        border-top: 1px solid #cbd5e1;
        border-bottom: 1px solid #94a3b8;
        white-space: nowrap;
      }
      .energy-table tbody th {
        text-align: left;
        font-weight: 500;
        padding: 6px 10px;
        border-bottom: 1px solid #e5e7eb;
        white-space: nowrap;
      }
      .energy-table tbody td {
        padding: 6px 10px;
        border-bottom: 1px solid #e5e7eb;
        vertical-align: top;
      }
      .energy-table tbody tr:nth-child(even) {
        background: rgba(148, 163, 184, 0.08);
      }
      .energy-table tbody tr:hover {
        background: rgba(148, 163, 184, 0.16);
      }
    """ + "\n".join(alignment_rules) + "\n</style>"

    return HTML(
        css
        + '<div class="energy-table-wrap">'
        + caption_html
        + table_html
        + "</div>"
    )


def plot_data_levels(
    levels: pd.DataFrame,
    last_obs: int | None = None,
    units: Mapping[str, str] | None = None,
):
    data = levels if last_obs is None else levels.iloc[-last_obs:]
    n = data.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    for ax, column in zip(_axes_array(axes), data.columns):
        ax.plot(data.index, data[column], linewidth=1.2)
        ax.set_title(column)
        ax.set_ylabel((units or {}).get(column, "EUR/MWh"))
        ax.grid(alpha=0.25)
    _axes_array(axes)[-1].set_xlabel("Date")
    fig.suptitle("Energy-model input series in levels")
    fig.tight_layout()
    return fig


def plot_absolute_differences(
    differences: pd.DataFrame,
    last_obs: int | None = None,
    units: Mapping[str, str] | None = None,
):
    data = differences if last_obs is None else differences.iloc[-last_obs:]
    n = data.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    for ax, column in zip(_axes_array(axes), data.columns):
        ax.plot(data.index, data[column], linewidth=1.0)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(rf"Absolute change: {column}")
        ax.set_ylabel((units or {}).get(column, "EUR/MWh"))
        ax.grid(alpha=0.25)
    _axes_array(axes)[-1].set_xlabel("Date")
    fig.suptitle(r"Model transformation: $\Delta P_t=P_t-P_{t-1}$")
    fig.tight_layout()
    return fig

def plot_minnesota_prior_by_equation(prior: Mapping, variables: Sequence[str], p: int):
    """Plot only the informative lag-decay profile, one figure per equation.

    The prior mean is zero for every lag coefficient, so a separate mean panel
    would be visually redundant. The constant is reported in the accompanying
    hyperparameter table; the figure focuses on the lag-specific standard
    deviations that determine Minnesota shrinkage.
    """
    variables = list(variables)
    n = len(variables)
    if n == 0 or p < 1:
        raise ValueError("variables must be non-empty and p must be positive.")

    prior_variances = np.asarray(prior["V0_diag"], dtype=float)
    if prior_variances.size % n:
        raise ValueError("The coefficient-prior vector is incompatible with the number of equations.")
    k = prior_variances.size // n
    if k < 1 + n * p:
        raise ValueError("The coefficient prior does not contain the requested VAR lag block.")
    sd = np.sqrt(prior_variances.reshape(k, n, order="F"))
    config = dict(prior.get("config", {}))
    lambda1 = float(config.get("lambda1", np.nan))
    lambda2 = float(config.get("lambda2", np.nan))
    lambda3 = float(config.get("lambda3", np.nan))
    lags = np.arange(1, p + 1)

    line_colors = ["#2563eb", "#e67e22", "#2a9d8f", "#8b5cf6"]
    figures = {}

    for eq, equation in enumerate(variables):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))

        for reg, regressor in enumerate(variables):
            rows = [1 + (lag - 1) * n + reg for lag in lags]
            values = sd[rows, eq]
            own = reg == eq
            label = f"{regressor} — {'own lags' if own else 'cross lags'}"
            ax.plot(
                lags,
                values,
                color=line_colors[reg % len(line_colors)],
                linestyle="-" if own else "--",
                marker="o" if own else "s",
                linewidth=2.2 if own else 1.8,
                markersize=5.2,
                label=label,
                zorder=3,
            )

            # Label the first and final lag only: enough information without
            # obscuring the decay profile.
            for index in sorted({0, len(lags) - 1}):
                is_first = index == 0
                ax.annotate(
                    format_number(values[index]),
                    (lags[index], values[index]),
                    xytext=(0, 8) if is_first else (-6, -5),
                    textcoords="offset points",
                    ha="center" if is_first else "right",
                    va="bottom" if is_first else "top",
                    fontsize=8.5,
                    color=line_colors[reg % len(line_colors)],
                )

        ax.set_yscale("log")
        ax.set_xticks(lags)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Prior standard deviation (log scale)")
        fig.suptitle(
            f"Minnesota lag-decay profile — {equation} equation",
            fontsize=13,
            y=0.985,
        )
        ax.grid(axis="y", which="both", color="#cbd5e1", alpha=0.50, linewidth=0.7)
        ax.grid(axis="x", which="major", color="#e2e8f0", alpha=0.55, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=1, loc="upper right")

        if np.all(np.isfinite([lambda1, lambda2, lambda3])):
            subtitle = (
                rf"White-noise mean; $\lambda_1={lambda1:.2f}$, "
                rf"$\lambda_2={lambda2:.2f}$, $\lambda_3={lambda3:.2f}$"
            )
            ax.set_title(
                subtitle,
                loc="left",
                pad=10,
                fontsize=9.2,
                color="#475569",
            )

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        figures[equation] = fig

    return figures

def plot_minnesota_prior(prior: Mapping, variables: Sequence[str], p: int):
    """Backward-compatible wrapper returning the equation-specific figures."""
    return plot_minnesota_prior_by_equation(prior, variables, p)

def plot_phi_and_outlier_priors(prior: Mapping, max_phi: float | None = None):
    phi_shape = float(prior["phi_shape"])
    phi_scale = float(prior["phi_scale"])
    alpha = float(prior["outlier_alpha"])
    beta = float(prior["outlier_beta"])

    try:
        from scipy.stats import beta as beta_distribution
        from scipy.stats import invgamma
    except ImportError as exc:
        raise ImportError("scipy is required for prior density plots.") from exc

    if max_phi is None:
        max_phi = float(invgamma.ppf(0.995, a=phi_shape, scale=phi_scale))
    phi_grid = np.linspace(max(max_phi / 1_000, 1e-8), max_phi, 500)
    p_upper = min(0.20, float(beta_distribution.ppf(0.999, alpha, beta)))
    p_grid = np.linspace(1e-6, max(p_upper, 0.03), 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    axes[0].fill_between(
        phi_grid,
        invgamma.pdf(phi_grid, a=phi_shape, scale=phi_scale),
        color="#c6dbef",
        alpha=0.85,
    )
    axes[0].plot(
        phi_grid,
        invgamma.pdf(phi_grid, a=phi_shape, scale=phi_scale),
        color="#2171b5",
        linewidth=1.5,
    )
    axes[0].axvline(
        prior["phi_prior_mean"], color="#cb181d", linestyle="--", linewidth=1.1,
        label=f"mean = {format_number(prior['phi_prior_mean'])}",
    )
    axes[0].set_title(r"Prior on the volatility innovation variance $\phi_j$")
    axes[0].set_xlabel(r"$\phi_j$")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.20)
    axes[0].legend(frameon=False)

    beta_density = beta_distribution.pdf(p_grid, alpha, beta)
    axes[1].fill_between(p_grid, beta_density, color="#fee6ce", alpha=0.90)
    axes[1].plot(p_grid, beta_density, color="#e6550d", linewidth=1.5)
    prior_mean = alpha / (alpha + beta)
    prior_frequency = str(prior.get("frequency", "monthly")).lower()
    prior_period = "weeks" if prior_frequency == "weekly" else "months"
    axes[1].axvline(
        prior_mean, color="#cb181d", linestyle="--", linewidth=1.1,
        label=(
            f"mean = {format_number(prior_mean)} "
            f"(1 in {1/prior_mean:.0f} {prior_period})"
        ),
    )
    axes[1].set_title(r"Prior on the outlier probability $p_j$")
    axes[1].set_xlabel(r"$p_j$")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.20)
    axes[1].legend(frameon=False)
    fig.suptitle("Priors for persistent and transitory volatility")
    fig.tight_layout()
    return fig

def plot_shrinkage_comparison(result: Mapping, equations: Sequence[str] | None = None):
    table = coefficient_posterior_table(result)
    variables = list(result["variables"])
    equations = variables if equations is None else list(equations)
    mask = np.array([index.split(":", 1)[0] in equations for index in table.index])
    data = table.loc[mask].copy()
    if len(equations) == 1:
        prefix = f"{equations[0]}:"
        data.index = [str(index).split(":", 1)[1].strip() for index in data.index]
        title = f"Prior and posterior coefficients — {equations[0]} equation"
    else:
        title = "Prior and posterior coefficients"
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.30 * len(data) + 1.8)))
    ax.hlines(
        y, data["q16"], data["q84"],
        linewidth=4.0, color="#9ecae1", label="68% posterior interval",
    )
    ax.scatter(
        data["posterior_median"], y,
        marker="o", color="#08519c", s=28, label="posterior median",
    )
    ax.scatter(
        data["prior_mean"], y,
        marker="x", color="#d95f0e", s=32, label="prior mean",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(data.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.35", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Coefficient value")
    ax.grid(axis="x", alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

def plot_coefficient_heatmap(
    result: Mapping,
    statistic: str = "median",
    equation: str | None = None,
):
    B = np.asarray(result["B"])
    if statistic == "median":
        centre = np.median(B, axis=0)
    elif statistic == "mean":
        centre = B.mean(axis=0)
    else:
        raise ValueError("statistic must be 'median' or 'mean'.")
    variables = list(result["variables"])
    n = len(variables)
    p = result["p"]
    equations = variables if equation is None else [equation]
    unknown = [name for name in equations if name not in variables]
    if unknown:
        raise ValueError(f"Unknown equations: {unknown}")
    rows = ["constant"] + [
        f"{name} lag {lag}" for lag in range(1, p + 1) for name in variables
    ] + list(result.get("exog_names", result.get("prep", {}).get("exog_names", [])))
    figures = {}
    for name in equations:
        eq = variables.index(name)
        matrix = centre[:, eq][:, None]
        vmax = max(float(np.max(np.abs(matrix))), 1e-8)
        fig, ax = plt.subplots(figsize=(5.5, max(5.5, 0.34 * len(rows) + 1.5)))
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for row, value in enumerate(matrix[:, 0]):
            text_color = "white" if abs(value) > 0.60 * vmax else "black"
            ax.text(0, row, format_number(value), ha="center", va="center", fontsize=8, color=text_color)
        ax.set_xticks([0])
        ax.set_xticklabels([name])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(f"Posterior {statistic} — {name} equation")
        fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04, label="Coefficient")
        fig.tight_layout()
        figures[name] = fig
    return figures[equations[0]] if equation is not None else figures

def plot_posterior_volatility(result: Mapping, include_outliers: bool = False):
    summary = posterior_volatility_summary(result, include_outliers=include_outliers)
    dates = summary["dates"]
    variables = summary["variables"]
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    for j, ax in enumerate(_axes_array(axes)):
        ax.fill_between(dates, summary["lower"][:, j], summary["upper"][:, j], alpha=0.25)
        ax.plot(dates, summary["median"][:, j], linewidth=1.2)
        ax.set_title(variables[j])
        ax.set_ylabel("Conditional sd")
        ax.grid(alpha=0.25)
    label = "total scale including outliers" if include_outliers else "persistent stochastic volatility"
    fig.suptitle(f"Posterior {label}")
    fig.tight_layout()
    return fig


def plot_outlier_probabilities(result: Mapping, threshold: float = 0.5):
    probabilities = posterior_outlier_probability(result)
    n = probabilities.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    for ax, column in zip(_axes_array(axes), probabilities.columns):
        ax.plot(probabilities.index, probabilities[column], linewidth=1.1)
        ax.axhline(threshold, linestyle="--", linewidth=0.9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Posterior probability")
        ax.set_title(column)
        ax.grid(alpha=0.25)
    fig.suptitle("Posterior probability of a transitory outlier")
    fig.tight_layout()
    return fig


def plot_vol_decomposition(dates, series, *, q=(0.16, 0.84), figsize=(13, 9)):
    r"""Aligned, interpretable view of the SV + outlier volatility decomposition.

    Parameters
    ----------
    dates : array-like of length T
        Observation dates.
    series : dict[name] -> dict
        For each variable name, a dictionary with keys:

        - ``persistent`` : array (n_draws, T) of :math:`\sqrt{\lambda_t}` draws.
        - ``total``      : array (n_draws, T) of :math:`o_t\sqrt{\lambda_t}` draws.
        - ``prob``       : array (T,) of posterior :math:`P(o_t \ge 2)`.

    q : tuple(float, float), default=(0.16, 0.84)
        Credible-band quantiles.
    figsize : tuple, default=(13, 9)
        Figure size.

    Notes
    -----
    The outlier multiplier is computed draw by draw,

    .. math::

        o_t = rac{o_t\sqrt{\lambda_t}}{\sqrt{\lambda_t}},

    and only then summarised. This is the correct decomposition because, in
    general, :math:`E[o_t\sqrt{\lambda_t}] 
eq E[o_t] E[\sqrt{\lambda_t}]`.
    """
    names = list(series)
    ncol = len(names)
    fig, ax = plt.subplots(
        3,
        ncol,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.8]},
    )

    BLUE = "#1f6feb"
    GREY = "#8a94a6"
    RED = "#d1495b"
    lo_q, hi_q = q

    for c, name in enumerate(names):
        sv = np.asarray(series[name]["persistent"], dtype=float)
        tot = np.asarray(series[name]["total"], dtype=float)
        prob = np.asarray(series[name]["prob"], dtype=float)
        mult = np.divide(tot, sv, out=np.ones_like(tot), where=sv > 0.0)

        sv_m = np.median(sv, axis=0)
        sv_lo = np.quantile(sv, lo_q, axis=0)
        sv_hi = np.quantile(sv, hi_q, axis=0)
        tot_m = np.median(tot, axis=0)
        mult_m = np.median(mult, axis=0)
        mult_hi = np.quantile(mult, hi_q, axis=0)

        # Row 0: persistent vs total conditional scale.
        a = ax[0, c]
        a.fill_between(dates, sv_lo, sv_hi, color=BLUE, alpha=0.18, lw=0)
        a.plot(dates, sv_m, color=BLUE, lw=1.8, label=r"persistent SV  $\sqrt{\lambda}$")
        a.plot(dates, tot_m, color=GREY, lw=1.0, label=r"total scale  $o\,\sqrt{\lambda}$")
        m = prob >= 0.5
        if np.any(m):
            date_array = np.asarray(dates)
            a.scatter(
                date_array[m],
                tot_m[m],
                s=14,
                color=RED,
                zorder=5,
                label="outlier periods (P>0.5)",
            )
        a.set_yscale("log")
        a.set_title(name, fontsize=11)
        a.grid(True, which="both", alpha=0.25)
        if c == 0:
            a.set_ylabel("conditional SD (log)")
        if c == ncol - 1:
            a.legend(fontsize=8, loc="upper left", framealpha=0.9)

        # Row 1: outlier multiplier.
        a = ax[1, c]
        a.axhline(1.0, color=GREY, lw=0.8, ls="--")
        a.fill_between(dates, mult_m, mult_hi, color=BLUE, alpha=0.18, lw=0)
        a.plot(dates, mult_m, color=BLUE, lw=1.1)
        a.grid(True, alpha=0.25)
        if c == 0:
            a.set_ylabel("outlier multiplier\n$o_t$ = total / persistent")

        # Row 2: outlier probability.
        a = ax[2, c]
        a.axhline(0.5, color=GREY, lw=0.8, ls="--")
        a.plot(dates, prob, color=BLUE, lw=1.0)
        a.set_ylim(-0.02, 1.02)
        a.grid(True, alpha=0.25)
        if c == 0:
            a.set_ylabel("P(outlier)")
        a.xaxis.set_major_locator(mdates.YearLocator(4))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Volatility decomposition — one aligned view per series", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig



def plot_volatility_decomposition(result: Mapping, q=(0.16, 0.84), figsize=(13, 9)):
    r"""Convenience wrapper around :func:`plot_vol_decomposition` for model output.

    .. math::

        	ext{total SD}_{j,t} = o_{j,t}\sqrt{\lambda_{j,t}},
        \qquad
        	ext{persistent SD}_{j,t} = \sqrt{\lambda_{j,t}}.
    """
    log_h = np.asarray(result["log_variance"], dtype=float)[:, 1:, :]
    persistent = np.exp(0.5 * np.clip(log_h, -745.0, 700.0))
    outlier_scales = np.asarray(result["outlier_scales"], dtype=float)
    total = persistent * outlier_scales
    prob = np.asarray(result["outlier_indicators"], dtype=float).mean(axis=0)
    dates = pd.DatetimeIndex(result["prep"]["dates"])
    variables = list(result["variables"])

    series = {
        name: {
            "persistent": persistent[:, :, j],
            "total": total[:, :, j],
            "prob": prob[:, j],
        }
        for j, name in enumerate(variables)
    }
    return plot_vol_decomposition(dates, series, q=q, figsize=figsize)


def plot_trace_grid(result: Mapping, quantities: Mapping[str, np.ndarray] | None = None, ncols: int = 2):
    quantities = key_parameter_draws(result) if quantities is None else dict(quantities)
    n = len(quantities)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 2.2 * nrows), squeeze=False)
    for ax, (name, draws) in zip(axes.ravel(), quantities.items()):
        ax.plot(np.asarray(draws), linewidth=0.45)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Retained iteration")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("MCMC trace plots")
    fig.tight_layout()
    return fig


def plot_chain_acf_grid(
    result: Mapping,
    quantities: Mapping[str, np.ndarray] | None = None,
    nlags: int = 60,
    ncols: int = 2,
):
    quantities = key_parameter_draws(result) if quantities is None else dict(quantities)
    n = len(quantities)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 2.4 * nrows), squeeze=False)
    for ax, (name, draws) in zip(axes.ravel(), quantities.items()):
        acf = chain_acf(np.asarray(draws), nlags=nlags)
        ax.vlines(np.arange(len(acf)), 0.0, acf, linewidth=1.0)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("MCMC lag")
        ax.set_ylabel("Autocorrelation")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Autocorrelation of retained MCMC draws")
    fig.tight_layout()
    return fig


def plot_spectral_radius(result: Mapping):
    radii = np.asarray(result["spectral_radius"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(radii, bins=40, alpha=0.8)
    ax.axvline(1.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Maximum companion-root modulus")
    ax.set_ylabel("Retained draws")
    ax.set_title(
        "VAR stability across retained draws\n"
        f"proposal rejection rate = {result['B_instability_rejection_rate']:.2%}"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _fan_quantiles(paths: np.ndarray):
    return np.percentile(paths, [5, 16, 50, 84, 95], axis=0)


# Forecast colours are deliberately stable across every figure.
_OBSERVED_COLOR = "0.20"
_NOWCAST_DARK = "#d94801"
_NOWCAST_MID = "#fdae6b"
_NOWCAST_LIGHT = "#fee6ce"
_FORECAST_DARK = "#08519c"
_FORECAST_MID = "#6baed6"
_FORECAST_LIGHT = "#c6dbef"
_CONDITIONAL_DARK = "#238b45"
_CONDITIONAL_MID = "#74c476"
_CONDITIONED_COLOR = "#31a354"

def _forecast_frequency(forecast: Mapping) -> str:
    frequency = str(forecast.get("frequency", "monthly")).lower()
    if frequency not in {"monthly", "weekly"}:
        raise ValueError(f"Unknown forecast frequency {frequency!r}.")
    return frequency


def _period_name_from_frequency(frequency: str) -> str:
    return "month" if frequency == "monthly" else "week"


def _horizon_unit_from_result(result: Mapping, requested: str | None) -> str:
    if requested is not None:
        return str(requested)
    frequency = str(result.get("frequency", "monthly")).lower()
    return "weeks" if frequency == "weekly" else "months"


def _validate_forecast_variable(
    forecast: Mapping,
    historical_levels: pd.DataFrame,
    variable: str,
) -> tuple[list[str], int, pd.Series]:
    variables = list(forecast["variables"])
    if variable not in variables:
        raise ValueError(
            f"Unknown variable {variable!r}. Available variables are {variables}."
        )
    if variable not in historical_levels.columns:
        raise ValueError(f"{variable!r} is not present in historical_levels.")
    history = historical_levels[variable].sort_index()
    if history.dropna().empty:
        raise ValueError(f"No observed historical value is available for {variable!r}.")
    return variables, variables.index(variable), history


def _prepend_anchor(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    anchor_date: pd.Timestamp,
    anchor_values,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Prepend one observed or draw-specific anchor to a simulated path."""
    values = np.asarray(anchor_values, dtype=float)
    if values.ndim == 0:
        values = np.full(paths.shape[0], float(values), dtype=float)
    values = values.reshape(-1)
    if values.size != paths.shape[0]:
        raise ValueError("Anchor values must contain one value per posterior draw.")
    anchored_paths = np.concatenate([values[:, None], paths], axis=1)
    anchored_dates = pd.DatetimeIndex([pd.Timestamp(anchor_date)]).append(dates)
    return anchored_paths, anchored_dates

def _forecast_segments(
    forecast: Mapping,
    historical_levels: pd.DataFrame,
    variable: str,
):
    """Return observed, nowcast and forecast segments with continuous anchors."""
    variables, j, history = _validate_forecast_variable(
        forecast, historical_levels, variable
    )
    frequency = _forecast_frequency(forecast)
    all_dates = pd.DatetimeIndex(forecast["path_dates"])
    all_paths = np.asarray(forecast["level_paths"], dtype=float)[:, :, j]
    if all_paths.ndim != 2 or all_paths.shape[1] != len(all_dates):
        raise ValueError(
            "level_paths must have shape (n_draws, n_path_dates, n_variables)."
        )

    tail_length = int(forecast.get("tail_length", 0))
    if tail_length < 0 or tail_length > len(all_dates):
        raise ValueError("Invalid tail_length in forecast object.")

    tail_dates = all_dates[:tail_length]
    tail_paths = all_paths[:, :tail_length]
    future_dates = all_dates[tail_length:]
    future_paths = all_paths[:, tail_length:]
    actual = history.dropna()

    nowcast_dates = pd.DatetimeIndex([])
    nowcast_paths = np.empty((all_paths.shape[0], 0), dtype=float)
    nowcast_anchor_date = None
    if tail_length:
        missing_tail = history.reindex(tail_dates).isna().to_numpy()
        missing_positions = np.flatnonzero(missing_tail)
        if missing_positions.size:
            first_missing = int(missing_positions[0])
            nowcast_dates = tail_dates[first_missing:]
            nowcast_paths = tail_paths[:, first_missing:]
            candidates = actual.loc[actual.index < nowcast_dates[0]]
            if candidates.empty:
                raise ValueError("No observed anchor exists before the nowcast segment.")
            nowcast_anchor_date = pd.Timestamp(candidates.index[-1])
            nowcast_paths, nowcast_dates = _prepend_anchor(
                nowcast_paths,
                nowcast_dates,
                nowcast_anchor_date,
                float(candidates.iloc[-1]),
            )

    conditioned = history.reindex(tail_dates).dropna()

    forecast_dates = pd.DatetimeIndex([])
    anchored_future_paths = np.empty((all_paths.shape[0], 0), dtype=float)
    forecast_anchor_date = pd.Timestamp(forecast["last_calendar_date"])
    if len(future_dates):
        if tail_length:
            anchor_values = tail_paths[:, -1]
        else:
            candidates = actual.loc[actual.index <= forecast_anchor_date]
            if candidates.empty:
                raise ValueError("No level is available at the forecast origin.")
            anchor_values = float(candidates.iloc[-1])
            forecast_anchor_date = pd.Timestamp(candidates.index[-1])
        anchored_future_paths, forecast_dates = _prepend_anchor(
            future_paths,
            future_dates,
            forecast_anchor_date,
            anchor_values,
        )

    return {
        "variables": variables,
        "frequency": frequency,
        "period_name": _period_name_from_frequency(frequency),
        "history": history,
        "actual": actual,
        "tail_dates": tail_dates,
        "conditioned": conditioned,
        "nowcast_dates": nowcast_dates,
        "nowcast_paths": nowcast_paths,
        "nowcast_anchor_date": nowcast_anchor_date,
        "forecast_dates": forecast_dates,
        "forecast_paths": anchored_future_paths,
        "forecast_anchor_date": forecast_anchor_date,
        "last_calendar_date": pd.Timestamp(forecast["last_calendar_date"]),
        "balanced_end": pd.Timestamp(forecast["balanced_end"]),
    }


def _plot_fan_segment(
    ax,
    dates: pd.DatetimeIndex,
    paths: np.ndarray,
    dark: str,
    mid: str,
    light: str,
    label_prefix: str,
    median_linestyle: str = "-",
    interval: str = "both",
    zorder: int = 2,
):
    if paths.size == 0 or len(dates) == 0:
        return
    q = _fan_quantiles(paths)
    if interval in {"both", "90"}:
        ax.fill_between(
            dates,
            q[0],
            q[4],
            color=light,
            alpha=0.72,
            label=f"{label_prefix} 90% interval",
            zorder=zorder,
        )
    if interval in {"both", "68"}:
        ax.fill_between(
            dates,
            q[1],
            q[3],
            color=mid,
            alpha=0.48,
            label=f"{label_prefix} 68% interval",
            zorder=zorder + 1,
        )
    ax.plot(
        dates,
        q[2],
        color=dark,
        linewidth=1.8,
        linestyle=median_linestyle,
        label=f"{label_prefix} median",
        zorder=zorder + 2,
    )


def _plot_observed_history(ax, actual: pd.Series, last_obs: int):
    if last_obs < 1:
        raise ValueError("last_obs must be at least 1.")
    shown = actual.iloc[-last_obs:]
    ax.plot(
        shown.index,
        shown.to_numpy(dtype=float),
        color=_OBSERVED_COLOR,
        linewidth=1.45,
        label="Observed",
        zorder=6,
    )


def _missing_method_title_suffix(obj: Mapping) -> str:
    """Surface approximate missing-data treatment on forecast figures."""
    if bool(obj.get("missing_treatment_exact", True)):
        return ""
    method = str(obj.get("missing_data_method", "linear")).upper()
    return f"  [{method} missing-data approximation]"


def plot_forecast_fan(
    forecast: Mapping,
    historical_levels: pd.DataFrame,
    variable: str,
    last_obs: int = 72,
    title: str | None = None,
    ylabel: str = "Level",
):
    """Plot observed data, orange ragged-edge nowcast and blue forecast."""
    segments = _forecast_segments(forecast, historical_levels, variable)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _plot_observed_history(ax, segments["actual"], last_obs)
    _plot_fan_segment(
        ax,
        segments["nowcast_dates"],
        segments["nowcast_paths"],
        _NOWCAST_DARK,
        _NOWCAST_MID,
        _NOWCAST_LIGHT,
        "Nowcast",
    )
    _plot_fan_segment(
        ax,
        segments["forecast_dates"],
        segments["forecast_paths"],
        _FORECAST_DARK,
        _FORECAST_MID,
        _FORECAST_LIGHT,
        "Forecast",
    )

    if not segments["conditioned"].empty:
        ax.scatter(
            segments["conditioned"].index,
            segments["conditioned"].to_numpy(dtype=float),
            color=_CONDITIONED_COLOR,
            edgecolor="white",
            linewidth=0.5,
            s=34,
            label="Released/conditioned observation",
            zorder=8,
        )

    last_observed_date = pd.Timestamp(segments["actual"].index[-1])
    ax.axvline(
        last_observed_date,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label=f'Last observed {segments["period_name"]}',
    )
    if segments["last_calendar_date"] != last_observed_date:
        ax.axvline(
            segments["last_calendar_date"],
            color="0.55",
            linestyle=":",
            linewidth=1.0,
            label="Forecast origin",
        )

    base_title = title or f"Ragged-edge nowcast and forecast — {variable}"
    ax.set_title(base_title + _missing_method_title_suffix(forecast))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_nowcast_fan(
    forecast: Mapping,
    historical_levels: pd.DataFrame,
    variable: str,
    last_obs: int = 48,
    title: str | None = None,
    ylabel: str = "Level",
):
    """Plot the ragged edge, with nowcasts in orange and released data in green."""
    segments = _forecast_segments(forecast, historical_levels, variable)
    tail_dates = segments["tail_dates"]
    if len(tail_dates) == 0:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.text(0.5, 0.5, "No ragged-edge months to nowcast", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(11, 4.5))
    _plot_observed_history(ax, segments["actual"], last_obs)
    _plot_fan_segment(
        ax,
        segments["nowcast_dates"],
        segments["nowcast_paths"],
        _NOWCAST_DARK,
        _NOWCAST_MID,
        _NOWCAST_LIGHT,
        "Nowcast",
    )

    if not segments["conditioned"].empty:
        ax.plot(
            segments["conditioned"].index,
            segments["conditioned"].to_numpy(dtype=float),
            color=_CONDITIONED_COLOR,
            linewidth=1.6,
            marker="o",
            markersize=4.5,
            label="Released/conditioned observation",
            zorder=8,
        )

    ax.axvline(
        segments["balanced_end"],
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label="Balanced-sample end",
    )
    base_title = title or f"Ragged-edge nowcast — {variable}"
    ax.set_title(base_title + _missing_method_title_suffix(forecast))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_short_forecast_fan(
    forecast: Mapping,
    historical_levels: pd.DataFrame,
    variable: str,
    last_obs: int = 36,
    title: str | None = None,
    ylabel: str = "Level",
):
    """Plot the short forecast while retaining the orange nowcast bridge."""
    return plot_forecast_fan(
        forecast=forecast,
        historical_levels=historical_levels,
        variable=variable,
        last_obs=last_obs,
        title=title or f"Short-horizon nowcast and forecast — {variable}",
        ylabel=ylabel,
    )


def plot_forecast_comparison(
    unconditional: Mapping,
    conditional: Mapping,
    historical_levels: pd.DataFrame,
    variable: str,
    last_obs: int = 72,
    title: str | None = None,
    ylabel: str = "Level",
):
    """Compare forecast scenarios after one clearly identified nowcast segment."""
    if list(unconditional["variables"]) != list(conditional["variables"]):
        raise ValueError("Forecast objects use different variable orderings.")

    seg_u = _forecast_segments(unconditional, historical_levels, variable)
    seg_c = _forecast_segments(conditional, historical_levels, variable)

    fig, ax = plt.subplots(figsize=(11, 4.9))
    _plot_observed_history(ax, seg_u["actual"], last_obs)
    _plot_fan_segment(
        ax,
        seg_u["nowcast_dates"],
        seg_u["nowcast_paths"],
        _NOWCAST_DARK,
        _NOWCAST_MID,
        _NOWCAST_LIGHT,
        "Nowcast",
    )
    _plot_fan_segment(
        ax,
        seg_u["forecast_dates"],
        seg_u["forecast_paths"],
        _FORECAST_DARK,
        _FORECAST_MID,
        _FORECAST_LIGHT,
        "Unconditional forecast",
        interval="68",
    )
    _plot_fan_segment(
        ax,
        seg_c["forecast_dates"],
        seg_c["forecast_paths"],
        _CONDITIONAL_DARK,
        _CONDITIONAL_MID,
        _CONDITIONAL_MID,
        "Conditional forecast",
        median_linestyle="--",
        interval="68",
        zorder=5,
    )

    if not seg_u["conditioned"].empty:
        ax.scatter(
            seg_u["conditioned"].index,
            seg_u["conditioned"].to_numpy(dtype=float),
            color=_CONDITIONED_COLOR,
            edgecolor="white",
            linewidth=0.5,
            s=34,
            label="Released/conditioned observation",
            zorder=9,
        )

    last_observed_date = pd.Timestamp(seg_u["actual"].index[-1])
    ax.axvline(
        last_observed_date,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label=f'Last observed {seg_u["period_name"]}',
    )
    if seg_u["last_calendar_date"] != last_observed_date:
        ax.axvline(
            seg_u["last_calendar_date"],
            color="0.55",
            linestyle=":",
            linewidth=1.0,
            label="Forecast origin",
        )

    base_title = title or f"Unconditional versus conditional forecast — {variable}"
    suffix = _missing_method_title_suffix(unconditional)
    if _missing_method_title_suffix(conditional) != suffix:
        suffix = "  [mixed missing-data treatments]"
    ax.set_title(base_title + suffix)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_hicp_forecast_fan(
    hicp_forecast: Mapping,
    kind: str = "yoy",
    last_obs: int = 72,
    component_label: str = "gas",
):
    """Plot HICP history, orange nowcast and blue forecast without a gap."""
    if kind == "yoy":
        paths = np.asarray(hicp_forecast["hicp_yoy_paths"], dtype=float)
        ylabel = "Year-on-year percent"
        title = f"HICP {component_label} inflation — nowcast and forecast"
        actual = 100.0 * (
            hicp_forecast["actual_hicp"]
            / hicp_forecast["actual_hicp"].shift(12)
            - 1.0
        )
    elif kind == "level":
        paths = np.asarray(hicp_forecast["hicp_level_paths"], dtype=float)
        ylabel = "Index"
        title = f"HICP {component_label} index — nowcast and forecast"
        actual = hicp_forecast["actual_hicp"]
    else:
        raise ValueError("kind must be 'yoy' or 'level'.")

    dates = pd.DatetimeIndex(hicp_forecast["path_dates"])
    if paths.ndim != 2 or paths.shape[1] != len(dates):
        raise ValueError("HICP paths and path_dates have incompatible dimensions.")
    tail_length = int(hicp_forecast.get("tail_length", 0))
    tail_dates = dates[:tail_length]
    tail_paths = paths[:, :tail_length]
    future_dates = dates[tail_length:]
    future_paths = paths[:, tail_length:]
    actual = actual.sort_index()
    actual_clean = actual.dropna()
    if actual_clean.empty:
        raise ValueError("No observed HICP history is available.")

    nowcast_dates = pd.DatetimeIndex([])
    nowcast_paths = np.empty((paths.shape[0], 0), dtype=float)
    if tail_length:
        missing_tail = actual.reindex(tail_dates).isna().to_numpy()
        missing_positions = np.flatnonzero(missing_tail)
        if missing_positions.size:
            first_missing = int(missing_positions[0])
            raw_dates = tail_dates[first_missing:]
            raw_paths = tail_paths[:, first_missing:]
            candidates = actual_clean.loc[actual_clean.index < raw_dates[0]]
            if candidates.empty:
                raise ValueError("No observed HICP anchor exists before the nowcast.")
            nowcast_paths, nowcast_dates = _prepend_anchor(
                raw_paths,
                raw_dates,
                pd.Timestamp(candidates.index[-1]),
                float(candidates.iloc[-1]),
            )

    forecast_dates = pd.DatetimeIndex([])
    anchored_future = np.empty((paths.shape[0], 0), dtype=float)
    if len(future_dates):
        if tail_length:
            anchor_date = pd.Timestamp(tail_dates[-1])
            anchor_values = tail_paths[:, -1]
        else:
            anchor_date = pd.Timestamp(future_dates[0] - pd.offsets.MonthBegin(1))
            candidates = actual_clean.loc[actual_clean.index <= anchor_date]
            if candidates.empty:
                raise ValueError("No HICP level is available at the forecast origin.")
            anchor_date = pd.Timestamp(candidates.index[-1])
            anchor_values = float(candidates.iloc[-1])
        anchored_future, forecast_dates = _prepend_anchor(
            future_paths, future_dates, anchor_date, anchor_values
        )

    conditioned = actual.reindex(tail_dates).dropna()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _plot_observed_history(ax, actual_clean, last_obs)
    _plot_fan_segment(
        ax,
        nowcast_dates,
        nowcast_paths,
        _NOWCAST_DARK,
        _NOWCAST_MID,
        _NOWCAST_LIGHT,
        "Nowcast",
    )
    _plot_fan_segment(
        ax,
        forecast_dates,
        anchored_future,
        _FORECAST_DARK,
        _FORECAST_MID,
        _FORECAST_LIGHT,
        "Forecast",
    )
    if not conditioned.empty:
        ax.scatter(
            conditioned.index,
            conditioned.to_numpy(dtype=float),
            color=_CONDITIONED_COLOR,
            edgecolor="white",
            linewidth=0.5,
            s=34,
            label="Released observation",
            zorder=8,
        )

    last_observed_date = pd.Timestamp(actual_clean.index[-1])
    ax.axvline(
        last_observed_date,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label="Last observed month",
    )
    if len(future_dates):
        forecast_origin = pd.Timestamp(future_dates[0] - pd.offsets.MonthBegin(1))
        if forecast_origin != last_observed_date:
            ax.axvline(
                forecast_origin,
                color="0.55",
                linestyle=":",
                linewidth=1.0,
                label="Forecast origin",
            )

    ax.set_title(title + _missing_method_title_suffix(hicp_forecast))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig

def _validated_tax_inflation_arrays(tax_forecast: Mapping):
    pre = np.asarray(tax_forecast["pre_tax_inflation_paths"], dtype=float)
    post = np.asarray(tax_forecast["post_tax_inflation_paths"], dtype=float)
    baseline_post = np.asarray(
        tax_forecast.get("baseline_post_tax_inflation_paths", post),
        dtype=float,
    )
    dates = pd.DatetimeIndex(tax_forecast["path_dates"])
    expected = (pre.ndim == 2 and pre.shape[1] == len(dates))
    if not expected or post.shape != pre.shape or baseline_post.shape != pre.shape:
        raise ValueError(
            "Tax inflation paths must all have shape (n_draws, len(path_dates))."
        )
    return pre, post, baseline_post, dates


def tax_inflation_spread_table(
    tax_forecast: Mapping,
    *,
    future_only: bool = True,
) -> pd.DataFrame:
    """Posterior tax wedge: post-tax inflation minus pre-tax inflation.

    The headline statistic is the median of the draw-wise wedge, not the
    difference between the two marginal medians. Both are retained so the
    vertical separation of plotted median curves can still be inspected.
    """
    pre, post, _, dates = _validated_tax_inflation_arrays(tax_forecast)
    wedge = post - pre
    pre_median = np.nanmedian(pre, axis=0)
    post_median = np.nanmedian(post, axis=0)
    frame = pd.DataFrame(
        {
            "pre_tax_median": pre_median,
            "post_tax_median": post_median,
            "spread_of_medians_pp": post_median - pre_median,
            "tax_wedge_median_pp": np.nanmedian(wedge, axis=0),
            "tax_wedge_q16_pp": np.nanquantile(wedge, 0.16, axis=0),
            "tax_wedge_q84_pp": np.nanquantile(wedge, 0.84, axis=0),
        },
        index=dates,
    )
    frame.index.name = "date"
    if future_only:
        frame = frame.reindex(pd.DatetimeIndex(tax_forecast["future_dates"]))
    return frame


def tax_scenario_impact_table(
    tax_forecast: Mapping,
    *,
    future_only: bool = True,
) -> pd.DataFrame:
    """Posterior impact of the tax scenario on full/post-tax inflation."""
    _, post, baseline_post, dates = _validated_tax_inflation_arrays(tax_forecast)
    impact = post - baseline_post
    frame = pd.DataFrame(
        {
            "baseline_post_tax_median": np.nanmedian(baseline_post, axis=0),
            "scenario_post_tax_median": np.nanmedian(post, axis=0),
            "scenario_impact_median_pp": np.nanmedian(impact, axis=0),
            "scenario_impact_q16_pp": np.nanquantile(impact, 0.16, axis=0),
            "scenario_impact_q84_pp": np.nanquantile(impact, 0.84, axis=0),
        },
        index=dates,
    )
    frame.index.name = "date"
    if future_only:
        frame = frame.reindex(pd.DatetimeIndex(tax_forecast["future_dates"]))
    return frame


def _annotation_dates(
    dates: pd.DatetimeIndex,
    future_dates: pd.DatetimeIndex,
    annotations: str | Sequence[pd.Timestamp] | None,
) -> pd.DatetimeIndex:
    if annotations is None or annotations == "none":
        return pd.DatetimeIndex([])
    if isinstance(annotations, str):
        key = annotations.lower()
        if key == "last":
            return future_dates[-1:]
        if key == "all":
            return future_dates
        raise ValueError("annotations must be 'last', 'all', None, or a date sequence.")
    selected = pd.DatetimeIndex(annotations)
    missing = selected.difference(dates)
    if len(missing):
        raise ValueError(
            "Annotation dates are absent from the forecast path: "
            f"{missing.strftime('%Y-%m-%d').tolist()}"
        )
    return selected


def _annotate_drawwise_effect(
    ax,
    dates: pd.DatetimeIndex,
    lower_curve: np.ndarray,
    upper_curve: np.ndarray,
    effect_median: np.ndarray,
    effect_q16: np.ndarray,
    effect_q84: np.ndarray,
    annotation_dates: pd.DatetimeIndex,
    decimals: int,
):
    positions = {pd.Timestamp(date): i for i, date in enumerate(dates)}
    for date in annotation_dates:
        i = positions[pd.Timestamp(date)]
        y0 = float(lower_curve[i])
        y1 = float(upper_curve[i])
        effect = float(effect_median[i])
        lo = float(effect_q16[i])
        hi = float(effect_q84[i])
        if not np.all(np.isfinite([y0, y1, effect, lo, hi])):
            continue
        ax.annotate(
            "",
            xy=(date, y1),
            xytext=(date, y0),
            arrowprops={"arrowstyle": "<->", "color": "0.25", "lw": 1.0},
            zorder=7,
        )
        ax.annotate(
            f"{effect:+.{decimals}f} pp\n68% [{lo:+.{decimals}f}, {hi:+.{decimals}f}]",
            xy=(date, 0.5 * (y0 + y1)),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": "0.70", "alpha": 0.92},
            zorder=8,
        )


def plot_tax_inflation_comparison(
    tax_forecast: Mapping,
    *,
    last_obs: int = 72,
    component_label: str = "energy",
    show_intervals: bool = True,
    spread_annotations: str | Sequence[pd.Timestamp] | None = "last",
    decimals: int = 2,
    title: str | None = None,
):
    """Plot pre-tax versus realised post-tax inflation from matched draws."""
    pre, post, _, dates = _validated_tax_inflation_arrays(tax_forecast)
    future_dates = pd.DatetimeIndex(tax_forecast["future_dates"])
    pre_q = np.nanquantile(pre, [0.16, 0.50, 0.84], axis=0)
    post_q = np.nanquantile(post, [0.16, 0.50, 0.84], axis=0)
    wedge = post - pre
    wedge_q = np.nanquantile(wedge, [0.16, 0.50, 0.84], axis=0)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    actual_pre = tax_forecast.get("actual_pre_tax_inflation")
    actual_post = tax_forecast.get("actual_post_tax_inflation")
    if actual_pre is not None:
        series = pd.Series(actual_pre).dropna().iloc[-last_obs:]
        ax.plot(series.index, series.to_numpy(dtype=float), color=_FORECAST_DARK,
                linewidth=1.15, alpha=0.60, label="Observed pre-tax inflation")
    if actual_post is not None:
        series = pd.Series(actual_post).dropna().iloc[-last_obs:]
        ax.plot(series.index, series.to_numpy(dtype=float), color=_CONDITIONAL_DARK,
                linewidth=1.15, alpha=0.60, label="Observed post-tax inflation")

    if show_intervals:
        ax.fill_between(dates, pre_q[0], pre_q[2], color=_FORECAST_MID, alpha=0.20,
                        label="Pre-tax 68% interval")
        ax.fill_between(dates, post_q[0], post_q[2], color=_CONDITIONAL_MID, alpha=0.18,
                        label="Post-tax 68% interval")
    ax.plot(dates, pre_q[1], color=_FORECAST_DARK, linewidth=2.0,
            label="Pre-tax median", zorder=5)
    ax.plot(dates, post_q[1], color=_CONDITIONAL_DARK, linewidth=2.0,
            label="Post-tax median", zorder=5)

    if len(future_dates):
        future_mask = dates.isin(future_dates)
        ax.fill_between(dates[future_mask], pre_q[1][future_mask], post_q[1][future_mask],
                        color="0.55", alpha=0.12, label="Median-line separation", zorder=1)
        ax.axvline(future_dates[0], color="0.50", linestyle=":", linewidth=1.0,
                   label="First forecast period")

    selected = _annotation_dates(dates, future_dates, spread_annotations)
    _annotate_drawwise_effect(
        ax, dates, pre_q[1], post_q[1], wedge_q[1], wedge_q[0], wedge_q[2],
        selected, decimals,
    )

    unit = str(tax_forecast.get("inflation_unit", "percent"))
    base_title = title or f"Pre-tax versus post-tax inflation — {component_label}"
    ax.set_title(base_title + _missing_method_title_suffix(tax_forecast))
    ax.set_ylabel(unit)
    ax.axhline(0.0, color="0.55", linewidth=0.8)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_tax_scenario_impact(
    tax_forecast: Mapping,
    *,
    last_obs: int = 72,
    component_label: str = "energy",
    show_intervals: bool = True,
    impact_annotations: str | Sequence[pd.Timestamp] | None = "last",
    decimals: int = 2,
    title: str | None = None,
):
    """Compare baseline full inflation with the conditioned tax scenario."""
    _, post, baseline_post, dates = _validated_tax_inflation_arrays(tax_forecast)
    future_dates = pd.DatetimeIndex(tax_forecast["future_dates"])
    scenario = dict(tax_forecast.get("tax_scenario", {}))
    if not scenario.get("active", False):
        raise ValueError("tax_forecast contains no active tax scenario.")

    baseline_q = np.nanquantile(baseline_post, [0.16, 0.50, 0.84], axis=0)
    scenario_q = np.nanquantile(post, [0.16, 0.50, 0.84], axis=0)
    impact = post - baseline_post
    impact_q = np.nanquantile(impact, [0.16, 0.50, 0.84], axis=0)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    actual_post = tax_forecast.get("actual_post_tax_inflation")
    if actual_post is not None:
        series = pd.Series(actual_post).dropna().iloc[-last_obs:]
        ax.plot(series.index, series.to_numpy(dtype=float), color=_OBSERVED_COLOR,
                linewidth=1.25, label="Observed full inflation")

    if show_intervals:
        ax.fill_between(dates, baseline_q[0], baseline_q[2], color=_FORECAST_MID,
                        alpha=0.18, label="Baseline 68% interval")
        ax.fill_between(dates, scenario_q[0], scenario_q[2], color=_CONDITIONAL_MID,
                        alpha=0.16, label="Tax scenario 68% interval")
    ax.plot(dates, baseline_q[1], color=_FORECAST_DARK, linewidth=2.0,
            label="Baseline full-inflation median")
    ax.plot(dates, scenario_q[1], color=_CONDITIONAL_DARK, linewidth=2.0,
            linestyle="--", label="Tax-scenario full-inflation median")

    start = scenario.get("effective_start_date")
    if start is not None:
        ax.axvline(pd.Timestamp(start), color="0.45", linestyle=":", linewidth=1.0,
                   label="Tax scenario start")

    selected = _annotation_dates(dates, future_dates, impact_annotations)
    _annotate_drawwise_effect(
        ax, dates, baseline_q[1], scenario_q[1], impact_q[1], impact_q[0], impact_q[2],
        selected, decimals,
    )

    unit = str(tax_forecast.get("inflation_unit", "percent"))
    base_title = title or f"Tax assumption impact on full inflation — {component_label}"
    ax.set_title(base_title + _missing_method_title_suffix(tax_forecast))
    ax.set_ylabel(unit)
    ax.axhline(0.0, color="0.55", linewidth=0.8)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig



def _tax_context_raw_series(tax_context: Mapping) -> tuple[pd.Series, pd.Series, str]:
    """Return raw VAT/excise observations and a source label."""
    if "data" in tax_context:
        data = tax_context["data"]
        if not isinstance(data, pd.DataFrame):
            raise TypeError("tax_context['data'] must be a DataFrame.")
        required = {"vat_percent", "excise"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"Weekly tax context is missing {sorted(missing)}.")
        return (
            data["vat_percent"].dropna().astype(float).sort_index(),
            data["excise"].dropna().astype(float).sort_index(),
            "European Commission Weekly Oil Bulletin",
        )
    if "vat_percent" not in tax_context or "excise" not in tax_context:
        raise KeyError("tax_context must contain VAT and excise history.")
    return (
        pd.Series(tax_context["vat_percent"]).dropna().astype(float).sort_index(),
        pd.Series(tax_context["excise"]).dropna().astype(float).sort_index(),
        "Eurostat",
    )


def _tax_history_frame(tax_forecast: Mapping) -> pd.DataFrame:
    history = tax_forecast.get("tax_history")
    if history is None:
        raise KeyError(
            "tax_forecast does not contain 'tax_history'. Re-run tax re-attribution "
            "with the updated component adapter."
        )
    history = pd.DataFrame(history).copy()
    required = {"applied_vat_percent", "applied_excise"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"tax_history is missing {sorted(missing)}.")
    history.index = pd.DatetimeIndex(history.index, name="date")
    return history.sort_index()


def _validate_tax_history_junction(
    tax_history: pd.DataFrame,
    tax_path: pd.DataFrame,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-10,
) -> None:
    """Guard against VAT/excise unit or dating discontinuities at the join."""
    overlap = tax_history.index.intersection(tax_path.index)
    if len(overlap):
        history_date = path_date = overlap[-1]
    else:
        if tax_history.empty or tax_path.empty:
            return
        history_date = tax_history.index[-1]
        path_date = tax_path.index[0]
        if history_date >= path_date:
            return
    checks = (
        ("VAT", "applied_vat_percent", "baseline_vat_percent"),
        ("excise", "applied_excise", "baseline_excise"),
    )
    for label, history_col, path_col in checks:
        left = float(tax_history.loc[history_date, history_col])
        right = float(tax_path.loc[path_date, path_col])
        if np.isfinite(left) and np.isfinite(right) and not np.isclose(
            left, right, rtol=rtol, atol=atol
        ):
            raise AssertionError(
                f"{label} history/baseline mismatch across the tax-path junction "
                f"({history_date.date()} -> {path_date.date()}): "
                f"historical applied={left:.12g}, baseline={right:.12g}. "
                "Check tax dating and units before interpreting the scenario."
            )


def _scenario_display_path(
    tax_path: pd.DataFrame,
    baseline_col: str,
    scenario_col: str,
    future_dates: pd.DatetimeIndex,
) -> pd.Series | None:
    baseline = tax_path[baseline_col].reindex(future_dates).astype(float)
    scenario = tax_path[scenario_col].reindex(future_dates).astype(float)
    if np.allclose(baseline.to_numpy(), scenario.to_numpy(), rtol=0.0, atol=1e-12):
        return None
    different = ~np.isclose(
        baseline.to_numpy(), scenario.to_numpy(), rtol=0.0, atol=1e-12
    )
    positions = np.flatnonzero(different)
    start = max(int(positions[0]) - 1, 0)
    return scenario.iloc[start:]


def plot_tax_assumptions(
    tax_context: Mapping,
    tax_forecast: Mapping,
    *,
    component_label: str = "energy",
    lookback_years: int = 5,
    title: str | None = None,
):
    """Plot historical VAT/excise and the future baseline/scenario assumptions.

    Monthly gas/electricity charts deliberately show two historical objects:
    semiannual source publications as markers and the monthly path actually
    applied by the model.  The overlay is a visual validation of the six-month
    expansion rule.  Weekly fuel taxes are already observed at model frequency,
    so a single historical line with markers is sufficient.
    """
    if lookback_years < 1:
        raise ValueError("lookback_years must be positive.")
    tax_path = pd.DataFrame(tax_forecast["tax_path"]).copy()
    tax_path.index = pd.DatetimeIndex(tax_path.index, name="date")
    tax_history = _tax_history_frame(tax_forecast)
    _validate_tax_history_junction(tax_history, tax_path)

    raw_vat, raw_excise, source_label = _tax_context_raw_series(tax_context)
    future_dates = pd.DatetimeIndex(tax_forecast["future_dates"], name="date")
    if len(future_dates) == 0:
        raise ValueError("Tax-assumption plot requires at least one future date.")
    scenario = dict(tax_forecast.get("tax_scenario", {}))
    scenario_start = (
        pd.Timestamp(scenario["effective_start_date"])
        if scenario.get("effective_start_date") is not None
        else future_dates[0]
    )
    forecast_origin = pd.Timestamp(
        tax_forecast.get("forecast_origin", future_dates[0])
    )
    frequency = str(tax_forecast.get("frequency", "monthly")).lower()

    future_tax_path = tax_path.reindex(future_dates)
    if future_tax_path.isna().any().any():
        raise ValueError("Future tax path is incomplete on one or more model dates.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    specs = (
        (
            axes[0], raw_vat, "applied_vat_percent", "baseline_vat_percent",
            "scenario_vat_percent", "VAT", "%",
        ),
        (
            axes[1], raw_excise, "applied_excise", "baseline_excise",
            "scenario_excise", "Excise", str(tax_forecast.get("excise_unit", "source unit")),
        ),
    )

    for ax, raw, history_col, baseline_col, scenario_col, name, unit in specs:
        history = tax_history[history_col].dropna()
        history = history.loc[history.index <= forecast_origin]
        if history.empty:
            raise ValueError(f"No historical applied {name} values are available.")

        if frequency == "monthly":
            ax.step(
                history.index,
                history.to_numpy(dtype=float),
                where="post",
                color="0.35",
                linewidth=1.6,
                label=f"{name} applied by model (monthly)",
            )
            raw_window = raw.loc[raw.index <= forecast_origin]
            ax.plot(
                raw_window.index,
                raw_window.to_numpy(dtype=float),
                linestyle="none",
                marker="o",
                markersize=4.2,
                color="0.10",
                label=f"{source_label} publications (semiannual)",
            )
        else:
            ax.plot(
                history.index,
                history.to_numpy(dtype=float),
                marker="o",
                markersize=2.8,
                markevery=max(len(history) // 35, 1),
                color="0.30",
                linewidth=1.3,
                label=f"Historical {source_label} observations (weekly)",
            )

        # Join the last applied historical value to the first future baseline.
        baseline_future = future_tax_path[baseline_col].astype(float)
        bridge_index = pd.DatetimeIndex([history.index[-1], *baseline_future.index])
        bridge_values = np.r_[float(history.iloc[-1]), baseline_future.to_numpy(dtype=float)]
        ax.plot(
            bridge_index,
            bridge_values,
            linestyle="--",
            linewidth=1.7,
            color=_FORECAST_DARK,
            label="Baseline, carried forward",
        )

        scenario_path = _scenario_display_path(
            tax_path, baseline_col, scenario_col, future_dates
        )
        if scenario_path is not None:
            ax.plot(
                scenario_path.index,
                scenario_path.to_numpy(dtype=float),
                linewidth=2.2,
                color=_CONDITIONAL_DARK,
                label="Scenario",
            )

        ax.axvline(
            forecast_origin,
            linestyle="--",
            linewidth=0.9,
            color="0.55",
            label="Forecast origin",
        )
        if scenario.get("active", False):
            ax.axvline(
                scenario_start,
                linestyle=":",
                linewidth=1.2,
                color="0.30",
                label="Tax scenario start",
            )
        ax.set_ylabel(f"{name} ({unit})")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    start = scenario_start - pd.DateOffset(years=int(lookback_years))
    end_offset = pd.DateOffset(months=2) if frequency == "monthly" else pd.DateOffset(weeks=8)
    axes[1].set_xlim(start, future_dates[-1] + end_offset)
    axes[0].set_title(
        (title or f"Tax assumptions — {component_label}")
        + _missing_method_title_suffix(tax_forecast)
    )
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    return fig


def _validated_tax_level_arrays(
    tax_forecast: Mapping,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    required = ("post_tax_level_paths", "baseline_post_tax_level_paths", "path_dates")
    missing = [name for name in required if name not in tax_forecast]
    if missing:
        raise KeyError(f"Tax forecast is missing {missing}.")
    scenario = np.asarray(tax_forecast["post_tax_level_paths"], dtype=float)
    baseline = np.asarray(tax_forecast["baseline_post_tax_level_paths"], dtype=float)
    dates = pd.DatetimeIndex(tax_forecast["path_dates"], name="date")
    if scenario.shape != baseline.shape or scenario.ndim != 2:
        raise ValueError("Baseline/scenario post-tax level paths must have matching 2D shapes.")
    if scenario.shape[1] != len(dates):
        raise ValueError("Post-tax level paths and path_dates have incompatible lengths.")
    if np.any(np.isclose(baseline, 0.0, rtol=0.0, atol=1e-14)):
        raise ValueError("Baseline post-tax level contains zero; relative level effect is undefined.")
    return scenario, baseline, dates


def tax_scenario_effect_summary_table(
    tax_forecast: Mapping,
    *,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Summarise the tax scenario's price-level and YoY inflation effects."""
    scenario_level, baseline_level, dates = _validated_tax_level_arrays(tax_forecast)
    _, scenario_yoy, baseline_yoy, yoy_dates = _validated_tax_inflation_arrays(tax_forecast)
    if not dates.equals(yoy_dates):
        raise ValueError("Level and inflation tax paths use different dates.")
    future_dates = pd.DatetimeIndex(tax_forecast["future_dates"], name="date")
    frequency = str(tax_forecast.get("frequency", "monthly")).lower()
    defaults = (1, 3, 6, 12) if frequency == "monthly" else (1, 4, 13, 26, 52)
    requested = list(defaults if horizons is None else horizons)
    requested = [int(h) for h in requested if int(h) >= 1 and int(h) <= len(future_dates)]
    if len(future_dates) and len(future_dates) not in requested:
        requested.append(len(future_dates))
    requested = sorted(set(requested))
    if not requested:
        raise ValueError("No requested tax-effect horizon lies inside the forecast path.")

    level_effect = 100.0 * (scenario_level / baseline_level - 1.0)
    yoy_effect = scenario_yoy - baseline_yoy
    positions = {pd.Timestamp(date): i for i, date in enumerate(dates)}
    period_word = "month" if frequency == "monthly" else "week"
    rows = []
    for horizon in requested:
        date = pd.Timestamp(future_dates[horizon - 1])
        i = positions[date]
        level_q = np.nanquantile(level_effect[:, i], [0.16, 0.50, 0.84])
        yoy_q = np.nanquantile(yoy_effect[:, i], [0.16, 0.50, 0.84])
        label = f"{horizon} {period_word}" + ("" if horizon == 1 else "s")
        if horizon == len(future_dates) and horizon not in defaults:
            label = f"end ({label})"
        rows.append(
            {
                "horizon": label,
                "periods_ahead": horizon,
                "target_date": date,
                "level_effect_q16_pct": float(level_q[0]),
                "level_effect_median_pct": float(level_q[1]),
                "level_effect_q84_pct": float(level_q[2]),
                "yoy_effect_q16_pp": float(yoy_q[0]),
                "yoy_effect_median_pp": float(yoy_q[1]),
                "yoy_effect_q84_pp": float(yoy_q[2]),
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def plot_tax_level_and_inflation_impact(
    tax_forecast: Mapping,
    *,
    component_label: str = "energy",
    lookback_years: int = 1,
    title: str | None = None,
    deterministic_tolerance: float = 1e-10,
):
    """Show the permanent price-level effect beside the temporary YoY effect.

    The level effect is computed draw by draw from the actual baseline and
    scenario post-tax paths, rather than from the VAT ratio alone.  Therefore a
    VAT-only scenario naturally has zero posterior width, while an excise
    scenario inherits uncertainty from the pre-tax price path.
    """
    scenario = dict(tax_forecast.get("tax_scenario", {}))
    if not scenario.get("active", False):
        raise ValueError("tax_forecast contains no active tax scenario.")
    scenario_level, baseline_level, dates = _validated_tax_level_arrays(tax_forecast)
    _, scenario_yoy, baseline_yoy, yoy_dates = _validated_tax_inflation_arrays(tax_forecast)
    if not dates.equals(yoy_dates):
        raise ValueError("Level and inflation tax paths use different dates.")

    level_effect = 100.0 * (scenario_level / baseline_level - 1.0)
    yoy_effect = scenario_yoy - baseline_yoy
    level_q = np.nanquantile(level_effect, [0.16, 0.50, 0.84], axis=0)
    yoy_q = np.nanquantile(yoy_effect, [0.16, 0.50, 0.84], axis=0)
    scenario_start = pd.Timestamp(scenario["effective_start_date"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharex=True)

    level_width = np.nanmax(level_q[2] - level_q[0])
    if np.isfinite(level_width) and level_width > deterministic_tolerance:
        axes[0].fill_between(
            dates, level_q[0], level_q[2], color=_CONDITIONAL_MID, alpha=0.22,
            label="68% posterior interval",
        )
    axes[0].plot(
        dates, level_q[1], color=_CONDITIONAL_DARK, linewidth=2.1,
        label="Median level effect",
    )
    if not np.isfinite(level_width) or level_width <= deterministic_tolerance:
        axes[0].text(
            0.02, 0.94,
            "Deterministic conditional on tax path",
            transform=axes[0].transAxes,
            ha="left", va="top", fontsize=8.5, color="0.35",
        )
    axes[0].set_title("Price-level impact")
    axes[0].set_ylabel("Scenario − baseline (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    axes[1].fill_between(
        dates, yoy_q[0], yoy_q[2], color=_CONDITIONAL_MID, alpha=0.22,
        label="68% posterior interval",
    )
    axes[1].plot(
        dates, yoy_q[1], color=_CONDITIONAL_DARK, linewidth=2.1,
        label="Median YoY effect",
    )
    axes[1].set_title("YoY inflation impact — base effect")
    axes[1].set_ylabel("Scenario − baseline (percentage points)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    for ax in axes:
        ax.axhline(0.0, color="0.55", linewidth=0.8)
        ax.axvline(
            scenario_start, linestyle=":", linewidth=1.2, color="0.35",
            label="Tax scenario start",
        )
        ax.grid(alpha=0.22)
        ax.set_xlim(
            scenario_start - pd.DateOffset(years=int(lookback_years)),
            dates[-1],
        )
        ax.set_xlabel("Date")

    fig.suptitle(
        (title or f"Tax-scenario effect — {component_label}")
        + _missing_method_title_suffix(tax_forecast),
        y=1.02,
    )
    fig.tight_layout()
    return fig

def plot_standardized_residuals(result: Mapping):
    eps = standardized_structural_residuals(result)
    n = eps.shape[1]
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.0 * n), squeeze=False)
    for j, column in enumerate(eps.columns):
        axes[j, 0].plot(eps.index, eps[column], linewidth=0.9)
        axes[j, 0].axhline(0.0, linewidth=0.8)
        axes[j, 0].set_title(f"Standardized structural residual: {column}")
        axes[j, 0].grid(alpha=0.25)
        axes[j, 1].hist(eps[column], bins=35, density=True, alpha=0.8)
        axes[j, 1].set_title(f"Distribution: {column}")
        axes[j, 1].grid(axis="y", alpha=0.25)
    fig.suptitle("Residual diagnostics after SV and outlier adjustment")
    fig.tight_layout()
    return fig


def plot_posterior_predictive_check(summary: pd.DataFrame, equation: str | None = None):
    data = summary.copy()
    if equation is not None:
        prefix = f"{equation}:"
        data = data.loc[[str(index).startswith(prefix) for index in data.index]].copy()
        data.index = [str(index).split(":", 1)[1].strip() for index in data.index]
        title = f"Posterior predictive checks — {equation} equation"
    else:
        title = "Posterior predictive checks on transformed data"
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.70 * len(data) + 1.3)))
    ax.hlines(
        y, data["replicated_q05"], data["replicated_q95"],
        linewidth=6.0, color="#c6dbef", label="90% replicated interval",
    )
    ax.scatter(data["replicated_median"], y, marker="o", color="#08519c", label="replicated median")
    ax.scatter(data["observed"], y, marker="x", color="#d95f0e", s=45, label="observed")
    ax.set_yticks(y)
    ax.set_yticklabels(data.index)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Structural-analysis plots
# -----------------------------------------------------------------------------


def plot_gas_irf(
    irf_result: Mapping,
    cumulative: bool = True,
    credible_intervals: Sequence[float] = (0.05, 0.16, 0.84, 0.95),
    title: str | None = None,
    units: Mapping[str, str] | None = None,
    horizon_unit: str | None = None,
):
    """Plot posterior IRFs for every response-shock pair."""
    horizon_unit = _horizon_unit_from_result(irf_result, horizon_unit)
    key = "cumulative_level_irfs" if cumulative else "change_irfs"
    draws = np.asarray(irf_result[key], dtype=float)
    variables = list(irf_result["variables"])
    shocks = list(irf_result["shock_names"])
    horizons = np.asarray(irf_result["horizons"])
    n_response, n_shock = len(variables), len(shocks)
    q05, q16, q84, q95 = np.quantile(draws, credible_intervals, axis=0)
    median = np.quantile(draws, 0.50, axis=0)

    fig, axes = plt.subplots(
        n_response,
        n_shock,
        figsize=(5.0 * n_shock, 3.3 * n_response),
        squeeze=False,
        sharex=True,
    )
    for i, response in enumerate(variables):
        response_unit = (units or {}).get(response, "response units")
        for j, shock in enumerate(shocks):
            ax = axes[i, j]
            ax.fill_between(horizons, q05[:, i, j], q95[:, i, j], alpha=0.15, label="90% interval")
            ax.fill_between(horizons, q16[:, i, j], q84[:, i, j], alpha=0.30, label="68% interval")
            ax.plot(horizons, median[:, i, j], linewidth=1.7, label="Posterior median")
            ax.axhline(0.0, linewidth=0.8, color="0.35")
            ax.set_title(f"{response} ← {shock}")
            ax.grid(alpha=0.22)
            if i == n_response - 1:
                ax.set_xlabel(f"{horizon_unit.capitalize()} after shock")
            if j == 0:
                ax.set_ylabel(
                    f"Cumulative {response_unit}"
                    if cumulative
                    else f"Change ({response_unit})"
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    identification = irf_result.get("identification", "structural")
    default_title = (
        f"{'Cumulative level' if cumulative else 'Change'} impulse responses "
        f"— {identification.replace('_', ' ')} identification"
    )
    fig.suptitle(title or default_title, y=1.01)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig

def plot_gas_fevd(
    fevd_result: Mapping,
    responses: Sequence[str] | None = None,
    title: str | None = None,
    horizon_unit: str | None = None,
):
    """Plot posterior-median FEVD shares as stacked areas by response variable."""
    horizon_unit = _horizon_unit_from_result(fevd_result, horizon_unit)
    draws = np.asarray(fevd_result["fevd_draws"], dtype=float)
    variables = list(fevd_result["variables"])
    shocks = list(fevd_result["shock_names"])
    horizons = np.asarray(fevd_result["horizons"])
    median = np.quantile(draws, 0.50, axis=0)
    if responses is None:
        responses = variables
    responses = list(responses)
    unknown = [name for name in responses if name not in variables]
    if unknown:
        raise ValueError(f"Unknown responses: {unknown}")

    fig, axes = plt.subplots(len(responses), 1, figsize=(10, 3.6 * len(responses)), squeeze=False, sharex=True)
    for row, response in enumerate(responses):
        i = variables.index(response)
        ax = axes[row, 0]
        ax.stackplot(horizons, *[median[:, i, j] for j in range(len(shocks))], labels=shocks, alpha=0.80)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Variance share")
        ax.set_title(f"Forecast-error variance of {response}")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, ncol=min(3, len(shocks)), loc="upper right")
    axes[-1, 0].set_xlabel(f"Forecast horizon ({horizon_unit})")
    default_title = f"Forecast error variance decomposition — {fevd_result.get('identification', 'structural').replace('_', ' ')}"
    fig.suptitle(title or default_title, y=1.01)
    fig.tight_layout()
    return fig


def plot_gas_historical_decomposition(
    hd_result: Mapping,
    response: str,
    last_obs: int | None = None,
    title: str | None = None,
    unit: str = "EUR/MWh",
):
    """Plot posterior-mean historical contributions, preserving exact additivity."""
    variables = list(hd_result["variables"])
    if response not in variables:
        raise ValueError(f"Unknown response {response!r}.")
    i = variables.index(response)
    dates = pd.DatetimeIndex(hd_result["dates"])
    contributions = np.asarray(hd_result["contributions"], dtype=float).mean(axis=0)[:, i, :]
    base = np.asarray(hd_result["base"], dtype=float).mean(axis=0)[:, i]
    observed = np.asarray(hd_result["observed"], dtype=float)[:, i]
    names = list(hd_result["component_names"])
    if last_obs is not None:
        sl = slice(max(0, len(dates) - int(last_obs)), None)
        dates = dates[sl]
        contributions = contributions[sl]
        base = base[sl]
        observed = observed[sl]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    positive_bottom = np.zeros(len(dates))
    negative_bottom = np.zeros(len(dates))
    for j, name in enumerate(names):
        values = contributions[:, j]
        positive = np.where(values > 0, values, 0.0)
        negative = np.where(values < 0, values, 0.0)
        ax.fill_between(dates, positive_bottom, positive_bottom + positive, alpha=0.65, label=name)
        ax.fill_between(dates, negative_bottom, negative_bottom + negative, alpha=0.65)
        positive_bottom += positive
        negative_bottom += negative
    ax.plot(dates, observed, linewidth=1.5, color="0.10", label="Observed monthly change")
    ax.plot(dates, base, linewidth=1.0, linestyle="--", color="0.40", label="Base / initial conditions")
    ax.axhline(0.0, linewidth=0.8, color="0.35")
    ax.set_ylabel(f"Period change ({unit})")
    ax.set_title(title or f"Historical decomposition — {response}")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig

__all__ = [
    "plot_data_levels",
    "plot_absolute_differences",
    "plot_minnesota_prior",
    "plot_minnesota_prior_by_equation",
    "plot_phi_and_outlier_priors",
    "plot_shrinkage_comparison",
    "plot_coefficient_heatmap",
    "plot_posterior_volatility",
    "plot_vol_decomposition",
    "plot_volatility_decomposition",
    "plot_outlier_probabilities",
    "plot_trace_grid",
    "plot_chain_acf_grid",
    "plot_spectral_radius",
    "plot_forecast_fan",
    "plot_nowcast_fan",
    "plot_short_forecast_fan",
    "plot_forecast_comparison",
    "plot_hicp_forecast_fan",
    "tax_inflation_spread_table",
    "tax_scenario_impact_table",
    "tax_scenario_effect_summary_table",
    "plot_tax_assumptions",
    "plot_tax_level_and_inflation_impact",
    "plot_tax_inflation_comparison",
    "plot_tax_scenario_impact",
    "plot_standardized_residuals",
    "plot_posterior_predictive_check",
    "plot_gas_irf",
    "plot_gas_fevd",
    "plot_gas_historical_decomposition",
    "plot_impulse_responses",
    "plot_fevd",
    "plot_historical_decomposition",
]


# Generic names used by all component notebooks.
plot_impulse_responses = plot_gas_irf
plot_fevd = plot_gas_fevd
plot_historical_decomposition = plot_gas_historical_decomposition
