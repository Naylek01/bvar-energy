"""Shared readable-results tables for the Inflation Dashboard.

Presentation contract
---------------------
This module never reads result files, scans registries, calls Haver, invokes an
API, or computes a model.  It only summarises already-materialised/frozen
objects that are also consumed by the dashboard charts.

Display horizons
----------------
M+1 / M+2 / M+3 are always prioritised.  M+6 and M+12 are included only when
the caller-selected/displayed horizon reaches them or, where no explicit
selector exists, when the frozen object itself reaches them.

Weekly Energy models are mapped to calendar-month targets using the closest
available weekly observation, so a weekly horizon is never mislabeled as a
monthly period number.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from dash import dash_table


_TABLE_STYLE_CELL = {
    "fontFamily": "Inter, Segoe UI, sans-serif",
    "fontSize": "11px",
    "padding": "7px 9px",
    "textAlign": "right",
    "border": "none",
    "borderBottom": "1px solid #EEF2F6",
    "whiteSpace": "nowrap",
}
_TABLE_STYLE_HEADER = {
    "fontWeight": 700,
    "fontSize": "10px",
    "textTransform": "uppercase",
    "letterSpacing": ".035em",
    "backgroundColor": "#F8FAFC",
    "color": "#667085",
    "border": "none",
    "borderBottom": "1px solid #D9E0E8",
    "padding": "8px 9px",
    "textAlign": "right",
}


def readable_table(table_id: str, columns: Sequence[Mapping[str, Any]], *, page_size: int = 10):
    """Return the dashboard's compact, read-first DataTable shell."""
    first = columns[0]["id"] if columns else None
    conditional = []
    if first:
        conditional.append({
            "if": {"column_id": first},
            "textAlign": "left",
            "fontWeight": 650,
            "color": "#111827",
        })
    return dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
        id=table_id,
        columns=list(columns),
        data=[],
        page_size=int(page_size),
        page_action="native",
        sort_action="none",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_cell=_TABLE_STYLE_CELL,
        style_header=_TABLE_STYLE_HEADER,
        style_data_conditional=conditional,
    )


def fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}{suffix}"


def fmt_signed(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:+.{digits}f}{suffix}"


def interval(low: Any, high: Any, *, digits: int = 2, suffix: str = "") -> str:
    a, b = _num(low), _num(high)
    if a is None or b is None:
        return "—"
    return f"[{a:.{digits}f}, {b:.{digits}f}]{suffix}"


def interval_signed(low: Any, high: Any, *, digits: int = 2, suffix: str = "") -> str:
    a, b = _num(low), _num(high)
    if a is None or b is None:
        return "—"
    return f"[{a:+.{digits}f}, {b:+.{digits}f}]{suffix}"


def _num(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def priority_months(max_months: int | None) -> list[int]:
    """M+1/2/3 always; 6/12 only when the selected object reaches them."""
    try:
        h = int(max_months) if max_months is not None else 3
    except (TypeError, ValueError):
        h = 3
    out = [1, 2, 3]
    if h >= 6:
        out.append(6)
    if h >= 12:
        out.append(12)
    return [m for m in out if m <= max(h, 3)]


def _dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if frame is None or frame.empty or "date" not in frame:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce")).dropna()


def _effective_available_months(origin: pd.Timestamp | None, future_dates) -> int:
    idx = pd.DatetimeIndex(pd.to_datetime(future_dates, errors="coerce")).dropna()
    if origin is None or not len(idx):
        return 3
    latest = pd.Timestamp(idx.max())
    origin = pd.Timestamp(origin)
    months = (latest.year - origin.year) * 12 + latest.month - origin.month
    # Weekly horizons can end a few days before the calendar month boundary.
    if latest.day + 10 >= origin.day:
        months = max(months, 0)
    return max(1, int(months))


def _nearest_target_row(frame: pd.DataFrame, target: pd.Timestamp, *, tolerance_days: int | None = None):
    if frame is None or frame.empty or "date" not in frame:
        return None
    part = frame.copy()
    part["date"] = pd.to_datetime(part["date"], errors="coerce")
    part = part.dropna(subset=["date"]).sort_values("date")
    if part.empty:
        return None
    delta = (part["date"] - pd.Timestamp(target)).abs()
    pos = int(delta.to_numpy().argmin())
    if tolerance_days is not None and delta.iloc[pos] > pd.Timedelta(days=int(tolerance_days)):
        return None
    return part.iloc[pos]


def _central(row: Mapping[str, Any]) -> float | None:
    for key in ("mean", "value", "posterior_mean", "q50"):
        if key in row:
            value = _num(row.get(key))
            if value is not None:
                return value
    return None


def _history_and_fan(frame: pd.DataFrame, *, series: str | None, metric: str | None):
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = frame.copy()
    if "date" in work:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if series is not None and "series" in work:
        work = work.loc[work["series"].astype(str).eq(str(series))]
    if metric is not None and "metric" in work:
        work = work.loc[work["metric"].astype(str).eq(str(metric))]
    hist = work.loc[work["record_type"].astype(str).eq("history")].copy() if "record_type" in work else pd.DataFrame()
    fan = work.loc[work["record_type"].astype(str).eq("fan")].copy() if "record_type" in work else pd.DataFrame()
    return hist.sort_values("date"), fan.sort_values("date")


def forecast_summary_records(
    frame: pd.DataFrame,
    *,
    series: str | None,
    metric: str | None,
    max_months: int | None = None,
) -> list[dict[str, Any]]:
    """Latest observed plus selected month-ahead posterior path points."""
    hist, fan = _history_and_fan(frame, series=series, metric=metric)
    if hist.empty and fan.empty:
        return []

    rows: list[dict[str, Any]] = []
    origin = None
    if not hist.empty:
        last = hist.dropna(subset=["date"]).iloc[-1]
        origin = pd.Timestamp(last["date"])
        rows.append({
            "horizon": "Latest observed",
            "date": origin.strftime("%Y-%m-%d"),
            "mean": fmt(last.get("value")),
            "median": "—",
            "interval68": "—",
            "interval90": "—",
        })
    elif not fan.empty:
        first_date = pd.Timestamp(fan["date"].dropna().min())
        origin = first_date - pd.DateOffset(months=1)

    if fan.empty or origin is None:
        return rows

    # Prefer actual predictive rows. Nowcast rows remain eligible because they
    # are genuinely month-ahead model paths from the latest published point.
    if "is_future" in fan:
        predictive = fan.loc[fan["is_future"].astype("boolean").fillna(False)].copy()
        if "segment" in fan:
            nowcast = fan.loc[fan["segment"].astype(str).eq("nowcast")].copy()
        else:
            nowcast = fan.iloc[0:0].copy()
        path = pd.concat([nowcast, predictive], ignore_index=True).drop_duplicates("date").sort_values("date")
    else:
        path = fan.copy()

    if path.empty:
        return rows
    available = _effective_available_months(origin, path["date"])
    display_max = available if max_months is None else min(int(max_months), available)
    for month in priority_months(display_max):
        target = pd.Timestamp(origin) + pd.DateOffset(months=int(month))
        item = _nearest_target_row(path, target, tolerance_days=20)
        if item is None:
            continue
        rows.append({
            "horizon": f"M+{month}",
            "date": pd.Timestamp(item["date"]).strftime("%Y-%m-%d"),
            "mean": fmt(_central(item)),
            "median": fmt(item.get("q50")),
            "interval68": interval(item.get("q16"), item.get("q84")),
            "interval90": interval(item.get("q05"), item.get("q95")),
        })
    return rows


def _records_frame(value: Any, *, name: str | None = None) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        out = value.copy()
    elif isinstance(value, str):
        try:
            out = pd.read_json(StringIO(value), orient="split")
        except Exception:
            return pd.DataFrame()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        out = pd.DataFrame(list(value))
    else:
        return pd.DataFrame()
    if name is not None and "name" in out:
        out = out.loc[out["name"].astype(str).eq(str(name))].copy()
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date") if "date" in out else out


def paired_effect_records(
    baseline: Any,
    scenario: Any,
    impact: Any,
    *,
    max_months: int | None,
    origin: Any = None,
    signed: bool = True,
) -> list[dict[str, Any]]:
    """Readable Baseline / Scenario / Effect table from paired fan records."""
    b = _records_frame(baseline)
    s = _records_frame(scenario)
    i = _records_frame(impact)
    frames = [x for x in (b, s, i) if not x.empty and "date" in x]
    if not frames:
        return []
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(pd.to_datetime(x["date"]).dropna()) for x in frames])))
    if not len(all_dates):
        return []
    if origin is None:
        origin_ts = pd.Timestamp(all_dates.min()) - pd.DateOffset(months=1)
    else:
        origin_ts = pd.Timestamp(origin)
    available = _effective_available_months(origin_ts, all_dates)
    display_max = available if max_months is None else min(int(max_months), available)
    out=[]
    for month in priority_months(display_max):
        target=origin_ts+pd.DateOffset(months=month)
        br=_nearest_target_row(b,target,tolerance_days=20)
        sr=_nearest_target_row(s,target,tolerance_days=20)
        ir=_nearest_target_row(i,target,tolerance_days=20)
        anchor = ir if ir is not None else sr if sr is not None else br
        if anchor is None:
            continue
        effect = _central(ir) if ir is not None else None
        out.append({
            "horizon": f"M+{month}",
            "date": pd.Timestamp(anchor["date"]).strftime("%Y-%m-%d"),
            "baseline": fmt(_central(br) if br is not None else None),
            "scenario": fmt(_central(sr) if sr is not None else None),
            "effect": (fmt_signed(effect, suffix=" pp") if signed else fmt(effect)),
            "interval68": (interval_signed(ir.get("q16"), ir.get("q84"), suffix=" pp") if ir is not None else "—"),
            "interval90": (interval_signed(ir.get("q05"), ir.get("q95"), suffix=" pp") if ir is not None else "—"),
        })
    return out


def paired_blocks_records(
    dates: Sequence[Any],
    baseline: Mapping[str, Any],
    scenario: Mapping[str, Any],
    impact: Mapping[str, Any],
    *,
    max_months: int | None,
) -> list[dict[str, Any]]:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates or []), errors="coerce")).dropna()
    if not len(idx):
        return []
    def make(block):
        block=dict(block or {})
        data={"date": idx}
        for key in ("mean","q05","q16","q50","q84","q95"):
            arr=np.asarray(block.get(key,[]),dtype=float)
            if len(arr)==len(idx): data[key]=arr
        return pd.DataFrame(data)
    return paired_effect_records(
        make(baseline), make(scenario), make(impact),
        max_months=max_months,
        origin=idx.min()-pd.DateOffset(months=1),
    )


def dual_impact_records(
    component_impact: Any,
    aggregate_impact: Any,
    *,
    max_months: int | None = None,
    origin: Any = None,
) -> list[dict[str, Any]]:
    c=_records_frame(component_impact); a=_records_frame(aggregate_impact)
    frames=[x for x in (c,a) if not x.empty and "date" in x]
    if not frames: return []
    dates=pd.DatetimeIndex(sorted(set().union(*[set(pd.to_datetime(x["date"]).dropna()) for x in frames])))
    origin_ts=(pd.Timestamp(dates.min())-pd.DateOffset(months=1)) if origin is None else pd.Timestamp(origin)
    available=_effective_available_months(origin_ts,dates)
    display_max=available if max_months is None else min(int(max_months),available)
    out=[]
    for month in priority_months(display_max):
        target=origin_ts+pd.DateOffset(months=month)
        cr=_nearest_target_row(c,target,tolerance_days=20); ar=_nearest_target_row(a,target,tolerance_days=20)
        anchor=ar if ar is not None else cr
        if anchor is None: continue
        out.append({
            "horizon":f"M+{month}",
            "date":pd.Timestamp(anchor["date"]).strftime("%Y-%m-%d"),
            "component":fmt_signed(_central(cr) if cr is not None else None,suffix=" pp"),
            "energy":fmt_signed(_central(ar) if ar is not None else None,suffix=" pp"),
            "component68":interval_signed(cr.get("q16"),cr.get("q84"),suffix=" pp") if cr is not None else "—",
            "energy68":interval_signed(ar.get("q16"),ar.get("q84"),suffix=" pp") if ar is not None else "—",
        })
    return out


def _structural_targets(payload: Mapping[str, Any], available_horizons, *, include_impact: bool):
    frequency=str((payload or {}).get("frequency") or "monthly").lower()
    available=sorted({int(x) for x in available_horizons})
    if not available: return []
    targets=[]
    if include_impact and 0 in available: targets.append(("Impact",0))
    mapping={1:4,2:9,3:13,6:26,12:52} if frequency=="weekly" else {1:1,2:2,3:3,6:6,12:12}
    max_h=max(available)
    for m,h in mapping.items():
        if h>max_h: continue
        # Exact for monthly; nearest weekly integer if custom horizon grid.
        nearest=min(available,key=lambda x:abs(x-h))
        if frequency=="weekly" and abs(nearest-h)>2: continue
        if frequency!="weekly" and nearest!=h: continue
        targets.append((f"M+{m}",nearest))
    return targets


def structural_irf_records(payload, *, response, shock, metric) -> list[dict[str, Any]]:
    if not payload or not payload.get("ok"): return []
    f=pd.DataFrame(payload.get("irf") or [])
    if f.empty: return []
    part=f.loc[(f["response"].astype(str)==str(response))&(f["shock"].astype(str)==str(shock))&(f["metric"].astype(str)==str(metric or "cumulative"))].copy()
    if part.empty: return []
    out=[]
    frequency=str(payload.get("frequency") or "monthly").lower()
    unit="w" if frequency=="weekly" else "m" if frequency=="monthly" else "p"
    for label,h in _structural_targets(payload,part["horizon"],include_impact=True):
        row=part.loc[pd.to_numeric(part["horizon"],errors="coerce").eq(h)]
        if row.empty: continue
        r=row.iloc[0]
        out.append({
            "horizon":label,
            "model_h":f"h={h}{unit}",
            "median":fmt(r.get("q50"),digits=4),
            "interval68":interval(r.get("q16"),r.get("q84"),digits=4),
            "interval90":interval(r.get("q05"),r.get("q95"),digits=4),
        })
    return out


def structural_fevd_table(payload, *, response):
    if not payload or not payload.get("ok"): return [], []
    f=pd.DataFrame(payload.get("fevd") or [])
    if f.empty: return [], []
    part=f.loc[f["response"].astype(str).eq(str(response))].copy()
    if part.empty: return [], []
    targets=_structural_targets(payload,part["horizon"],include_impact=False)
    columns=[{"name":"Shock","id":"shock"}]+[{"name":label,"id":f"h{h}"} for label,h in targets]
    rows=[]
    shock_order=list(payload.get("variables") or part["shock"].astype(str).drop_duplicates())
    for shock in shock_order:
        rec={"shock":str(shock).replace("_"," ").title()}
        block=part.loc[part["shock"].astype(str).eq(str(shock))]
        for label,h in targets:
            rr=block.loc[pd.to_numeric(block["horizon"],errors="coerce").eq(h)]
            rec[f"h{h}"]=fmt(rr.iloc[0].get("q50") if not rr.empty else None,suffix="%")
        rows.append(rec)
    total={"shock":"Total"}
    for _,h in targets:
        vals=[]
        for shock in shock_order:
            rr=part.loc[(part["shock"].astype(str)==str(shock))&pd.to_numeric(part["horizon"],errors="coerce").eq(h)]
            if not rr.empty:
                x=_num(rr.iloc[0].get("q50"))
                if x is not None: vals.append(x)
        total[f"h{h}"]=fmt(sum(vals) if vals else None,suffix="%")
    rows.append(total)
    return rows, columns


def structural_hd_records(payload, *, response) -> list[dict[str, Any]]:
    if not payload or not payload.get("ok"): return []
    f=pd.DataFrame(payload.get("hd") or [])
    if f.empty or "date" not in f: return []
    part=f.loc[f["response"].astype(str).eq(str(response))].copy()
    part["date"]=pd.to_datetime(part["date"],errors="coerce"); part=part.dropna(subset=["date"]).sort_values("date")
    if part.empty: return []
    latest=pd.Timestamp(part["date"].max())
    targets=[("Latest",latest),("−1M",latest-pd.DateOffset(months=1)),("−3M",latest-pd.DateOffset(months=3)),("−6M",latest-pd.DateOffset(months=6))]
    components=[c for c in (payload.get("hd_component_names") or []) if c in part.columns]
    display=[(str(c).replace("_"," ").title(),c) for c in components]
    if "base" in part: display.append(("Base / initial conditions","base"))
    if "reconstructed" in part: display.append(("Reconstructed","reconstructed"))
    if "observed" in part: display.append((str(payload.get("hd_observed_label") or "Observed"),"observed"))
    out=[]
    for label,col in display:
        rec={"component":label}
        for key,target in targets:
            row=_nearest_target_row(part,target,tolerance_days=20)
            rec[key]=fmt_signed(row.get(col) if row is not None else None,digits=3)
        out.append(rec)
    return out


def _regime(z: Any) -> str:
    x=_num(z)
    if x is None: return "—"
    if x>=1: return "High"
    if x>=0.5: return "Above norm"
    if x>-0.5: return "Near norm"
    if x>-1: return "Below norm"
    return "Low"


def economic_conditions_records(snapshot: Mapping[str, Any] | None, series_config: Mapping[str, Any], selected_series=None):
    if not snapshot: return []
    combined=snapshot.get("combined"); transformed=snapshot.get("transformed"); zscore=snapshot.get("zscore")
    if not isinstance(combined,pd.DataFrame) or not isinstance(transformed,pd.DataFrame) or not isinstance(zscore,pd.DataFrame): return []
    selected=set(selected_series or list(series_config))
    out=[]
    for name,cfg in series_config.items():
        if name not in selected or name not in combined or name not in transformed or name not in zscore: continue
        valid=zscore[name].dropna()
        if valid.empty: continue
        stamp=pd.Timestamp(valid.index.max())
        raw=combined[name].reindex([stamp]).iloc[0] if stamp in combined.index else np.nan
        tr=transformed[name].reindex([stamp]).iloc[0] if stamp in transformed.index else np.nan
        z=valid.loc[stamp]
        transform={"yoy_pct":"YoY %","diff12":"12m change","level":"Level"}.get(str(cfg.get("transform")),str(cfg.get("transform")))
        if cfg.get("invert"): transform += " · inverted z"
        if not cfg.get("standardise",True): transform += " · published z"
        out.append({
            "indicator":name,
            "raw":fmt(raw,digits=3),
            "transform":transform,
            "transformed":fmt(tr,digits=3),
            "zscore":fmt(z,digits=2),
            "regime":_regime(z),
            "date":stamp.strftime("%Y-%m"),
        })
    return out


FORECAST_COLUMNS = [
    {"name":"Horizon","id":"horizon"},{"name":"Date","id":"date"},
    {"name":"Posterior mean","id":"mean"},{"name":"Median","id":"median"},
    {"name":"68% interval","id":"interval68"},{"name":"90% interval","id":"interval90"},
]
PAIRED_EFFECT_COLUMNS = [
    {"name":"Horizon","id":"horizon"},{"name":"Date","id":"date"},
    {"name":"Baseline","id":"baseline"},{"name":"Conditional / scenario","id":"scenario"},
    {"name":"Effect","id":"effect"},{"name":"68% effect","id":"interval68"},{"name":"90% effect","id":"interval90"},
]
DUAL_IMPACT_COLUMNS = [
    {"name":"Horizon","id":"horizon"},{"name":"Date","id":"date"},
    {"name":"Component effect","id":"component"},{"name":"HICP Energy effect","id":"energy"},
    {"name":"Component 68%","id":"component68"},{"name":"Energy 68%","id":"energy68"},
]
IRF_COLUMNS = [
    {"name":"Horizon","id":"horizon"},{"name":"Model horizon","id":"model_h"},
    {"name":"Posterior median","id":"median"},{"name":"68% interval","id":"interval68"},{"name":"90% interval","id":"interval90"},
]
HD_COLUMNS = [
    {"name":"Shock / component","id":"component"},{"name":"Latest","id":"Latest"},
    {"name":"−1M","id":"−1M"},{"name":"−3M","id":"−3M"},{"name":"−6M","id":"−6M"},
]
ECONOMIC_CONDITIONS_COLUMNS = [
    {"name":"Indicator","id":"indicator"},{"name":"Latest raw","id":"raw"},
    {"name":"Transformation","id":"transform"},{"name":"Latest transformed","id":"transformed"},
    {"name":"Z-score","id":"zscore"},{"name":"Regime","id":"regime"},{"name":"Latest data","id":"date"},
]

__all__ = [
    "readable_table","priority_months","forecast_summary_records",
    "paired_effect_records","paired_blocks_records","dual_impact_records",
    "structural_irf_records","structural_fevd_table","structural_hd_records",
    "economic_conditions_records","FORECAST_COLUMNS","PAIRED_EFFECT_COLUMNS",
    "DUAL_IMPACT_COLUMNS","IRF_COLUMNS","HD_COLUMNS","ECONOMIC_CONDITIONS_COLUMNS",
]
