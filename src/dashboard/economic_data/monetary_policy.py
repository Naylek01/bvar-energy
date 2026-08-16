"""Official central-bank policy + liquidity snapshot loader.

This module is called ONLY from dashboard bootstrap / explicit ``Refresh snapshot``.
Presentation callbacks never access the internet.

Official sources
----------------
ECB
    - DFR: FM.B.U2.EUR.4F.KR.DFR.LEV
    - Excess liquidity: ILM.D.U2.C.EXLIQ.U2.EUR
    - Current accounts: ILM.D.U2.C.L020100.U2.EUR
    - Minimum reserve requirements: ILM.D.U2.C.MRR.U2.EUR
    - Deposit facility usage: ILM.D.U2.C.L020200.U2.EUR
    - Marginal lending facility usage: ILM.D.U2.C.A050500.U2.EUR
    - Open-market operations excluding monetary-policy portfolios:
      ILM.D.U2.C.TOMO.U2.EUR

Bank of England
    - Bank Rate: IUDBEDR
    - Reserve balances: RPWB56A
    - Short-Term Repo: RPWB67A
    - Indexed Long-Term Repo: RPWZ4TJ
    - APF gilt stock at purchase proceeds: YWWB9T9

Liquidity levels are source values in millions of euro / sterling.  The
dashboard converts these to bn/tn only for presentation.  Δ1M and Δ3M are
absolute level changes versus the latest available observation at or before
30/90 calendar days before the current observation.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


MONETARY_POLICY_CONTRACT_VERSION = "monetary-policy-liquidity-snapshot-v2"

ECB_SERIES_KEY = "FM.B.U2.EUR.4F.KR.DFR.LEV"
ECB_DATA_URL = (
    "https://data-api.ecb.europa.eu/service/data/FM/"
    "B.U2.EUR.4F.KR.DFR.LEV?format=csvdata&lastNObservations=16"
)
ECB_CALENDAR_URL = (
    "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
)
ECB_DATA_API_BASE = "https://data-api.ecb.europa.eu/service/data"

ECB_LIQUIDITY_SERIES = {
    "excess_liquidity": {
        "label": "Excess liquidity",
        "series": "ILM.D.U2.C.EXLIQ.U2.EUR",
    },
    "current_accounts": {
        "label": "Current accounts",
        "series": "ILM.D.U2.C.L020100.U2.EUR",
    },
    "minimum_reserves": {
        "label": "Minimum reserve requirements",
        "series": "ILM.D.U2.C.MRR.U2.EUR",
    },
    "deposit_facility": {
        "label": "Deposit facility usage",
        "series": "ILM.D.U2.C.L020200.U2.EUR",
    },
    "marginal_lending": {
        "label": "Marginal lending facility usage",
        "series": "ILM.D.U2.C.A050500.U2.EUR",
    },
    "open_market_operations": {
        "label": "Open-market operations excl. MonPol portfolios",
        "series": "ILM.D.U2.C.TOMO.U2.EUR",
    },
}

BOE_SERIES_CODE = "IUDBEDR"
BOE_DATA_BASE = (
    "https://www.bankofengland.co.uk/boeapps/database/"
    "_iadb-fromshowcolumns.asp"
)
BOE_CURRENT_RATE_URL = (
    "https://www.bankofengland.co.uk/monetary-policy/"
    "the-interest-rate-bank-rate."
)
BOE_HISTORY_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp"
)

BOE_LIQUIDITY_SERIES = {
    "reserve_balances": {
        "label": "Reserve balances",
        "series": "RPWB56A",
    },
    "short_term_repo": {
        "label": "Short-Term Repo",
        "series": "RPWB67A",
    },
    "indexed_long_term_repo": {
        "label": "Indexed Long-Term Repo",
        "series": "RPWZ4TJ",
    },
    "apf_gilts": {
        "label": "APF gilt holdings · purchase proceeds",
        "series": "YWWB9T9",
    },
}

_DEFAULT_TIMEOUT = 8.0
_USER_AGENT = "InflationDashboard/1.0 (official central-bank data snapshot)"


class MonetaryPolicySourceError(RuntimeError):
    pass


class _VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = " ".join(str(data).split())
        if value:
            self.parts.append(value)

    def text(self) -> str:
        return " ".join(self.parts)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_date(value: str) -> date:
    raw = str(value).strip()
    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%d %b %y",
        "%d %B %y",
    ):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported date {raw!r}")


def _fetch_text(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    accept: str | None = None,
) -> str:
    headers = {"User-Agent": _USER_AGENT}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def _float(value: Any) -> float:
    out = float(str(value).strip())
    if not math.isfinite(out):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return out


def _compress_changes(
    observations: Iterable[tuple[date, float]],
) -> list[tuple[date, float]]:
    ordered = sorted(
        ((d, float(v)) for d, v in observations),
        key=lambda x: x[0],
    )
    out: list[tuple[date, float]] = []
    for stamp, value in ordered:
        if not out or abs(value - out[-1][1]) > 1e-12:
            out.append((stamp, value))
    return out


def _record_from_changes(
    changes,
    *,
    area,
    institution,
    rate_name,
    source_series,
    source_url,
):
    if not changes:
        raise MonetaryPolicySourceError(
            f"No observations returned for {institution}."
        )
    effective_date, current = changes[-1]
    if len(changes) >= 2:
        previous_date, previous = changes[-2]
        last_move = float(current - previous)
    else:
        previous_date = previous = last_move = None
    return {
        "area": area,
        "institution": institution,
        "rate_name": rate_name,
        "value": float(current),
        "effective_date": effective_date.isoformat(),
        "previous_value": None if previous is None else float(previous),
        "previous_effective_date": (
            None if previous_date is None else previous_date.isoformat()
        ),
        "last_move_pp": last_move,
        "last_move_date": (
            effective_date.isoformat() if previous is not None else None
        ),
        "next_meeting": None,
        "source_series": source_series,
        "source_url": source_url,
        "source_status": "live",
    }


# ---------------------------------------------------------------------------
# ECB policy + liquidity
# ---------------------------------------------------------------------------


def _parse_ecb_observations(text: str) -> list[tuple[date, float]]:
    reader = csv.DictReader(io.StringIO(text))
    fields = [str(x or "") for x in (reader.fieldnames or [])]
    upper = {x.upper(): x for x in fields}
    date_col = next(
        (upper[x] for x in ("TIME_PERIOD", "TIME", "DATE") if x in upper),
        None,
    )
    value_col = next(
        (upper[x] for x in ("OBS_VALUE", "VALUE") if x in upper),
        None,
    )
    if not date_col or not value_col:
        raise MonetaryPolicySourceError(
            "ECB CSV missing TIME_PERIOD/OBS_VALUE columns."
        )

    observations: list[tuple[date, float]] = []
    for row in reader:
        try:
            observations.append(
                (_as_date(row[date_col]), _float(row[value_col]))
            )
        except Exception:
            pass
    observations.sort(key=lambda item: item[0])
    if not observations:
        raise MonetaryPolicySourceError(
            "ECB CSV contained no parseable observations."
        )
    return observations


def _parse_ecb_csv(text: str):
    return _compress_changes(_parse_ecb_observations(text))


def _ecb_series_url(series: str, *, last_n: int = 190) -> str:
    dataset, key = str(series).split(".", 1)
    return (
        f"{ECB_DATA_API_BASE}/{dataset}/{key}"
        f"?format=csvdata&lastNObservations={int(last_n)}"
    )


def _parse_ecb_next_meeting(html_text: str, *, today: date) -> date:
    parser = _VisibleTextParser()
    parser.feed(html_text)
    text = parser.text()
    date_matches = list(
        re.finditer(r"\b\d{2}/\d{2}/\d{4}\b", text)
    )
    candidates = []
    for idx, match in enumerate(date_matches):
        end = (
            date_matches[idx + 1].start()
            if idx + 1 < len(date_matches)
            else min(len(text), match.end() + 900)
        )
        segment = text[match.end():end]
        if (
            re.search(r"monetary policy meeting", segment, re.I)
            and re.search(r"Day\s*2", segment, re.I)
        ):
            try:
                stamp = _as_date(match.group())
                if stamp >= today:
                    candidates.append(stamp)
            except Exception:
                pass
    if not candidates:
        raise MonetaryPolicySourceError(
            "ECB calendar exposed no future Day-2 monetary-policy meeting."
        )
    return min(candidates)


def fetch_ecb_policy(
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    today: date | None = None,
):
    today = today or _utc_now().date()
    record = _record_from_changes(
        _parse_ecb_csv(
            _fetch_text(
                ECB_DATA_URL,
                timeout=timeout,
                accept="text/csv",
            )
        ),
        area="Euro area",
        institution="ECB",
        rate_name="Deposit facility rate",
        source_series=ECB_SERIES_KEY,
        source_url=ECB_DATA_URL,
    )
    try:
        html = _fetch_text(
            ECB_CALENDAR_URL,
            timeout=timeout,
            accept="text/html",
        )
        record["next_meeting"] = _parse_ecb_next_meeting(
            html,
            today=today,
        ).isoformat()
    except Exception as exc:
        record["calendar_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _previous_observation(
    observations: list[tuple[date, float]],
    target: date,
) -> tuple[date, float] | None:
    candidate = None
    for stamp, value in observations:
        if stamp <= target:
            candidate = (stamp, value)
        else:
            break
    return candidate


def _liquidity_metric(
    observations: list[tuple[date, float]],
    *,
    label: str,
    series: str,
    currency: str,
    source_url: str,
) -> dict[str, Any]:
    if not observations:
        raise MonetaryPolicySourceError(
            f"No liquidity observations for {series}."
        )
    observations = sorted(observations, key=lambda item: item[0])
    latest_date, latest_value = observations[-1]

    prior_1m = _previous_observation(
        observations,
        latest_date - timedelta(days=30),
    )
    prior_3m = _previous_observation(
        observations,
        latest_date - timedelta(days=90),
    )

    return {
        "label": label,
        "series": series,
        "currency": currency,
        "unit": "million",
        "value_mn": float(latest_value),
        "date": latest_date.isoformat(),
        "delta_1m_mn": (
            None
            if prior_1m is None
            else float(latest_value - prior_1m[1])
        ),
        "delta_1m_base_date": (
            None if prior_1m is None else prior_1m[0].isoformat()
        ),
        "delta_3m_mn": (
            None
            if prior_3m is None
            else float(latest_value - prior_3m[1])
        ),
        "delta_3m_base_date": (
            None if prior_3m is None else prior_3m[0].isoformat()
        ),
        "source_url": source_url,
        "source_status": "live",
    }


def fetch_ecb_liquidity(
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    errors: dict[str, str] = {}
    metrics: dict[str, dict[str, Any]] = {}

    def fetch_one(key: str, spec: dict[str, str]):
        url = _ecb_series_url(spec["series"], last_n=190)
        obs = _parse_ecb_observations(
            _fetch_text(url, timeout=timeout, accept="text/csv")
        )
        return _liquidity_metric(
            obs,
            label=spec["label"],
            series=spec["series"],
            currency="EUR",
            source_url=url,
        )

    with ThreadPoolExecutor(
        max_workers=min(6, len(ECB_LIQUIDITY_SERIES))
    ) as executor:
        futures = {
            key: executor.submit(fetch_one, key, spec)
            for key, spec in ECB_LIQUIDITY_SERIES.items()
        }
        for key, future in futures.items():
            try:
                metrics[key] = future.result()
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"

    if not metrics:
        raise MonetaryPolicySourceError(
            "No ECB liquidity series could be loaded."
        )
    return {
        "institution": "ECB",
        "area": "Euro area",
        "source_status": (
            "live" if not errors else "partial"
        ),
        "metrics": metrics,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Bank of England policy + liquidity
# ---------------------------------------------------------------------------


def _boe_data_url(*, today: date) -> str:
    params = {
        "csv.x": "yes",
        "Datefrom": f"01/Jan/{today.year - 3}",
        "Dateto": "now",
        "SeriesCodes": BOE_SERIES_CODE,
        "CSVF": "TN",
        "UsingCodes": "Y",
        "VPD": "Y",
        "VFD": "N",
    }
    return BOE_DATA_BASE + "?" + urllib.parse.urlencode(params)


def _boe_multi_url(
    series_codes: Iterable[str],
    *,
    today: date,
) -> str:
    params = {
        "csv.x": "yes",
        "Datefrom": f"01/Jan/{today.year - 1}",
        "Dateto": "now",
        "SeriesCodes": ",".join(map(str, series_codes)),
        "CSVF": "TN",
        "UsingCodes": "Y",
        "VPD": "Y",
        "VFD": "N",
    }
    return BOE_DATA_BASE + "?" + urllib.parse.urlencode(params)


def _parse_boe_csv(text: str):
    observations = []
    for row in csv.reader(io.StringIO(text)):
        parsed_date = None
        date_idx = None
        for idx, cell in enumerate(row):
            try:
                parsed_date = _as_date(
                    str(cell).strip().strip('"')
                )
                date_idx = idx
                break
            except Exception:
                pass
        if parsed_date is None:
            continue
        for idx, cell in enumerate(row):
            if idx == date_idx:
                continue
            try:
                observations.append(
                    (
                        parsed_date,
                        _float(
                            str(cell)
                            .strip()
                            .strip('"')
                            .replace("%", "")
                        ),
                    )
                )
                break
            except Exception:
                pass
    if not observations:
        raise MonetaryPolicySourceError(
            "BoE CSV contained no parseable Bank Rate observations."
        )
    return _compress_changes(observations)


def _parse_boe_multi_csv(
    text: str,
    *,
    series_codes: list[str],
) -> dict[str, list[tuple[date, float]]]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise MonetaryPolicySourceError("BoE multi-series CSV is empty.")

    codes = [str(code) for code in series_codes]
    header_idx = None
    column_map: dict[str, int] = {}

    for idx, row in enumerate(rows):
        normalised = [
            str(cell).strip().strip('"')
            for cell in row
        ]
        exact = {
            code: normalised.index(code)
            for code in codes
            if code in normalised
        }
        if exact:
            header_idx = idx
            column_map = exact
            break

    # Official endpoint with UsingCodes=Y normally exposes code headers.
    # If the envelope uses title rows instead, data columns retain the
    # requested series order after the date column, so use that as fallback.
    if header_idx is None:
        header_idx = -1
        column_map = {
            code: pos + 1 for pos, code in enumerate(codes)
        }

    out = {code: [] for code in codes}
    for row in rows[header_idx + 1:]:
        parsed_date = None
        for cell in row[:2]:
            try:
                parsed_date = _as_date(
                    str(cell).strip().strip('"')
                )
                break
            except Exception:
                pass
        if parsed_date is None:
            continue

        for code, col in column_map.items():
            if col >= len(row):
                continue
            raw = str(row[col]).strip().strip('"')
            if raw.lower() in {"", "n/a", "na", "-", ".."}:
                continue
            try:
                out[code].append(
                    (parsed_date, _float(raw.replace(",", "")))
                )
            except Exception:
                pass

    if not any(out.values()):
        raise MonetaryPolicySourceError(
            "BoE multi-series CSV contained no parseable observations."
        )
    for code in out:
        out[code].sort(key=lambda item: item[0])
    return out


def _parse_boe_history_html(text: str):
    parser = _VisibleTextParser()
    parser.feed(text)
    plain = parser.text()
    obs = []
    for stamp, value in re.findall(
        r"\b(\d{2}\s+[A-Za-z]{3}\s+\d{2,4})\s+"
        r"(-?\d+(?:\.\d+)?)\b",
        plain,
    ):
        try:
            obs.append((_as_date(stamp), _float(value)))
        except Exception:
            pass
    if not obs:
        raise MonetaryPolicySourceError(
            "BoE history page contained no parseable Bank Rate changes."
        )
    return _compress_changes(obs)


def _parse_boe_next_meeting(text: str, *, today: date) -> date:
    parser = _VisibleTextParser()
    parser.feed(text)
    plain = parser.text()
    for pattern in (
        r"Next due:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"next decision.{0,100}?"
        r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    ):
        match = re.search(pattern, plain, re.I | re.S)
        if match:
            stamp = _as_date(match.group(1))
            if stamp >= today:
                return stamp
    raise MonetaryPolicySourceError(
        "BoE page exposed no future MPC decision date."
    )


def fetch_boe_policy(
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    today: date | None = None,
):
    today = today or _utc_now().date()
    data_url = _boe_data_url(today=today)
    try:
        changes = _parse_boe_csv(
            _fetch_text(
                data_url,
                timeout=timeout,
                accept="text/csv",
            )
        )
        source_url = data_url
    except Exception:
        changes = _parse_boe_history_html(
            _fetch_text(
                BOE_HISTORY_URL,
                timeout=timeout,
                accept="text/html",
            )
        )
        source_url = BOE_HISTORY_URL

    record = _record_from_changes(
        changes,
        area="United Kingdom",
        institution="Bank of England",
        rate_name="Bank Rate",
        source_series=BOE_SERIES_CODE,
        source_url=source_url,
    )
    try:
        html = _fetch_text(
            BOE_CURRENT_RATE_URL,
            timeout=timeout,
            accept="text/html",
        )
        record["next_meeting"] = _parse_boe_next_meeting(
            html,
            today=today,
        ).isoformat()
    except Exception as exc:
        record["calendar_error"] = f"{type(exc).__name__}: {exc}"
    return record


def fetch_boe_liquidity(
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or _utc_now().date()
    codes = [
        spec["series"]
        for spec in BOE_LIQUIDITY_SERIES.values()
    ]
    url = _boe_multi_url(codes, today=today)
    series_obs = _parse_boe_multi_csv(
        _fetch_text(url, timeout=timeout, accept="text/csv"),
        series_codes=codes,
    )

    metrics: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for key, spec in BOE_LIQUIDITY_SERIES.items():
        observations = series_obs.get(spec["series"]) or []
        if not observations:
            errors[key] = "No observations returned."
            continue
        try:
            metrics[key] = _liquidity_metric(
                observations,
                label=spec["label"],
                series=spec["series"],
                currency="GBP",
                source_url=url,
            )
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"

    if not metrics:
        raise MonetaryPolicySourceError(
            "No BoE liquidity series could be loaded."
        )
    return {
        "institution": "Bank of England",
        "area": "United Kingdom",
        "source_status": (
            "live" if not errors else "partial"
        ),
        "metrics": metrics,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Persisted snapshot / source-by-source fallback
# ---------------------------------------------------------------------------


def _read_cache(path: Path):
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _cached_bank(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict) or value.get("value") is None:
        return None
    out = dict(value)
    out["source_status"] = "cached"
    return out


def _cached_liquidity(payload, key):
    liquidity = payload.get("liquidity")
    if not isinstance(liquidity, dict):
        return None
    value = liquidity.get(key)
    if not isinstance(value, dict):
        return None
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None

    out = dict(value)
    out["source_status"] = "cached"
    out_metrics = {}
    for metric_key, metric in metrics.items():
        if isinstance(metric, dict):
            item = dict(metric)
            item["source_status"] = "cached"
            out_metrics[str(metric_key)] = item
    out["metrics"] = out_metrics
    return out if out_metrics else None


def _merge_liquidity_group(
    live_group,
    cached_group,
):
    """Keep live metrics; fill individual failed metrics from last cache."""
    live_group = dict(live_group or {})
    cached_group = dict(cached_group or {})
    live_metrics = dict(live_group.get("metrics") or {})
    cached_metrics = dict(cached_group.get("metrics") or {})

    merged = {}
    for key in sorted(set(live_metrics) | set(cached_metrics)):
        if isinstance(live_metrics.get(key), dict):
            merged[key] = dict(live_metrics[key])
        elif isinstance(cached_metrics.get(key), dict):
            item = dict(cached_metrics[key])
            item["source_status"] = "cached"
            merged[key] = item

    if not merged:
        return None

    live_count = sum(
        1
        for item in merged.values()
        if item.get("source_status") == "live"
    )
    cached_count = sum(
        1
        for item in merged.values()
        if item.get("source_status") == "cached"
    )
    status = (
        "live"
        if live_count == len(merged)
        else "partial"
        if live_count
        else "cached"
    )

    return {
        "institution": (
            live_group.get("institution")
            or cached_group.get("institution")
        ),
        "area": live_group.get("area") or cached_group.get("area"),
        "source_status": status,
        "metrics": merged,
        "errors": dict(live_group.get("errors") or {}),
    }


def _persist(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    tmp.replace(path)


def build_monetary_policy_snapshot(
    *,
    cache_path: str | Path,
    timeout: float = _DEFAULT_TIMEOUT,
    today: date | None = None,
):
    cache_path = Path(cache_path)
    today = today or _utc_now().date()
    cached = _read_cache(cache_path)

    errors = {}
    live = {}

    def fetch_one(key):
        if key == "ecb":
            return fetch_ecb_policy(timeout=timeout, today=today)
        if key == "boe":
            return fetch_boe_policy(timeout=timeout, today=today)
        if key == "ecb_liquidity":
            return fetch_ecb_liquidity(timeout=timeout)
        if key == "boe_liquidity":
            return fetch_boe_liquidity(
                timeout=timeout,
                today=today,
            )
        raise KeyError(key)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            key: executor.submit(fetch_one, key)
            for key in (
                "ecb",
                "boe",
                "ecb_liquidity",
                "boe_liquidity",
            )
        }
        for key, future in futures.items():
            try:
                live[key] = future.result()
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"

    banks = {
        key: (
            dict(live[key])
            if key in live
            else _cached_bank(cached, key)
        )
        for key in ("ecb", "boe")
    }

    cached_ecb_liq = _cached_liquidity(cached, "ecb")
    cached_boe_liq = _cached_liquidity(cached, "boe")
    ecb_liq = _merge_liquidity_group(
        live.get("ecb_liquidity"),
        cached_ecb_liq,
    )
    boe_liq = _merge_liquidity_group(
        live.get("boe_liquidity"),
        cached_boe_liq,
    )

    available_banks = [
        key for key, value in banks.items()
        if isinstance(value, dict)
    ]
    live_banks = [
        key for key in available_banks
        if banks[key].get("source_status") == "live"
    ]
    cached_banks = [
        key for key in available_banks
        if banks[key].get("source_status") == "cached"
    ]
    status = (
        "live"
        if len(live_banks) == 2
        else "partial"
        if live_banks
        else "cached_fallback"
        if cached_banks
        else "unavailable"
    )

    liquidity_groups = {
        "ecb": ecb_liq,
        "boe": boe_liq,
    }
    liq_available = [
        key for key, value in liquidity_groups.items()
        if isinstance(value, dict)
    ]
    liq_live = [
        key for key in liq_available
        if liquidity_groups[key].get("source_status") == "live"
    ]
    liq_partial = [
        key for key in liq_available
        if liquidity_groups[key].get("source_status") == "partial"
    ]
    if len(liq_live) == 2:
        liquidity_status = "live"
    elif liq_live or liq_partial:
        liquidity_status = "partial"
    elif liq_available:
        liquidity_status = "cached_fallback"
    else:
        liquidity_status = "unavailable"

    payload = {
        "contract_version": MONETARY_POLICY_CONTRACT_VERSION,
        "snapshot_date": today.isoformat(),
        "fetched_at_utc": _utc_now().isoformat(),
        "status": status,
        "liquidity_status": liquidity_status,
        "ecb": banks["ecb"],
        "boe": banks["boe"],
        "liquidity": liquidity_groups,
        "errors": errors,
        "cache_path": str(cache_path),
    }

    if available_banks or liq_available:
        try:
            _persist(cache_path, payload)
        except Exception as exc:
            payload.setdefault("errors", {})["cache_write"] = (
                f"{type(exc).__name__}: {exc}"
            )
    return payload


__all__ = [
    "MONETARY_POLICY_CONTRACT_VERSION",
    "ECB_SERIES_KEY",
    "ECB_LIQUIDITY_SERIES",
    "BOE_SERIES_CODE",
    "BOE_LIQUIDITY_SERIES",
    "fetch_ecb_policy",
    "fetch_boe_policy",
    "fetch_ecb_liquidity",
    "fetch_boe_liquidity",
    "build_monetary_policy_snapshot",
]
