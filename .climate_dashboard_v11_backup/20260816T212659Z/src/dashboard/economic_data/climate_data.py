"""Climate/hydrology data backend for the Economic Data dashboard.

Design contract
---------------
* Dash never downloads the long GloFAS history inside a plotting callback.
* GloFAS is pinned to the current v5.0 historical contract and retrieved from
  the CEMS Early Warning Data Store (EWDS) with ``cdsapi``.
* The long-history modelled discharge series and derived monthly stress
  indicators are persisted below ``data/climate``.
* PEGELONLINE is used only for the observed Kaub live/near-live panel and is
  cached independently; it is never spliced into the GloFAS history.
* No credential value is written to logs or dashboard payloads.

The first implementation intentionally focuses on hydrology:
    Rhine at Kaub + Danube at the Vienna reach.
EDO/GDO drought rasters can be added later without changing this persisted
monthly contract.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests


CLIMATE_DATA_CONTRACT_VERSION = "economic-climate-hydrology-v1"
GLOFAS_DATASET = "cems-glofas-historical"
GLOFAS_SYSTEM_VERSION = "version_5_0"
GLOFAS_SOURCE_VERSION = "GloFAS v5.0"
GLOFAS_HYDROLOGICAL_MODEL = "lisflood"
GLOFAS_VARIABLE = "average_river_discharge_in_the_last_24_hours"
GLOFAS_TIMESPAN = "time_mean"
GLOFAS_CLIMATOLOGY_START = 1991
GLOFAS_CLIMATOLOGY_END = 2020
GLOFAS_DEFAULT_START_YEAR = 1991
GLOFAS_CHUNK_YEARS = 5

PEGELONLINE_BASE = "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
KAUB_UUID = "1d26e504-7f9e-480a-b52c-5932be6549ab"
KAUB_LATITUDE = 50.085438
KAUB_LONGITUDE = 7.764962


@dataclass(frozen=True)
class RiverPoint:
    key: str
    label: str
    river: str
    latitude: float
    longitude: float
    bbox_half_lat: float = 0.14
    bbox_half_lon: float = 0.18

    @property
    def area(self) -> list[float]:
        # EWDS order: North, West, South, East.
        return [
            float(self.latitude + self.bbox_half_lat),
            float(self.longitude - self.bbox_half_lon),
            float(self.latitude - self.bbox_half_lat),
            float(self.longitude + self.bbox_half_lon),
        ]


RIVER_POINTS: dict[str, RiverPoint] = {
    "rhine_kaub": RiverPoint(
        key="rhine_kaub",
        label="Rhine · Kaub",
        river="Rhine",
        latitude=KAUB_LATITUDE,
        longitude=KAUB_LONGITUDE,
    ),
    # Transparent default for the Danube leg of the dashboard.  This is not
    # presented as an official gauge: it is a GloFAS main-stem model point in
    # the Vienna reach, selected inside the bbox by the river-flow diagnostic.
    "danube_vienna": RiverPoint(
        key="danube_vienna",
        label="Danube · Vienna reach",
        river="Danube",
        latitude=48.235,
        longitude=16.405,
        bbox_half_lat=0.16,
        bbox_half_lon=0.22,
    ),
}

SERIES_LABELS = {
    "rhine_discharge_stress": "Rhine · discharge stress",
    "rhine_low_flow_intensity": "Rhine · low-flow intensity",
    "danube_discharge_stress": "Danube · discharge stress",
    "danube_low_flow_intensity": "Danube · low-flow intensity",
}
SERIES_ORDER = tuple(SERIES_LABELS)


class ClimateDataError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def climate_paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).expanduser().resolve()
    climate_root = root / "data" / "climate"
    raw = climate_root / "raw"
    processed = climate_root / "processed"
    cache = root / ".dashboard_cache"
    for path in (raw, processed, cache):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": climate_root,
        "raw": raw,
        "processed": processed,
        "monthly": processed / "climate_monthly.parquet",
        "metadata": processed / "climate_metadata.json",
        "kaub_cache": cache / "economic_data_climate_kaub.json",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _daily_path(project_root: str | Path, point_key: str) -> Path:
    return climate_paths(project_root)["processed"] / f"glofas_{point_key}_daily.parquet"


def glofas_credentials_status() -> dict[str, Any]:
    """Return presence/configuration flags without ever returning a token."""
    path = Path.home() / ".cdsapirc"
    url_env = str(os.getenv("CDSAPI_URL") or "").strip()
    key_env = bool(str(os.getenv("CDSAPI_KEY") or "").strip())
    file_url = ""
    file_has_key = False
    if path.is_file():
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                key, sep, raw = line.partition(":")
                if not sep:
                    continue
                key = key.strip().lower()
                raw = raw.strip()
                if key == "url":
                    file_url = raw
                elif key == "key" and raw:
                    file_has_key = True
        except OSError:
            pass
    effective_url = url_env or file_url
    configured = bool((key_env or file_has_key) and effective_url)
    return {
        "configured": configured,
        "cdsapirc_exists": path.is_file(),
        "url_is_ewds": effective_url.rstrip("/") == "https://ewds.climate.copernicus.eu/api",
        "url_source": "environment" if url_env else (".cdsapirc" if file_url else None),
        "key_present": bool(key_env or file_has_key),
    }


def build_glofas_request(
    *,
    years: Sequence[int | str],
    area: Sequence[float],
    product_type: str = "consolidated",
) -> dict[str, Any]:
    """Build the current EWDS v5.0 GloFAS request contract.

    Field names deliberately match the current EWDS process schema.  In
    particular, v5 uses ``lisflood``, ``year/month/day`` and
    ``average_river_discharge_in_the_last_24_hours`` rather than legacy v4-era
    request names.
    """
    years = [str(int(y)) for y in years]
    if not years:
        raise ValueError("At least one GloFAS year is required.")
    if product_type not in {"consolidated", "intermediate"}:
        raise ValueError("product_type must be consolidated or intermediate.")
    area = [float(x) for x in area]
    if len(area) != 4:
        raise ValueError("area must be [north, west, south, east].")
    return {
        "system_version": [GLOFAS_SYSTEM_VERSION],
        "hydrological_model": [GLOFAS_HYDROLOGICAL_MODEL],
        "product_type": [product_type],
        "timespan": [GLOFAS_TIMESPAN],
        "variable": [GLOFAS_VARIABLE],
        "year": years,
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area,
    }


def _cds_client():
    status = glofas_credentials_status()
    if not status["configured"]:
        raise ClimateDataError(
            "EWDS credentials are not configured. Create %USERPROFILE%/.cdsapirc "
            "from the EWDS API setup page and accept the GloFAS dataset licence."
        )
    if not status["url_is_ewds"]:
        raise ClimateDataError(
            "The configured CDS API URL is not the CEMS Early Warning Data Store. "
            "Expected https://ewds.climate.copernicus.eu/api."
        )
    try:
        import cdsapi  # type: ignore
    except ImportError as exc:
        raise ClimateDataError(
            "cdsapi is not installed in this Python environment."
        ) from exc
    return cdsapi.Client()


def _download_glofas_archive(
    *,
    project_root: str | Path,
    point: RiverPoint,
    years: Sequence[int],
    product_type: str,
    overwrite: bool = False,
) -> Path:
    paths = climate_paths(project_root)
    year_text = f"{min(years)}-{max(years)}"
    directory = paths["raw"] / "glofas" / GLOFAS_SYSTEM_VERSION / point.key / product_type
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{point.key}_{product_type}_{year_text}.zip"
    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return target

    client = _cds_client()
    request = build_glofas_request(
        years=years,
        area=point.area,
        product_type=product_type,
    )
    temp = target.with_suffix(".download")
    if temp.exists():
        temp.unlink()
    client.retrieve(GLOFAS_DATASET, request, str(temp))
    if not temp.is_file() or temp.stat().st_size == 0:
        raise ClimateDataError(f"EWDS returned no file for {point.key} {year_text}.")
    temp.replace(target)
    return target


def _safe_extract_zip(path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    with zipfile.ZipFile(path) as zf:
        root = destination.resolve()
        for member in zf.infolist():
            candidate = (destination / member.filename).resolve()
            if root not in candidate.parents and candidate != root:
                raise ClimateDataError(f"Unsafe archive member: {member.filename!r}")
            if member.is_dir():
                continue
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, candidate.open("wb") as dst:
                dst.write(src.read())
            files.append(candidate)
    return files


def _open_xarray(path: Path):
    try:
        import xarray as xr  # type: ignore
    except ImportError as exc:
        raise ClimateDataError("xarray is required to read GloFAS NetCDF files.") from exc
    errors = []
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            kwargs = {} if engine is None else {"engine": engine}
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:  # try next installed backend
            errors.append(f"{engine or 'default'}: {exc}")
    raise ClimateDataError(
        f"Could not open NetCDF {path.name}: " + " | ".join(errors[-3:])
    )


def _coord_name(dataset, candidates: Sequence[str]) -> str | None:
    names = set(dataset.coords) | set(dataset.dims)
    lower = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _data_variable(dataset, lat_name: str, lon_name: str) -> str:
    preferred = (
        GLOFAS_VARIABLE,
        "dis24",
        "river_discharge",
        "discharge",
    )
    for name in preferred:
        if name in dataset.data_vars:
            return str(name)
    for name, var in dataset.data_vars.items():
        dims = set(map(str, var.dims))
        if lat_name in dims and lon_name in dims and np.issubdtype(var.dtype, np.number):
            return str(name)
    raise ClimateDataError("Could not identify the river-discharge variable in GloFAS NetCDF.")


def _filename_date(path: Path) -> pd.Timestamp | None:
    match = re.search(r"(19|20)\d{6}", path.name)
    if not match:
        return None
    try:
        return pd.Timestamp(datetime.strptime(match.group(0), "%Y%m%d").date())
    except ValueError:
        return None


def _load_discharge_collection(
    files: Sequence[Path],
    *,
    selected_cell: Mapping[str, float] | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    """Read one EWDS response and extract one stable river grid cell."""
    try:
        import xarray as xr  # type: ignore
    except ImportError as exc:
        raise ClimateDataError("xarray is required to read GloFAS data.") from exc

    arrays = []
    lat_name = lon_name = None
    for path in sorted(files):
        if path.suffix.lower() not in {".nc", ".nc4", ".netcdf"}:
            continue
        ds = _open_xarray(path)
        try:
            this_lat = _coord_name(ds, ("latitude", "lat", "y"))
            this_lon = _coord_name(ds, ("longitude", "lon", "x"))
            if not this_lat or not this_lon:
                continue
            name = _data_variable(ds, this_lat, this_lon)
            da = ds[name]
            # Load the tiny requested bbox before the dataset file is closed.
            da = da.load()
            time_name = _coord_name(ds, ("valid_time", "time", "date"))
            if time_name and time_name in da.dims:
                if time_name != "time":
                    da = da.rename({time_name: "time"})
            else:
                stamp = _filename_date(path)
                if stamp is None:
                    # A singleton non-spatial dimension may be time-like even
                    # when not exposed as a coordinate.
                    non_spatial = [d for d in da.dims if d not in {this_lat, this_lon}]
                    if len(non_spatial) == 1 and da.sizes[non_spatial[0]] == 1:
                        da = da.rename({non_spatial[0]: "time"})
                        stamp = pd.Timestamp("1970-01-01")
                    else:
                        continue
                if "time" not in da.dims:
                    da = da.expand_dims(time=[stamp])
                elif "time" not in da.coords:
                    da = da.assign_coords(time=[stamp])
            if this_lat != "latitude":
                da = da.rename({this_lat: "latitude"})
            if this_lon != "longitude":
                da = da.rename({this_lon: "longitude"})
            # Remove singleton metadata dimensions, but never spatial/time.
            for dim in list(da.dims):
                if dim not in {"time", "latitude", "longitude"} and da.sizes[dim] == 1:
                    da = da.isel({dim: 0}, drop=True)
            extra = [d for d in da.dims if d not in {"time", "latitude", "longitude"}]
            if extra:
                da = da.mean(dim=extra, skipna=True)
            arrays.append(da)
            lat_name = "latitude"
            lon_name = "longitude"
        finally:
            ds.close()

    if not arrays or not lat_name or not lon_name:
        raise ClimateDataError("No readable NetCDF discharge fields were found in the EWDS archive.")

    combined = xr.concat(arrays, dim="time").sortby("time")
    combined = combined.where(np.isfinite(combined))
    if selected_cell is None:
        med = combined.median(dim="time", skipna=True)
        values = np.asarray(med.values, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).any():
            raise ClimateDataError("The GloFAS bbox contains no finite river-discharge cells.")
        flat_index = int(np.nanargmax(values))
        i, j = np.unravel_index(flat_index, values.shape)
        lat = float(np.asarray(med["latitude"].values)[i])
        lon = float(np.asarray(med["longitude"].values)[j])
        median_discharge = float(values[i, j])
    else:
        lat = float(selected_cell["latitude"])
        lon = float(selected_cell["longitude"])
        picked = combined.sel(latitude=lat, longitude=lon, method="nearest")
        lat = float(picked["latitude"].values)
        lon = float(picked["longitude"].values)
        median_discharge = float(np.nanmedian(np.asarray(picked.values, dtype=float)))

    selected = combined.sel(latitude=lat, longitude=lon, method="nearest")
    times = pd.to_datetime(np.asarray(selected["time"].values), errors="coerce")
    values = np.asarray(selected.values, dtype=float).reshape(-1)
    frame = pd.DataFrame({"date": times, "value": values}).dropna(subset=["date", "value"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    series = pd.Series(frame["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(frame["date"], name="date"))
    series = series.where(series >= 0).dropna().rename("discharge_m3s")
    if series.empty:
        raise ClimateDataError("Selected GloFAS cell contains no finite non-negative discharge.")
    return series, {
        "latitude": float(lat),
        "longitude": float(lon),
        "median_discharge_m3s": float(median_discharge),
    }


def _archive_series(
    archive: Path,
    *,
    selected_cell: Mapping[str, float] | None,
) -> tuple[pd.Series, dict[str, float]]:
    with tempfile.TemporaryDirectory(prefix="glofas_extract_") as tmp:
        tmpdir = Path(tmp)
        if zipfile.is_zipfile(archive):
            files = _safe_extract_zip(archive, tmpdir)
        else:
            copy = tmpdir / (archive.stem + ".nc")
            copy.write_bytes(archive.read_bytes())
            files = [copy]
        return _load_discharge_collection(files, selected_cell=selected_cell)


def _year_chunks(start_year: int, end_year: int, size: int = GLOFAS_CHUNK_YEARS):
    year = int(start_year)
    while year <= int(end_year):
        end = min(int(end_year), year + int(size) - 1)
        yield list(range(year, end + 1))
        year = end + 1


def _merge_daily(parts: Iterable[pd.Series]) -> pd.Series:
    valid = [part.dropna().astype(float) for part in parts if part is not None and len(part)]
    if not valid:
        return pd.Series(dtype=float, name="discharge_m3s")
    merged = pd.concat(valid).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.index = pd.DatetimeIndex(merged.index, name="date")
    return merged.rename("discharge_m3s")


def _download_point_history(
    *,
    project_root: str | Path,
    point: RiverPoint,
    start_year: int,
    end_year: int,
    selected_cell: Mapping[str, float] | None = None,
    overwrite_downloads: bool = False,
    include_intermediate: bool = True,
) -> tuple[pd.Series, dict[str, float]]:
    parts = []
    cell = dict(selected_cell or {}) or None
    for years in _year_chunks(start_year, end_year):
        archive = _download_glofas_archive(
            project_root=project_root,
            point=point,
            years=years,
            product_type="consolidated",
            overwrite=overwrite_downloads,
        )
        series, found = _archive_series(archive, selected_cell=cell)
        if cell is None:
            cell = found
        parts.append(series)

    consolidated = _merge_daily(parts)
    if include_intermediate:
        # Near-real-time stream: current and previous calendar years are enough
        # to bridge the consolidated latency without re-downloading history.
        recent_years = [y for y in (end_year - 1, end_year) if y >= 1979]
        try:
            archive = _download_glofas_archive(
                project_root=project_root,
                point=point,
                years=recent_years,
                product_type="intermediate",
                overwrite=True,
            )
            intermediate, _ = _archive_series(archive, selected_cell=cell)
            if len(intermediate):
                # Consolidated is final and wins on overlap; intermediate only
                # fills dates that consolidated has not published yet.
                extra = intermediate.loc[~intermediate.index.isin(consolidated.index)]
                consolidated = _merge_daily([consolidated, extra])
        except Exception:
            # The historical build remains valid if the near-real-time stream
            # is temporarily unavailable.  The failure is recorded by the CLI
            # status/metadata through the latest date rather than corrupting the
            # long history.
            pass

    if cell is None:
        raise ClimateDataError(f"Could not select a GloFAS cell for {point.key}.")
    return consolidated, cell


def _seasonal_stats(values: pd.Series, *, start_year: int, end_year: int) -> tuple[pd.Series, pd.Series]:
    base = values.loc[(values.index.year >= start_year) & (values.index.year <= end_year)].dropna()
    if base.empty:
        raise ClimateDataError(
            f"No observations in climatology {start_year}-{end_year}."
        )
    grouped = base.groupby(base.index.month)
    mean = grouped.mean()
    std = grouped.std(ddof=1)
    return mean, std


def monthly_discharge_features(
    daily: pd.Series,
    *,
    prefix: str,
    label_prefix: str,
    climatology_start: int = GLOFAS_CLIMATOLOGY_START,
    climatology_end: int = GLOFAS_CLIMATOLOGY_END,
) -> pd.DataFrame:
    """Create seasonally-standardised discharge and low-flow stress indicators."""
    q = daily.astype(float).dropna().sort_index()
    if q.empty:
        raise ClimateDataError(f"{prefix}: empty daily discharge series.")
    if (q <= 0).all():
        raise ClimateDataError(f"{prefix}: discharge is non-positive throughout.")

    monthly_mean = q.resample("MS").mean()
    log_monthly = np.log(monthly_mean.clip(lower=1e-6))
    mu, sigma = _seasonal_stats(
        log_monthly,
        start_year=climatology_start,
        end_year=climatology_end,
    )
    month_number = pd.Series(log_monthly.index.month, index=log_monthly.index)
    mu_t = month_number.map(mu).astype(float)
    sd_t = month_number.map(sigma).astype(float)
    sd_t = sd_t.where(sd_t > 1e-8)
    discharge_stress = -(log_monthly - mu_t) / sd_t

    base_daily = q.loc[
        (q.index.year >= climatology_start) & (q.index.year <= climatology_end)
    ]
    thresholds = base_daily.groupby(base_daily.index.month).quantile(0.10)
    threshold_t = pd.Series(q.index.month, index=q.index).map(thresholds).astype(float)
    low_indicator = (q < threshold_t).astype(float)
    low_share = 100.0 * low_indicator.resample("MS").mean()
    low_mu, low_sd = _seasonal_stats(
        low_share,
        start_year=climatology_start,
        end_year=climatology_end,
    )
    low_month = pd.Series(low_share.index.month, index=low_share.index)
    low_mu_t = low_month.map(low_mu).astype(float)
    low_sd_t = low_month.map(low_sd).astype(float)
    fallback = float(low_share.loc[
        (low_share.index.year >= climatology_start)
        & (low_share.index.year <= climatology_end)
    ].std(ddof=1))
    if not np.isfinite(fallback) or fallback <= 1e-8:
        fallback = 10.0
    low_sd_t = low_sd_t.where(low_sd_t > 1e-8, fallback)
    low_stress = (low_share - low_mu_t) / low_sd_t

    blocks = []
    for name, label, raw, raw_unit, stress in (
        (
            f"{prefix}_discharge_stress",
            f"{label_prefix} · discharge stress",
            monthly_mean,
            "m³/s",
            discharge_stress,
        ),
        (
            f"{prefix}_low_flow_intensity",
            f"{label_prefix} · low-flow intensity",
            low_share,
            "% days below seasonal P10",
            low_stress,
        ),
    ):
        frame = pd.DataFrame(
            {
                "date": stress.index,
                "series": name,
                "label": label,
                "stress_score": pd.to_numeric(stress, errors="coerce").to_numpy(),
                "raw_value": pd.to_numeric(raw.reindex(stress.index), errors="coerce").to_numpy(),
                "raw_unit": raw_unit,
                "source": "Copernicus CEMS GloFAS",
                "source_version": GLOFAS_SOURCE_VERSION,
            }
        ).dropna(subset=["date", "stress_score"])
        blocks.append(frame)
    return pd.concat(blocks, ignore_index=True)


def _point_prefix(point_key: str) -> str:
    if point_key == "rhine_kaub":
        return "rhine"
    if point_key == "danube_vienna":
        return "danube"
    return point_key


def _save_processed_monthly(
    project_root: str | Path,
    *,
    daily_by_point: Mapping[str, pd.Series],
    cell_by_point: Mapping[str, Mapping[str, float]],
    build_mode: str,
) -> tuple[Path, Path]:
    paths = climate_paths(project_root)
    blocks = []
    point_meta = {}
    for key, point in RIVER_POINTS.items():
        daily = daily_by_point.get(key)
        if daily is None or len(daily) == 0:
            continue
        prefix = _point_prefix(key)
        blocks.append(
            monthly_discharge_features(
                daily,
                prefix=prefix,
                label_prefix=point.river,
            )
        )
        cell = dict(cell_by_point.get(key) or {})
        point_meta[key] = {
            "label": point.label,
            "river": point.river,
            "reference_latitude": point.latitude,
            "reference_longitude": point.longitude,
            "bbox": point.area,
            "selected_cell": cell,
            "selection_method": "maximum long-run median discharge within local bbox",
            "selection_note": (
                "Main-stem proxy used for v1. The selected model cell is persisted "
                "and reused on refresh; it is not presented as the PEGELONLINE gauge."
            ),
            "first_date": str(pd.Timestamp(daily.index.min()).date()),
            "last_date": str(pd.Timestamp(daily.index.max()).date()),
            "observations": int(len(daily)),
        }

    if not blocks:
        raise ClimateDataError("No GloFAS point produced monthly climate indicators.")
    monthly = pd.concat(blocks, ignore_index=True).sort_values(["series", "date"])
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly.to_parquet(paths["monthly"], index=False)

    metadata = {
        "contract_version": CLIMATE_DATA_CONTRACT_VERSION,
        "built_at_utc": _utc_now().isoformat(),
        "build_mode": str(build_mode),
        "glofas_dataset": GLOFAS_DATASET,
        "glofas_system_version": GLOFAS_SYSTEM_VERSION,
        "glofas_source_version": GLOFAS_SOURCE_VERSION,
        "glofas_hydrological_model": GLOFAS_HYDROLOGICAL_MODEL,
        "glofas_variable": GLOFAS_VARIABLE,
        "glofas_timespan": GLOFAS_TIMESPAN,
        "climatology": {
            "start_year": GLOFAS_CLIMATOLOGY_START,
            "end_year": GLOFAS_CLIMATOLOGY_END,
            "method": "calendar-month standardisation",
            "discharge_transform": "-z(log monthly mean discharge)",
            "low_flow_definition": "share of daily discharge below calendar-month climatological P10",
        },
        "series": list(SERIES_ORDER),
        "points": point_meta,
    }
    _write_json(paths["metadata"], metadata)
    return paths["monthly"], paths["metadata"]


def build_climate_history(
    project_root: str | Path,
    *,
    start_year: int = GLOFAS_DEFAULT_START_YEAR,
    end_year: int | None = None,
    overwrite_downloads: bool = False,
) -> dict[str, Any]:
    """One-time long-history GloFAS build for the two dashboard river points."""
    if int(start_year) > GLOFAS_CLIMATOLOGY_START:
        raise ClimateDataError(
            f"start_year must be <= {GLOFAS_CLIMATOLOGY_START} so the "
            "1991-2020 climatology is complete."
        )
    end_year = int(end_year or datetime.now().year)
    daily_by_point: dict[str, pd.Series] = {}
    cells: dict[str, Mapping[str, float]] = {}
    for key, point in RIVER_POINTS.items():
        daily, cell = _download_point_history(
            project_root=project_root,
            point=point,
            start_year=int(start_year),
            end_year=end_year,
            selected_cell=None,
            overwrite_downloads=overwrite_downloads,
            include_intermediate=True,
        )
        daily.to_frame().to_parquet(_daily_path(project_root, key))
        daily_by_point[key] = daily
        cells[key] = cell
    monthly_path, metadata_path = _save_processed_monthly(
        project_root,
        daily_by_point=daily_by_point,
        cell_by_point=cells,
        build_mode="bootstrap",
    )
    return {
        "monthly": str(monthly_path),
        "metadata": str(metadata_path),
        "points": {k: dict(v) for k, v in cells.items()},
    }


def refresh_climate_history(project_root: str | Path) -> dict[str, Any]:
    """Refresh only the recent GloFAS edge while preserving the long history."""
    paths = climate_paths(project_root)
    metadata = _read_json(paths["metadata"])
    if not metadata:
        raise ClimateDataError("No climate history exists yet; run the build command first.")
    daily_by_point: dict[str, pd.Series] = {}
    cells: dict[str, Mapping[str, float]] = {}
    current_year = datetime.now().year
    for key, point in RIVER_POINTS.items():
        daily_path = _daily_path(project_root, key)
        if not daily_path.is_file():
            raise ClimateDataError(f"Missing daily GloFAS cache: {daily_path}")
        old = pd.read_parquet(daily_path)
        if "discharge_m3s" in old.columns:
            old_series = old["discharge_m3s"]
        elif old.shape[1] == 1:
            old_series = old.iloc[:, 0]
        else:
            raise ClimateDataError(f"Unexpected daily cache schema: {daily_path}")
        old_series.index = pd.DatetimeIndex(pd.to_datetime(old_series.index), name="date")
        cell = (((metadata.get("points") or {}).get(key) or {}).get("selected_cell") or {})
        if not cell:
            raise ClimateDataError(f"Metadata lacks the persisted GloFAS cell for {key}.")
        recent, _ = _download_point_history(
            project_root=project_root,
            point=point,
            start_year=current_year - 1,
            end_year=current_year,
            selected_cell=cell,
            overwrite_downloads=True,
            include_intermediate=True,
        )
        merged = _merge_daily([old_series, recent])
        merged.to_frame().to_parquet(daily_path)
        daily_by_point[key] = merged
        cells[key] = cell
    monthly_path, metadata_path = _save_processed_monthly(
        project_root,
        daily_by_point=daily_by_point,
        cell_by_point=cells,
        build_mode="refresh",
    )
    return {"monthly": str(monthly_path), "metadata": str(metadata_path)}


# ---------------------------------------------------------------------------
# PEGELONLINE live Kaub panel
# ---------------------------------------------------------------------------


def _get_json(url: str, *, timeout: float = 8.0):
    response = requests.get(
        url,
        timeout=float(timeout),
        headers={"User-Agent": "InflationDashboard/1.0 climate module"},
    )
    response.raise_for_status()
    return response.json()


def fetch_kaub_live(*, timeout: float = 8.0) -> dict[str, Any]:
    current_url = f"{PEGELONLINE_BASE}/stations/{KAUB_UUID}/W/currentmeasurement.json"
    history_url = f"{PEGELONLINE_BASE}/stations/{KAUB_UUID}/W/measurements.json?start=P31D"
    forecast_url = f"{PEGELONLINE_BASE}/stations/{KAUB_UUID}/WV/measurements.json"
    current = _get_json(current_url, timeout=timeout)
    history = _get_json(history_url, timeout=timeout)
    try:
        forecast = _get_json(forecast_url, timeout=timeout)
    except Exception:
        forecast = []
    return {
        "source": "PEGELONLINE WSV",
        "station": "KAUB",
        "uuid": KAUB_UUID,
        "river": "RHEIN",
        "latitude": KAUB_LATITUDE,
        "longitude": KAUB_LONGITUDE,
        "fetched_at_utc": _utc_now().isoformat(),
        "status": "live",
        "current": current if isinstance(current, dict) else {},
        "history": history if isinstance(history, list) else [],
        "forecast": forecast if isinstance(forecast, list) else [],
    }


def load_kaub_live_cached(
    project_root: str | Path,
    *,
    refresh: bool = False,
    max_age_minutes: float = 15.0,
) -> dict[str, Any]:
    path = climate_paths(project_root)["kaub_cache"]
    cached = _read_json(path)
    fresh = False
    if cached.get("fetched_at_utc"):
        try:
            age = _utc_now() - pd.Timestamp(cached["fetched_at_utc"]).to_pydatetime()
            fresh = age.total_seconds() <= float(max_age_minutes) * 60.0
        except Exception:
            fresh = False
    if cached and fresh and not refresh:
        cached["status"] = "cached"
        return cached
    try:
        live = fetch_kaub_live()
        _write_json(path, live)
        return live
    except Exception as exc:
        if cached:
            cached["status"] = "cached_stale"
            cached["refresh_error"] = str(exc)
            return cached
        return {
            "source": "PEGELONLINE WSV",
            "station": "KAUB",
            "status": "unavailable",
            "error": str(exc),
            "current": {},
            "history": [],
            "forecast": [],
        }


def read_climate_monthly(project_root: str | Path) -> pd.DataFrame:
    path = climate_paths(project_root)["monthly"]
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "date", "series", "label", "stress_score", "raw_value",
                "raw_unit", "source", "source_version",
            ]
        )
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("stress_score", "raw_value"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values(["series", "date"])


def climate_status(project_root: str | Path) -> dict[str, Any]:
    paths = climate_paths(project_root)
    frame = read_climate_monthly(project_root)
    metadata = _read_json(paths["metadata"])
    per_series = {}
    if not frame.empty:
        for name, block in frame.groupby("series"):
            block = block.dropna(subset=["stress_score"]).sort_values("date")
            if not block.empty:
                per_series[str(name)] = {
                    "first": str(block["date"].iloc[0].date()),
                    "last": str(block["date"].iloc[-1].date()),
                    "observations": int(len(block)),
                }
    return {
        "contract_version": CLIMATE_DATA_CONTRACT_VERSION,
        "credentials": glofas_credentials_status(),
        "monthly_exists": paths["monthly"].is_file(),
        "metadata_exists": paths["metadata"].is_file(),
        "metadata": metadata,
        "series": per_series,
    }


def dashboard_snapshot(
    project_root: str | Path,
    *,
    refresh_live: bool = False,
) -> dict[str, Any]:
    monthly = read_climate_monthly(project_root)
    paths = climate_paths(project_root)
    meta = _read_json(paths["metadata"])
    kaub = load_kaub_live_cached(project_root, refresh=refresh_live)
    return {
        "contract_version": CLIMATE_DATA_CONTRACT_VERSION,
        "created_at_utc": _utc_now().isoformat(),
        "monthly_json": monthly.to_json(orient="split", date_format="iso", double_precision=15),
        "metadata": meta,
        "credentials": glofas_credentials_status(),
        "kaub": kaub,
    }


def _print_status(status: Mapping[str, Any]) -> None:
    print("=" * 78)
    print("ECONOMIC DATA · CLIMATE STATUS")
    print("=" * 78)
    cred = dict(status.get("credentials") or {})
    print(f"EWDS credentials : {'READY' if cred.get('configured') and cred.get('url_is_ewds') else 'NOT READY'}")
    print(f"  .cdsapirc      : {cred.get('cdsapirc_exists')}")
    print(f"  EWDS URL       : {cred.get('url_is_ewds')}")
    print(f"  key present    : {cred.get('key_present')}")
    print(f"monthly cache    : {status.get('monthly_exists')}")
    for name, info in dict(status.get("series") or {}).items():
        print(f"  {name:28s} {info.get('first')} -> {info.get('last')}  n={info.get('observations')}")
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/refresh climate data for Economic Data.")
    parser.add_argument("command", choices=("status", "build", "refresh"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--start-year", type=int, default=GLOFAS_DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--overwrite-downloads", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "status":
        _print_status(climate_status(args.project_root))
        return 0
    if args.command == "build":
        result = build_climate_history(
            args.project_root,
            start_year=args.start_year,
            end_year=args.end_year,
            overwrite_downloads=args.overwrite_downloads,
        )
        print(json.dumps(result, indent=2, default=str))
        _print_status(climate_status(args.project_root))
        return 0
    if args.command == "refresh":
        result = refresh_climate_history(args.project_root)
        print(json.dumps(result, indent=2, default=str))
        _print_status(climate_status(args.project_root))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
