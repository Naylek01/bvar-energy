from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "dashboard" / "economic_data" / "climate_data.py"
spec = importlib.util.spec_from_file_location("economic_climate_data_test", MODULE_PATH)
assert spec and spec.loader
climate_data = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = climate_data
spec.loader.exec_module(climate_data)


def test_current_glofas_v5_request_contract():
    req = climate_data.build_glofas_request(
        years=[2025, 2026],
        area=[50.2, 7.6, 49.9, 7.9],
        product_type="consolidated",
    )
    assert req["system_version"] == ["version_5_0"]
    assert req["hydrological_model"] == ["lisflood"]
    assert req["variable"] == ["average_river_discharge_in_the_last_24_hours"]
    assert req["timespan"] == ["time_mean"]
    assert req["year"] == ["2025", "2026"]
    assert req["data_format"] == "netcdf"
    assert req["download_format"] == "zip"
    assert "hyear" not in req
    assert "hmonth" not in req
    assert "hday" not in req


def test_seasonal_discharge_stress_has_expected_sign():
    idx = pd.date_range("1991-01-01", "2021-12-31", freq="D")
    # Add small deterministic year-to-year variation so climatological SD is non-zero.
    seasonal = 1500.0 + 400.0 * np.cos(2.0 * np.pi * (idx.dayofyear.to_numpy() - 20) / 365.25)
    year_effect = 1.0 + 0.04 * np.sin((idx.year.to_numpy() - 1991) * 1.7)
    daily = pd.Series(seasonal * year_effect, index=idx, name="discharge_m3s")
    mask = (daily.index.year == 2021) & (daily.index.month == 8)
    daily.loc[mask] *= 0.35

    out = climate_data.monthly_discharge_features(
        daily,
        prefix="rhine",
        label_prefix="Rhine",
    )
    discharge = out.loc[out["series"].eq("rhine_discharge_stress")].set_index("date")
    lowflow = out.loc[out["series"].eq("rhine_low_flow_intensity")].set_index("date")

    aug = pd.Timestamp("2021-08-01")
    assert discharge.loc[aug, "stress_score"] > 2.0
    assert lowflow.loc[aug, "stress_score"] > 0.0
    assert lowflow.loc[aug, "raw_value"] > 50.0


def test_feature_integration_is_additive():
    text = (ROOT / "src" / "dashboard" / "economic_data" / "feature.py").read_text(encoding="utf-8")
    assert "from .climate import climate_section, register_climate_callbacks" in text
    assert "climate_section()," in text
    assert "register_climate_callbacks(" in text
