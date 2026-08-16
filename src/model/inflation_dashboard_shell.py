"""Pure domain/routing helpers for the unified Inflation Dashboard."""

from __future__ import annotations

from energy_bvar_pipeline import CANONICAL_MODEL_IDS, model_spec

HEADLINE_MODEL_ID = "headline_joint"
ENERGY_MODEL_IDS = tuple(CANONICAL_MODEL_IDS)
HEADLINE_MODEL_IDS = (HEADLINE_MODEL_ID,)

ENERGY_ROUTES = {
    "forecast": "/forecast",
    "aggregate": "/aggregate",
    "scenarios": "/scenarios",
    "structural": "/structural",
    "estimation": "/estimation",
}
HEADLINE_ROUTES = {
    "overview": "/headline/overview",
    "forecast": "/headline/forecast",
    "contributions": "/headline/contributions",
    "components": "/headline/components",
    "structural": "/headline/structural",
    "diagnostics": "/headline/diagnostics",
}


def domain_from_path(pathname: str | None) -> str:
    path = str(pathname or "/forecast")
    return (
        "headline"
        if path == "/headline" or path.startswith("/headline/")
        else "energy"
    )


def models_for_domain(model_ids, domain: str) -> list[str]:
    allowed = (
        set(HEADLINE_MODEL_IDS)
        if str(domain) == "headline"
        else set(ENERGY_MODEL_IDS)
    )
    return sorted(
        str(model_id)
        for model_id in model_ids
        if str(model_id) in allowed
    )


def model_label(model_id: str) -> str:
    value = str(model_id)
    if value == HEADLINE_MODEL_ID:
        return "Headline HICP"
    try:
        return str(model_spec(value).label)
    except Exception:
        return value.replace("_", " ").title()


def default_path(domain: str) -> str:
    return (
        HEADLINE_ROUTES["overview"]
        if str(domain) == "headline"
        else ENERGY_ROUTES["forecast"]
    )


__all__ = [
    "HEADLINE_MODEL_ID",
    "ENERGY_MODEL_IDS",
    "HEADLINE_MODEL_IDS",
    "ENERGY_ROUTES",
    "HEADLINE_ROUTES",
    "domain_from_path",
    "models_for_domain",
    "model_label",
    "default_path",
]
