"""Strict repository-versioned production-state restoration."""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

PRODUCTION_MANIFEST_SCHEMA_VERSION = 1
PRODUCTION_MANIFEST_CONTRACT = "ecb-var-production-manifest-v1"


class ProductionManifestError(RuntimeError):
    pass


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProductionManifestError(f"Production manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProductionManifestError(
            f"Invalid production manifest {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProductionManifestError("Production manifest root must be an object.")
    return payload


def load_production_manifest(
    project_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else root / "configs" / "production_manifest.json"
    )
    payload = _read_manifest(path)

    if int(payload.get("schema_version") or 0) != PRODUCTION_MANIFEST_SCHEMA_VERSION:
        raise ProductionManifestError(
            f"Unsupported production manifest schema_version={payload.get('schema_version')!r}."
        )
    if str(payload.get("contract") or "") != PRODUCTION_MANIFEST_CONTRACT:
        raise ProductionManifestError(
            f"Unsupported production manifest contract={payload.get('contract')!r}."
        )

    vintage = str(payload.get("production_vintage") or "").strip()
    forecast_name = str(payload.get("forecast_name") or "").strip()
    if not vintage or not forecast_name:
        raise ProductionManifestError(
            "production_vintage and forecast_name are required."
        )

    components = payload.get("energy_components")
    if not isinstance(components, dict) or len(components) != 7:
        raise ProductionManifestError(
            "energy_components must contain exactly seven entries."
        )
    for model_id, run_id in components.items():
        if not str(model_id).strip() or not str(run_id).strip():
            raise ProductionManifestError(
                f"Invalid Energy identity {model_id!r} -> {run_id!r}."
            )

    headline = payload.get("headline")
    aggregate = payload.get("energy_aggregate")
    if not isinstance(headline, dict) or not str(headline.get("run_id") or "").strip():
        raise ProductionManifestError("headline.run_id is required.")
    if (
        not isinstance(aggregate, dict)
        or not str(aggregate.get("aggregate_run_id") or "").strip()
    ):
        raise ProductionManifestError("energy_aggregate.aggregate_run_id is required.")

    serialised = json.dumps(payload, sort_keys=True)
    for token in (":\\\\", "\\\\Users\\\\", "/Users/", "/home/"):
        if token in serialised:
            raise ProductionManifestError(
                "Production manifest contains an absolute machine path."
            )
    return payload


def _connect_ro(path: Path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _one(con, sql: str, params: tuple[Any, ...], label: str):
    rows = con.execute(sql, params).fetchall()
    if len(rows) != 1:
        raise ProductionManifestError(
            f"{label}: expected exactly one usable registry row, found {len(rows)}."
        )
    return rows[0]


def validate_production_manifest(
    manifest: Mapping[str, Any],
    *,
    registry_path: str | Path,
) -> dict[str, Any]:
    registry = Path(registry_path).expanduser().resolve()
    if not registry.is_file():
        raise ProductionManifestError(f"Registry missing after scan: {registry}")

    vintage = str(manifest["production_vintage"])
    forecast_name = str(manifest["forecast_name"])
    components = dict(manifest["energy_components"])
    headline_run_id = str(manifest["headline"]["run_id"])
    aggregate_run_id = str(manifest["energy_aggregate"]["aggregate_run_id"])
    aggregate_forecast_name = str(
        manifest["energy_aggregate"].get("forecast_name") or forecast_name
    )

    con = _connect_ro(registry)
    try:
        for model_id, run_id in components.items():
            _one(
                con,
                """
                SELECT r.run_id
                FROM runs r
                JOIN forecasts f
                  ON f.model_id=r.model_id
                 AND f.vintage=r.vintage
                 AND f.run_id=r.run_id
                WHERE r.model_id=? AND CAST(r.vintage AS TEXT)=?
                  AND r.run_id=?
                  AND r.status='complete'
                  AND r.present_on_disk=1
                  AND f.forecast_name=?
                  AND f.valid=1
                  AND f.present_on_disk=1
                """,
                (str(model_id), vintage, str(run_id), forecast_name),
                f"Energy component {model_id}/{vintage}/{run_id}",
            )

        _one(
            con,
            """
            SELECT r.run_id
            FROM runs r
            JOIN forecasts f
              ON f.model_id=r.model_id
             AND f.vintage=r.vintage
             AND f.run_id=r.run_id
            WHERE r.model_id='headline_joint'
              AND CAST(r.vintage AS TEXT)=?
              AND r.run_id=?
              AND r.status='complete'
              AND r.present_on_disk=1
              AND f.forecast_name=?
              AND f.valid=1
              AND f.present_on_disk=1
            """,
            (vintage, headline_run_id, forecast_name),
            f"Headline/{vintage}/{headline_run_id}",
        )

        cols = {row[1] for row in con.execute("PRAGMA table_info(aggregates)").fetchall()}
        required = {"aggregate_run_id", "vintage", "status", "present_on_disk"}
        missing = required - cols
        if missing:
            raise ProductionManifestError(
                f"Aggregate registry schema missing {sorted(missing)}."
            )
        where = [
            "CAST(vintage AS TEXT)=?",
            "aggregate_run_id=?",
            "status='complete'",
            "present_on_disk=1",
        ]
        params: list[Any] = [vintage, aggregate_run_id]
        if "forecast_name" in cols:
            where.append("forecast_name=?")
            params.append(aggregate_forecast_name)
        _one(
            con,
            "SELECT aggregate_run_id FROM aggregates WHERE " + " AND ".join(where),
            tuple(params),
            f"HICP Energy aggregate/{vintage}/{aggregate_run_id}",
        )
    finally:
        con.close()

    return {
        "production_vintage": vintage,
        "forecast_name": forecast_name,
        "energy_components": components,
        "headline_run_id": headline_run_id,
        "energy_aggregate_run_id": aggregate_run_id,
        "energy_aggregate_forecast_name": aggregate_forecast_name,
    }


def _invoke_existing(fn, values: Mapping[str, Any], label: str) -> Any:
    sig = inspect.signature(fn)
    aliases = {"path": "registry_path", "registry": "registry_path"}
    args = []
    kwargs = {}

    for name, param in sig.parameters.items():
        key = aliases.get(name, name)
        if key in values:
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(values[key])
            else:
                kwargs[name] = values[key]
            continue
        if (
            param.default is inspect._empty
            and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ):
            raise ProductionManifestError(
                f"{label}: unsupported required parameter {name!r}; signature={sig}."
            )
    return fn(*args, **kwargs)


def _verify_promotions(manifest: Mapping[str, Any], registry_path: Path) -> None:
    vintage = str(manifest["production_vintage"])
    expected = dict(manifest["energy_components"])
    expected["headline_joint"] = str(manifest["headline"]["run_id"])
    aggregate_run_id = str(manifest["energy_aggregate"]["aggregate_run_id"])

    con = _connect_ro(registry_path)
    try:
        for model_id, run_id in expected.items():
            got = [
                str(row[0])
                for row in con.execute(
                    """
                    SELECT run_id FROM runs
                    WHERE model_id=? AND CAST(vintage AS TEXT)=? AND promoted=1
                    """,
                    (str(model_id), vintage),
                ).fetchall()
            ]
            if got != [str(run_id)]:
                raise ProductionManifestError(
                    f"Promotion mismatch for {model_id}: expected {[str(run_id)]}, found {got}."
                )

        got_aggs = [
            str(row[0])
            for row in con.execute(
                """
                SELECT aggregate_run_id FROM aggregates
                WHERE CAST(vintage AS TEXT)=? AND promoted=1
                """,
                (vintage,),
            ).fetchall()
        ]
        if aggregate_run_id not in got_aggs:
            raise ProductionManifestError(
                f"Aggregate promotion mismatch: expected {aggregate_run_id!r} among {got_aggs}."
            )
    finally:
        con.close()


def restore_production_state(
    *,
    project_root: str | Path,
    results_root: str | Path,
    registry_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Restore exact production promotions after scan_results()."""
    root = Path(project_root).expanduser().resolve()
    results = Path(results_root).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()

    manifest = load_production_manifest(root, manifest_path=manifest_path)
    state = validate_production_manifest(manifest, registry_path=registry)

    from inflation_bvar_registry import promote_run
    from energy_bvar_registry import promote_aggregate

    common = {
        "registry_path": registry,
        "vintage": state["production_vintage"],
        "forecast_name": state["forecast_name"],
        "results_root": results,
        "project_root": root,
    }

    for model_id, run_id in state["energy_components"].items():
        _invoke_existing(
            promote_run,
            {**common, "model_id": model_id, "run_id": run_id},
            f"promote_run({model_id})",
        )

    _invoke_existing(
        promote_run,
        {
            **common,
            "model_id": "headline_joint",
            "run_id": state["headline_run_id"],
        },
        "promote_run(headline_joint)",
    )

    _invoke_existing(
        promote_aggregate,
        {
            **common,
            "forecast_name": state["energy_aggregate_forecast_name"],
            "aggregate_run_id": state["energy_aggregate_run_id"],
            "run_id": state["energy_aggregate_run_id"],
        },
        "promote_aggregate(HICP Energy)",
    )

    _verify_promotions(manifest, registry)
    return state
