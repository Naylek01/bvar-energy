"""Dataset-build orchestration for the Energy BVAR dashboard.

This module is deliberately separate from the Dash callbacks:
- it discovers the canonical workbook and builder;
- serialises dataset builds with an atomic lock;
- invokes the existing builder *in process* (no second Python environment);
- validates the production output contract;
- never writes below ``results/``.

The econometric/data transformations remain owned by the canonical
``build_dataset_headline_joint_v*_FINAL.py`` script. Legacy six-model builder
names remain discoverable only as a backward-compatible fallback.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import date, datetime, timezone
from io import StringIO
import hashlib
import importlib.util
import json
import os
import re
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable


EXPECTED_ENERGY_OUTPUTS = (
    "car_fuels_weekly.csv",
    "liquid_fuels_weekly.csv",
    "gas_monthly.csv",
    "electricity_monthly.csv",
    "heat_energy_monthly.csv",
    "solid_fuels_monthly.csv",
)

EXPECTED_HICP_OUTPUTS = (
    "hicp_indices_monthly.csv",
    "hicp_weights_annual.csv",
    "hicp_series_metadata.csv",
    "hicp_flags.csv",
    "hicp_weight_identity_diagnostics.csv",
)

EXPECTED_HEADLINE_OUTPUTS = (
    "headline_joint_monthly.csv",
    "headline_joint_weights_annual.csv",
    "headline_joint_weight_identity_diagnostics.csv",
    "headline_hicp_indices_monthly.csv",
    "headline_hicp_weights_annual.csv",
    "headline_hicp_weight_identity_diagnostics.csv",
)

EXPECTED_AUDIT_OUTPUTS = (
    "manifest.json",
    "model_datasets_diagnostics.csv",
    "source_vintages.csv",
    "aggregation_coverage.csv",
    "partial_period_inputs.csv",
)

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

DATASET_LOCK_NAME = ".dataset_build.lock"


class DatasetBuildError(RuntimeError):
    """Raised when the dashboard dataset-build contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_vintage_name(name: str) -> date | None:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(name), fmt).date()
        except ValueError:
            pass
    return None


def _version_key(path: Path) -> tuple[int, int, int]:
    """Rank builder filenames such as v10, v10_1, v11_FINAL."""
    match = re.search(r"_v(\d+)(?:[_\.](\d+))?", path.stem, flags=re.IGNORECASE)
    major = int(match.group(1)) if match else -1
    minor = int(match.group(2) or 0) if match else 0
    return major, minor, int(path.stat().st_mtime_ns)


def discover_dataset_builder(
    project_root: str | Path,
    explicit: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()

    candidate = explicit or os.getenv("ENERGY_BVAR_DATASET_BUILDER")
    if candidate:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise DatasetBuildError(f"Dataset builder does not exist: {path}")
        return path

    pipeline_dir = root / "src" / "data_pipeline"
    if not pipeline_dir.is_dir():
        raise DatasetBuildError(
            f"Dataset pipeline folder does not exist: {pipeline_dir}"
        )

    # Prefer the final joint-Headline builder family. Legacy six-model names
    # remain a fallback so old checkouts are still diagnosable.
    families = (
        (2, "build_dataset_headline_joint_v*.py"),
        (1, "build_dataset_paper_six_models_v*.py"),
    )
    ranked: list[tuple[int, Path]] = []
    for family_rank, pattern in families:
        for p in pipeline_dir.glob(pattern):
            if (
                p.is_file()
                and not p.name.startswith("~")
                and "backup" not in p.name.casefold()
                and "old" not in p.name.casefold()
            ):
                ranked.append((family_rank, p))
    if not ranked:
        raise DatasetBuildError(
            "No canonical dataset builder found in "
            f"{pipeline_dir}. Expected build_dataset_headline_joint_v*.py "
            "or the legacy build_dataset_paper_six_models_v*.py family. "
            "Set ENERGY_BVAR_DATASET_BUILDER to override."
        )
    return max(
        ranked,
        key=lambda item: (item[0], *_version_key(item[1])),
    )[1].resolve()


def discover_raw_workbook(
    project_root: str | Path,
    explicit: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()

    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise DatasetBuildError(f"Raw workbook does not exist: {path}")
        return path

    raw_root = root / "data" / "raw"
    if not raw_root.is_dir():
        raise DatasetBuildError(f"Raw-data folder does not exist: {raw_root}")

    # The production contract is one living top-level workbook.  Prefer it
    # deterministically over any archived/datestamped copies below data/raw.
    for name in PREFERRED_RAW_WORKBOOK_NAMES:
        canonical = raw_root / name
        if canonical.is_file() and not canonical.name.startswith("~$"):
            return canonical.resolve()

    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in RAW_WORKBOOK_PATTERNS:
        for path in raw_root.rglob(pattern):
            if not path.is_file() or path.name.startswith("~$"):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)

    if not candidates:
        top_level = [
            p
            for suffix in (".xlsm", ".xlsx")
            for p in raw_root.glob(f"*{suffix}")
            if p.is_file() and not p.name.startswith("~$")
        ]
        if len(top_level) == 1:
            candidates = top_level
        elif len(top_level) > 1:
            listing = "\n  ".join(str(p.relative_to(root)) for p in top_level)
            raise DatasetBuildError(
                "Could not auto-select the production workbook because several "
                f"top-level Excel files exist:\n  {listing}\n"
                "Enter the workbook path explicitly in the Data page."
            )

    if not candidates:
        raise DatasetBuildError(
            f"No production .xlsm/.xlsx workbook found below {raw_root}. "
            "Expected bvar_energy_raw_data* or raw_energy_bvar*."
        )

    preference = {
        name.casefold(): len(PREFERRED_RAW_WORKBOOK_NAMES) - i
        for i, name in enumerate(PREFERRED_RAW_WORKBOOK_NAMES)
    }

    def rank(path: Path) -> tuple[int, int, int, int]:
        vintage = _parse_vintage_name(path.parent.name)
        vintage_ord = vintage.toordinal() if vintage else -1
        exact_preference = preference.get(path.name.casefold(), 0)
        macro_enabled = int(path.suffix.casefold() == ".xlsm")
        return (
            vintage_ord,
            exact_preference,
            macro_enabled,
            int(path.stat().st_mtime_ns),
        )

    return max(candidates, key=rank).resolve()


def default_build_vintage(raw_path: str | Path) -> str:
    raw = Path(raw_path)
    parent_vintage = _parse_vintage_name(raw.parent.name)
    if parent_vintage is not None:
        return parent_vintage.strftime("%Y%m%d")
    return datetime.now().strftime("%Y%m%d")


def dataset_lock_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    return processed / DATASET_LOCK_NAME


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def dataset_build_lock_state(
    project_root: str | Path,
    *,
    clear_stale: bool = True,
) -> dict[str, Any]:
    path = dataset_lock_path(project_root)
    payload = _read_json(path)
    if not payload:
        if path.exists() and clear_stale:
            try:
                path.unlink()
            except OSError:
                pass
        return {"active": False, "path": str(path)}

    pid = int(payload.get("pid", -1) or -1)
    alive = _pid_is_alive(pid)
    if not alive and clear_stale:
        try:
            path.unlink()
        except OSError:
            pass
        return {"active": False, "path": str(path), "stale_cleared": True}

    return {
        **payload,
        "active": bool(alive),
        "path": str(path),
    }


@contextmanager
def _dataset_build_lock(
    project_root: str | Path,
    *,
    raw_path: Path,
    build_vintage: str,
):
    path = dataset_lock_path(project_root)
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "raw_path": str(raw_path),
        "build_vintage": str(build_vintage),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            state = dataset_build_lock_state(project_root, clear_stale=True)
            if attempt == 0 and not state.get("active"):
                continue
            raise DatasetBuildError(
                "Another dataset build is already running"
                + (
                    f" for vintage {state.get('build_vintage')}"
                    if state.get("build_vintage")
                    else ""
                )
                + "."
            )
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            break

    try:
        yield payload
    finally:
        try:
            current = _read_json(path)
            if current.get("token") == token:
                path.unlink(missing_ok=True)
        except Exception:
            pass


def _load_builder_module(path: Path):
    module_name = f"_energy_bvar_dataset_builder_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DatasetBuildError(f"Could not import dataset builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_dataset"):
        raise DatasetBuildError(
            f"{path.name} does not expose the required build_dataset(**kwargs) API."
        )
    return module


def inspect_dataset_build_environment(
    project_root: str | Path,
    *,
    raw_path: str | Path | None = None,
    builder_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    builder = discover_dataset_builder(root, builder_path)
    raw = discover_raw_workbook(root, raw_path)
    return {
        "builder_path": str(builder),
        "builder_relative": (
            str(builder.relative_to(root))
            if builder.is_relative_to(root)
            else str(builder)
        ),
        "raw_path": str(raw),
        "raw_relative": (
            str(raw.relative_to(root))
            if raw.is_relative_to(root)
            else str(raw)
        ),
        "raw_modified_at": datetime.fromtimestamp(
            raw.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds"),
        "raw_size_bytes": int(raw.stat().st_size),
        "default_build_vintage": default_build_vintage(raw),
        "lock": dataset_build_lock_state(root),
    }


def _validate_outputs(
    output_dir: Path,
    *,
    require_hicp: bool,
    require_headline: bool,
) -> list[str]:
    required = list(EXPECTED_ENERGY_OUTPUTS) + list(EXPECTED_AUDIT_OUTPUTS)
    if require_hicp:
        required.extend(EXPECTED_HICP_OUTPUTS)
    if require_headline:
        required.extend(EXPECTED_HEADLINE_OUTPUTS)
    return [name for name in required if not (output_dir / name).is_file()]


def run_dataset_build(
    project_root: str | Path,
    *,
    raw_path: str | Path | None = None,
    builder_path: str | Path | None = None,
    build_vintage: str | None = None,
    overwrite: bool = False,
    require_hicp: bool = True,
    require_headline: bool = True,
    hicp_discovery: bool = False,
    progress_callback: Callable[[int, str, str], None] | None = None,
) -> dict[str, Any]:
    """Run the canonical workbook -> processed-data builder.

    ``progress_callback`` receives ``(percent, phase, detail)``. The canonical
    builder is intentionally not modified for UI progress, so the dashboard
    reports coarse orchestration milestones around the validated builder call.
    """
    root = Path(project_root).resolve()
    started = time.monotonic()

    builder = discover_dataset_builder(root, builder_path)
    raw = discover_raw_workbook(root, raw_path)
    effective_vintage = str(build_vintage or "").strip() or default_build_vintage(raw)

    if progress_callback:
        progress_callback(
            5,
            "Preflight",
            f"Workbook {raw.name} · target processed vintage {effective_vintage}.",
        )

    module = _load_builder_module(builder)
    script_version = str(getattr(module, "SCRIPT_VERSION", builder.stem))

    stdout = StringIO()
    stderr = StringIO()
    output_root = root / "data" / "processed"

    with _dataset_build_lock(
        root,
        raw_path=raw,
        build_vintage=effective_vintage,
    ):
        if progress_callback:
            progress_callback(
                15,
                "Building datasets",
                f"Running {builder.name} · HICP "
                f"{'required' if require_hicp else 'optional'} · Headline "
                f"{'required' if require_headline else 'optional'}.",
            )
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                outputs = module.build_dataset(
                    raw_path=raw,
                    raw_root=root / "data" / "raw",
                    output_root=output_root,
                    build_vintage=effective_vintage,
                    weekly_commodity_shift_weeks=1,
                    overwrite=bool(overwrite),
                    max_tax_carry_months=12,
                    hicp_discovery=bool(hicp_discovery),
                    require_hicp=bool(require_hicp),
                    require_headline=bool(require_headline),
                )
        except Exception as exc:
            log = (stdout.getvalue() + "\n" + stderr.getvalue()).strip()
            raise DatasetBuildError(
                f"{type(exc).__name__}: {exc}"
                + (f"\n\nBuilder log:\n{log}" if log else "")
            ) from exc

    output_dir = output_root / effective_vintage
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("build_vintage"):
        effective_vintage = str(manifest["build_vintage"])
        output_dir = output_root / effective_vintage

    missing = _validate_outputs(
        output_dir,
        require_hicp=require_hicp,
        require_headline=require_headline,
    )
    if missing:
        raise DatasetBuildError(
            "Builder returned without the complete production output contract. "
            f"Missing below {output_dir}: {missing}"
        )

    if progress_callback:
        progress_callback(
            90,
            "Validating outputs",
            "Processed Energy datasets, complete HICP aggregation lineage, "
            "joint Headline inputs and audit manifest found.",
        )

    raw_hash = _sha256(raw)
    log = (stdout.getvalue() + "\n" + stderr.getvalue()).strip()
    model_files = {
        name: str((output_dir / name).resolve())
        for name in EXPECTED_ENERGY_OUTPUTS
    }
    hicp_files = {
        name: str((output_dir / name).resolve())
        for name in EXPECTED_HICP_OUTPUTS
        if (output_dir / name).is_file()
    }
    headline_files = {
        name: str((output_dir / name).resolve())
        for name in EXPECTED_HEADLINE_OUTPUTS
        if (output_dir / name).is_file()
    }

    payload = {
        "ok": True,
        "status": "complete",
        "build_vintage": effective_vintage,
        "output_dir": str(output_dir.resolve()),
        "builder_path": str(builder),
        "builder_version": script_version,
        "raw_path": str(raw),
        "raw_sha256": raw_hash,
        "raw_modified_at": datetime.fromtimestamp(
            raw.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds"),
        "overwrite": bool(overwrite),
        "require_hicp": bool(require_hicp),
        "require_headline": bool(require_headline),
        "model_files": model_files,
        "hicp_files": hicp_files,
        "headline_files": headline_files,
        "manifest_path": str(manifest_path.resolve()),
        "manifest": manifest,
        "log": log,
        "elapsed_seconds": float(time.monotonic() - started),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if progress_callback:
        progress_callback(
            100,
            "Processed vintage ready",
            f"Vintage {effective_vintage} is ready for the seven-model Energy suite "
            "and the joint Headline BVAR.",
        )
    return payload


__all__ = [
    "DatasetBuildError",
    "EXPECTED_ENERGY_OUTPUTS",
    "EXPECTED_HICP_OUTPUTS",
    "EXPECTED_HEADLINE_OUTPUTS",
    "dataset_build_lock_state",
    "default_build_vintage",
    "discover_dataset_builder",
    "discover_raw_workbook",
    "inspect_dataset_build_environment",
    "run_dataset_build",
]
