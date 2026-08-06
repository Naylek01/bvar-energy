"""Patch ``src/model/energy_bvar_model.py`` for ragged-edge level conditioning.

The patch keeps the fitted VAR in absolute differences, but augments the
Durbin--Koopman forecast state with cumulative levels.  Future level conditions
are therefore imposed directly inside the same simulation-smoother draw that
fills the ragged edge.

Run from the bvar-energy project root::

    python apply_energy_bvar_ragged_edge_fix.py

A timestamped backup is created beside the model file before replacement.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
import py_compile
import shutil
import tempfile
import textwrap


DEFAULT_TARGET = Path("src/model/energy_bvar_model.py")
PATCH_MARKER = "RAGGED_EDGE_LEVEL_STATE_FIX_2026_08_06"


NORMALISE_FUNCTION = textwrap.dedent(
    '''
    def _normalise_level_conditions(
        conditions: Mapping[str, float | Sequence[float]] | None,
        variables: Sequence[str],
        H: int,
        levels: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Validate future level paths without requiring a balanced panel edge.

        A conditioned variable only needs at least one observed historical
        level.  Any missing months between its last observation and the first
        future condition remain missing observations and are drawn jointly by
        the augmented Durbin--Koopman smoother.
        """
        if conditions is None:
            return {}

        out: dict[str, np.ndarray] = {}
        for variable, value in conditions.items():
            if variable not in variables:
                raise ValueError(f"Unknown conditioned variable {variable!r}.")

            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = np.repeat(float(arr), H)
            if arr.ndim != 1 or len(arr) != H:
                raise ValueError(
                    f"Condition for {variable!r} must be scalar or length H={H}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Level conditions must be finite; omit unconstrained "
                    "variables instead."
                )
            if levels[variable].dropna().empty:
                raise ValueError(
                    f"No observed historical level is available for "
                    f"conditioned variable {variable!r}."
                )
            out[variable] = arr

        return out
    '''
).strip() + "\n"


LEVEL_SMOOTHER_FUNCTION = textwrap.dedent(
    f'''
    # {PATCH_MARKER}
    def _durbin_koopman_level_draw(
        level_endog: np.ndarray,
        B: np.ndarray,
        covariance_sequence: np.ndarray,
        initial_companion_state: np.ndarray,
        rng: np.random.Generator,
        *extra_args,
        **extra_kwargs,
    ) -> np.ndarray:
        """Draw differences and cumulative levels jointly.

        The original companion state contains the current difference and its
        lags.  This function appends

            cumulative_t = level_t - level_at_balanced_end

        to that state.  ``level_endog`` observes the appended block directly.
        Missing ragged-edge levels stay NaN; future conditioned levels are exact
        observations.  The same reduced-form innovation drives both the new
        difference and the cumulative-level update.

        Optional deterministic/exogenous information is accepted in either of
        two forms:

        * an ``(L, d)`` future-exog matrix, multiplied by the coefficient rows
          after ``constant + n*p lag rows``;
        * an already assembled ``(L, n)`` reduced-form intercept path.

        Common keyword aliases used by earlier project versions are accepted.
        """
        level_endog = np.asarray(level_endog, dtype=float)
        covariance_sequence = np.asarray(covariance_sequence, dtype=float)
        B = np.asarray(B, dtype=float)
        initial_companion_state = np.asarray(
            initial_companion_state, dtype=float
        ).reshape(-1)

        if level_endog.ndim != 2:
            raise ValueError("level_endog must be a two-dimensional array.")
        L, n = level_endog.shape
        if L < 1:
            raise ValueError("The state-space forecast path is empty.")
        if covariance_sequence.shape != (L, n, n):
            raise ValueError(
                "covariance_sequence must have shape (L, n, n)."
            )
        if initial_companion_state.size % n:
            raise ValueError(
                "initial_companion_state length is not divisible by n."
            )

        inferred_p = initial_companion_state.size // n
        supplied_p = extra_kwargs.pop("p", None)
        p = inferred_p if supplied_p is None else int(supplied_p)
        if p != inferred_p or p < 1:
            raise ValueError(
                "p is inconsistent with initial_companion_state."
            )

        # Recover a deterministic input supplied by the existing forecast code.
        deterministic = None
        deterministic_is_intercept = False
        alias_order = (
            ("state_intercept_sequence", True),
            ("state_intercepts", True),
            ("intercept_sequence", True),
            ("deterministic_intercept", True),
            ("deterministic_future", False),
            ("future_exog", False),
            ("exog_future", False),
        )
        for key, is_intercept in alias_order:
            if key in extra_kwargs:
                if deterministic is not None:
                    raise TypeError(
                        "Pass only one deterministic/intercept path."
                    )
                deterministic = extra_kwargs.pop(key)
                deterministic_is_intercept = is_intercept

        if len(extra_args) > 1:
            raise TypeError(
                "At most one extra positional deterministic path is supported."
            )
        if extra_args:
            if deterministic is not None:
                raise TypeError(
                    "The deterministic path was supplied both positionally "
                    "and by keyword."
                )
            deterministic = extra_args[0]

        if extra_kwargs:
            unknown = ", ".join(sorted(extra_kwargs))
            raise TypeError(
                f"Unknown _durbin_koopman_level_draw arguments: {{unknown}}"
            )

        minimum_rows = 1 + n * p
        if B.ndim != 2 or B.shape[1] != n or B.shape[0] < minimum_rows:
            raise ValueError(
                f"B must have at least {{minimum_rows}} rows and {{n}} columns; "
                f"got {{B.shape}}."
            )

        # Difference companion: constant first, lag-major blocks next.
        companion_size = n * p
        F = np.zeros((companion_size, companion_size))
        F[:n] = B[1:minimum_rows].T
        if p > 1:
            F[n:, :-n] = np.eye(n * (p - 1))
        selection = np.vstack(
            [np.eye(n), np.zeros((n * (p - 1), n))]
        )
        J = np.hstack(
            [np.eye(n), np.zeros((n, companion_size - n))]
        )

        n_exog = B.shape[0] - minimum_rows
        intercept_path = np.repeat(B[0][None, :], L, axis=0)

        if deterministic is not None:
            deterministic = np.asarray(deterministic, dtype=float)
            if deterministic.ndim == 1:
                deterministic = deterministic[:, None]
            if deterministic.ndim != 2:
                raise ValueError(
                    "The deterministic path must be one- or two-dimensional."
                )
            if deterministic.shape[0] != L and deterministic.shape[1] == L:
                deterministic = deterministic.T
            if deterministic.shape[0] != L:
                raise ValueError(
                    f"The deterministic path has {{deterministic.shape[0]}} "
                    f"rows; expected L={{L}}."
                )
            if not np.all(np.isfinite(deterministic)):
                raise ValueError(
                    "Future deterministic/exogenous values must be finite."
                )

            if deterministic_is_intercept:
                if deterministic.shape[1] != n:
                    raise ValueError(
                        "An intercept path must have one column per variable."
                    )
                intercept_path = deterministic
            elif n_exog and deterministic.shape[1] == n_exog:
                intercept_path = (
                    B[0][None, :]
                    + deterministic @ B[minimum_rows:]
                )
            elif deterministic.shape[1] == n:
                # Backward-compatible interpretation for project versions that
                # already assembled the reduced-form intercept path upstream.
                intercept_path = deterministic
            elif n_exog == 0 and deterministic.shape[1] == 0:
                pass
            else:
                raise ValueError(
                    "The deterministic path must contain either the model's "
                    f"{{n_exog}} exogenous columns or {{n}} intercept columns."
                )
        elif n_exog:
            raise ValueError(
                f"This posterior draw has {{n_exog}} exogenous coefficient "
                "rows; provide the deterministic future path."
            )

        # Augmented transition for [difference companion, cumulative level].
        F_aug = np.block(
            [
                [F, np.zeros((companion_size, n))],
                [J @ F, np.eye(n)],
            ]
        )
        selection_aug = np.vstack([selection, np.eye(n)])
        design_aug = np.hstack(
            [np.zeros((n, companion_size)), np.eye(n)]
        )

        first_companion_mean = (
            intercept_path[0] + F @ initial_companion_state
        )
        initial_mean = np.concatenate(
            [first_companion_mean, J @ first_companion_mean]
        )
        initial_cov = (
            selection_aug
            @ covariance_sequence[0]
            @ selection_aug.T
        )

        # statsmodels column t is used for the transition into state t+1.
        if L > 1:
            transition_intercepts = np.vstack(
                [intercept_path[1:], intercept_path[-1:]]
            )
        else:
            transition_intercepts = intercept_path.copy()
        state_intercept_aug = np.vstack(
            [transition_intercepts.T, J @ transition_intercepts.T]
        )

        model = MLEModel(
            endog=level_endog,
            k_states=companion_size + n,
            k_posdef=n,
        )
        model["design"] = design_aug
        model["obs_cov"] = np.zeros((n, n))
        model["transition"] = F_aug
        model["state_intercept"] = state_intercept_aug
        model["selection"] = selection_aug

        transition_covariance = np.empty_like(covariance_sequence)
        if L > 1:
            transition_covariance[:-1] = covariance_sequence[1:]
        transition_covariance[-1] = covariance_sequence[-1]
        model["state_cov"] = np.moveaxis(
            transition_covariance, 0, -1
        )
        model.initialize_known(
            initial_mean,
            0.5 * (initial_cov + initial_cov.T),
        )

        simulator = model.simulation_smoother(method="kfs")
        simulator.simulate(random_state=rng)
        return simulator.simulated_state.T.copy()
    '''
).strip() + "\n"


FORECAST_LEVEL_TEMPLATE = textwrap.indent(
    textwrap.dedent(
        '''
        # Conditions are observations of the cumulative-level state, not
        # pre-computed monthly differences.  Observed ragged-edge levels stay
        # observed; unpublished months stay NaN and are drawn jointly.
        base_level = np.asarray(
            prep["level_at_balanced_end"], dtype=float
        )
        level_endog_template = (
            levels.reindex(path_dates)
            .to_numpy(dtype=float, copy=True)
            - base_level[None, :]
        )
        if H:
            level_endog_template[tail_length:] = np.nan
        for variable, path in conditions.items():
            column = variables.index(variable)
            level_endog_template[tail_length:, column] = (
                path - base_level[column]
            )
        '''
    ).strip("\n"),
    "    ",
)


FORECAST_PATH_EXTRACTION = textwrap.indent(
    textwrap.dedent(
        '''
        companion_size = len(prep["last_companion_state"])
        diff_path = state_path[:, :n]
        cumulative_level_path = state_path[
            :, companion_size:companion_size + n
        ]
        diff_paths[out_index] = diff_path
        level_paths[out_index] = (
            base_level[None, :] + cumulative_level_path
        )
        '''
    ).strip("\n"),
    "        ",
)


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level function {name!r}; "
            f"found {len(matches)}."
        )
    return matches[0]  # type: ignore[return-value]


def _target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_conditions_loop(node: ast.stmt) -> bool:
    if not isinstance(node, ast.For):
        return False
    if not isinstance(node.target, (ast.Tuple, ast.List)):
        return False
    names = [_target_name(item) for item in node.target.elts]
    if names != ["variable", "path"]:
        return False
    call = node.iter
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "conditions"
        and call.func.attr == "items"
    )


def _assignment_to(node: ast.stmt, name: str) -> bool:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)


def _subscript_assignment_to(node: ast.stmt, name: str) -> bool:
    if not isinstance(node, ast.Assign):
        return False
    for target in node.targets:
        if isinstance(target, ast.Subscript):
            value = target.value
            if isinstance(value, ast.Name) and value.id == name:
                return True
    return False


def _patch_forecast_function(function_source: str) -> str:
    """Patch only the architecture-specific statements, preserving exog code."""
    tree = ast.parse(function_source)
    function = _function_node(tree, "forecast_bvar_sv_outlier")
    body = function.body

    observed = next(
        (node for node in body if _assignment_to(node, "observed_changes")),
        None,
    )
    condition_loop = next((node for node in body if _is_conditions_loop(node)), None)
    state_assignment = next(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "state_path"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_durbin_koopman_companion_draw"
        ),
        None,
    )
    diff_assignment = next(
        (node for node in ast.walk(function) if _assignment_to(node, "diff_path")),
        None,
    )
    level_assignment = next(
        (
            node
            for node in ast.walk(function)
            if _subscript_assignment_to(node, "level_paths")
        ),
        None,
    )

    missing = [
        name
        for name, node in (
            ("observed_changes assignment", observed),
            ("conditions loop", condition_loop),
            ("DK state_path assignment", state_assignment),
            ("diff_path assignment", diff_assignment),
            ("level_paths assignment", level_assignment),
        )
        if node is None
    ]
    if missing:
        raise RuntimeError(
            "Could not identify the forecast blocks to patch: "
            + ", ".join(missing)
        )

    assert observed is not None
    assert condition_loop is not None
    assert state_assignment is not None
    assert diff_assignment is not None
    assert level_assignment is not None
    if observed.lineno > condition_loop.end_lineno:
        raise RuntimeError("Unexpected forecast block order.")
    if diff_assignment.lineno > level_assignment.end_lineno:
        raise RuntimeError("Unexpected path-extraction order.")

    lines = function_source.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []

    # Replace the old level.diff() / fixed-anchor pre-differencing block.
    edits.append(
        (
            observed.lineno - 1,
            condition_loop.end_lineno,
            FORECAST_LEVEL_TEMPLATE + "\n",
        )
    )

    # Preserve all arguments, including any future-exog path assembled by the
    # current engine; only route the call through the augmented-level smoother.
    start = state_assignment.lineno - 1
    end = state_assignment.end_lineno
    call_text = "".join(lines[start:end])
    replaced_call = call_text.replace(
        "_durbin_koopman_companion_draw",
        "_durbin_koopman_level_draw",
        1,
    ).replace("endog_template", "level_endog_template")
    if replaced_call == call_text:
        raise RuntimeError("Could not rewrite the DK forecast call.")
    edits.append((start, end, replaced_call))

    # Read levels from the appended cumulative state rather than re-cumulating
    # a separately conditioned difference path.
    edits.append(
        (
            diff_assignment.lineno - 1,
            level_assignment.end_lineno,
            FORECAST_PATH_EXTRACTION + "\n",
        )
    )

    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    patched = "".join(lines)
    ast.parse(patched)
    return patched


def patch_model_file(target: Path) -> tuple[Path, Path]:
    target = target.resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Model file not found: {target}")

    original = target.read_text(encoding="utf-8")
    if PATCH_MARKER in original:
        raise RuntimeError(
            f"The ragged-edge level-state patch is already present in {target}."
        )

    tree = ast.parse(original)
    normalise = _function_node(tree, "_normalise_level_conditions")
    smoother = _function_node(tree, "_durbin_koopman_companion_draw")
    forecast = _function_node(tree, "forecast_bvar_sv_outlier")

    original_lines = original.splitlines(keepends=True)
    forecast_source = "".join(
        original_lines[forecast.lineno - 1 : forecast.end_lineno]
    )
    patched_forecast = _patch_forecast_function(forecast_source)

    edits = [
        (
            normalise.lineno - 1,
            normalise.end_lineno,
            NORMALISE_FUNCTION + "\n",
        ),
        (
            smoother.end_lineno,
            smoother.end_lineno,
            "\n" + LEVEL_SMOOTHER_FUNCTION + "\n",
        ),
        (
            forecast.lineno - 1,
            forecast.end_lineno,
            patched_forecast,
        ),
    ]
    for start, end, replacement in sorted(edits, reverse=True):
        original_lines[start:end] = [replacement]
    patched = "".join(original_lines)

    # Parse and compile before touching the user's source file.
    ast.parse(patched)
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / target.name
        candidate.write_text(patched, encoding="utf-8")
        py_compile.compile(str(candidate), doraise=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(
        f"{target.stem}.before_ragged_edge_fix_{timestamp}{target.suffix}"
    )
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)
    return target, backup


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the energy BVAR forecast so level conditions bridge the "
            "ragged edge inside an augmented Durbin--Koopman smoother."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TARGET,
        help=(
            "Path to energy_bvar_model.py. Default: "
            "src/model/energy_bvar_model.py"
        ),
    )
    args = parser.parse_args()
    target, backup = patch_model_file(args.path)
    print(f"Updated: {target}")
    print(f"Backup : {backup}")
    print("Next: restart the notebook kernel and run all cells.")


if __name__ == "__main__":
    main()
