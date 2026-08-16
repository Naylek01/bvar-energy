"""Immutable process-local snapshot cache for dashboard presentation data.

Result/registry artefacts are loaded or parsed once per snapshot key. The cache
has no TTL: an explicit dashboard snapshot refresh is the only invalidation
event. Background estimation/scenario computation keeps its existing Diskcache
contract and is intentionally separate.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from io import StringIO
from threading import RLock
from typing import Any, Callable, Hashable

import pandas as pd


_MAX_ENTRIES = 512
_LOCK = RLock()
_CACHE: OrderedDict[tuple[str, Hashable], Any] = OrderedDict()


def _normalise_key(key: Any) -> Hashable:
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    if isinstance(key, tuple):
        return tuple(_normalise_key(item) for item in key)
    if isinstance(key, list):
        return tuple(_normalise_key(item) for item in key)
    if isinstance(key, dict):
        return tuple(
            sorted(
                (str(k), _normalise_key(v))
                for k, v in key.items()
            )
        )
    return repr(key)


def get(namespace: str, key: Any, default=None):
    cache_key = (str(namespace), _normalise_key(key))
    with _LOCK:
        if cache_key not in _CACHE:
            return default
        value = _CACHE.pop(cache_key)
        _CACHE[cache_key] = value
        return value


def put(namespace: str, key: Any, value: Any):
    cache_key = (str(namespace), _normalise_key(key))
    with _LOCK:
        _CACHE.pop(cache_key, None)
        _CACHE[cache_key] = value
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return value


def get_or_build(namespace: str, key: Any, builder: Callable[[], Any]):
    cached = get(namespace, key, default=None)
    if cached is not None:
        return cached
    return put(namespace, key, builder())


def clear_all() -> None:
    with _LOCK:
        _CACHE.clear()


def put_frame(reference: str, frame: pd.DataFrame) -> str:
    ref = str(reference)
    put("dataframe", ref, frame)
    return ref


def frame_from_store(
    store: dict | None,
    *,
    json_key: str = "frame_json",
    reference_key: str = "snapshot_ref",
) -> pd.DataFrame:
    if not store:
        return pd.DataFrame()

    reference = store.get(reference_key)
    if reference:
        cached = get("dataframe", str(reference))
        if isinstance(cached, pd.DataFrame):
            return cached

    payload = store.get(json_key)
    if not payload:
        return pd.DataFrame()

    if not reference:
        reference = f"json::{sha256(str(payload).encode('utf-8')).hexdigest()}"

    cached = get("dataframe", str(reference))
    if isinstance(cached, pd.DataFrame):
        return cached

    frame = pd.read_json(StringIO(payload), orient="split")
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    put_frame(str(reference), frame)
    return frame


def cache_stats() -> dict[str, int]:
    with _LOCK:
        counts: dict[str, int] = {}
        for namespace, _key in _CACHE:
            counts[namespace] = counts.get(namespace, 0) + 1
        counts["total"] = len(_CACHE)
        return counts
