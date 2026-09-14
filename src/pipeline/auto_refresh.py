"""Keeps the served corpus current without anyone remembering to do it.

WHY THIS EXISTS
---------------
The dashboard's status badge is computed from the newest detection it holds, so
it degrades honestly to `STALE - 5.2 d old` when nothing has been ingested.
That is the correct behaviour for a badge and the wrong behaviour for a
surveillance system: a near-real-time pipeline that only ingests when a human
runs a script is not near-real-time, it is a manual import with a clock on it.

This module closes that loop. While the API is running it pulls FIRMS, runs the
spatial join and recurrence state machine, and reseeds the incident database on
an interval. FIRMS NRT publishes roughly three hours behind the overpass, so
there is nothing to gain from polling faster than that.

WHAT IT REFUSES TO DO
---------------------
**It never blocks startup.** The refresh runs on a worker thread. If FIRMS is
slow, rate-limited or unreachable, the dashboard still comes up serving whatever
it already had, and the badge says how old that is. A surveillance console that
will not open because an upstream API is down is worse than one showing
yesterday's data with yesterday's date on it.

**It never reports a refresh it did not perform.** Every outcome is recorded
with a status and a timestamp, exposed at `GET /api/v1/system/refresh`. A failed
pull leaves `last_success_utc` untouched rather than advancing it.

**It never runs twice at once.** A lock guards the pipeline, because two
concurrent runs would write the same parquet and seed the same rows.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("auto_refresh")

# FIRMS NRT lands about 3 hours behind the overpass. Polling faster than the
# publication cadence spends the map key's quota to re-download what we have.
DEFAULT_INTERVAL_MINUTES = 180
MIN_INTERVAL_MINUTES = 30

# Two days rather than one: a single day's window can fall entirely between
# passes for part of the country, and the seeder skips detections it already
# holds, so the overlap costs nothing but a slightly larger CSV.
DEFAULT_DAY_RANGE = 2

SENSORS = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT")
RAW_OUTPUT = Path("data/raw/firms_latest.parquet")

# The live pipeline writes HERE, and never to PROCESSED_CORPUS.
#
# Those were the same path, and it destroyed data: PROCESSED_CORPUS is what the
# corpus-scale endpoints read, and a deployment points it at the 12-month
# archive so /map/hexes and /compliance/register see all 2,044,295 detections.
# The scheduled refresh then overwrote that archive with two days of NRT every
# three hours, silently, and the only symptom was the archive layer quietly
# reporting a thousand detections instead of two million.
#
# Live ingest and the analytical archive are different things with different
# lifetimes. They get different files.
LIVE_JOINED = Path("data/processed/firms_industrial_joined.parquet")

# A live window replacing a corpus this large is never intentional.
ARCHIVE_ROW_FLOOR = 100_000
SHRINK_FACTOR = 10

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "enabled": False,
    "interval_minutes": DEFAULT_INTERVAL_MINUTES,
    "running": False,
    "runs": 0,
    "failures": 0,
    "last_attempt_utc": None,
    "last_success_utc": None,
    "last_status": "NEVER_RUN",
    "last_detail": None,
    "last_detections": None,
    "next_attempt_utc": None,
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def refresh_enabled() -> bool:
    """Whether the background refresh should run at all.

    Off under pytest regardless of configuration. A test suite that reaches a
    third-party API is a test suite whose result depends on someone else's
    uptime, and this project already learned that from the weather provider.

    Also off in replay mode. A loop retrying a dead network during a
    demonstration achieves nothing except log noise and a stalled worker thread
    at the moment attention is least available.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False

    from src.pipeline import replay_mode
    if replay_mode.enabled():
        return False

    return _env_flag("AUTO_REFRESH_ENABLED", True)


def refresh_interval_minutes() -> int:
    try:
        value = int(os.getenv("AUTO_REFRESH_INTERVAL_MIN", DEFAULT_INTERVAL_MINUTES))
    except ValueError:
        value = DEFAULT_INTERVAL_MINUTES
    return max(MIN_INTERVAL_MINUTES, value)


def status() -> Dict[str, Any]:
    """Current refresh state, safe to serialise straight to an API response."""
    return dict(_state)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_firms_window(day_range: int = DEFAULT_DAY_RANGE) -> Optional[gpd.GeoDataFrame]:
    """Pulls the last `day_range` days across every NRT sensor, deduplicated.

    Returns None when no sensor produced anything -- which is different from an
    empty dataframe, and the caller treats it as a failed pull rather than as a
    day with no fires in India.
    """
    from src.ingestion.firms_client import fetch_firms_nrt

    frames: List[gpd.GeoDataFrame] = []
    for sensor in SENSORS:
        try:
            g = fetch_firms_nrt(satellite=sensor, day_range=day_range)
        except Exception as exc:                      # noqa: BLE001 - one sensor must not sink the run
            logger.warning("FIRMS pull failed for %s: %s", sensor, exc)
            continue
        if g is not None and len(g):
            frames.append(g)

    if not frames:
        return None

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    return combined.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"]
    )


def would_shrink_corpus(target: Path, incoming_rows: int) -> Optional[str]:
    """Whether writing `incoming_rows` to `target` would destroy an archive.

    Pure, so it can be tested without running the pipeline. The obvious
    alternative -- comparing the live path against PROCESSED_CORPUS -- looks
    right and is not: in the default configuration they are legitimately the
    same file, so that test refuses every safe refresh and catches no dangerous
    one. What matters is the shape of the write. A two-day NRT window replacing
    a multi-month archive is always a mistake, whichever path it lands on.

    Returns a reason string when the write should be refused, else None.
    """
    if not target.exists():
        return None
    try:
        existing = pq.ParquetFile(target).metadata.num_rows
    except Exception:                                  # noqa: BLE001 - unreadable is not a reason to refuse
        return None
    if existing > ARCHIVE_ROW_FLOOR and existing > incoming_rows * SHRINK_FACTOR:
        return (f"{target} holds {existing:,} detections and this window has "
                f"{incoming_rows:,}. Refusing to replace an archive with a live "
                "pull. Point the live ingest elsewhere, or remove the archive "
                "deliberately.")
    return None


def refresh_once(day_range: int = DEFAULT_DAY_RANGE) -> Dict[str, Any]:
    """Pull, join, reseed. Returns the outcome; never raises.

    The three stages are separated in the result so a partial failure is
    legible: a successful pull followed by a failed join is a different problem
    from FIRMS being unreachable, and "refresh failed" tells an operator
    neither.
    """
    if not _lock.acquire(blocking=False):
        return {"status": "ALREADY_RUNNING", "detail": "A refresh is already in progress."}

    _state["running"] = True
    _state["last_attempt_utc"] = _now_iso()
    started = time.time()

    try:
        # 1. Ingest
        gdf = fetch_firms_window(day_range)
        if gdf is None or gdf.empty:
            return _record("PULL_EMPTY", "FIRMS returned no detections for the window.")
        RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(RAW_OUTPUT, index=False)
        pulled = len(gdf)

        # 2. Spatial join, recurrence, alert tiers
        from src.pipeline.spatial_join import process_live_pipeline

        refusal = would_shrink_corpus(LIVE_JOINED, pulled)
        if refusal:
            return _record("REFUSED_WOULD_SHRINK_CORPUS", refusal)

        # Optical validation is left off here on purpose: it is a metered
        # third-party API, and a scheduled job that silently consumes a quota
        # every few hours is how a free tier disappears before a demo.
        process_live_pipeline(
            firms_file=RAW_OUTPUT,
            output_file=LIVE_JOINED,
            optical_validation=False,
        )

        # 3. Reseed. Imported late so this module stays importable without the API.
        from app.database import SessionLocal, init_db
        from app.main import seed_database_from_parquet

        init_db()
        db = SessionLocal()
        try:
            added = seed_database_from_parquet(db, LIVE_JOINED)
        finally:
            db.close()

        _state["last_success_utc"] = _now_iso()
        _state["last_detections"] = int(pulled)
        return _record(
            "OK",
            f"Pulled {pulled:,} detections, seeded {added:,} new "
            f"in {time.time() - started:.1f}s.",
            success=True,
        )

    except Exception as exc:                          # noqa: BLE001 - a scheduler must not die
        logger.exception("Automatic refresh failed.")
        return _record("FAILED", f"{type(exc).__name__}: {exc}")
    finally:
        _state["running"] = False
        _lock.release()


def _record(status_code: str, detail: str, success: bool = False) -> Dict[str, Any]:
    _state["last_status"] = status_code
    _state["last_detail"] = detail
    _state["runs"] += 1
    if not success:
        _state["failures"] += 1
        logger.warning("Auto-refresh %s: %s", status_code, detail)
    else:
        logger.info("Auto-refresh OK: %s", detail)
    return {"status": status_code, "detail": detail}


def start_scheduler(interval_minutes: Optional[int] = None,
                    run_immediately: bool = True) -> Optional[threading.Thread]:
    """Starts the background refresh loop. Returns None when disabled.

    `run_immediately` exists for the case that actually bites: someone opens the
    dashboard for a demonstration after it has sat unused for a week. Waiting a
    full interval before the first pull would leave the first -- and possibly
    only -- look at the system showing week-old data.
    """
    if not refresh_enabled():
        _state["enabled"] = False
        _state["last_status"] = "DISABLED"
        _state["last_detail"] = "AUTO_REFRESH_ENABLED is off, or running under pytest."
        logger.info("Automatic refresh is disabled.")
        return None

    interval = interval_minutes or refresh_interval_minutes()
    _state["enabled"] = True
    _state["interval_minutes"] = interval

    def _loop() -> None:
        if run_immediately:
            refresh_once()
        while True:
            _state["next_attempt_utc"] = datetime.fromtimestamp(
                time.time() + interval * 60, tz=timezone.utc
            ).isoformat(timespec="seconds")
            time.sleep(interval * 60)
            refresh_once()

    thread = threading.Thread(target=_loop, name="firms-auto-refresh", daemon=True)
    thread.start()
    logger.info("Automatic FIRMS refresh every %d minutes (first run immediately).", interval)
    return thread
