"""Event-level features, including the ones the labelling rule cannot see.

WHY THIS MATTERS MORE THAN THE USUAL FEATURE MODULE
---------------------------------------------------
The circularity audit measures how much of the model's score is it reciting its
own labelling rule: ablate the 14 columns `weak_label_real_detection()` reads and
macro F1 falls from 0.9976 to 0.7345, a delta of 0.2631. The audit's own note
says what closes that gap -- "verified labels, not more data" -- but there is a
second lever, and it is cheaper: **evidence the rule never consulted.**

`thermal_physics.py` was the first attempt at that. It implemented the Dozier
bi-spectral retrieval correctly, and it was rejected twice on validation because
every combustion regime collapsed into a 443-522 K band. The rejection stands and
that module is still unwired.

These features are the second attempt, and they are cheaper because they need no
new instrument. They are properties of a *group* of detections, so they simply do
not exist at the granularity the labelling rule operates on:

    duration_h            how long it burned
    centroid_drift_km     whether the source moved
    drift_rate_km_per_day how fast
    extent_km             how far it spread
    dispersion_km         how tightly the detections cluster
    frp_trend_mw_per_day  whether it was growing or dying

`centroid_drift_km` is the one worth arguing for explicitly. A flare stack is
bolted to the ground; a fire front is not. That distinction is physical, it is
independent of OpenStreetMap coverage, and it is the discriminator that would
have caught Rumaila -- 200 detections from dozens of intermittent stacks across
an 80 km field, which recurrence read as scattered transients because no single
~900 m cell reached the persistence threshold. Drift asks a different question of
the same data: did the heat *move*, or was it always there?

HONEST NOTE ON WHAT IS CLAIMED HERE
-----------------------------------
That these features are structurally independent of the labelling rule is a fact
about the code and is enforced by test. That they carry useful signal is a
*hypothesis*, and the circularity delta after retraining is what will decide it.
Nothing in this module should be quoted as evidence until that number exists.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("event_features")

EARTH_RADIUS_KM = 6371.0088

# Features that are properties of an event rather than of a detection, and so
# cannot appear in LABEL_RULE_FEATURES. The circularity audit must NOT ablate
# these -- that is the entire reason for computing them -- and a test asserts the
# two lists stay disjoint.
EVENT_ONLY_FEATURES = [
    "duration_h",
    "n_detections",
    "n_overpasses",
    "centroid_drift_km",
    "drift_rate_km_per_day",
    "extent_km",
    "dispersion_km",
    "frp_trend_mw_per_day",
    "night_fraction",
    "bearing_deg",
]


def _haversine_km(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in kilometres.

    The same formula `pipeline/responders.py` uses; kept local rather than
    imported because that one is private to its module and pre-converts to
    radians for a layer it holds in memory.
    """
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, in degrees east of north."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def _thirds(n: int) -> int:
    """How many detections make up the leading or trailing third of an event.

    At least one, so a two-detection event still has a first and a last sample
    to measure drift between.
    """
    return max(1, n // 3)


def _bool_fraction(series: pd.Series) -> float:
    """Fraction of a column that is true, tolerating the stringified form.

    Parquet round-trips turn booleans into "True"/"False", exactly as
    `FireFeaturePipeline._bool_col` already documents.
    """
    if series.empty:
        return 0.0
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        truthy = series.astype(str).str.strip().str.lower().isin(["true", "1"])
    else:
        truthy = series.fillna(False).astype(bool)
    return float(truthy.mean())


def _mode_or_none(series: pd.Series) -> Optional[str]:
    """The most common non-null value, or None when there is none."""
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    counts = cleaned.astype(str).value_counts()
    return None if counts.empty else str(counts.index[0])


def _numeric(group: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    """A numeric column from an event's detections, tolerating its absence."""
    if column not in group.columns:
        return pd.Series(default, index=group.index, dtype=float)
    return pd.to_numeric(group[column], errors="coerce")


def event_identity_frame(detections: pd.DataFrame) -> pd.DataFrame:
    """One row per event carrying only identity and extent -- no model inputs.

    Separated from `build_event_features` so that a caller who only needs to know
    which events exist does not pay to compute drift over the whole corpus.
    """
    if detections.empty or "event_id" not in detections.columns:
        return pd.DataFrame(
            columns=["event_id", "n_detections", "first_seen_utc", "last_seen_utc"]
        )

    work = detections.copy()
    work["_ts"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    grouped = work.groupby("event_id", sort=True)
    return pd.DataFrame(
        {
            "event_id": list(grouped.groups.keys()),
            "n_detections": grouped.size().to_numpy(),
            "first_seen_utc": grouped["_ts"].min().to_numpy(),
            "last_seen_utc": grouped["_ts"].max().to_numpy(),
        }
    )


def _one_event(event_id: str, group: pd.DataFrame) -> Dict:
    """Features for a single event, from its member detections."""
    ordered = group.sort_values("_ts", kind="mergesort")
    n = len(ordered)

    lats = pd.to_numeric(ordered["latitude"], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(ordered["longitude"], errors="coerce").to_numpy(dtype=float)
    times = ordered["_ts"]
    frp = _numeric(ordered, "frp", 0.0).fillna(0.0).to_numpy(dtype=float)

    first_seen, last_seen = times.min(), times.max()
    if pd.isna(first_seen) or pd.isna(last_seen):
        duration_h = 0.0
    else:
        duration_h = float((last_seen - first_seen).total_seconds() / 3600.0)
    duration_days = max(duration_h / 24.0, 1e-9)

    centroid_lat, centroid_lon = float(np.mean(lats)), float(np.mean(lons))

    # Drift: where the heat was early against where it was late. A third at each
    # end rather than a single detection, so one off-nadir outlier cannot invent
    # motion that did not happen.
    k = _thirds(n)
    head_lat, head_lon = float(np.mean(lats[:k])), float(np.mean(lons[:k]))
    tail_lat, tail_lon = float(np.mean(lats[-k:])), float(np.mean(lons[-k:]))
    drift_km = float(_haversine_km(head_lat, head_lon, np.array([tail_lat]),
                                   np.array([tail_lon]))[0])
    bearing = _bearing_deg(head_lat, head_lon, tail_lat, tail_lon) if drift_km > 0 else np.nan

    # Extent and dispersion describe shape rather than motion: a dispersed flare
    # field is wide and static, a front is narrow and moving.
    from_centroid = _haversine_km(centroid_lat, centroid_lon, lats, lons)
    extent_km = float(from_centroid.max() * 2.0) if n > 1 else 0.0
    dispersion_km = float(np.sqrt(np.mean(from_centroid ** 2))) if n > 1 else 0.0

    # Trend: least-squares slope of FRP against time. Positive means the event was
    # intensifying over its own life, which no single detection can express.
    if n > 1 and duration_h > 0 and np.isfinite(frp).all():
        elapsed_days = (times - first_seen).dt.total_seconds().to_numpy() / 86400.0
        if float(np.ptp(elapsed_days)) > 0:
            frp_trend = float(np.polyfit(elapsed_days, frp, 1)[0])
        else:
            frp_trend = 0.0
    else:
        frp_trend = 0.0

    # An overpass is one satellite pass, so detections sharing a timestamp are one
    # observation of a larger fire rather than repeated evidence over time.
    n_overpasses = int(times.dt.floor("min").nunique())

    row: Dict = {
        "event_id": event_id,
        # --- identity and extent ---
        "first_seen_utc": first_seen,
        "last_seen_utc": last_seen,
        "centroid_latitude": centroid_lat,
        "centroid_longitude": centroid_lon,
        "recurrence_key": _mode_or_none(ordered.get("recurrence_key", pd.Series(dtype=object))),

        # --- event-only features: not visible to the labelling rule ---
        "duration_h": round(duration_h, 3),
        "n_detections": n,
        "n_overpasses": n_overpasses,
        "centroid_drift_km": round(drift_km, 4),
        "drift_rate_km_per_day": round(drift_km / duration_days, 4) if duration_h > 0 else 0.0,
        "extent_km": round(extent_km, 4),
        "dispersion_km": round(dispersion_km, 4),
        "frp_trend_mw_per_day": round(frp_trend, 4),
        "night_fraction": round(
            float((ordered["daynight"].astype(str).str.upper() == "N").mean())
            if "daynight" in ordered.columns else 0.0, 4),
        "bearing_deg": None if np.isnan(bearing) else round(bearing, 1),

        # --- radiometry, aggregated ---
        "frp_max_mw": round(float(np.max(frp)), 3),
        "frp_median_mw": round(float(np.median(frp)), 3),
        "frp_total_mw": round(float(np.sum(frp)), 3),
        "bright_ti4_max": round(float(_numeric(ordered, "bright_ti4").max()), 3)
            if "bright_ti4" in ordered.columns else None,
        "bright_ti5_max": round(float(_numeric(ordered, "bright_ti5").max()), 3)
            if "bright_ti5" in ordered.columns else None,

        # --- spatial context, carried up unchanged ---
        "inside_industrial_fraction": round(
            _bool_fraction(ordered.get("inside_industrial", pd.Series(dtype=bool))), 4),
        "in_forest_fraction": round(
            _bool_fraction(ordered.get("in_forest", pd.Series(dtype=bool))), 4),
        "facility_name": _mode_or_none(ordered.get("facility_name", pd.Series(dtype=object))),
        "facility_type": _mode_or_none(ordered.get("facility_type", pd.Series(dtype=object)))
            or "non_industrial",
        "dist_to_industrial_km": round(
            float(_numeric(ordered, "dist_to_industrial_km", 999.0).min()), 4),

        # --- recurrence context from the state machine, at its extreme ---
        # Aggregated as a maximum rather than a mean: an event's strongest
        # recurrence evidence is what the rule acted on, and averaging it over a
        # long event would dilute exactly the moment that mattered.
        "n_30d_max": float(_numeric(ordered, "n_30d", 0.0).max()),
        "z_frp_max": float(_numeric(ordered, "z_frp", 0.0).max()),
        "onset_lag_days_max": float(_numeric(ordered, "onset_lag_days", -1.0).max()),
        "burst_ratio_max": float(_numeric(ordered, "burst_ratio", 0.0).max()),
        "neighbourhood_active_keys_max": float(
            _numeric(ordered, "neighbourhood_active_keys", 0.0).max()),
    }

    # Optical validation, where it was measured. Absent stays absent: a dNBR that
    # was never computed must not become a 0.0, which is the rule
    # `sentinel2_client` already enforces per detection.
    if "dnbr" in ordered.columns:
        dnbr = _numeric(ordered, "dnbr").dropna()
        row["dnbr_mean"] = round(float(dnbr.mean()), 4) if not dnbr.empty else None
        row["dnbr_measured_n"] = int(len(dnbr))
    else:
        row["dnbr_mean"] = None
        row["dnbr_measured_n"] = 0

    return row


def build_event_features(detections: pd.DataFrame) -> pd.DataFrame:
    """One row of features per event, from a frame carrying `event_id`.

    Expects the output of `event_builder.assign_events`. Detections are never
    modified; this reads them.
    """
    if detections.empty:
        return pd.DataFrame(columns=["event_id"] + EVENT_ONLY_FEATURES)
    if "event_id" not in detections.columns:
        raise ValueError(
            "build_event_features needs an event_id column; run "
            "event_builder.assign_events first."
        )

    work = detections.copy()
    work["_ts"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")

    rows: List[Dict] = [
        _one_event(str(event_id), group)
        for event_id, group in work.groupby("event_id", sort=True)
    ]

    out = pd.DataFrame(rows)
    logger.info(
        "Built features for %d events over %d detections; median duration %.1f h, "
        "median drift %.3f km.",
        len(out), len(detections),
        float(out["duration_h"].median()) if len(out) else 0.0,
        float(out["centroid_drift_km"].median()) if len(out) else 0.0,
    )
    return out
