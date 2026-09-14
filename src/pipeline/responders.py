"""Nearest emergency responders to an incident.

Answers the question a SitRep already implies but could not previously address:
the briefing recommends an evacuation cordon, and until now the system could not
say who would enforce it or where casualties would go.

The layer is extracted from the same OpenStreetMap India PBF as the industrial
and forest layers (`src/ingestion/pbf_extractor.py --layer responders`), so it
needs no new credential, no new API and no network at serving time.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
Distance here is **geodesic** -- great-circle metres between two coordinates.
That is a measurement.

Road distance and travel time are **not** computed, because computing them
honestly requires a routing engine over a road graph, and this project does not
have one wired. What is offered instead is an explicitly indicative figure
derived from the geodesic distance and two stated assumptions -- a circuity
factor and an average speed -- both returned in the payload beside the number
so a reader can see exactly what produced it.

That distinction is the point. A road ETA presented as "real-time" when it is a
straight line divided by an assumed speed is a fabricated measurement of the
same family as the synthetic weather this project removed and the CO2 figure it
refuses to print. The number is useful for triage; the claim that it is routed
would not be true, so it is not made.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("responders")

RESPONDER_LAYER = Path("data/reference/osm_india_responders_from_pbf.parquet")

EARTH_RADIUS_KM = 6371.0088

# Assumptions behind the indicative travel estimate. Named, returned in the
# response, and deliberately not buried.
#
# CIRCUITY_FACTOR: roads are not straight lines. 1.3 is a common planning
# figure for the ratio of road distance to straight-line distance in mixed
# urban/rural terrain. It is an assumption, not a measurement for any
# particular route.
#
# AVG_RESPONSE_SPEED_KMH: an emergency vehicle under priority, averaged over
# Indian mixed traffic. Also an assumption.
CIRCUITY_FACTOR = 1.3
AVG_RESPONSE_SPEED_KMH = 40.0

RESPONDER_KINDS = ("FIRE", "HOSPITAL", "POLICE")

_layer: Optional[pd.DataFrame] = None
_lock = threading.Lock()
_load_attempted = False


def _load() -> Optional[pd.DataFrame]:
    """Loads the responder layer once, or returns None when it is absent.

    Absent is a supported state: the layer is derived from a 1.8 GB PBF that is
    deliberately not committed, so a fresh clone has no responders until the
    extractor is run. Callers degrade to "withheld" rather than guessing, the
    same contract `forest_cover.py` uses for the land-cover layer.
    """
    global _layer, _load_attempted
    with _lock:
        if _load_attempted:
            return _layer
        _load_attempted = True

        if not RESPONDER_LAYER.exists():
            logger.warning(
                "Responder layer not found at %s. Nearest-responder lookup will "
                "report UNAVAILABLE. Build it with: python -m "
                "src.ingestion.pbf_extractor --layer responders",
                RESPONDER_LAYER,
            )
            return None

        df = pd.read_parquet(
            RESPONDER_LAYER,
            columns=["responder_kind", "responder_name", "latitude", "longitude"],
        )
        df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        # Radians precomputed once: the haversine below runs over the whole
        # layer on every query, and 61k rows is small enough that a KD-tree
        # would add a dependency and an approximation for no measurable gain.
        df["_lat_rad"] = np.radians(df["latitude"].to_numpy())
        df["_lon_rad"] = np.radians(df["longitude"].to_numpy())
        _layer = df
        logger.info(
            "Loaded %d emergency responders (%s).",
            len(df), df["responder_kind"].value_counts().to_dict(),
        )
        return _layer


def layer_available() -> bool:
    return _load() is not None


def layer_summary() -> Dict:
    df = _load()
    if df is None:
        return {"status": "UNAVAILABLE", "total": 0, "by_kind": {}}
    return {
        "status": "OK",
        "total": int(len(df)),
        "by_kind": {k: int(v) for k, v in df["responder_kind"].value_counts().items()},
        "source": "OpenStreetMap India extract, amenity=fire_station|hospital|police",
    }


def _haversine_km(lat_rad: float, lon_rad: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    dlat = lats - lat_rad
    dlon = lons - lon_rad
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat_rad) * np.cos(lats) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _indicative_travel(distance_km: float) -> Dict:
    """An explicitly non-routed travel estimate.

    Returns the assumptions alongside the figure so the number cannot travel
    without them.
    """
    road_km = distance_km * CIRCUITY_FACTOR
    minutes = (road_km / AVG_RESPONSE_SPEED_KMH) * 60.0
    return {
        "indicative_road_km": round(road_km, 2),
        "indicative_minutes": round(minutes, 1),
        "is_routed": False,
        "basis": (
            f"Geodesic distance x circuity factor {CIRCUITY_FACTOR}, at an assumed "
            f"{AVG_RESPONSE_SPEED_KMH:.0f} km/h average response speed. This is an "
            "indicative planning figure, NOT a routed travel time: no road graph "
            "is consulted. Treat the geodesic distance as the measurement."
        ),
    }


def nearest_responders(
    lat: float,
    lon: float,
    kinds: Optional[List[str]] = None,
    per_kind: int = 1,
    max_km: float = 100.0,
) -> Dict:
    """Nearest responders to a point, one group per category.

    Grouped by category rather than returned as a flat nearest-N list, because
    the three are not interchangeable: an incident commander needs the nearest
    fire station *and* the nearest hospital, and a flat list of the five closest
    facilities in a city centre would be five hospitals and no fire station.
    """
    df = _load()
    if df is None:
        return {
            "status": "UNAVAILABLE",
            "reason": (
                "The responder layer is not on disk. Build it with: "
                "python -m src.ingestion.pbf_extractor --layer responders"
            ),
            "responders": {},
        }

    wanted = [k.upper() for k in (kinds or RESPONDER_KINDS)]
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    distances = _haversine_km(
        lat_rad, lon_rad, df["_lat_rad"].to_numpy(), df["_lon_rad"].to_numpy()
    )

    out: Dict[str, List[Dict]] = {}
    for kind in wanted:
        mask = (df["responder_kind"].to_numpy() == kind) & (distances <= max_km)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            out[kind] = []
            continue
        nearest = idx[np.argsort(distances[idx])[:per_kind]]
        out[kind] = [
            {
                "name": (df.at[i, "responder_name"] or "").strip() or f"Unnamed {kind.lower()}",
                "kind": kind,
                "latitude": float(df.at[i, "latitude"]),
                "longitude": float(df.at[i, "longitude"]),
                "distance_km": round(float(distances[i]), 3),
                "travel_estimate": _indicative_travel(float(distances[i])),
            }
            for i in nearest
        ]

    empty = [k for k, v in out.items() if not v]
    return {
        "status": "OK",
        "query": {"latitude": lat, "longitude": lon, "max_km": max_km},
        "responders": out,
        # Nothing within range is a finding, not an empty result to be glossed.
        "none_within_range": empty,
        "caveat": (
            "Responder locations are OpenStreetMap community data and are not a "
            "complete register. OSM maps 741 fire stations across India, which is "
            "certainly an undercount; an absent facility means absent from the map, "
            "not absent from the ground. Distances are geodesic, not road distances."
        ),
    }
