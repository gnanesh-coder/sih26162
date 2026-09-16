"""Groups detections into events: the object the rest of the system should reason about.

WHY THIS EXISTS
---------------
Until this module the unit of analysis was the satellite pixel. `Incident` is one
row per detection, the classifier scores a detection, and the verified register --
which defines 21 *events*, each with a space-time window and a citation -- was
scored per detection. Jharia contributed 23,633 rows to that average and
Buncefield contributed 5.

That mismatch causes four problems that look unrelated until you name the unit:

1. **Validation is a category error.** The truth set is event-shaped and the
   measurement is pixel-shaped, so the headline number is a detection-weighted
   average over events that differ in size by four orders of magnitude.

2. **Ground truth cannot be grown.** Nobody will hand-adjudicate 2,044,295
   detections. 300 events is an afternoon's work per analyst. The circularity
   audit says the gap "closes with ground truth"; event granularity is what
   makes that affordable.

3. **The features that would break circularity do not exist at pixel level.**
   Duration, centroid drift, footprint and growth rate are properties of an
   event. None of them is readable by `weak_label_real_detection()`, so they are
   evidence the labelling rule never saw.

4. **Baghjan.** A five-month blowout is 705 detections that gradually become
   their own baseline, which is why recurrence reports it as routine. As one
   event with a 150-day duration it is not a hysteresis problem at all.

An operator sees one alert per fire rather than one per overpass, which is the
same thing the alert-fatigue mandate already asks for, applied one level up.

HOW GROUPING WORKS, AND WHY NOT DBSCAN
--------------------------------------
The obvious implementation is ST-DBSCAN over (lat, lon, t). This does something
simpler, because the hard half is already done.

`H3RecurrenceTracker.recurrence_key()` decides what counts as one source, and it
was tuned against a measured failure: at Reliance Jamnagar, 43 detections spread
across 18 resolution-9 cells and 42 were misclassified, because a refinery is
one source scattered over tens of square kilometres. Inside a mapped polygon the
key is therefore the facility; outside it, a resolution-8 cell (~900 m), matching
the "spatial jitter up to 1,000 metres caused by satellite parallax effects and
viewing geometry" the design documents call for.

So grouping runs in three stages:

    1. Group by recurrence key.          One refinery is one source, already.
    2. Split on temporal gaps.           A source that goes quiet and restarts
                                         is two events, not one.
    3. Merge adjacent hex events.        A fire front crossing a cell boundary
                                         is one fire, not one event per cell.

Stage 3 applies only to hex-keyed events. A facility polygon already states the
extent of its source, so merging two facilities because they are neighbours
would undo the evidence the polygon provides.

WHAT IS NOT YET MEASURED
------------------------
`MAX_GAP_HOURS` and `LINK_KM` are reasoned defaults, not swept ones. `LINK_KM`
at least inherits a number the project already committed to elsewhere -- the
1 km parallax jitter budget behind `persistence_resolution` -- but 48 hours is
an argument about overpass cadence, not a measurement.

The project's own discipline is that a threshold chosen by argument is
provisional until it is measured against the verified set: that is how
`INSIDE_SURGE_Z` was rejected and `BURST_*` adopted. That sweep is owed, and
until it is run these two numbers are a starting point rather than a result.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    import h3
except (ImportError, OSError):  # pragma: no cover - mirrors state_machine
    h3 = None

from src.alerting.state_machine import H3RecurrenceTracker, latlng_to_h3

logger = logging.getLogger("event_builder")

# How long a source may go unobserved before its next detection starts a new
# event.
#
# VIIRS gives roughly two overpasses a day, so 48 h is about four consecutive
# missed opportunities. Below that, ordinary overpass geometry and a single
# cloudy day would split one fire into several.
#
# The cost runs the other way too, and is worth stating: cloud can hide a
# genuinely continuous fire for longer than two days, and this will split it.
# That is the safer direction of error -- two events that should be one are
# both still reported, whereas one event that should be two hides an ignition
# inside an existing record.
MAX_GAP_HOURS = 48.0

# How far apart two detections may be and still belong to the same event.
#
# 1.0 km is the "spatial jitter up to 1,000 metres caused by satellite parallax
# effects and viewing geometry" the design documents already specify, and which
# `H3RecurrenceTracker.persistence_resolution` was chosen to match. Using the
# same number here means one physical claim governs both: below it, two
# detections are the same source seen twice; above it, they are not.
#
# Expressed in kilometres rather than H3 rings deliberately. An earlier draft
# linked on cell adjacency alone, and the result depended on where cell
# boundaries happened to fall relative to the fire -- a source moving 1.2 km a
# day fragmented into events of three, two, two and one detections purely by
# geometry. Rings are still used to *find* candidates cheaply; the decision is
# made on the real distance.
LINK_KM = 1.0

# H3 rings searched when generating candidate neighbours. Derived from LINK_KM
# rather than set independently, so the prefilter cannot silently become tighter
# than the rule it is prefiltering for.
def _rings_for(link_km: float, resolution: int) -> int:
    """Smallest ring count whose reach covers `link_km`."""
    if h3 is None or not hasattr(h3, "average_hexagon_edge_length"):
        return 2
    try:
        edge_km = float(h3.average_hexagon_edge_length(resolution, unit="km"))
    except Exception:  # pragma: no cover - older h3 signatures
        return 2
    # Adjacent cell centres sit about edge*sqrt(3) apart.
    step_km = max(edge_km * 1.7320508, 1e-6)
    return max(1, int(np.ceil(link_km / step_km)) + 1)

# Prefix on every generated identifier, so an event id is recognisable in a log
# line without context.
EVENT_ID_PREFIX = "EVT"


@dataclass(frozen=True)
class EventBuildConfig:
    """Parameters governing how detections are grouped."""

    max_gap_hours: float = MAX_GAP_HOURS
    link_km: float = LINK_KM
    persistence_resolution: int = 8
    # Stage 3 is the only part that can chain across cells, and chaining is
    # DBSCAN's known failure mode: in a dense agricultural region a continuous
    # path of ignitions can link into one implausible mega-event. Off is not
    # the default, because a fire front really is one fire -- but it is one
    # switch, so the effect can be measured rather than argued about.
    merge_adjacent_cells: bool = True


class _UnionFind:
    """Disjoint-set over integer labels, path-compressed."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Lower root wins, so the result does not depend on call order.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self._parent[hi] = lo


@dataclass
class _Segment:
    """One source's detections between two temporal gaps -- a candidate event."""

    key: str
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    lats: np.ndarray
    lons: np.ndarray
    cells: Tuple[str, ...]

    @property
    def is_facility(self) -> bool:
        return self.key.startswith("fac:")


def _detection_identity(df: pd.DataFrame) -> pd.Series:
    """A stable per-detection string, for hashing an event id.

    `detection_id` is used where the corpus carries one. Where it does not, the
    tuple that FIRMS itself de-duplicates on is reconstructed, so the identity is
    the same one the ingestion layer already treats as unique.
    """
    if "detection_id" in df.columns:
        ident = df["detection_id"].astype(str)
        if ident.notna().all() and (ident.str.len() > 0).all():
            return ident

    # Each part is mapped through str() elementwise rather than cast with
    # .astype(str). A column holding an unparseable timestamp comes back as NaT,
    # and on this pandas version casting that column yields a float NaN rather
    # than the string "NaT" -- which then fails the join. An undated detection
    # still needs an identity, because it is still going to become its own
    # event.
    parts = [
        df["latitude"].astype(float).round(5).map(str),
        df["longitude"].astype(float).round(5).map(str),
        pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce").map(str),
    ]
    if "satellite" in df.columns:
        parts.append(df["satellite"].map(str))
    return pd.Series(["|".join(t) for t in zip(*parts)], index=df.index)


def _ensure_recurrence_key(df: pd.DataFrame, resolution: int) -> pd.Series:
    """The recurrence key for every row, computed only where it is absent.

    The processed corpus already carries `recurrence_key` because
    `evaluate_dataframe` writes it. A frame that has not been through the state
    machine -- a test fixture, a hand-built request -- has not, so the same rule
    is applied here rather than requiring callers to run the state machine first.
    """
    if "recurrence_key" in df.columns:
        key = df["recurrence_key"].astype(str)
        if key.notna().all() and (key.str.len() > 0).all():
            return key

    if "inside_industrial" in df.columns:
        col = df["inside_industrial"]
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            inside = col.astype(str).str.strip().str.lower().isin(["true", "1"])
        else:
            inside = col.fillna(False).astype(bool)
    else:
        inside = pd.Series(False, index=df.index)

    # Same precedence as evaluate_dataframe: OSM id, then the facility name, then
    # whatever the join matched.
    fac = pd.Series(None, index=df.index, dtype=object)
    for candidate_col in ("osm_id", "facility_name", "index_right"):
        if candidate_col not in df.columns:
            continue
        candidate = df[candidate_col].astype(str)
        usable = fac.isna() & candidate.str.strip().str.lower().isin(
            ["nan", "none", "null", ""]
        ).eq(False)
        fac = fac.mask(usable, candidate)

    keys = []
    for i in df.index:
        facility_id = fac.at[i] if bool(inside.at[i]) else None
        cell = latlng_to_h3(
            float(df.at[i, "latitude"]), float(df.at[i, "longitude"]), resolution
        )
        keys.append(H3RecurrenceTracker.recurrence_key(cell, facility_id))
    return pd.Series(keys, index=df.index)


def _split_on_time_gaps(times: np.ndarray, max_gap_hours: float) -> np.ndarray:
    """Segment labels for one source's chronologically sorted timestamps.

    A new segment starts wherever the gap since the previous detection exceeds
    the threshold. The first detection always opens segment 0.
    """
    if len(times) == 0:
        return np.empty(0, dtype=int)
    gaps_h = np.diff(times).astype("timedelta64[s]").astype(float) / 3600.0
    return np.concatenate([[0], np.cumsum(gaps_h > max_gap_hours)]).astype(int)


def _min_separation_km(a: "_Segment", b: "_Segment") -> float:
    """Closest approach between any detection of `a` and any detection of `b`."""
    lat1 = np.radians(a.lats)[:, None]
    lon1 = np.radians(a.lons)[:, None]
    lat2 = np.radians(b.lats)[None, :]
    lon2 = np.radians(b.lons)[None, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float((2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))).min())


def _merge_adjacent_cell_events(
    segments: List["_Segment"],
    config: EventBuildConfig,
) -> _UnionFind:
    """Unions hex-keyed segments that are close in space and touch in time.

    A fire front moves. Keyed on a ~900 m cell it must cross boundaries, and
    without this every crossing would open a new event -- which is the
    fragmentation the recurrence tracker already documents for wildfires. There
    the fragmentation is correct, because recurrence asks "is this source
    stationary?". Here the question is "is this one fire?", and the answer can
    span cells.

    Two segments are joined when their closest detections lie within
    `link_km` AND their time intervals are within one `max_gap_hours` of each
    other. Both conditions are required, and the time one is what stops runaway
    chaining: proximity alone would link every field in Punjab during October
    into a single state-sized event.

    Facility-keyed segments are never touched. A polygon already states its
    source's extent, so merging two neighbouring plants would discard the
    evidence the polygon provides.

    **A source moving faster than `link_km` between overpasses still fragments,
    and that is the intended direction of error.** Two events that should be one
    are both still detected, classified and alerted. One event that should be
    two hides a fresh ignition inside an existing record, which is the failure
    this system exists to prevent.
    """
    uf = _UnionFind(len(segments))
    if not config.merge_adjacent_cells:
        return uf
    if h3 is None or not hasattr(h3, "grid_disk"):
        logger.warning(
            "h3.grid_disk unavailable; cross-cell merging is disabled and a "
            "moving fire front will be reported as one event per cell."
        )
        return uf

    rings = _rings_for(config.link_km, config.persistence_resolution)

    # A segment is indexed under every cell it touches, so a long segment is
    # reachable from either end rather than from its centroid alone.
    by_cell: Dict[str, List[int]] = {}
    for idx, seg in enumerate(segments):
        if seg.is_facility:
            continue
        for cell in seg.cells:
            by_cell.setdefault(cell, []).append(idx)

    gap = pd.Timedelta(hours=config.max_gap_hours)
    considered: set = set()

    for cell, members in by_cell.items():
        try:
            neighbourhood = h3.grid_disk(cell, rings)
        except Exception:  # pragma: no cover - cells from the pure-python fallback
            continue
        candidates = {j for n in neighbourhood for j in by_cell.get(n, ())}
        for i in members:
            for j in candidates:
                if i == j or uf.find(i) == uf.find(j):
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in considered:
                    continue
                considered.add(pair)

                a, b = segments[i], segments[j]
                # Intervals overlap, or sit within one gap of each other.
                if not (a.t_start - gap <= b.t_end and b.t_start - gap <= a.t_end):
                    continue
                if _min_separation_km(a, b) <= config.link_km:
                    uf.union(i, j)
    return uf


def _event_id(first_seen: pd.Timestamp, identities: Sequence[str]) -> str:
    """A deterministic identifier for one event.

    The hash runs over the event's *sorted* member identities, so the same
    detections produce the same id regardless of the order the corpus happened
    to arrive in. That matters more than it sounds: an id that moved when rows
    were reordered would make every downstream join and every adjudication
    record unreproducible.
    """
    digest = hashlib.sha1("\n".join(sorted(identities)).encode("utf-8")).hexdigest()
    stamp = "unknown" if pd.isna(first_seen) else f"{first_seen:%Y%m%dT%H%M}"
    return f"{EVENT_ID_PREFIX}-{stamp}-{digest[:8]}"


def assign_events(
    df: pd.DataFrame,
    config: Optional[EventBuildConfig] = None,
) -> pd.DataFrame:
    """Returns `df` with an `event_id` column naming the event each row belongs to.

    Rows are never dropped, reordered or altered. The detections remain the
    evidence behind an event, which is what the dossier already renders; this
    only says which of them belong together.
    """
    config = config or EventBuildConfig()

    if df.empty:
        out = df.copy()
        out["event_id"] = pd.Series(dtype=str)
        return out

    required = {"latitude", "longitude", "timestamp_utc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"assign_events needs {sorted(required)}; missing {sorted(missing)}"
        )

    work = df.copy()
    work["_key"] = _ensure_recurrence_key(work, config.persistence_resolution)
    work["_ts"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    work["_ident"] = _detection_identity(work)

    # A row with no usable timestamp cannot be placed in time, so it cannot be
    # grouped honestly. It becomes its own event rather than being silently
    # attached to whichever neighbour happens to be nearest.
    undated = work["_ts"].isna()
    if undated.any():
        logger.warning(
            "%d detections have no parseable timestamp; each becomes a "
            "single-detection event rather than being grouped by guesswork.",
            int(undated.sum()),
        )

    # --- stages 1 and 2: per source, split on temporal gaps ------------------
    segment_of_row = np.full(len(work), -1, dtype=int)
    segments: List[_Segment] = []

    positions = {label: i for i, label in enumerate(work.index)}

    def _record(key: str, member_index: pd.Index) -> int:
        members = work.loc[member_index]
        times = members["_ts"]
        lats = pd.to_numeric(members["latitude"], errors="coerce").to_numpy(dtype=float)
        lons = pd.to_numeric(members["longitude"], errors="coerce").to_numpy(dtype=float)
        cells = tuple(sorted({
            latlng_to_h3(float(la), float(lo), config.persistence_resolution)
            for la, lo in zip(lats, lons)
        }))
        segments.append(_Segment(key, times.min(), times.max(), lats, lons, cells))
        seg_id = len(segments) - 1
        for label in member_index:
            segment_of_row[positions[label]] = seg_id
        return seg_id

    for key, group in work[~undated].groupby("_key", sort=True):
        ordered = group.sort_values("_ts", kind="mergesort")
        labels = _split_on_time_gaps(ordered["_ts"].to_numpy(), config.max_gap_hours)
        for local_label in np.unique(labels):
            _record(str(key), ordered.index[labels == local_label])

    # A detection with no usable timestamp cannot be placed in time, so it
    # becomes its own event and never participates in stage 3.
    for label in work.index[undated]:
        _record(str(work.at[label, "_key"]), pd.Index([label]))

    # --- stage 3: merge nearby segments that touch in time -------------------
    datable = [s for s in segments if not pd.isna(s.t_start)]
    uf = _merge_adjacent_cell_events(segments, config) if datable else _UnionFind(len(segments))

    root_of_segment = np.array([uf.find(i) for i in range(len(segments))])
    root_of_row = root_of_segment[segment_of_row]

    # --- identifiers ---------------------------------------------------------
    work["_root"] = root_of_row
    ids: Dict[int, str] = {}
    for root, group in work.groupby("_root", sort=True):
        ids[int(root)] = _event_id(group["_ts"].min(), group["_ident"].tolist())

    out = df.copy()
    out["event_id"] = [ids[int(r)] for r in root_of_row]

    logger.info(
        "Grouped %d detections into %d events (%.1f detections per event).",
        len(out), len(ids), len(out) / max(len(ids), 1),
    )
    return out


def build_events(
    df: pd.DataFrame,
    config: Optional[EventBuildConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assigns events and returns `(detections_with_event_id, events)`.

    The second frame carries one row per event with only the identity columns.
    Everything a model would read is computed in `event_features.py`, so a
    caller that only needs the grouping does not pay for the features.
    """
    from src.pipeline.event_features import event_identity_frame

    detections = assign_events(df, config)
    return detections, event_identity_frame(detections)


def _main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/processed/firms_industrial_joined.parquet")
    parser.add_argument("--output", default="data/processed/events.parquet")
    parser.add_argument("--max-gap-hours", type=float, default=MAX_GAP_HOURS)
    parser.add_argument("--no-merge-adjacent", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    source = Path(args.input)
    if not source.exists():
        logger.error("No corpus at %s.", source)
        return 1

    df = pd.read_parquet(source)
    config = EventBuildConfig(
        max_gap_hours=args.max_gap_hours,
        merge_adjacent_cells=not args.no_merge_adjacent,
    )
    detections, events = build_events(df, config)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(destination, index=False)
    logger.info("Wrote %d events to %s.", len(events), destination)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
