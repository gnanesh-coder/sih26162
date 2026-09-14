"""Server-side H3 aggregation, so the map never ships 2 million points.

THE PROBLEM
-----------
The dashboard draws detections with a Leaflet heat layer. That works at the
scale of a day's detections and does not work at the scale of the corpus: the
12-month national archive is 2,044,295 rows, and serialising them to the browser
is roughly 200 MB of JSON before a single marker is drawn. The browser is not
the bottleneck to fix; sending the points at all is.

The problem statement's own architecture note reaches for PostGIS and deck.gl
here. This project uses neither, and the reason is worth stating plainly rather
than hiding: the aggregation PostGIS would perform is a group-by on a column the
pipeline already computes. Every detection already carries `h3_index` at
resolution 8 -- roughly 900 m across -- because the recurrence tracker needs it.
Rolling those up to a coarser resolution is `h3.cell_to_parent`, which is a bit
shift on the index, not a spatial query. A 2M-row roll-up takes well under a
second in pandas, and the answer is a few thousand cells rather than millions of
points.

deck.gl's H3HexagonLayer then draws those cells directly from the index, so the
server never sends geometry either -- only the cell id and its numbers.

WHAT A CELL REPORTS, AND WHY EACH FIELD IS THERE
-----------------------------------------------
Detection count alone makes a dense agricultural region look like the most
urgent place in the country. So each cell also carries the operational fields
the alerting pipeline already assigned: the highest priority reached, how many
detections were suppressed as routine, and the maximum FRP. A cell with 4,000
detections that are all PERSISTENT_BASELINE is a steel plant; a cell with 30
that include a P0 is the one an operator needs to look at.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h3
import numpy as np
import pandas as pd

logger = logging.getLogger("hex_aggregator")

# Resolution 8 is what the pipeline stores (~900 m). Resolutions below 3 cover
# whole subcontinents in a handful of cells and tell a viewer nothing; above 8
# would require re-indexing the raw coordinates rather than rolling up, and the
# stored index cannot be refined after the fact.
MIN_RESOLUTION = 3
MAX_RESOLUTION = 8
STORED_RESOLUTION = 8
DEFAULT_RESOLUTION = 6

# A hard ceiling on cells returned. The point of aggregating is to bound the
# payload; an unbounded response at resolution 8 would reintroduce the problem
# this module exists to remove.
MAX_CELLS = 20000

# Ordered most to least urgent. Used to pick a cell's headline priority.
PRIORITY_ORDER = ("P0_EMERGENCY", "P1_ALERT", "P2_ADVISORY", "NON_ALERT", "SUPPRESSED")
_PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITY_ORDER)}


def _rank(priority: str) -> int:
    return _PRIORITY_RANK.get(str(priority), len(PRIORITY_ORDER))


def roll_up_to(h3_index: pd.Series, resolution: int) -> pd.Series:
    """Coarsens stored resolution-8 cells to the requested resolution.

    `cell_to_parent` is an index operation, not a geometric one, so this stays
    cheap at corpus scale. Cells already at or below the target resolution pass
    through unchanged rather than raising -- a corpus built with a different
    stored resolution should degrade, not fail.
    """
    idx = h3_index.astype(str)
    unique = pd.Index(idx.unique())

    mapping: Dict[str, str] = {}
    for cell in unique:
        if not cell or cell == "nan":
            continue
        try:
            if h3.get_resolution(cell) <= resolution:
                mapping[cell] = cell
            else:
                mapping[cell] = h3.cell_to_parent(cell, resolution)
        except (ValueError, TypeError):
            continue
    return idx.map(mapping)


def aggregate_hexes(
    df: pd.DataFrame,
    resolution: int = DEFAULT_RESOLUTION,
    bbox: Optional[Sequence[float]] = None,
    states: Optional[Sequence[str]] = None,
    min_detections: int = 1,
    limit: int = MAX_CELLS,
) -> Dict[str, Any]:
    """Aggregates detections into H3 cells for map rendering.

    Args:
        df: Processed detections carrying h3_index (or latitude/longitude).
        resolution: Target H3 resolution, MIN_RESOLUTION..MAX_RESOLUTION.
        bbox: Optional (west, south, east, north) filter in degrees.
        states: Optional whitelist of pipeline states.
        min_detections: Cells below this are omitted.
        limit: Maximum cells returned, busiest first.
    """
    resolution = int(np.clip(resolution, MIN_RESOLUTION, MAX_RESOLUTION))
    limit = int(np.clip(limit, 1, MAX_CELLS))

    n_input = len(df)
    if n_input == 0:
        return _empty(resolution, "The corpus is empty.")

    work = df
    if bbox is not None and len(bbox) == 4:
        west, south, east, north = (float(v) for v in bbox)
        lat = pd.to_numeric(work["latitude"], errors="coerce")
        lon = pd.to_numeric(work["longitude"], errors="coerce")
        work = work[(lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)]

    if states:
        wanted = {str(s).upper() for s in states}
        work = work[work["state"].astype(str).str.upper().isin(wanted)]

    if work.empty:
        return _empty(resolution, "No detections matched the requested filters.")

    work = work.copy()

    # Prefer the stored index; fall back to indexing coordinates directly so
    # this still works on a corpus produced before h3_index existed.
    if "h3_index" in work.columns and work["h3_index"].notna().any():
        work["_cell"] = roll_up_to(work["h3_index"], resolution)
    else:
        logger.info("No h3_index column; indexing %s coordinates directly.", f"{len(work):,}")
        lat = pd.to_numeric(work["latitude"], errors="coerce")
        lon = pd.to_numeric(work["longitude"], errors="coerce")
        work["_cell"] = [
            h3.latlng_to_cell(la, lo, resolution) if np.isfinite(la) and np.isfinite(lo) else None
            for la, lo in zip(lat, lon)
        ]

    work = work[work["_cell"].notna()]
    if work.empty:
        return _empty(resolution, "No detection carried a usable H3 cell.")

    work["_frp"] = pd.to_numeric(work.get("frp"), errors="coerce").fillna(0.0)
    dates = pd.to_datetime(work.get("acq_date", work.get("timestamp_utc")),
                           errors="coerce", utc=True)
    work["_day"] = dates.dt.date

    priority = work.get("priority", pd.Series("NON_ALERT", index=work.index)).astype(str)
    work["_prank"] = priority.map(_rank).astype(int)
    work["_p0"] = (priority == "P0_EMERGENCY").astype(int)
    work["_p1"] = (priority == "P1_ALERT").astype(int)
    if "suppressed" in work.columns:
        work["_suppressed"] = work["suppressed"].astype(bool).astype(int)
    else:
        work["_suppressed"] = 0

    grouped = work.groupby("_cell", sort=False)
    agg = grouped.agg(
        detections=("_frp", "size"),
        days_observed=("_day", "nunique"),
        frp_max=("_frp", "max"),
        frp_median=("_frp", "median"),
        frp_total=("_frp", "sum"),
        p0=("_p0", "sum"),
        p1=("_p1", "sum"),
        suppressed=("_suppressed", "sum"),
        top_priority_rank=("_prank", "min"),
    )
    agg = agg[agg["detections"] >= max(1, int(min_detections))]
    if agg.empty:
        return _empty(resolution, f"No cell reached {min_detections} detections.")

    # Dominant priority per cell, alongside the highest one reached.
    #
    # Both are needed, and colouring a map by the wrong one is why. At
    # resolution 4 a cell is 22 km across, so almost every populated cell in
    # India contains at least one P0 somewhere in a year -- rendered by highest
    # priority, the national view came out uniformly red and said nothing. The
    # highest priority is the right key to SORT by; the most common one is the
    # right thing to COLOUR by, with the emergencies marked separately.
    modes_p = (work.groupby(["_cell", priority], sort=False)
               .size().rename("n").reset_index()
               .sort_values("n", ascending=False)
               .drop_duplicates("_cell").set_index("_cell"))
    agg["dominant_priority"] = agg.index.map(modes_p.iloc[:, 0]).fillna("NON_ALERT")

    # Dominant state per cell, resolved without a second full pass.
    if "state" in work.columns:
        modes = (work.groupby(["_cell", work["state"].astype(str)], sort=False)
                 .size().rename("n").reset_index()
                 .sort_values("n", ascending=False)
                 .drop_duplicates("_cell").set_index("_cell")["state"])
        agg["dominant_state"] = agg.index.map(modes).fillna("UNKNOWN")
    else:
        agg["dominant_state"] = "UNKNOWN"

    # Rank by urgency first, then by volume. Sorting on count alone buries a
    # single P0 cell underneath thousands of routine agricultural ones, which
    # is the exact failure the alerting tier exists to avoid.
    agg = agg.sort_values(["top_priority_rank", "detections"], ascending=[True, False])
    truncated = len(agg) > limit
    agg = agg.head(limit)

    cells: List[Dict[str, Any]] = []
    for cell, row in agg.iterrows():
        rank = int(row["top_priority_rank"])
        cells.append({
            "h3": str(cell),
            "detections": int(row["detections"]),
            "days_observed": int(row["days_observed"]),
            "frp_max_mw": round(float(row["frp_max"]), 2),
            "frp_median_mw": round(float(row["frp_median"]), 2),
            "frp_total_mw": round(float(row["frp_total"]), 1),
            "p0": int(row["p0"]),
            "p1": int(row["p1"]),
            "suppressed": int(row["suppressed"]),
            "top_priority": (PRIORITY_ORDER[rank] if rank < len(PRIORITY_ORDER) else "UNKNOWN"),
            "dominant_priority": str(row["dominant_priority"]),
            "dominant_state": str(row["dominant_state"]),
        })

    logger.info("Aggregated %s detections into %d cell(s) at resolution %d.",
                f"{n_input:,}", len(cells), resolution)

    return {
        "status": "OK",
        "resolution": resolution,
        "approx_cell_edge_km": round(_edge_km(resolution), 2),
        "detections_aggregated": int(len(work)),
        "detections_in_corpus": int(n_input),
        "cells_returned": len(cells),
        "truncated": truncated,
        "compression_ratio": round(len(work) / max(len(cells), 1), 1),
        "ordering": ("Highest priority reached first, then detection count. A cell "
                     "holding one P0 outranks one holding ten thousand suppressed "
                     "agricultural detections."),
        "cells": cells,
    }


def _edge_km(resolution: int) -> float:
    """Nominal H3 edge length, from the published resolution table."""
    table = {0: 1107.71, 1: 418.68, 2: 158.24, 3: 59.81, 4: 22.61,
             5: 8.54, 6: 3.23, 7: 1.22, 8: 0.46}
    return table.get(int(resolution), float("nan"))


def _empty(resolution: int, detail: str) -> Dict[str, Any]:
    return {
        "status": "NO_CELLS",
        "resolution": resolution,
        "detail": detail,
        "cells_returned": 0,
        "cells": [],
    }
