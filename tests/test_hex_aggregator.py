"""Tests for server-side H3 aggregation.

The aggregator exists so the map stops shipping two million points. The
properties worth defending are therefore about what survives the compression:
the payload must be bounded, the roll-up must not move detections between
places, and a single emergency must not vanish into a crowd of routine
agricultural detections.
"""

import sys
from pathlib import Path

import h3
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.hex_aggregator import (
    MAX_CELLS,
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    aggregate_hexes,
    roll_up_to,
)


def _frame(rows):
    df = pd.DataFrame(rows)
    if "h3_index" not in df.columns:
        df["h3_index"] = [h3.latlng_to_cell(la, lo, 8)
                          for la, lo in zip(df.latitude, df.longitude)]
    for col, default in (("frp", 5.0), ("state", "MONITORING"),
                         ("priority", "P2_ADVISORY"), ("suppressed", False),
                         ("acq_date", "2026-03-07")):
        if col not in df.columns:
            df[col] = default
    return df


def _rows(n, lat=22.34, lon=69.87, **over):
    base = [{"latitude": lat + i * 1e-4, "longitude": lon + i * 1e-4} for i in range(n)]
    return [{**b, **over} for b in base]


# --------------------------------------------------------------------------
# Roll-up
# --------------------------------------------------------------------------

def test_roll_up_lands_in_the_parent_of_the_original_cell():
    """A coarsened cell must contain the fine cell it came from."""
    fine = h3.latlng_to_cell(22.3450, 69.8700, 8)
    coarse = roll_up_to(pd.Series([fine]), 5).iloc[0]

    assert h3.get_resolution(coarse) == 5
    assert h3.cell_to_parent(fine, 5) == coarse


def test_roll_up_leaves_cells_already_coarse_enough_alone():
    coarse = h3.latlng_to_cell(22.3450, 69.8700, 4)
    assert roll_up_to(pd.Series([coarse]), 6).iloc[0] == coarse


def test_roll_up_skips_unparseable_cells_rather_than_raising():
    out = roll_up_to(pd.Series(["not-a-cell", h3.latlng_to_cell(22.0, 69.0, 8)]), 5)
    assert out.isna().iloc[0]
    assert out.notna().iloc[1]


# --------------------------------------------------------------------------
# The payload must be bounded
# --------------------------------------------------------------------------

def test_aggregation_compresses_many_detections_into_few_cells():
    df = _frame(_rows(400, lat=22.34, lon=69.87))
    out = aggregate_hexes(df, resolution=5)

    assert out["status"] == "OK"
    assert out["cells_returned"] < 400
    assert out["compression_ratio"] > 1.0
    assert sum(c["detections"] for c in out["cells"]) == 400


def test_the_cell_count_is_capped_and_the_truncation_is_declared():
    """An unbounded response would reintroduce the problem this module fixes."""
    df = _frame([{"latitude": 8.0 + i * 0.03, "longitude": 70.0 + i * 0.03}
                 for i in range(300)])
    out = aggregate_hexes(df, resolution=8, limit=10)

    assert out["cells_returned"] == 10
    assert out["truncated"] is True


def test_limit_cannot_exceed_the_hard_ceiling():
    df = _frame(_rows(10))
    out = aggregate_hexes(df, resolution=6, limit=10 ** 9)
    assert out["cells_returned"] <= MAX_CELLS


@pytest.mark.parametrize("asked,expected", [(-5, MIN_RESOLUTION), (99, MAX_RESOLUTION)])
def test_resolution_is_clamped_to_what_the_stored_index_supports(asked, expected):
    out = aggregate_hexes(_frame(_rows(5)), resolution=asked)
    assert out["resolution"] == expected


# --------------------------------------------------------------------------
# An emergency must not be buried
# --------------------------------------------------------------------------

def test_a_single_p0_outranks_a_crowd_of_suppressed_detections():
    """Sorting on count alone is the alert-fatigue failure, drawn on a map."""
    busy = _rows(300, lat=30.75, lon=75.50, priority="SUPPRESSED",
                 state="PERSISTENT_BASELINE", suppressed=True)
    one = _rows(1, lat=22.34, lon=69.87, priority="P0_EMERGENCY",
                state="ESCALATED_FLAREUP")
    out = aggregate_hexes(_frame(busy + one), resolution=5, limit=50)

    assert out["cells"][0]["top_priority"] == "P0_EMERGENCY"
    assert out["cells"][0]["detections"] == 1
    assert out["cells"][1]["detections"] > 1


def test_a_cell_reports_its_highest_priority_not_its_most_common():
    rows = _rows(40, priority="SUPPRESSED") + _rows(1, priority="P1_ALERT")
    out = aggregate_hexes(_frame(rows), resolution=4)

    assert out["cells"][0]["top_priority"] == "P1_ALERT"
    assert out["cells"][0]["p1"] == 1


def test_a_cell_reports_both_its_highest_and_its_most_common_priority():
    """Colouring a map by the highest priority turns every coarse cell red.

    At resolution 4 a cell is 22km across and almost every populated one
    contains a P0 somewhere in a year. The highest is the right key to sort by;
    the most common is the right one to colour by.
    """
    rows = _rows(40, priority="SUPPRESSED") + _rows(1, priority="P0_EMERGENCY")
    cell = aggregate_hexes(_frame(rows), resolution=4)["cells"][0]

    assert cell["top_priority"] == "P0_EMERGENCY"
    assert cell["dominant_priority"] == "SUPPRESSED"


def test_dominant_state_is_the_most_common_one():
    rows = _rows(30, state="PERSISTENT_BASELINE") + _rows(5, state="MONITORING")
    out = aggregate_hexes(_frame(rows), resolution=4)
    assert out["cells"][0]["dominant_state"] == "PERSISTENT_BASELINE"


# --------------------------------------------------------------------------
# Filters and degenerate inputs
# --------------------------------------------------------------------------

def test_bbox_excludes_detections_outside_it():
    rows = _rows(20, lat=22.34, lon=69.87) + _rows(20, lat=28.60, lon=77.20)
    out = aggregate_hexes(_frame(rows), resolution=5, bbox=[69.0, 22.0, 70.0, 23.0])

    assert out["detections_aggregated"] == 20
    assert out["detections_in_corpus"] == 40


def test_state_filter_is_case_insensitive():
    rows = _rows(10, state="PERSISTENT_BASELINE") + _rows(10, state="MONITORING")
    out = aggregate_hexes(_frame(rows), resolution=5, states=["persistent_baseline"])
    assert out["detections_aggregated"] == 10


def test_min_detections_drops_thin_cells():
    rows = _rows(20, lat=22.34, lon=69.87) + _rows(1, lat=28.60, lon=77.20)
    out = aggregate_hexes(_frame(rows), resolution=5, min_detections=5)
    assert all(c["detections"] >= 5 for c in out["cells"])


def test_an_empty_corpus_reports_no_cells_rather_than_failing():
    out = aggregate_hexes(pd.DataFrame(), resolution=6)
    assert out["status"] == "NO_CELLS"
    assert out["cells"] == []


def test_filters_that_match_nothing_say_so():
    out = aggregate_hexes(_frame(_rows(10)), resolution=6, states=["NOTHING_LIKE_THIS"])
    assert out["status"] == "NO_CELLS"
    assert "filters" in out["detail"]


def test_a_corpus_without_h3_index_is_indexed_from_coordinates():
    """Corpora predating the recurrence tracker must still render."""
    df = _frame(_rows(12)).drop(columns=["h3_index"])
    out = aggregate_hexes(df, resolution=6)

    assert out["status"] == "OK"
    assert sum(c["detections"] for c in out["cells"]) == 12
