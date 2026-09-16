"""Guards the grouping that decides what one fire is.

Every number the system reports downstream is now conditional on this module
drawing the right boundaries. Two ways it can be wrong, and they cost different
things:

  * **Over-merging** hides a fresh ignition inside an existing record. A new
    fire at a site that already burns is exactly the Deonar failure the
    labelling rule needed three attempts to see, and merging it into its own
    baseline would put it beyond reach of any of them.
  * **Under-merging** splits one fire into several. Each is still detected,
    classified and alerted, so the cost is duplicate alerts rather than a
    missed event.

These tests pin the first shut and allow the second, which is why a source
moving faster than the link budget is asserted to fragment rather than asserted
to merge.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.pipeline.event_builder import (
    EventBuildConfig,
    MAX_GAP_HOURS,
    assign_events,
    build_events,
)

BASE = pd.Timestamp("2025-03-01T06:00:00Z")


def _det(lat, lon, ts, frp=10.0, inside=False, facility=None, daynight="N"):
    return {
        "latitude": lat, "longitude": lon, "timestamp_utc": ts, "frp": frp,
        "inside_industrial": inside, "facility_name": facility, "daynight": daynight,
    }


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 1: one source is one event
# ---------------------------------------------------------------------------

def test_a_refinery_scattered_across_cells_is_one_event():
    """The Jamnagar defect, one level up.

    43 detections at Jamnagar spread across 18 resolution-9 cells and 42 were
    misclassified, because a refinery is one source occupying tens of square
    kilometres. Recurrence fixed that by keying on facility identity; grouping
    has to inherit it, or the same complex becomes eighteen events.
    """
    rng = np.random.default_rng(0)
    rows = [
        _det(22.42 + rng.normal(0, 0.012), 69.85 + rng.normal(0, 0.012),
             BASE + pd.Timedelta(days=d, hours=h), 30.0, True, "Reliance Jamnagar")
        for d in range(20) for h in (0, 12)
    ]
    out = assign_events(_frame(rows))

    assert out["event_id"].nunique() == 1, (
        "a continuously flaring facility must be one event, not one per cell"
    )


def test_two_facilities_are_never_merged():
    """A polygon states its source's extent; merging two discards that."""
    rows = [
        _det(22.420, 69.850, BASE + pd.Timedelta(hours=h), 30.0, True, "Plant A")
        for h in range(0, 72, 12)
    ] + [
        # Deliberately within the link budget of Plant A.
        _det(22.424, 69.850, BASE + pd.Timedelta(hours=h), 30.0, True, "Plant B")
        for h in range(0, 72, 12)
    ]
    out = assign_events(_frame(rows))

    assert out["event_id"].nunique() == 2


# ---------------------------------------------------------------------------
# Stage 2: a source that stops and restarts is two events
# ---------------------------------------------------------------------------

def test_a_gap_longer_than_the_threshold_opens_a_new_event():
    rows = [_det(19.05, 72.93, BASE + pd.Timedelta(days=d)) for d in (0, 1, 9, 10)]
    out = assign_events(_frame(rows))

    assert out["event_id"].nunique() == 2
    # The split falls at the gap, not anywhere else.
    assert out["event_id"].iloc[0] == out["event_id"].iloc[1]
    assert out["event_id"].iloc[2] == out["event_id"].iloc[3]
    assert out["event_id"].iloc[1] != out["event_id"].iloc[2]


def test_a_gap_inside_the_threshold_does_not_split():
    """Ordinary overpass cadence and one cloudy day must not fragment a fire."""
    hours = MAX_GAP_HOURS - 1.0
    rows = [
        _det(19.05, 72.93, BASE),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=hours)),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=2 * hours)),
    ]
    assert assign_events(_frame(rows))["event_id"].nunique() == 1


# ---------------------------------------------------------------------------
# Stage 3: movement within the link budget is one fire
# ---------------------------------------------------------------------------

def test_a_front_moving_inside_the_link_budget_stays_one_event():
    """0.55 km a day is inside the 1 km parallax budget, so it is one fire."""
    rows = [_det(21.00 + d * 0.005, 79.00, BASE + pd.Timedelta(days=d)) for d in range(10)]
    assert assign_events(_frame(rows))["event_id"].nunique() == 1


def test_a_front_moving_faster_than_the_link_budget_fragments():
    """Asserted deliberately, and it is the safe direction of error.

    A source that moves further than the link budget between overpasses is
    reported as successive events. Each is still classified and alerted, so the
    cost is duplicate alerts on a moving wildfire -- against the alternative,
    which is a link rule loose enough to chain every burning field in Punjab
    into one state-sized event during October.
    """
    rows = [_det(23.00 + d * 0.015, 80.00, BASE + pd.Timedelta(days=d)) for d in range(10)]
    assert assign_events(_frame(rows))["event_id"].nunique() > 1


def test_distant_simultaneous_fires_are_separate_events():
    rows = [
        _det(30.9, 75.0, BASE), _det(30.9, 75.5, BASE + pd.Timedelta(days=1)),
        _det(30.9, 76.0, BASE + pd.Timedelta(days=2)),
    ]
    assert assign_events(_frame(rows))["event_id"].nunique() == 3


def test_merging_can_be_switched_off():
    """The chaining switch exists so its effect can be measured, not argued."""
    rows = [_det(21.00 + d * 0.005, 79.00, BASE + pd.Timedelta(days=d)) for d in range(10)]
    merged = assign_events(_frame(rows), EventBuildConfig(merge_adjacent_cells=True))
    split = assign_events(_frame(rows), EventBuildConfig(merge_adjacent_cells=False))

    assert merged["event_id"].nunique() < split["event_id"].nunique()


# ---------------------------------------------------------------------------
# Identifiers must not depend on how the corpus arrived
# ---------------------------------------------------------------------------

def test_event_ids_are_stable_under_row_reordering():
    """An id that moved when rows were reordered would make every downstream
    join and every adjudication record unreproducible."""
    rng = np.random.default_rng(3)
    rows = [
        _det(22.42 + rng.normal(0, 0.01), 69.85 + rng.normal(0, 0.01),
             BASE + pd.Timedelta(hours=6 * i), 12.0, True, "Plant A")
        for i in range(30)
    ]
    df = _frame(rows)
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)

    assert set(assign_events(df)["event_id"]) == set(assign_events(shuffled)["event_id"])


def test_event_id_changes_when_membership_changes():
    """The id is a claim about which detections belong together, so it must
    move when that claim does -- otherwise a re-run silently reuses an id for a
    different set of evidence."""
    rows = [_det(19.05, 72.93, BASE + pd.Timedelta(hours=6 * i)) for i in range(4)]
    full = assign_events(_frame(rows))["event_id"].iloc[0]
    partial = assign_events(_frame(rows[:3]))["event_id"].iloc[0]

    assert full != partial


# ---------------------------------------------------------------------------
# Degenerate input must degrade, not explode
# ---------------------------------------------------------------------------

def test_empty_frame_returns_an_empty_event_column():
    out = assign_events(pd.DataFrame(columns=["latitude", "longitude", "timestamp_utc"]))
    assert "event_id" in out.columns
    assert out.empty


def test_a_single_detection_is_still_an_event():
    """A one-pixel fire is the hardest case to verify and the easiest to drop."""
    out = assign_events(_frame([_det(19.05, 72.93, BASE)]))
    assert out["event_id"].nunique() == 1


def test_a_detection_with_no_timestamp_becomes_its_own_event():
    """It cannot be placed in time, so it must not be attached to a neighbour
    by guesswork."""
    rows = [
        _det(19.05, 72.93, BASE),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=6)),
        _det(19.05, 72.93, None),
    ]
    out = assign_events(_frame(rows))
    assert out["event_id"].nunique() == 2
    assert out["event_id"].iloc[2] not in set(out["event_id"].iloc[:2])


def test_missing_required_columns_raise_rather_than_guess():
    with pytest.raises(ValueError, match="timestamp_utc"):
        assign_events(pd.DataFrame({"latitude": [1.0], "longitude": [2.0]}))


def test_detections_are_never_dropped_or_reordered():
    """Grouping annotates the corpus; it must not edit it."""
    rows = [_det(21.0 + i * 0.01, 79.0, BASE + pd.Timedelta(days=i)) for i in range(12)]
    df = _frame(rows)
    out = assign_events(df)

    assert len(out) == len(df)
    pd.testing.assert_frame_equal(out.drop(columns=["event_id"]), df)


def test_build_events_returns_one_row_per_event():
    rows = [_det(19.05, 72.93, BASE + pd.Timedelta(days=d)) for d in (0, 1, 9, 10)]
    detections, events = build_events(_frame(rows))

    assert len(events) == detections["event_id"].nunique() == 2
    assert set(events["event_id"]) == set(detections["event_id"])
    assert events["n_detections"].sum() == len(detections)
