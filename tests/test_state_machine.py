"""Unit tests for Recurrence & Suppression State Machine."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.alerting.state_machine import (
    AlertPriority,
    AlertState,
    H3RecurrenceTracker,
    RecurrenceStateMachine,
    evaluate_dataframe,
    latlng_to_h3,
)


def test_latlng_to_h3():
    """Verify H3 binning produces a valid string index at resolution 9."""
    h3_idx = latlng_to_h3(19.0760, 72.8777, resolution=9)
    assert isinstance(h3_idx, str)
    assert len(h3_idx) > 10


def test_state_persistent_baseline():
    """N_30 >= 8 and Z_FRP <= 2.5 -> PERSISTENT_BASELINE, suppressed."""
    decision = RecurrenceStateMachine.evaluate(
        n_30d=12,
        frp=25.0,
        mu_frp=24.0,
        var_frp=4.0,  # std = 2.0, Z = (25-24)/2 = 0.5 <= 2.5
        inside_industrial=True,
    )
    assert decision.state == AlertState.PERSISTENT_BASELINE
    assert decision.priority == AlertPriority.SUPPRESSED
    assert decision.suppressed is True
    assert decision.tag == "flare_or_kiln"


def test_state_escalated_flareup_by_z_score():
    """N_30 >= 8 and Z_FRP > 3.0 -> ESCALATED_FLAREUP, P1 Alert."""
    decision = RecurrenceStateMachine.evaluate(
        n_30d=15,
        frp=45.0,
        mu_frp=20.0,
        var_frp=25.0,  # std = 5.0, Z = (45-20)/5 = 5.0 > 3.0
        inside_industrial=True,
    )
    assert decision.state == AlertState.ESCALATED_FLAREUP
    assert decision.priority == AlertPriority.P1_ALERT
    assert decision.suppressed is False
    assert decision.tag == "escalated_flareup"


def test_state_escalated_flareup_by_ratio():
    """N_30 >= 8 and FRP > 3 * mu_FRP -> ESCALATED_FLAREUP, P1 Alert."""
    decision = RecurrenceStateMachine.evaluate(
        n_30d=10,
        frp=70.0,
        mu_frp=20.0,  # 70 > 3 * 20 = 60
        var_frp=400.0,  # std = 20.0, Z = 2.5 (ratio triggers rule)
        inside_industrial=True,
    )
    assert decision.state == AlertState.ESCALATED_FLAREUP
    assert decision.priority == AlertPriority.P1_ALERT
    assert decision.suppressed is False


def test_state_accidental_fire():
    """N_30 < 3, inside industrial polygon, FRP >= 10 MW -> ACCIDENTAL_FIRE, P0 Emergency."""
    decision = RecurrenceStateMachine.evaluate(
        n_30d=1,
        frp=22.5,
        mu_frp=15.0,
        var_frp=2.0,
        inside_industrial=True,
    )
    assert decision.state == AlertState.ACCIDENTAL_FIRE
    assert decision.priority == AlertPriority.P0_EMERGENCY
    assert decision.suppressed is False
    assert decision.tag == "accidental_industrial_fire"


def test_state_transient_suspicion():
    """N_30 < 3, outside industrial polygon -> TRANSIENT_SUSPICION, Non-alert background."""
    decision = RecurrenceStateMachine.evaluate(
        n_30d=0,
        frp=15.0,
        mu_frp=0.0,
        var_frp=0.0,
        inside_industrial=False,
    )
    assert decision.state == AlertState.TRANSIENT_SUSPICION
    assert decision.priority == AlertPriority.NON_ALERT
    assert decision.suppressed is True
    assert decision.tag == "background_agri_burn"


def test_h3_recurrence_tracker_progression():
    """Verify H3RecurrenceTracker correctly tracks sequential hotspots at a single site."""
    tracker = H3RecurrenceTracker(resolution=9, window_days=30)
    lat, lng = 22.3039, 70.8022  # Refinery location

    # First event inside industrial zone with FRP >= 10MW -> Accidental Fire P0
    t0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    _, d0 = tracker.process_hotspot(lat, lng, frp=15.0, timestamp=t0, inside_industrial=True)
    assert d0.state == AlertState.ACCIDENTAL_FIRE
    assert d0.priority == AlertPriority.P0_EMERGENCY

    # Simulate 7 more recurring detections over following days (total 8 in 30 days)
    for i in range(1, 8):
        t_i = datetime(2026, 9, 1 + i, 10, 0, tzinfo=timezone.utc)
        tracker.process_hotspot(lat, lng, frp=15.0, timestamp=t_i, inside_industrial=True)

    # 9th event with normal FRP -> Now classified as PERSISTENT_BASELINE (flare/kiln)
    t9 = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)
    _, d9 = tracker.process_hotspot(lat, lng, frp=15.2, timestamp=t9, inside_industrial=True)
    assert d9.state == AlertState.PERSISTENT_BASELINE
    assert d9.suppressed is True


def test_evaluate_dataframe():
    """Verify batch evaluation over a DataFrame."""
    df = pd.DataFrame(
        {
            "latitude": [22.5, 28.6],
            "longitude": [88.3, 77.2],
            "frp": [25.0, 5.0],
            "timestamp_utc": [
                "2026-09-07 05:00:00+00:00",
                "2026-09-07 06:00:00+00:00",
            ],
            "inside_industrial": [True, False],
        }
    )
    result = evaluate_dataframe(df)
    assert len(result) == 2
    assert "h3_index" in result.columns
    assert "priority" in result.columns
    assert result.iloc[0]["state"] == AlertState.ACCIDENTAL_FIRE.value
    assert result.iloc[1]["state"] == AlertState.TRANSIENT_SUSPICION.value


# --------------------------------------------------------------------------
# Facility-level recurrence
#
# Regression for a measured defect: keying recurrence on an H3 res-9 cell
# (~174m edge) fragmented a single large refinery across 18 cells, so n_30d
# stayed near 1 against a threshold of 8 and 42 of 43 detections at a
# continuously flaring complex were misclassified. The design documents call for
# tolerating ~1000m of parallax jitter; a res-9 cell is ~6x tighter.
# --------------------------------------------------------------------------

def test_recurrence_key_prefers_facility_identity():
    from src.alerting.state_machine import H3RecurrenceTracker as T
    assert T.recurrence_key("8928308280fffff", "way/12345") == "fac:way/12345"
    assert T.recurrence_key("8928308280fffff", None) == "hex:8928308280fffff"


def test_blank_and_null_facility_ids_fall_back_to_the_cell():
    """Parquet round-trips turn missing ids into the strings 'nan'/'None'."""
    from src.alerting.state_machine import H3RecurrenceTracker as T
    for empty in (None, "", "   ", "nan", "None", "null"):
        assert T.recurrence_key("8928308280fffff", empty).startswith("hex:")


def test_scattered_detections_at_one_facility_share_a_baseline():
    """The Jamnagar case: detections spread across a complex must accumulate.

    Each coordinate here lands in a different H3 res-9 cell, as successive
    detections of a real multi-square-kilometre facility do.
    """
    tracker = H3RecurrenceTracker()
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    offsets = [0.000, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.032]

    cells = {latlng_to_h3(22.33 + o, 69.86 + o, 9) for o in offsets}
    assert len(cells) > 1, "fixture must span multiple H3 cells to be meaningful"

    decision = None
    for i, o in enumerate(offsets):
        _, decision = tracker.process_hotspot(
            lat=22.33 + o, lng=69.86 + o, frp=18.0,
            timestamp=base + timedelta(days=i),
            inside_industrial=True, facility_id="way/999",
        )

    assert len(tracker.registry) == 1, "one facility must occupy one registry entry"
    assert decision.state == AlertState.PERSISTENT_BASELINE
    assert decision.priority == AlertPriority.SUPPRESSED


def test_without_facility_id_the_same_points_stay_fragmented():
    """Confirms the fix is what changes the outcome, not the fixture."""
    tracker = H3RecurrenceTracker()
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    offsets = [0.000, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.032]

    decision = None
    for i, o in enumerate(offsets):
        _, decision = tracker.process_hotspot(
            lat=22.33 + o, lng=69.86 + o, frp=18.0,
            timestamp=base + timedelta(days=i),
            inside_industrial=True, facility_id=None,
        )

    assert len(tracker.registry) > 1, "hex keying should fragment these"
    assert decision.state != AlertState.PERSISTENT_BASELINE


def test_moving_wildfire_still_fragments_correctly():
    """A spreading front is genuinely moving and must NOT be pooled.

    Outside mapped infrastructure there is no facility id, so hex keying applies
    and the fire is never mistaken for a persistent industrial source.
    """
    tracker = H3RecurrenceTracker()
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)

    decision = None
    for i in range(10):
        _, decision = tracker.process_hotspot(
            lat=24.0 + i * 0.02, lng=80.0 + i * 0.02, frp=30.0,
            timestamp=base + timedelta(hours=6 * i),
            inside_industrial=False, facility_id=None,
        )

    assert len(tracker.registry) > 1
    assert decision.state != AlertState.PERSISTENT_BASELINE


def test_two_facilities_keep_separate_baselines():
    tracker = H3RecurrenceTracker()
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)

    for i in range(9):
        tracker.process_hotspot(lat=22.33, lng=69.86, frp=18.0,
                                timestamp=base + timedelta(days=i),
                                inside_industrial=True, facility_id="way/111")
    _, first = tracker.process_hotspot(lat=30.10, lng=75.50, frp=18.0,
                                       timestamp=base + timedelta(days=9),
                                       inside_industrial=True, facility_id="way/222")

    assert len(tracker.registry) == 2
    # The second facility has no history of its own and must not inherit one.
    assert first.state != AlertState.PERSISTENT_BASELINE


def test_evaluate_dataframe_emits_the_recurrence_key():
    df = pd.DataFrame({
        "latitude": [22.33, 22.37],
        "longitude": [69.86, 69.90],
        "frp": [20.0, 22.0],
        "timestamp_utc": pd.to_datetime(["2026-07-01", "2026-07-02"], utc=True),
        "inside_industrial": [True, True],
        "osm_id": ["way/555", "way/555"],
    })
    out = evaluate_dataframe(df)

    assert "recurrence_key" in out.columns
    assert out["recurrence_key"].nunique() == 1
    assert out["recurrence_key"].iloc[0] == "fac:way/555"
