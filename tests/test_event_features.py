"""Guards the features that exist to be independent of the labelling rule.

The circularity audit measures how much of the model's score is it reciting its
own rule back: ablate the 14 columns `weak_label_real_detection()` reads and
macro F1 falls 0.9976 -> 0.7345. Event-level features are the cheap half of
closing that, because they are properties of a *group* of detections and so
cannot appear in a rule that sees one detection at a time.

That independence is a property of the code, and code drifts. The most important
test here is `test_event_only_features_are_not_label_rule_features`: if someone
later teaches the labelling rule to read `centroid_drift_km`, the audit must stop
counting it as independent evidence, and this fails until they do.

The rest hold the individual measurements to what they claim. `centroid_drift_km`
in particular carries an operational claim -- that a bolted-down flare stack and
a moving fire front separate on it -- so it is tested in both directions rather
than only for absence of error.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.models.train_classifier import LABEL_RULE_FEATURES
from src.pipeline.event_builder import assign_events
from src.pipeline.event_features import (
    EVENT_ONLY_FEATURES,
    build_event_features,
    event_identity_frame,
)

BASE = pd.Timestamp("2025-03-01T06:00:00Z")


def _det(lat, lon, ts, **kw):
    row = {
        "latitude": lat, "longitude": lon, "timestamp_utc": ts, "frp": 10.0,
        "inside_industrial": False, "facility_name": None, "daynight": "N",
    }
    row.update(kw)
    return row


def _features(rows):
    return build_event_features(assign_events(pd.DataFrame(rows)))


# ---------------------------------------------------------------------------
# The independence claim
# ---------------------------------------------------------------------------

def test_event_only_features_are_not_label_rule_features():
    """The whole argument for these features is that the rule cannot see them.

    If the labelling rule ever starts reading one, the circularity audit must
    ablate it like any other rule input -- and this test is what forces that
    conversation instead of letting the audit quietly flatter itself.
    """
    overlap = set(EVENT_ONLY_FEATURES) & set(LABEL_RULE_FEATURES)
    assert not overlap, (
        f"{sorted(overlap)} are declared as event-only evidence but the "
        "labelling rule now reads them. Either stop the rule reading them, or "
        "move them into LABEL_RULE_FEATURES so the audit ablates them."
    )


def test_every_declared_event_only_feature_is_actually_produced():
    """A feature declared independent but never computed is a claim with no
    measurement behind it."""
    rows = [_det(21.0 + d * 0.004, 79.0, BASE + pd.Timedelta(days=d)) for d in range(6)]
    produced = set(_features(rows).columns)
    missing = set(EVENT_ONLY_FEATURES) - produced
    assert not missing, f"declared but not computed: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Drift: the discriminator this module exists for
# ---------------------------------------------------------------------------

def test_a_static_source_has_near_zero_drift():
    """A flare stack is bolted to the ground. Pixel jitter is not movement."""
    rng = np.random.default_rng(0)
    rows = [
        _det(22.42 + rng.normal(0, 0.004), 69.85 + rng.normal(0, 0.004),
             BASE + pd.Timedelta(hours=12 * i), inside_industrial=True,
             facility_name="Reliance Jamnagar")
        for i in range(40)
    ]
    feats = _features(rows)
    assert len(feats) == 1
    assert feats["drift_rate_km_per_day"].iloc[0] < 0.2, (
        "a static facility must not look like it is moving"
    )


def test_a_moving_front_has_clear_drift():
    rows = [_det(21.00 + d * 0.005, 79.00, BASE + pd.Timedelta(days=d)) for d in range(10)]
    feats = _features(rows)
    assert len(feats) == 1
    assert feats["centroid_drift_km"].iloc[0] > 1.0
    assert feats["drift_rate_km_per_day"].iloc[0] > 0.2


def test_drift_separates_a_static_source_from_a_moving_one():
    """The claim in one assertion: these two must not be confusable."""
    rng = np.random.default_rng(1)
    static = _features([
        _det(22.42 + rng.normal(0, 0.004), 69.85 + rng.normal(0, 0.004),
             BASE + pd.Timedelta(hours=12 * i), inside_industrial=True,
             facility_name="Plant A")
        for i in range(20)
    ])["drift_rate_km_per_day"].iloc[0]

    moving = _features([
        _det(21.00 + d * 0.005, 79.00, BASE + pd.Timedelta(days=d)) for d in range(10)
    ])["drift_rate_km_per_day"].iloc[0]

    assert moving > static * 3, (
        f"drift rate does not separate the two regimes: static {static:.4f}, "
        f"moving {moving:.4f} km/day"
    )


def test_bearing_is_absent_when_nothing_moved():
    """A direction of travel for a source that did not travel is a fabricated
    measurement of the same family the project refuses elsewhere."""
    rows = [_det(19.05, 72.93, BASE + pd.Timedelta(hours=12 * i)) for i in range(4)]
    assert _features(rows)["bearing_deg"].iloc[0] is None


def test_bearing_points_the_way_the_fire_went():
    """Due north, so the bearing is near 0 or 360."""
    rows = [_det(21.00 + d * 0.005, 79.00, BASE + pd.Timedelta(days=d)) for d in range(10)]
    bearing = _features(rows)["bearing_deg"].iloc[0]
    assert bearing is not None
    assert min(abs(bearing - 0.0), abs(bearing - 360.0)) < 5.0


# ---------------------------------------------------------------------------
# Duration and cadence
# ---------------------------------------------------------------------------

def test_duration_spans_the_whole_event():
    rows = [_det(19.05, 72.93, BASE + pd.Timedelta(hours=12 * i)) for i in range(5)]
    assert _features(rows)["duration_h"].iloc[0] == pytest.approx(48.0)


def test_duration_tolerates_an_overpass_gap():
    """A missed overpass inside the threshold must not shorten the event."""
    rows = [
        _det(19.05, 72.93, BASE),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=36)),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=48)),
    ]
    feats = _features(rows)
    assert len(feats) == 1
    assert feats["duration_h"].iloc[0] == pytest.approx(48.0)


def test_overpasses_count_passes_not_pixels():
    """A large fire lighting four pixels in one pass is one observation of one
    fire, not four observations over time."""
    rows = [_det(19.05 + i * 0.002, 72.93, BASE) for i in range(4)]
    rows += [_det(19.05 + i * 0.002, 72.93, BASE + pd.Timedelta(hours=12)) for i in range(4)]
    feats = _features(rows)
    assert feats["n_detections"].iloc[0] == 8
    assert feats["n_overpasses"].iloc[0] == 2


def test_a_single_detection_event_reports_zero_rather_than_nothing():
    """The smallest event must not produce NaN in columns a model will read."""
    feats = _features([_det(19.05, 72.93, BASE)])
    assert feats["duration_h"].iloc[0] == 0.0
    assert feats["centroid_drift_km"].iloc[0] == 0.0
    assert feats["drift_rate_km_per_day"].iloc[0] == 0.0
    assert feats["extent_km"].iloc[0] == 0.0
    assert feats["n_detections"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Radiometry and context carried up from the detections
# ---------------------------------------------------------------------------

def test_frp_trend_is_positive_for_an_intensifying_event():
    rows = [
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=12 * i), frp=5.0 * (i + 1))
        for i in range(6)
    ]
    assert _features(rows)["frp_trend_mw_per_day"].iloc[0] > 0


def test_frp_trend_is_negative_for_a_dying_event():
    rows = [
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=12 * i), frp=60.0 - 8.0 * i)
        for i in range(6)
    ]
    assert _features(rows)["frp_trend_mw_per_day"].iloc[0] < 0


def test_night_fraction_reflects_the_detections():
    rows = [
        _det(19.05, 72.93, BASE, daynight="N"),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=12), daynight="D"),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=24), daynight="D"),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=36), daynight="D"),
    ]
    assert _features(rows)["night_fraction"].iloc[0] == pytest.approx(0.25)


def test_inside_industrial_survives_the_parquet_string_round_trip():
    """Parquet turns booleans into "True"/"False", as FireFeaturePipeline
    already documents. A fraction of 0.0 for a facility event would tell the
    model the opposite of the truth."""
    rows = [
        _det(22.42, 69.85, BASE + pd.Timedelta(hours=12 * i),
             inside_industrial="True", facility_name="Plant A")
        for i in range(4)
    ]
    assert _features(rows)["inside_industrial_fraction"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# dNBR: an absence must stay an absence
# ---------------------------------------------------------------------------

def test_dnbr_absent_is_none_and_never_zero():
    """`sentinel2_client` refuses to report an unmeasured dNBR as 0.0, because
    0.0 reads as "measured, no burn scar". Aggregating must not undo that."""
    rows = [_det(19.05, 72.93, BASE + pd.Timedelta(hours=12 * i)) for i in range(3)]
    feats = _features(rows)
    assert feats["dnbr_mean"].iloc[0] is None
    assert feats["dnbr_measured_n"].iloc[0] == 0


def test_dnbr_averages_only_what_was_measured():
    rows = [
        _det(19.05, 72.93, BASE, dnbr=0.30),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=12), dnbr=None),
        _det(19.05, 72.93, BASE + pd.Timedelta(hours=24), dnbr=0.50),
    ]
    feats = _features(rows)
    assert feats["dnbr_mean"].iloc[0] == pytest.approx(0.40)
    assert feats["dnbr_measured_n"].iloc[0] == 2


# ---------------------------------------------------------------------------
# Shape of the output
# ---------------------------------------------------------------------------

def test_one_row_per_event():
    rows = [_det(19.05, 72.93, BASE + pd.Timedelta(days=d)) for d in (0, 1, 9, 10)]
    detections = assign_events(pd.DataFrame(rows))
    feats = build_event_features(detections)

    assert len(feats) == detections["event_id"].nunique() == 2
    assert feats["n_detections"].sum() == len(detections)


def test_identity_frame_agrees_with_the_feature_frame():
    rows = [_det(19.05, 72.93, BASE + pd.Timedelta(days=d)) for d in (0, 1, 9, 10)]
    detections = assign_events(pd.DataFrame(rows))

    identity = event_identity_frame(detections).sort_values("event_id")
    feats = build_event_features(detections).sort_values("event_id")

    assert list(identity["event_id"]) == list(feats["event_id"])
    assert list(identity["n_detections"]) == list(feats["n_detections"])


def test_features_require_event_ids():
    with pytest.raises(ValueError, match="event_id"):
        build_event_features(pd.DataFrame({
            "latitude": [1.0], "longitude": [2.0], "timestamp_utc": [BASE],
        }))


def test_empty_input_produces_an_empty_frame():
    out = build_event_features(pd.DataFrame())
    assert out.empty
    assert set(EVENT_ONLY_FEATURES) <= set(out.columns)
