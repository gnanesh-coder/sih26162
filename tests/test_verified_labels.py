"""Tests for the verified-label harness.

Verified labels are the only measurement in this project that is independent of
its own labelling rule, which makes them the scarce and load-bearing resource.
These tests guard the two ways that value can be quietly destroyed: admitting an
uncited label, and mis-locating a real one so it matches nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.verified_labels import (
    evaluate_events_against_verified,
    CLASS_NAME_TO_INDEX,
    KNOWN_UNDETECTED_INCIDENTS,
    VERIFIED_EVENTS,
    VerifiedEvent,
    attach_verified_labels,
    evaluate_against_verified,
    load_verified_events,
)


# --------------------------------------------------------------------------
# Provenance is mandatory
# --------------------------------------------------------------------------

def test_label_without_a_source_is_rejected():
    """A verified label with no citation is just a guess wearing a badge."""
    with pytest.raises(ValueError, match="how it was established"):
        VerifiedEvent(
            name="Unsourced claim", lat=22.0, lon=70.0,
            start_date="2026-01-01", end_date="2026-01-31",
            label="ACCIDENTAL_FIRE", confidence="high", source="",
        )


def test_unknown_class_is_rejected():
    with pytest.raises(ValueError, match="unknown label"):
        VerifiedEvent(
            name="Bad class", lat=22.0, lon=70.0,
            start_date="2026-01-01", end_date="2026-01-31",
            label="REFINERY_ON_FIRE", confidence="high", source="cited",
        )


def test_confidence_must_be_declared():
    with pytest.raises(ValueError, match="confidence"):
        VerifiedEvent(
            name="No confidence", lat=22.0, lon=70.0,
            start_date="2026-01-01", end_date="2026-01-31",
            label="ACCIDENTAL_FIRE", confidence="probably", source="cited",
        )


def test_every_seeded_event_carries_a_substantive_source():
    """The seed set must model the standard it asks contributors to meet."""
    for ev in VERIFIED_EVENTS:
        assert len(ev.source.strip()) > 40, f"{ev.name}: source is too thin to audit"
        assert ev.confidence in {"high", "medium"}
        assert ev.label in CLASS_NAME_TO_INDEX


def test_seeded_events_have_sane_geometry_and_dates():
    for ev in VERIFIED_EVENTS:
        assert -90 <= ev.lat <= 90 and -180 <= ev.lon <= 180
        assert pd.Timestamp(ev.start_date) <= pd.Timestamp(ev.end_date)
        assert 0 < ev.radius_km <= 50


# --------------------------------------------------------------------------
# Spatio-temporal matching
# --------------------------------------------------------------------------

@pytest.fixture
def event():
    return VerifiedEvent(
        name="Test refinery", lat=22.0, lon=70.0,
        start_date="2026-03-01", end_date="2026-03-31",
        label="PERSISTENT_BASELINE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=5.0,
    )


def _detections(rows):
    return pd.DataFrame(rows, columns=["latitude", "longitude", "acq_date"])


def test_detection_inside_window_is_labelled(event):
    df = _detections([(22.0, 70.0, "2026-03-15")])
    out = attach_verified_labels(df, [event])

    assert out.loc[0, "verified_label"] == CLASS_NAME_TO_INDEX["PERSISTENT_BASELINE"]
    assert out.loc[0, "verified_event"] == "Test refinery"
    assert out.loc[0, "verified_confidence"] == "high"


def test_detection_outside_radius_is_not_labelled(event):
    # ~0.5 degrees north is ~55km, well beyond the 5km radius.
    df = _detections([(22.5, 70.0, "2026-03-15")])
    out = attach_verified_labels(df, [event])
    assert pd.isna(out.loc[0, "verified_label"])


def test_detection_outside_date_window_is_not_labelled(event):
    df = _detections([(22.0, 70.0, "2026-05-15")])
    out = attach_verified_labels(df, [event])
    assert pd.isna(out.loc[0, "verified_label"])


def test_window_boundaries_are_inclusive(event):
    df = _detections([
        (22.0, 70.0, "2026-03-01"),
        (22.0, 70.0, "2026-03-31"),
        (22.0, 70.0, "2026-02-28"),
    ])
    out = attach_verified_labels(df, [event])
    assert out.verified_label.notna().tolist() == [True, True, False]


def test_longitude_radius_accounts_for_latitude(event):
    """A degree of longitude is ~half a degree of latitude at 60N.

    Without the cosine correction the match region becomes an ellipse and points
    are wrongly included or excluded east-west.
    """
    high_lat = VerifiedEvent(
        name="High latitude", lat=60.0, lon=70.0,
        start_date="2026-03-01", end_date="2026-03-31",
        label="ACCIDENTAL_FIRE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=10.0,
    )
    # 0.15 deg lon at 60N is ~8.3km on the ground: inside a 10km radius.
    df = _detections([(60.0, 70.15, "2026-03-15")])
    out = attach_verified_labels(df, [high_lat])
    assert out.verified_label.notna().all()

    # 0.30 deg lon at 60N is ~16.7km: outside.
    df2 = _detections([(60.0, 70.30, "2026-03-15")])
    out2 = attach_verified_labels(df2, [high_lat])
    assert out2.verified_label.isna().all()


def test_more_specific_event_wins_an_overlap():
    """A point incident inside a wide regional window is the sharper claim."""
    regional = VerifiedEvent(
        name="Regional burn", lat=30.0, lon=75.0,
        start_date="2026-10-01", end_date="2026-11-30",
        label="AGRICULTURAL_BURN", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=25.0,
    )
    incident = VerifiedEvent(
        name="Plant fire", lat=30.0, lon=75.0,
        start_date="2026-10-15", end_date="2026-10-16",
        label="ACCIDENTAL_FIRE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=2.0,
    )

    df = _detections([(30.0, 75.0, "2026-10-15")])
    out = attach_verified_labels(df, [regional, incident])

    assert out.loc[0, "verified_event"] == "Plant fire"
    assert out.loc[0, "verified_label"] == CLASS_NAME_TO_INDEX["ACCIDENTAL_FIRE"]


def test_empty_detections_frame_is_handled(event):
    out = attach_verified_labels(_detections([]), [event])
    assert out.empty
    assert "verified_label" in out.columns


# --------------------------------------------------------------------------
# Evaluation harness
# --------------------------------------------------------------------------

class _StubModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return np.full(len(X), self.prediction)


class _StubPipeline:
    def transform(self, df):
        return pd.DataFrame({"f": np.zeros(len(df))}, index=df.index)


def test_evaluation_reports_no_matches_rather_than_a_fake_score(event):
    """Zero overlap must be stated, not silently reported as 0.0 accuracy."""
    df = _detections([(10.0, 10.0, "2020-01-01")])
    res = evaluate_against_verified(df, _StubModel(0), _StubPipeline(), [event])

    assert res["status"] == "NO_VERIFIED_MATCHES"
    assert "accuracy" not in res


def test_evaluation_scores_only_verified_detections(event):
    """Unverified rows must not dilute the independent measurement."""
    df = _detections([
        (22.0, 70.0, "2026-03-15"),   # verified, model will get this right
        (10.0, 10.0, "2026-03-15"),   # unverified, must be excluded entirely
    ])
    correct = CLASS_NAME_TO_INDEX["PERSISTENT_BASELINE"]
    res = evaluate_against_verified(df, _StubModel(correct), _StubPipeline(), [event],
                                    min_support=1)

    assert res["n_verified_detections"] == 1
    assert res["accuracy"] == 1.0


def test_evaluation_flags_low_support(event):
    df = _detections([(22.0, 70.0, "2026-03-15")])
    res = evaluate_against_verified(df, _StubModel(0), _StubPipeline(), [event],
                                    min_support=20)

    assert res["status"] == "LOW_SUPPORT"
    assert "defensible accuracy claim" in res["detail"]


def test_evaluation_detects_a_wrong_model(event):
    """The harness must be able to fail the model, or it is not a measurement."""
    df = _detections([(22.0, 70.0, "2026-03-15")])
    wrong = CLASS_NAME_TO_INDEX["ACCIDENTAL_FIRE"]
    res = evaluate_against_verified(df, _StubModel(wrong), _StubPipeline(), [event],
                                    min_support=1)

    assert res["accuracy"] == 0.0


def test_evaluation_carries_its_caveat(event):
    df = _detections([(22.0, 70.0, "2026-03-15")])
    res = evaluate_against_verified(df, _StubModel(0), _StubPipeline(), [event],
                                    min_support=1)
    assert "rule-derived labels" in res["caveat"]


def test_loader_returns_seed_set_without_csv(tmp_path):
    events = load_verified_events(csv_path=tmp_path / "absent.csv")
    assert len(events) == len(VERIFIED_EVENTS)


def test_loader_rejects_uncited_csv_rows_but_keeps_valid_ones(tmp_path):
    csv_path = tmp_path / "verified_events.csv"
    csv_path.write_text(
        "name,lat,lon,start_date,end_date,label,confidence,source,radius_km,notes\n"
        "Cited event,22.0,70.0,2026-01-01,2026-01-31,ACCIDENTAL_FIRE,high,"
        "Incident report ref 123,3.0,\n"
        "Uncited event,23.0,71.0,2026-01-01,2026-01-31,ACCIDENTAL_FIRE,high,,3.0,\n",
        encoding="utf-8",
    )

    events = load_verified_events(csv_path=csv_path)
    names = [e.name for e in events]

    assert "Cited event" in names
    assert "Uncited event" not in names, "a row without a source must be rejected"


# --------------------------------------------------------------------------
# Documented incidents the sensor could not see
# --------------------------------------------------------------------------

def test_undetected_incidents_carry_a_source_and_a_reason():
    """A recorded miss is only useful if it says why and cites the incident."""
    from src.models.verified_labels import KNOWN_UNDETECTED_INCIDENTS

    assert KNOWN_UNDETECTED_INCIDENTS
    for inc in KNOWN_UNDETECTED_INCIDENTS:
        assert inc.source.strip(), f"{inc.name} has no source"
        assert inc.why_missed.strip(), f"{inc.name} does not say why it was missed"
        assert inc.evidence.strip(), f"{inc.name} cites no corpus evidence"


def test_undetected_incidents_are_not_in_the_evaluation_set():
    """They must never inflate the verified event count -- nothing scores them."""
    from src.models.verified_labels import KNOWN_UNDETECTED_INCIDENTS, load_verified_events

    verified_names = {e.name for e in load_verified_events()}
    for inc in KNOWN_UNDETECTED_INCIDENTS:
        assert inc.name not in verified_names


# ---------------------------------------------------------------------------
# The generated register (section 5c)
# ---------------------------------------------------------------------------
#
# The register exists so the evidence base is readable outside Python. That only
# works if it stays in step with the objects it was generated from, and the
# failure mode is silent: a stale table still reads as authoritative. These
# tests fail when the register has drifted, which is the moment to re-run
# `scripts/generate_verified_register.py`.


class TestVerifiedRegisterSection:
    """Guards the generated register in PROJECT_DOCUMENTATION.md."""

    DOC = PROJECT_ROOT / "PROJECT_DOCUMENTATION.md"
    BEGIN = "<!-- BEGIN GENERATED: verified-event register -->"
    END = "<!-- END GENERATED: verified-event register -->"

    def _section(self) -> str:
        text = self.DOC.read_text(encoding="utf-8")
        assert self.BEGIN in text and self.END in text, (
            "The generated-register markers are missing from "
            "PROJECT_DOCUMENTATION.md. Section 5c cannot be regenerated without "
            "them; re-insert both comment lines."
        )
        return text[text.index(self.BEGIN): text.index(self.END)]

    def test_generator_script_exists(self):
        script = PROJECT_ROOT / "scripts" / "generate_verified_register.py"
        assert script.exists(), (
            "Section 5c is generated; its generator must stay in the repository "
            "or the section becomes hand-maintained prose that will drift."
        )

    def test_every_event_appears_in_the_register(self):
        section = self._section()
        missing = [ev.name for ev in load_verified_events()
                   if ev.name not in section]
        assert not missing, (
            f"{len(missing)} verified event(s) are defined but absent from the "
            f"register: {missing}. Re-run scripts/generate_verified_register.py."
        )

    def test_every_undetected_incident_appears(self):
        section = self._section()
        missing = [inc.name for inc in KNOWN_UNDETECTED_INCIDENTS
                   if inc.name not in section]
        assert not missing, (
            f"Undetected incidents missing from the counter-register: {missing}. "
            "These bound what the system may claim and must not be dropped."
        )

    def test_every_incident_has_a_named_mechanism(self):
        """A counter-register entry with no mechanism bounds nothing.

        The register groups incidents by *how* the sensor failed, and the
        grouping lives beside the generator. Adding a seventh incident without
        adding its mechanism prints a bare "--" in the table, which reads as an
        entry nobody understood -- worse than not listing it.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_genreg", PROJECT_ROOT / "scripts" / "generate_verified_register.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        missing = [f"{i.date} {i.name}" for i in KNOWN_UNDETECTED_INCIDENTS
                   if i.date not in mod.MECHANISMS]
        assert not missing, (
            f"No mechanism recorded for: {missing}. Add an entry to MECHANISMS "
            "in scripts/generate_verified_register.py naming what defeated the "
            "sensor."
        )

    def test_every_incident_states_its_evidence(self):
        """The zero has to be measured, not asserted."""
        for inc in KNOWN_UNDETECTED_INCIDENTS:
            assert any(ch.isdigit() for ch in inc.evidence), (
                f"{inc.name}: `evidence` records no counts. A claim that a fire "
                "was undetected is only worth keeping if the window was pulled "
                "and what landed was counted."
            )
            assert inc.why_missed.strip(), f"{inc.name}: no mechanism given."

    def test_event_count_is_current(self):
        section = self._section()
        n = len(load_verified_events())
        assert f"**{n} events are defined**" in section, (
            f"The register does not state the current event count ({n}). "
            "Re-run scripts/generate_verified_register.py."
        )

    def test_every_source_citation_is_carried_over(self):
        """The citation is the whole point; a register without it proves nothing."""
        section = self._section()
        for ev in load_verified_events():
            head = ev.source.strip()[:60]
            assert head in section, (
                f"{ev.name}: its source citation is not in the register. "
                "The register is the only place a reader sees provenance."
            )


# --------------------------------------------------------------------------
# Event-level evaluation: one vote per citation, not one per pixel
#
# The register defines events; the old harness scored detections inside them.
# Jharia supplies 23,633 and Buncefield supplies 5, so the detection-weighted
# number answers "did we classify the biggest sites right" when the question
# asked was "did we classify this fire right".
# --------------------------------------------------------------------------

def _timed_detections(rows):
    """Detections with a real timestamp, so they can be grouped into events."""
    return pd.DataFrame(rows, columns=["latitude", "longitude", "timestamp_utc"])


def test_event_evaluation_gives_every_event_one_vote():
    """A 100-detection event and a 1-detection event must weigh the same.

    This is the whole reason the function exists. Scored per detection the
    model below is 99% accurate; scored per event it is 50%.
    """
    big = VerifiedEvent(
        name="Big persistent site", lat=22.0, lon=70.0,
        start_date="2026-03-01", end_date="2026-03-31",
        label="PERSISTENT_BASELINE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=5.0,
    )
    small = VerifiedEvent(
        name="Small accident", lat=28.0, lon=77.0,
        start_date="2026-03-01", end_date="2026-03-02",
        label="ACCIDENTAL_FIRE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=3.0,
    )

    rows = [(22.0, 70.0, f"2026-03-{d:02d}T06:00:00Z") for d in range(1, 26)]
    rows += [(28.0, 77.0, "2026-03-01T06:00:00Z")]

    persistent = CLASS_NAME_TO_INDEX["PERSISTENT_BASELINE"]
    res = evaluate_events_against_verified(
        _timed_detections(rows), _StubModel(persistent), _StubPipeline(),
        [big, small], min_support=1,
    )

    assert res["n_verified_events_matched"] == 2
    # The model is right about the big site and wrong about the accident.
    assert res["event_accuracy"] == pytest.approx(0.5)
    assert res["per_verified_event"]["Small accident"]["accuracy"] == 0.0
    assert res["per_verified_event"]["Big persistent site"]["accuracy"] == 1.0


def test_event_evaluation_reports_grouping_purity():
    """Every number here is conditional on the boundaries being right, so the
    boundaries are measured too."""
    event = VerifiedEvent(
        name="Test refinery", lat=22.0, lon=70.0,
        start_date="2026-03-01", end_date="2026-03-31",
        label="PERSISTENT_BASELINE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=5.0,
    )
    rows = [(22.0, 70.0, f"2026-03-{d:02d}T06:00:00Z") for d in range(1, 8)]
    res = evaluate_events_against_verified(
        _timed_detections(rows), _StubModel(CLASS_NAME_TO_INDEX["PERSISTENT_BASELINE"]),
        _StubPipeline(), [event], min_support=1,
    )

    # One verified event, one truth class: grouping cannot be impure here.
    assert res["mean_grouping_purity"] == 1.0
    assert res["impure_events"] == 0


def test_event_evaluation_says_so_when_nothing_matched():
    event = VerifiedEvent(
        name="Test refinery", lat=22.0, lon=70.0,
        start_date="2026-03-01", end_date="2026-03-31",
        label="PERSISTENT_BASELINE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=5.0,
    )
    res = evaluate_events_against_verified(
        _timed_detections([(10.0, 10.0, "2020-01-01T06:00:00Z")]),
        _StubModel(0), _StubPipeline(), [event],
    )

    assert res["status"] == "NO_VERIFIED_MATCHES"
    assert "event_accuracy" not in res


def test_event_evaluation_flags_low_support():
    """21 events is enough to expose defects and not enough to certify."""
    event = VerifiedEvent(
        name="Test refinery", lat=22.0, lon=70.0,
        start_date="2026-03-01", end_date="2026-03-31",
        label="PERSISTENT_BASELINE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=5.0,
    )
    res = evaluate_events_against_verified(
        _timed_detections([(22.0, 70.0, "2026-03-01T06:00:00Z")]),
        _StubModel(0), _StubPipeline(), [event], min_support=50,
    )

    assert res["status"] == "LOW_SUPPORT"
    assert "defensible accuracy claim" in res["detail"]


def test_event_evaluation_declares_that_it_is_not_an_event_level_model():
    """The prediction is a plurality vote over a per-detection model. Quoting
    this as an event-level model's score would be the fabricated measurement
    the project refuses everywhere else."""
    event = VerifiedEvent(
        name="Test refinery", lat=22.0, lon=70.0,
        start_date="2026-03-01", end_date="2026-03-31",
        label="PERSISTENT_BASELINE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=5.0,
    )
    res = evaluate_events_against_verified(
        _timed_detections([(22.0, 70.0, "2026-03-01T06:00:00Z")]),
        _StubModel(0), _StubPipeline(), [event], min_support=1,
    )

    joined = " ".join(res["caveats"])
    assert "NOT an event-level model" in joined
    assert "mean_grouping_purity" in joined


def test_event_evaluation_can_fail_the_model():
    """A harness that cannot report a wrong answer is not a measurement."""
    event = VerifiedEvent(
        name="Test refinery", lat=22.0, lon=70.0,
        start_date="2026-03-01", end_date="2026-03-31",
        label="PERSISTENT_BASELINE", confidence="high",
        source="Synthetic fixture used only in tests.", radius_km=5.0,
    )
    rows = [(22.0, 70.0, f"2026-03-{d:02d}T06:00:00Z") for d in range(1, 6)]
    res = evaluate_events_against_verified(
        _timed_detections(rows), _StubModel(CLASS_NAME_TO_INDEX["ACCIDENTAL_FIRE"]),
        _StubPipeline(), [event], min_support=1,
    )

    assert res["event_accuracy"] == 0.0
