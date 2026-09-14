"""Guards the classify response contract across the serving-guard boundary.

The defect these exist to prevent: `predicted_class` reported the **served**
class while `predicted_class_id` reported the **model's** index, so the two
disagreed whenever a guard fired. Bandipur came back as

    {"predicted_class": "FOREST_FIRE", "predicted_class_id": 2}

and index 2 is `AGRICULTURAL_BURN`. Deer Park came back as `TRANSIENT_HOTSPOT`
with the same id 2, where the correct index is 3.

That is worse than a cosmetic inconsistency in this system specifically. The
whole argument for `apply_serving_guards()` is that an override is *visible
rather than silent* -- and for any consumer keying on the numeric id, it was
silent. The guard did its work and the JSON hid it.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.models.verified_labels import CLASS_NAME_TO_INDEX, SERVED_ONLY_CLASSES

client = TestClient(app)


def _classify(lat, lon, frp=12.5):
    r = client.post(
        "/api/v1/classify",
        json={"latitude": lat, "longitude": lon, "frp": frp},
    )
    assert r.status_code == 200, r.text
    return r.json()


# Bandipur National Park: inside mapped forest cover, so the forest guard
# upgrades AGRICULTURAL_BURN to the served-only FOREST_FIRE class.
BANDIPUR = (11.67, 76.63)
# Deer Park, Texas: outside SEASONALITY_DOMAIN, so the harvest calendar is
# withheld and the detection is served as an unclassified transient.
DEER_PARK = (29.72, -95.13)
# Inside the Pavagada Solar Park polygon: no guard needed, model is correct.
PAVAGADA = (14.23654, 77.42073)


@pytest.mark.parametrize("lat,lon", [BANDIPUR, DEER_PARK, PAVAGADA])
def test_class_id_always_matches_its_own_class_name(lat, lon):
    """The id must describe `predicted_class`, guard fired or not."""
    d = _classify(lat, lon)
    served = d["predicted_class"]
    if served in SERVED_ONLY_CLASSES:
        assert d["predicted_class_id"] is None
    else:
        assert d["predicted_class_id"] == CLASS_NAME_TO_INDEX[served], (
            f"{served} reported id {d['predicted_class_id']}, "
            f"which names {CLASS_NAMES_BY_INDEX.get(d['predicted_class_id'])}"
        )


CLASS_NAMES_BY_INDEX = {v: k for k, v in CLASS_NAME_TO_INDEX.items()}


@pytest.mark.parametrize("lat,lon", [BANDIPUR, DEER_PARK, PAVAGADA])
def test_model_class_id_always_matches_the_model_class_name(lat, lon):
    """The model's own pair must stay internally consistent too."""
    d = _classify(lat, lon)
    assert d["model_predicted_class_id"] == CLASS_NAME_TO_INDEX[d["model_predicted_class"]]


def test_served_only_class_has_no_model_index():
    """FOREST_FIRE is decided by land cover; the model has no index for it.

    Returning a stand-in integer would re-create the confusion this fixes.
    """
    d = _classify(*BANDIPUR)
    assert d["predicted_class"] == "FOREST_FIRE"
    assert d["served_only_class"] is True
    assert d["predicted_class_id"] is None
    # The model still reports what it actually thought.
    assert d["model_predicted_class"] == "AGRICULTURAL_BURN"
    assert d["model_predicted_class_id"] == 2


def test_out_of_domain_guard_reports_the_served_index_not_the_model_one():
    d = _classify(*DEER_PARK)
    assert d["serving_guard_applied"] is True
    assert d["predicted_class"] == "TRANSIENT_HOTSPOT"
    assert d["predicted_class_id"] == 3, "was 2 (AGRICULTURAL_BURN) before the fix"
    assert d["model_predicted_class"] == "AGRICULTURAL_BURN"
    assert d["model_predicted_class_id"] == 2


def test_confidence_says_which_class_it_describes():
    """A guard leaves the confidence attached to the class the model scored.

    Bandipur reported 100% confidence beside FOREST_FIRE, a class the model
    never evaluated. The figure is not wrong, but unlabelled it reads as
    certainty about the served answer.
    """
    guarded = _classify(*BANDIPUR)
    assert guarded["serving_guard_applied"] is True
    assert guarded["confidence_describes"] == "model_predicted_class"

    plain = _classify(*PAVAGADA)
    assert plain["serving_guard_applied"] is False
    assert plain["confidence_describes"] == "predicted_class"


@pytest.mark.parametrize("lat,lon", [BANDIPUR, DEER_PARK, PAVAGADA])
def test_unguarded_responses_keep_both_pairs_in_agreement(lat, lon):
    """With no override, served and model answers must be identical."""
    d = _classify(lat, lon)
    if not d["serving_guard_applied"]:
        assert d["predicted_class"] == d["model_predicted_class"]
        assert d["predicted_class_id"] == d["model_predicted_class_id"]


@pytest.mark.parametrize("lat,lon", [BANDIPUR, DEER_PARK, PAVAGADA])
def test_confidence_percent_is_still_present_and_numeric(lat, lon):
    """The UI reads this key; the fix must not remove it."""
    d = _classify(lat, lon)
    assert isinstance(d["confidence_percent"], (int, float))
    assert 0.0 <= d["confidence_percent"] <= 100.0


# ---------------------------------------------------------------------------
# A confidence that was never computed must stay absent
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unscored_detections_are_not_given_a_class_or_a_confidence():
    """Detections outside any industrial polygon skip the classifier.

    They used to be written as `CONTROLLED_PROCESS` with a hard-coded
    confidence of 0.99 -- a class absent from CLASS_NAMES and from the
    documentation, whose name contradicted its own rationale string, on 74% of
    all rows. The dashboard then rendered that as "THERMOSCOPE AI classifier
    assigned 99% confidence", crediting a model that had not run.
    """
    src = (PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "CONTROLLED_PROCESS" not in src.replace(
        '# It used to set predicted_class="CONTROLLED_PROCESS" with a', ""
    ), "the retired class name is back in a live code path"
    assert 'pred_class = "NOT_ASSESSED"' in src
    assert "conf = None" in src


def test_confidence_column_accepts_null():
    """A confidence that was never computed must be storable as absent."""
    from app.database import Incident

    assert Incident.__table__.c.confidence.nullable, (
        "confidence is NOT NULL again, which forces a placeholder value onto "
        "rows the model never scored"
    )


def test_ui_never_substitutes_a_default_confidence():
    """The dashboard must not invent a figure where none exists."""
    ui = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "confidence_percent || 95" not in ui
    assert "inc.confidence || 0.95" not in ui


def test_ui_offers_only_classes_the_system_can_produce():
    """The reclassify dropdown writes straight into the audit log.

    `AGRICULTURAL_STUBBLE` sat in it and existed nowhere in the Python, so an
    analyst could file a label no model could produce and no report could
    reconcile.
    """
    ui = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    import re

    offered = set(re.findall(r'<option value="([A-Z_]+)"', ui))
    allowed = set(CLASS_NAME_TO_INDEX) | SERVED_ONLY_CLASSES
    assert offered <= allowed, f"dropdown offers unknown classes: {offered - allowed}"
