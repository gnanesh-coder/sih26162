"""Guards the emergency-responder layer and its honesty boundary.

The layer answers a question the SitRep already implied: it recommends an
evacuation cordon, and until now the system could not say who would enforce it
or where casualties would go.

The thing most worth protecting here is the line between what is measured and
what is assumed. Distance is geodesic -- a measurement. Travel time is NOT
routed: it is the geodesic distance times a circuity factor divided by an
assumed speed, and the payload has to keep saying so. A straight line divided
by an assumed speed, presented as a "real-time ETA", is a fabricated
measurement of the same family as the synthetic weather this project removed
and the CO2 figure it refuses to print.
"""

import math

import pytest

from src.pipeline import responders as R


pytestmark = pytest.mark.skipif(
    not R.layer_available(),
    reason="responder layer not built (python -m src.ingestion.pbf_extractor --layer responders)",
)

JHARIA = (23.75, 86.42)
MUMBAI = (19.0760, 72.8777)


# ---------------------------------------------------------------------------
# The honesty boundary
# ---------------------------------------------------------------------------

def test_travel_estimate_declares_it_is_not_routed():
    r = R.nearest_responders(*JHARIA)
    for items in r["responders"].values():
        for item in items:
            t = item["travel_estimate"]
            assert t["is_routed"] is False
            assert "NOT a routed travel time" in t["basis"]


def test_travel_estimate_names_both_assumptions():
    """The figure must not be able to travel without what produced it."""
    r = R.nearest_responders(*JHARIA)
    sample = next(i for v in r["responders"].values() if v for i in v)
    basis = sample["travel_estimate"]["basis"]
    assert str(R.CIRCUITY_FACTOR) in basis
    assert f"{R.AVG_RESPONSE_SPEED_KMH:.0f}" in basis


def test_indicative_time_follows_from_the_stated_assumptions():
    """Reproduce the number from the assumptions alone."""
    est = R._indicative_travel(10.0)
    expected_km = 10.0 * R.CIRCUITY_FACTOR
    expected_min = (expected_km / R.AVG_RESPONSE_SPEED_KMH) * 60.0
    assert est["indicative_road_km"] == pytest.approx(expected_km, abs=0.01)
    assert est["indicative_minutes"] == pytest.approx(expected_min, abs=0.1)


def test_result_carries_the_coverage_caveat():
    """An incomplete register must not read as a complete one."""
    r = R.nearest_responders(*JHARIA)
    assert "not a complete register" in r["caveat"].lower()
    assert "geodesic" in r["caveat"].lower()


# ---------------------------------------------------------------------------
# Distance correctness
# ---------------------------------------------------------------------------

def test_distance_is_geodesic_and_ordered():
    r = R.nearest_responders(*JHARIA, per_kind=3)
    for items in r["responders"].values():
        dists = [i["distance_km"] for i in items]
        assert dists == sorted(dists), "results must be nearest-first"


def test_distance_matches_an_independent_haversine():
    r = R.nearest_responders(*JHARIA, per_kind=1)
    lat0, lon0 = JHARIA
    for items in r["responders"].values():
        for item in items:
            dlat = math.radians(item["latitude"] - lat0)
            dlon = math.radians(item["longitude"] - lon0)
            a = (math.sin(dlat / 2) ** 2
                 + math.cos(math.radians(lat0)) * math.cos(math.radians(item["latitude"]))
                 * math.sin(dlon / 2) ** 2)
            expected = 2 * 6371.0088 * math.asin(math.sqrt(a))
            assert item["distance_km"] == pytest.approx(expected, abs=0.01)


def test_radius_is_respected():
    r = R.nearest_responders(*JHARIA, max_km=2.0)
    for items in r["responders"].values():
        for item in items:
            assert item["distance_km"] <= 2.0


# ---------------------------------------------------------------------------
# Grouping, and reporting absence
# ---------------------------------------------------------------------------

def test_categories_are_grouped_not_pooled():
    """A flat nearest-N in a city would be all hospitals and no fire station.

    Mumbai is the case that proves it: hospitals are mapped far more densely
    than fire stations, so pooling would bury the one category that matters.
    """
    r = R.nearest_responders(*MUMBAI, per_kind=1)
    assert set(r["responders"]) == set(R.RESPONDER_KINDS)


def test_nothing_within_range_is_reported_not_glossed():
    """An empty category must be named, not silently omitted."""
    # 1 km around a point in the Bay of Bengal reaches nothing.
    r = R.nearest_responders(15.0, 88.0, max_km=1.0)
    assert set(r["none_within_range"]) == set(R.RESPONDER_KINDS)
    assert all(v == [] for v in r["responders"].values())


def test_per_kind_limit_is_respected():
    r = R.nearest_responders(*MUMBAI, per_kind=2)
    for items in r["responders"].values():
        assert len(items) <= 2


def test_kinds_filter_returns_only_what_was_asked_for():
    r = R.nearest_responders(*MUMBAI, kinds=["FIRE"])
    assert set(r["responders"]) == {"FIRE"}


# ---------------------------------------------------------------------------
# Layer summary
# ---------------------------------------------------------------------------

def test_summary_reports_all_three_categories():
    s = R.layer_summary()
    assert s["status"] == "OK"
    assert set(s["by_kind"]) == set(R.RESPONDER_KINDS)
    assert s["total"] == sum(s["by_kind"].values())


def test_fire_stations_are_the_sparse_category():
    """Recorded because it shapes how the feature must be read.

    OSM maps hospitals densely and fire stations sparsely, so the nearest
    mapped fire station is routinely much further than the nearest real one.
    If this ever inverts, the caveat text needs revisiting.
    """
    s = R.layer_summary()
    assert s["by_kind"]["FIRE"] < s["by_kind"]["HOSPITAL"]
