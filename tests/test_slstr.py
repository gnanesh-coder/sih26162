"""Tests for the Sentinel-3 SLSTR fire-channel retrieval.

The properties defended here are the ones that decide whether a number printed
under the word "temperature" can be trusted:

  * an unconfigured client makes no network call and says so;
  * a failed solve reports that it failed instead of returning a number;
  * the background comes from the same acquisition as the source, never a
    different day's weather;
  * the background annulus really is an annulus, so the fire is not measured
    against itself.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import slstr_client as sc
from src.ingestion.slstr_client import (
    STATUS_NO_CREDENTIALS,
    STATUS_NO_SCENE,
    STATUS_NO_SOLUTION,
    STATUS_OK,
    SlstrClient,
    _best_acquisition,
    annulus_geometry,
    disc_geometry,
)


def _interval(day, f1_max, f2_max, s7_max=None, n=9, f2_p25=None):
    def band(stats):
        return {"bands": {"B0": {"stats": stats}}}

    outputs = {
        "f1": band({"max": f1_max, "mean": f1_max - 1, "sampleCount": n, "noDataCount": 0}),
        "f2": band({"max": f2_max, "mean": f2_max - 1, "sampleCount": n, "noDataCount": 0,
                    "percentiles": {"25.0": f2_p25 if f2_p25 is not None else f2_max - 2}}),
    }
    if s7_max is not None:
        outputs["s7"] = band({"max": s7_max, "sampleCount": n, "noDataCount": 0})
    return {"interval": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"},
            "outputs": outputs}


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def test_annulus_has_a_hole_so_the_fire_is_not_its_own_background():
    """A plain disc would include the hot pixels the background must exclude."""
    geom = annulus_geometry(22.345, 69.87, 5.0, 15.0)

    assert geom["type"] == "Polygon"
    assert len(geom["coordinates"]) == 2, "outer ring plus one hole"

    outer, inner = geom["coordinates"]
    assert outer[0] == outer[-1] and inner[0] == inner[-1], "rings must close"
    assert _signed_area(outer) * _signed_area(inner) < 0, (
        "the hole must wind opposite to the outer ring or it is not a hole"
    )


def test_annulus_inner_radius_clears_the_source():
    inner_pts = annulus_geometry(22.345, 69.87, 5.0, 15.0)["coordinates"][1]
    dists = [_km(22.345, 69.87, p[1], p[0]) for p in inner_pts]
    assert min(dists) > 4.0, "the punched-out hole must be larger than the hot area"


def test_disc_radius_is_about_what_was_asked_for():
    pts = disc_geometry(10.0, 76.0, 1.5)["coordinates"][0]
    dists = [_km(10.0, 76.0, p[1], p[0]) for p in pts]
    assert 1.4 < min(dists) <= max(dists) < 1.6


def _signed_area(ring):
    return sum((ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1])
               for i in range(len(ring) - 1)) / 2.0


def _km(lat1, lon1, lat2, lon2):
    dl = (lat2 - lat1) * 111.32
    dn = (lon2 - lon1) * 111.32 * np.cos(np.radians(lat1))
    return float(np.hypot(dl, dn))


# --------------------------------------------------------------------------
# Acquisition selection
# --------------------------------------------------------------------------

def test_the_hottest_acquisition_is_chosen_not_the_average():
    """A flare is intermittent; a window mean is dominated by the quiet days."""
    data = [_interval("2026-03-01", 295.0, 294.0),
            _interval("2026-03-25", 320.8, 303.4),
            _interval("2026-03-28", 297.0, 295.0)]
    assert _best_acquisition(data)["from"].startswith("2026-03-25")


def test_background_is_pinned_to_the_source_acquisition():
    """Ambient ground from another day would smuggle in different weather."""
    data = [_interval("2026-03-01", 291.0, 290.0, f2_p25=289.0),
            _interval("2026-03-25", 293.0, 292.0, f2_p25=291.6)]
    rec = _best_acquisition(data, prefer="background", match_day="2026-03-25")

    assert rec["from"].startswith("2026-03-25")
    assert rec["f2_background"] == pytest.approx(291.6)


def test_no_matching_day_is_reported_as_nothing_rather_than_the_nearest():
    data = [_interval("2026-03-01", 291.0, 290.0)]
    assert _best_acquisition(data, prefer="background", match_day="2026-03-25") is None


def test_intervals_without_valid_samples_are_skipped():
    empty = _interval("2026-03-10", 300.0, 299.0, n=0)
    empty["outputs"]["f1"]["bands"]["B0"]["stats"]["noDataCount"] = 0
    empty["outputs"]["f1"]["bands"]["B0"]["stats"]["sampleCount"] = 0
    assert _best_acquisition([empty]) is None


# --------------------------------------------------------------------------
# The client contract
# --------------------------------------------------------------------------

def test_unconfigured_client_makes_no_request(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(sc.requests, "post",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    result = SlstrClient(client_id="", client_secret="").retrieve_temperature(
        22.3, 69.8, "2026-03-01", "2026-03-31")

    assert result["status"] == STATUS_NO_CREDENTIALS
    assert calls["n"] == 0
    assert "fire_temperature_k" not in result


def test_empty_string_credentials_are_not_backfilled_from_the_environment(monkeypatch):
    monkeypatch.setenv("CDSE_CLIENT_ID", "real-id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "real-secret")
    assert SlstrClient(client_id="", client_secret="").is_configured() is False
    assert SlstrClient().is_configured() is True


def test_a_missing_scene_never_becomes_a_temperature(monkeypatch):
    client = SlstrClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "tok")
    monkeypatch.setattr(client, "_statistics", lambda *a, **k: (STATUS_OK, []))

    result = client.retrieve_temperature(22.3, 69.8, "2026-03-01", "2026-03-31")

    assert result["status"] == STATUS_NO_SCENE
    assert "fire_temperature_k" not in result


def test_an_unsolvable_mixture_reports_no_solution_with_the_evidence(monkeypatch):
    """A source too small to lift F1 above background has no admissible solve.

    It must still return what was measured, so the failure can be read rather
    than guessed at.
    """
    client = SlstrClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "tok")

    def fake_stats(geometry, d0, d1, token, interval="P1D"):
        hole = len(geometry["coordinates"]) == 2
        # Background annulus and source agree: nothing is burning.
        return STATUS_OK, [_interval("2026-03-25", 292.4, 291.8, s7_max=291.5,
                                     f2_p25=291.2 if hole else 291.0)]

    monkeypatch.setattr(client, "_statistics", fake_stats)
    result = client.retrieve_temperature(30.75, 75.5, "2026-03-01", "2026-03-31")

    assert result["status"] == STATUS_NO_SOLUTION
    assert "fire_temperature_k" not in result
    assert result["f1_max_k"] == pytest.approx(292.4)
    assert result["background_k"] == pytest.approx(291.2)


def test_a_solved_retrieval_reports_its_background_as_measured(monkeypatch):
    """The whole point of SLSTR here is that the background is not a guess."""
    client = SlstrClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "tok")

    def fake_stats(geometry, d0, d1, token, interval="P1D"):
        hole = len(geometry["coordinates"]) == 2
        if hole:
            return STATUS_OK, [_interval("2026-03-25", 301.0, 302.0, f2_p25=300.1)]
        return STATUS_OK, [_interval("2026-03-25", 316.5, 303.0, s7_max=312.2)]

    monkeypatch.setattr(client, "_statistics", fake_stats)
    result = client.retrieve_temperature(15.17, 76.64, "2026-03-01", "2026-03-31")

    assert result["status"] == STATUS_OK
    assert result["background_basis"] == "MEASURED_ANNULUS"
    assert result["background_k"] == pytest.approx(300.1)
    assert result["fire_temperature_k"] > sc.MIN_FIRE_TEMP_K
    assert 0.0 < result["area_fraction"] < 1.0


def test_the_saturation_gap_is_reported(monkeypatch):
    """S7 clips near 311 K by design; F1 does not. The gap is the evidence."""
    client = SlstrClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "tok")

    def fake_stats(geometry, d0, d1, token, interval="P1D"):
        hole = len(geometry["coordinates"]) == 2
        if hole:
            return STATUS_OK, [_interval("2026-03-25", 296.0, 295.0, f2_p25=291.6)]
        return STATUS_OK, [_interval("2026-03-25", 320.84, 303.36, s7_max=312.28)]

    monkeypatch.setattr(client, "_statistics", fake_stats)
    result = client.retrieve_temperature(22.345, 69.87, "2026-03-01", "2026-03-31")

    assert result["s7_f1_gap_k"] == pytest.approx(8.56, abs=0.01)


def test_every_status_returned_is_a_declared_one(monkeypatch):
    client = SlstrClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "tok")
    monkeypatch.setattr(client, "_statistics", lambda *a, **k: (sc.STATUS_API_ERROR, []))

    result = client.retrieve_temperature(22.3, 69.8, "2026-03-01", "2026-03-31")
    assert result["status"] in sc.VALID_STATUSES
