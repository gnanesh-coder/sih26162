"""Unit & Integration Tests for Tactical Situation Report (SitRep) Briefing Generator.

Verifies:
  - MGRS & UTM coordinate conversion accuracy
  - Atmospheric dispersion & chemical emissions estimation
  - REST API endpoint for SitRep JSON export (/api/v1/incident/{id}/sitrep?format=json)
  - REST API endpoint for SitRep A4 printable HTML export (/api/v1/incident/{id}/sitrep?format=html)
  - 404 handling for invalid incident IDs
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.alerting.sitrep_generator import (
    compute_atmospheric_dispersion,
    estimate_chemical_emissions,
    wgs84_to_utm_mgrs,
)


@pytest.fixture(scope="module")
def client():
    """Initializes test client under application lifespan context."""
    with TestClient(app) as c:
        yield c


def test_wgs84_to_utm_mgrs():
    """Verifies dual-coordinate conversion from WGS84 to UTM and MGRS."""
    # Jamnagar refinery approximate coordinates: 22.42°N, 69.85°E
    res = wgs84_to_utm_mgrs(22.42, 69.85)
    assert res["utm_zone"] == 42
    assert res["hemisphere"] == "N"
    assert res["latitude_band"] == "Q"
    assert "Zone 42N" in res["utm_string"]
    assert "42Q" in res["mgrs_string"]


def test_chemical_emissions_estimation():
    """Verifies chemical mass release rate estimation based on FRP and facility type."""
    # Petrochemical facility
    petro = estimate_chemical_emissions(100.0, "petrochemical_refinery")
    assert petro["co2_kg_h"] > 0
    assert petro["so2_kg_h"] > 0
    assert petro["nox_kg_h"] > 0
    assert petro["pm25_kg_h"] > 0
    assert "Flare Soot" in petro["primary_hazard"] or "Hydrocarbons" in petro["primary_hazard"]

    # Steel foundry
    steel = estimate_chemical_emissions(50.0, "steel_mill")
    assert steel["co2_kg_h"] > 0
    assert "Metallic Particulates" in steel["primary_hazard"]


def test_atmospheric_dispersion_computation():
    """Verifies weather synthesis, wind heading, and hazard evacuation radius."""
    disp = compute_atmospheric_dispersion(22.42, 69.85, 85.0)
    assert 10 <= disp["temperature_c"] <= 50
    assert 20 <= disp["humidity_pct"] <= 100
    assert disp["wind_speed_kmh"] > 0
    assert 0 <= disp["wind_direction_deg"] <= 360
    assert disp["hazard_length_km"] > 0
    assert disp["recommended_evac_radius_km"] > 0


def test_sitrep_json_api_endpoint(client, seeded_incidents):
    """Verifies programmatic JSON SitRep endpoint retrieval."""
    inc_id = seeded_incidents[0]["id"]
    sitrep_resp = client.get(f"/api/v1/incident/{inc_id}/sitrep?format=json")
    assert sitrep_resp.status_code == 200
    data = sitrep_resp.json()

    assert "report_metadata" in data
    assert "incident" in data
    assert "geospatial" in data
    assert "thermal_telemetry" in data
    assert "recurrence_metrics" in data
    assert "atmospheric_hazard" in data
    assert "chemical_emissions" in data
    assert "audit_trail" in data

    assert data["report_metadata"]["sitrep_id"].startswith("SITREP-2026-")
    assert "RESTRICTED" in data["report_metadata"]["classification"]
    assert data["incident"]["id"] == inc_id
    assert "utm_string" in data["geospatial"]
    assert "mgrs_string" in data["geospatial"]


def test_sitrep_html_api_endpoint(client, seeded_incidents):
    """Verifies printable A4 HTML SitRep endpoint retrieval."""
    inc_id = seeded_incidents[0]["id"]
    sitrep_resp = client.get(f"/api/v1/incident/{inc_id}/sitrep?format=html")
    assert sitrep_resp.status_code == 200
    assert "text/html" in sitrep_resp.headers["content-type"]

    html = sitrep_resp.text
    assert "TACTICAL SITUATION REPORT (SITREP)" in html
    assert "RESTRICTED // OPERATIONAL INTELLIGENCE" in html
    assert "Executive Threat Assessment" in html
    assert "Geospatial Intelligence" in html
    assert "Atmospheric Dispersion & Chemical Emissions Hazard" in html
    assert "window.print()" in html


def test_sitrep_nonexistent_incident_404(client):
    """Verifies 404 response for invalid incident ID."""
    resp = client.get("/api/v1/incident/99999999/sitrep")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# --------------------------------------------------------------------------
# Fabricated values must never present as measurements
# --------------------------------------------------------------------------
#
# The weather in every SitRep is generated from the incident coordinates by a
# fixed formula, and it drives the plume geometry and the evacuation radius. It
# shipped with the docstring "Calculates weather context" and no disclosure
# anywhere a reader would see it. These tests keep the disclosure attached.

def test_synthetic_weather_declares_itself(monkeypatch):
    """With no provider result, the fallback must flag itself unmistakably.

    Mocked rather than live: a test that reaches the internet passes or fails on
    someone else's uptime, and this one would silently start exercising the
    observed path the day the archive gained today's date.
    """
    from src.alerting import sitrep_generator as sg
    from src.alerting.sitrep_generator import compute_atmospheric_dispersion

    monkeypatch.setattr(sg, "fetch_observed_weather", lambda *a, **k: None)
    ctx = compute_atmospheric_dispersion(22.4, 70.05, 35.0)

    assert ctx["weather_source"] == "SYNTHETIC_FORMULA"
    assert ctx["weather_is_measured"] is False
    assert "NOT A MEASUREMENT" in ctx["weather_disclaimer"]
    assert ctx["dispersion_is_illustrative"] is True


def test_weather_provider_is_wired_to_a_real_archive():
    """The hook is no longer empty: it queries a real historical archive.

    It was deliberately unimplemented while no provider existed, because a
    formula pretending to be a provider is worse than an absent one.
    """
    from src.alerting.sitrep_generator import OPEN_METEO_ARCHIVE, fetch_observed_weather
    import inspect

    assert "archive-api.open-meteo.com" in OPEN_METEO_ARCHIVE
    src = inspect.getsource(fetch_observed_weather)
    assert "requests.get" in src
    assert "return None" in src  # still fails closed


def test_observed_weather_flips_the_provenance(monkeypatch):
    from src.alerting import sitrep_generator as sg

    monkeypatch.setattr(sg, "fetch_observed_weather", lambda lat, lon, when=None: {
        "temperature_c": 31.0, "humidity_pct": 55, "wind_speed_kmh": 12.0,
        "wind_direction_deg": 270, "wind_compass": "W",
        "downwind_deg": 90, "downwind_compass": "E",
        "observed_at": "2026-09-13T00:00:00Z",
    })

    ctx = sg.compute_atmospheric_dispersion(22.4, 70.05, 35.0)

    assert ctx["weather_source"] == "OBSERVED"
    assert ctx["weather_is_measured"] is True
    assert ctx["dispersion_is_illustrative"] is False
    assert "weather_disclaimer" not in ctx


def test_emission_estimates_do_not_claim_a_published_source():
    from src.alerting.sitrep_generator import estimate_chemical_emissions

    em = estimate_chemical_emissions(35.0, "petrochemical_refinery")

    assert em["estimate_basis"] == "HAND_TUNED_SCALING"
    assert "not traced to EPA AP-42" in em["estimate_disclaimer"]


def test_refinery_outranks_open_biomass_on_sulphur():
    """The ratios are the part these coefficients are meant to carry."""
    from src.alerting.sitrep_generator import estimate_chemical_emissions

    refinery = estimate_chemical_emissions(35.0, "petrochemical_refinery")
    biomass = estimate_chemical_emissions(35.0, "non_industrial")

    assert refinery["so2_kg_h"] > biomass["so2_kg_h"]
    assert biomass["pm25_kg_h"] > refinery["pm25_kg_h"]



# --------------------------------------------------------------------------
# The weather provider is now real -- and must fail honestly
# --------------------------------------------------------------------------

def test_provider_failure_falls_back_to_labelled_synthetic(monkeypatch):
    """No network must give clearly-flagged synthetic values, never a guess."""
    from src.alerting import sitrep_generator as sg

    def boom(*a, **k):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(sg.requests, "get", boom)
    ctx = sg.compute_atmospheric_dispersion(22.34, 69.87, 35.0, when="2026-03-15T12:00:00Z")

    assert ctx["weather_source"] == "SYNTHETIC_FORMULA"
    assert ctx["weather_is_measured"] is False
    assert "NOT A MEASUREMENT" in ctx["weather_disclaimer"]


def test_empty_archive_response_is_not_treated_as_observation(monkeypatch):
    from src.alerting import sitrep_generator as sg

    class Resp:
        status_code = 200
        def json(self):
            return {"hourly": {"temperature_2m": []}}

    monkeypatch.setattr(sg.requests, "get", lambda *a, **k: Resp())
    assert sg.fetch_observed_weather(22.34, 69.87, "2026-03-15") is None


def test_observation_is_taken_at_the_incident_hour(monkeypatch):
    """A March fire must not be briefed with today's weather stamped OBSERVED."""
    from src.alerting import sitrep_generator as sg

    captured = {}

    class Resp:
        status_code = 200
        def json(self):
            return {"hourly": {
                "time": [f"2026-03-15T{h:02d}:00" for h in range(24)],
                "temperature_2m": list(range(24)),
                "relative_humidity_2m": [50] * 24,
                "wind_speed_10m": [10.0] * 24,
                "wind_direction_10m": [270] * 24,
            }}

    def fake_get(url, params=None, timeout=None):
        captured.update(params or {})
        return Resp()

    monkeypatch.setattr(sg.requests, "get", fake_get)
    obs = sg.fetch_observed_weather(22.34, 69.87, "2026-03-15T07:00:00Z")

    assert captured["start_date"] == "2026-03-15"
    assert obs["temperature_c"] == 7.0          # the 07:00 reading, not a daily mean
    assert obs["observed_at"].startswith("2026-03-15T07:00")


def test_observed_weather_removes_the_synthetic_banner(monkeypatch):
    from src.alerting import sitrep_generator as sg

    monkeypatch.setattr(sg, "fetch_observed_weather", lambda lat, lon, when=None: {
        "temperature_c": 31.0, "humidity_pct": 55, "wind_speed_kmh": 12.0,
        "wind_direction_deg": 270, "wind_compass": "W",
        "downwind_deg": 90, "downwind_compass": "E", "observed_at": "2026-03-15T12:00Z",
    })
    ctx = sg.compute_atmospheric_dispersion(22.34, 69.87, 35.0, when="2026-03-15")

    assert ctx["weather_source"] == "OBSERVED"
    assert "weather_disclaimer" not in ctx
    assert ctx["dispersion_is_illustrative"] is False
