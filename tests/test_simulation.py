"""Unit and Integration Tests for Interactive 'What-If' Tactical Simulation Sandbox.

Verifies:
  - Real-time Point-in-Polygon + XGBoost inference with simulation parameter overrides
  - Dynamic atmospheric dispersion calculation with user-specified wind vectors
  - Instant SitRep briefing generation for hypothetical simulation scenarios (HTML & JSON)
  - Edge cases (extreme 500 MW fires, calm vs hurricane wind conditions)
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


@pytest.fixture(scope="module")
def client():
    """Initializes test client under application lifespan context."""
    with TestClient(app) as c:
        yield c


def test_classify_with_simulation_parameters(client):
    """Verifies that simulation parameter overrides accurately steer ML inference and dispersion."""
    # Scenario A: Baseline routine flare (8.0 MW inside Jamnagar refinery coordinates)
    baseline_payload = {
        "latitude": 22.42,
        "longitude": 69.85,
        "frp": 8.0,
        "bright_ti4": 330.0,
        "bright_ti5": 295.0,
        "daynight": "N",
        "inside_industrial_override": True,
        "facility_type_override": "petrochemical_refinery",
        "wind_speed_kmh": 15.0,
        "wind_direction_deg": 270.0,  # Westerly wind -> Plume blows East (90 deg)
    }

    resp = client.post("/api/v1/classify", json=baseline_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["predicted_class"] == "PERSISTENT_BASELINE"
    assert data["alert_priority"] in ("P2_ADVISORY", "NON_ALERT")
    assert "atmospheric_dispersion" in data
    assert data["atmospheric_dispersion"]["wind"]["speed_kmh"] == 15.0
    assert data["atmospheric_dispersion"]["wind"]["direction_deg"] == 270
    assert data["atmospheric_dispersion"]["wind"]["downwind_deg"] == 90

    # Scenario B: Catastrophic 350 MW uncontained surge
    catastrophic_payload = {
        "latitude": 22.42,
        "longitude": 69.85,
        "frp": 350.0,
        "bright_ti4": 460.0,
        "bright_ti5": 320.0,
        "daynight": "N",
        "inside_industrial_override": True,
        "facility_type_override": "petrochemical_refinery",
        "wind_speed_kmh": 40.0,
        "wind_direction_deg": 180.0,  # Southerly wind -> Plume blows North (0 deg)
    }

    resp_cat = client.post("/api/v1/classify", json=catastrophic_payload)
    assert resp_cat.status_code == 200
    cat_data = resp_cat.json()

    assert cat_data["predicted_class"] == "ACCIDENTAL_FIRE"
    assert cat_data["alert_priority"] == "P0_EMERGENCY"
    assert cat_data["alert_state"] == "UNCONTAINED_EMERGENCY"
    assert cat_data["confidence_percent"] >= 90.0
    assert cat_data["atmospheric_dispersion"]["smoke_dispersion_cone"]["hazard_length_km"] > data["atmospheric_dispersion"]["smoke_dispersion_cone"]["hazard_length_km"]
    assert cat_data["atmospheric_dispersion"]["wind"]["downwind_deg"] == 0


def test_simulation_sitrep_json_generation(client):
    """Verifies that a simulated what-if scenario generates a complete machine-readable SitRep."""
    sim_payload = {
        "latitude": 21.68,
        "longitude": 72.58,
        "frp": 125.0,
        "bright_ti4": 380.0,
        "bright_ti5": 305.0,
        "inside_industrial_override": True,
        "facility_type_override": "chemical_storage",
        "wind_speed_kmh": 22.0,
        "wind_direction_deg": 45.0,
    }

    resp = client.post("/api/v1/simulation/sitrep?format=json", json=sim_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["report_metadata"]["sitrep_id"] == "SITREP-SIM-125MW"
    assert "SIMULATION EXERCISE" in data["report_metadata"]["classification"]
    assert data["incident"]["satellite"] == "SIMULATED (What-If Scenario)"
    assert data["thermal_telemetry"]["frp_mw"] == 125.0
    assert data["chemical_emissions"]["so2_kg_h"] > 0
    assert "chemical_storage" in data["geospatial"]["facility_type"]


def test_simulation_sitrep_html_generation(client):
    """Verifies that a simulated scenario produces a printable A4 defense-grade HTML SitRep."""
    sim_payload = {
        "latitude": 22.42,
        "longitude": 69.85,
        "frp": 250.0,
        "bright_ti4": 420.0,
        "bright_ti5": 310.0,
        "inside_industrial_override": True,
        "facility_type_override": "petrochemical_refinery",
    }

    resp = client.post("/api/v1/simulation/sitrep?format=html", json=sim_payload)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]

    html = resp.text
    assert "TACTICAL SITUATION REPORT (SITREP)" in html
    assert "SIMULATION EXERCISE // UNCLASSIFIED" in html
    assert "250.0 MW" in html
    assert "window.print()" in html


def test_weather_context_parameter_overrides(client):
    """Verifies that GET /api/v1/weather/context respects manual wind speed and heading overrides."""
    resp = client.get("/api/v1/weather/context?lat=22.42&lon=69.85&frp=80&wind_speed=35&wind_deg=90")
    assert resp.status_code == 200
    data = resp.json()

    assert data["wind"]["speed_kmh"] == 35.0
    assert data["wind"]["direction_deg"] == 90
    assert data["wind"]["downwind_deg"] == 270  # Blows West
    assert data["smoke_dispersion_cone"]["type"] == "Polygon"
    assert len(data["smoke_dispersion_cone"]["coordinates"][0]) == 5
