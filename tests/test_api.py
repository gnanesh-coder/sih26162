"""Comprehensive Test Suite for FastAPI Serving Backend & Incident API.

Verifies:
  - Tactical GIS Dashboard UI serving (GET /)
  - System health and database connectivity (GET /api/v1/health)
  - Active incidents retrieval & filtering (GET /api/v1/alerts/active)
  - Incident details retrieval (GET /api/v1/incident/{id})
  - Operator incident status transitions (POST /api/v1/incident/{id}/status)
  - Executive & Tactical KPI metrics (GET /api/v1/stats)
  - Telemetry syncing from parquet (POST /api/v1/sync)
  - Real-time Point-in-Polygon + XGBoost + TreeSHAP classification (POST /api/v1/classify)
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


def test_dashboard_ui_serving(client):
    """Verifies that root URL serves the interactive Leaflet GIS dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "PROJECT SIH26162" in response.text
    assert "Leaflet" in response.text or "map" in response.text


def test_health_check_endpoint(client):
    """Verifies system health, DB mode, and model readiness."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database_mode" in data
    assert "records_count" in data
    assert data["model_loaded"] is True


def test_active_alerts_and_filtering(client):
    """Verifies active alerts retrieval with optional filtering."""
    response = client.get("/api/v1/alerts/active")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["incidents"], list)

    # Test industrial filter
    ind_resp = client.get("/api/v1/alerts/active?industrial_only=true")
    assert ind_resp.status_code == 200
    ind_data = ind_resp.json()
    for inc in ind_data["incidents"]:
        assert inc["inside_industrial"] is True


def test_stats_kpi_metrics(client):
    """Verifies KPI analytics calculation."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "total_detections" in stats
    assert "p0_emergencies" in stats
    assert "p2_advisories" in stats
    assert "fatigue_reduction_pct" in stats
    assert 0.0 <= stats["fatigue_reduction_pct"] <= 100.0


def test_custom_hotspot_classification_industrial(client):
    """Tests real-time inference on a simulated hotspot inside Rourkela Steel Plant."""
    payload = {
        "latitude": 22.209,
        "longitude": 84.861,
        "frp": 75.0,
        "bright_ti4": 378.0,
        "bright_ti5": 298.0,
        "daynight": "N",
    }
    response = client.post("/api/v1/classify", json=payload)
    assert response.status_code == 200
    result = response.json()

    # Spatial matching verification
    assert result["spatial_enrichment"]["inside_industrial"] is True
    assert "steel" in result["spatial_enrichment"]["facility_type"]

    # ML & SHAP verification
    assert result["predicted_class"] in ["ACCIDENTAL_FIRE", "PERSISTENT_BASELINE"]
    assert "confidence_percent" in result
    assert len(result["top_driving_factors"]) > 0
    assert "narrative_rationale" in result
    assert result["inference_time_ms"] < 200.0  # sub-200ms verification


def test_custom_hotspot_classification_open_field(client):
    """Tests real-time inference on an agricultural background hotspot outside industrial bounds."""
    payload = {
        "latitude": 30.5,
        "longitude": 75.8,
        "frp": 12.0,
        "bright_ti4": 315.0,
        "bright_ti5": 298.0,
        "daynight": "D",
    }
    response = client.post("/api/v1/classify", json=payload)
    assert response.status_code == 200
    result = response.json()

    assert result["spatial_enrichment"]["inside_industrial"] is False
    assert result["alert_priority"] == "NON_ALERT"
    assert result["alert_state"] == "TRANSIENT_SUSPICION"


def test_incident_lifecycle_status_update(client):
    """Tests operator dispatch and acknowledgment status progression."""
    # First fetch any incident
    active_resp = client.get("/api/v1/alerts/active?limit=1")
    incidents = active_resp.json()["incidents"]
    if not incidents:
        pytest.skip("No incidents available to test lifecycle status.")

    target_id = incidents[0]["id"]

    # Update to ACKNOWLEDGED
    ack_resp = client.post(
        f"/api/v1/incident/{target_id}/status",
        json={"status": "ACKNOWLEDGED", "operator_notes": "Duty officer verified via thermal camera."},
    )
    assert ack_resp.status_code == 200
    assert ack_resp.json()["incident"]["status"] == "ACKNOWLEDGED"

    # Verify retrieval
    get_resp = client.get(f"/api/v1/incident/{target_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "ACKNOWLEDGED"
    assert "Duty officer" in get_resp.json()["operator_notes"]


def test_telemetry_sync_endpoint(client, processed_corpus):
    """Verifies that POST /api/v1/sync triggers ingestion into database."""
    sync_resp = client.post("/api/v1/sync")
    assert sync_resp.status_code == 200
    data = sync_resp.json()
    assert data["status"] == "success"
    assert "records_synced" in data
    assert "total_in_db" in data


def test_analytics_charts_endpoint(client):
    """Verifies analytical chart distributions endpoint for Chart.js."""
    resp = client.get("/api/v1/analytics/charts")
    assert resp.status_code == 200
    data = resp.json()
    assert "frp_distribution" in data
    assert "facility_types" in data
    assert "alert_priorities" in data
    assert "top_hotspots" in data
    assert len(data["frp_distribution"]["labels"]) == len(data["frp_distribution"]["counts"])


def test_analyst_action_and_audit_logs(client):
    """Verifies human-in-the-loop analyst action recording and audit trail logging."""
    active_resp = client.get("/api/v1/alerts/active?limit=1")
    records = active_resp.json().get("records") or active_resp.json().get("incidents", [])
    if not records:
        pytest.skip("No incidents available to test analyst actions.")

    target_id = records[0]["id"]

    # 1. Execute CONFIRM action
    action_resp = client.post(
        f"/api/v1/incident/{target_id}/action",
        json={
            "action": "CONFIRM",
            "operator_notes": "Tactical operator verified thermal spike via optical satellite pass.",
        },
    )
    assert action_resp.status_code == 200
    res_data = action_resp.json()
    assert res_data["status"] == "success"
    assert res_data["action"] == "CONFIRM"
    assert res_data["incident"]["status"] == "DISPATCHED"

    # 2. Verify Audit Log entry
    logs_resp = client.get("/api/v1/audit/logs")
    assert logs_resp.status_code == 200
    logs = logs_resp.json()
    assert isinstance(logs, list)
    assert len(logs) > 0
    assert logs[0]["action"] == "CONFIRM"
    assert "Tactical operator" in logs[0]["operator_notes"]


def test_weather_and_dispersion_cone(client):
    """Verifies meteorological synthesis and Gaussian smoke plume dispersion polygon."""
    resp = client.get("/api/v1/weather/context?lat=22.42&lon=69.85&frp=85.0")
    assert resp.status_code == 200
    data = resp.json()
    assert "temperature_c" in data
    assert "wind" in data
    assert "speed_kmh" in data["wind"]
    assert "smoke_dispersion_cone" in data
    assert data["smoke_dispersion_cone"]["type"] == "Polygon"
    assert len(data["smoke_dispersion_cone"]["coordinates"][0]) >= 4
    assert data["smoke_dispersion_cone"]["hazard_length_km"] > 0.0



def _first_incident_id(client):
    alerts = client.get("/api/v1/alerts/active?limit=1").json()
    items = alerts if isinstance(alerts, list) else alerts.get("alerts", alerts.get("incidents", []))
    return items[0]["id"] if items else None


def test_optical_validation_endpoint_without_credentials(client, monkeypatch):
    """Unconfigured, the endpoint must explain itself rather than error out."""
    from src.ingestion import sentinel2_client as s2

    incident_id = _first_incident_id(client)
    if incident_id is None:
        pytest.skip("No incidents in database to validate")

    # Force the unconfigured path regardless of what is in the developer's .env.
    monkeypatch.setattr(s2.Sentinel2Client, "is_configured", property(lambda self: False))

    body = client.get(f"/api/v1/incident/{incident_id}/optical-validation").json()

    assert body["status"] == "NO_CREDENTIALS"
    assert body["dnbr"] is None
    assert body["severity"] == "UNKNOWN"
    assert "dataspace.copernicus.eu" in body["detail"]


def test_optical_validation_endpoint_passes_through_result(client, monkeypatch):
    """A successful validation reaches the caller with its interpretation."""
    from src.ingestion import sentinel2_client as s2

    incident_id = _first_incident_id(client)
    if incident_id is None:
        pytest.skip("No incidents in database to validate")

    monkeypatch.setattr(s2.Sentinel2Client, "is_configured", property(lambda self: True))
    monkeypatch.setattr(
        s2.Sentinel2Client,
        "compute_dnbr",
        lambda self, lat, lon, event_date, **kw: {
            "lat": lat, "lon": lon, "event_date": event_date,
            "dnbr": 0.02, "nbr_pre": 0.30, "nbr_post": 0.28,
            "severity": "UNBURNED", "status": "OK", "cached": False,
        },
    )

    body = client.get(f"/api/v1/incident/{incident_id}/optical-validation").json()

    assert body["status"] == "OK"
    assert body["dnbr"] == 0.02
    assert body["severity"] == "UNBURNED"
    # A contained industrial fire is the whole point of a near-zero dNBR.
    assert "contained" in body["interpretation"].lower()


def test_optical_validation_never_fabricates_a_zero(client, monkeypatch):
    """Any non-OK status must yield dnbr=None, never a measured-looking 0.0."""
    from src.ingestion import sentinel2_client as s2

    incident_id = _first_incident_id(client)
    if incident_id is None:
        pytest.skip("No incidents in database to validate")

    monkeypatch.setattr(s2.Sentinel2Client, "is_configured", property(lambda self: True))

    for status in ("NO_CLEAR_SCENE", "PARSE_MISMATCH", "NOT_AUTHORIZED", "BAD_REQUEST"):
        monkeypatch.setattr(
            s2.Sentinel2Client,
            "compute_dnbr",
            lambda self, lat, lon, event_date, _s=status, **kw: {
                "lat": lat, "lon": lon, "event_date": event_date,
                "dnbr": None, "nbr_pre": None, "nbr_post": None,
                "severity": "UNKNOWN", "status": _s, "cached": False,
            },
        )

        body = client.get(f"/api/v1/incident/{incident_id}/optical-validation").json()
        assert body["status"] in s2.VALID_STATUSES
        assert body["dnbr"] is None, f"{status} must not produce a numeric dNBR"
        assert body["severity"] == "UNKNOWN"


def test_optical_validation_unknown_incident_returns_404(client):
    assert client.get("/api/v1/incident/999999/optical-validation").status_code == 404


# --------------------------------------------------------------------------
# Corpus-scale map aggregation (GET /api/v1/map/hexes)
# --------------------------------------------------------------------------

def test_map_hexes_returns_bounded_aggregate(client, processed_corpus):
    """The whole point is a bounded payload: cells, never raw detections."""
    response = client.get("/api/v1/map/hexes?resolution=5&limit=200")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"OK", "NO_CELLS"}
    assert len(body["cells"]) <= 200
    if body["status"] == "OK":
        assert body["cells_returned"] <= body["detections_aggregated"]
        # No cell may carry a per-detection payload.
        assert set(body["cells"][0]) >= {"h3", "detections", "top_priority"}
        assert "latitude" not in body["cells"][0]


def test_map_hexes_rejects_a_malformed_bbox(client, processed_corpus):
    assert client.get("/api/v1/map/hexes?bbox=1,2").status_code == 422
    assert client.get("/api/v1/map/hexes?bbox=a,b,c,d").status_code == 422


def test_map_hexes_rejects_a_resolution_the_index_cannot_support(client):
    """Resolution 12 would need re-indexing raw coordinates, not a roll-up."""
    assert client.get("/api/v1/map/hexes?resolution=12").status_code == 422


def test_map_hexes_is_cached_between_identical_requests(client, processed_corpus):
    """Re-reading 2M rows on every map pan makes aggregation slower than not."""
    first = client.get("/api/v1/map/hexes?resolution=4&limit=50").json()
    second = client.get("/api/v1/map/hexes?resolution=4&limit=50").json()
    assert first == second


# --------------------------------------------------------------------------
# SLSTR thermal probe (GET /api/v1/incident/{id}/thermal-probe)
# --------------------------------------------------------------------------

def test_thermal_probe_unknown_incident_returns_404(client):
    assert client.get("/api/v1/incident/999999/thermal-probe").status_code == 404


def test_thermal_probe_never_returns_a_temperature_without_a_solve(
        client, monkeypatch, seeded_incidents):
    """A refused retrieval must stay refused all the way out of the API."""
    from app import main as api

    monkeypatch.setattr(
        api.SlstrClient, "retrieve_temperature",
        lambda self, *a, **k: {"status": "NO_SOLUTION", "f1_max_k": 292.4,
                               "background_k": 291.2, "background_basis": "MEASURED_ANNULUS"},
    )
    probe = f"/api/v1/incident/{seeded_incidents[0]['id']}/thermal-probe"
    body = client.get(probe).json()

    assert body["status"] == "NO_SOLUTION"
    assert "fire_temperature_k" not in body
    assert "not" in body["interpretation"].lower()


def test_thermal_probe_attaches_the_caveat_to_any_temperature(
        client, monkeypatch, seeded_incidents):
    """The retrieval returned 466K for a solar park. Nothing ships uncaveated."""
    from app import main as api

    monkeypatch.setattr(
        api.SlstrClient, "retrieve_temperature",
        lambda self, *a, **k: {"status": "OK", "fire_temperature_k": 473.3,
                               "background_basis": "MEASURED_ANNULUS"},
    )
    probe = f"/api/v1/incident/{seeded_incidents[0]['id']}/thermal-probe"
    body = client.get(probe).json()

    assert body["fire_temperature_k"] == 473.3
    assert "466" in body["interpretation"]
    assert body["background_basis"] == "MEASURED_ANNULUS"


# --------------------------------------------------------------------------
# Per-feature attribution must be per-feature (GET /api/v1/incident/{id})
# --------------------------------------------------------------------------

def test_incident_detail_exposes_parsed_shap_factors(client, seeded_incidents):
    """The dashboard drew three fixed bars for every incident. Never again."""
    body = client.get(f"/api/v1/incident/{seeded_incidents[0]['id']}").json()
    assert "shap_factors" in body and "shap_basis" in body
    assert isinstance(body["shap_factors"], list)

    if body["shap_basis"] == "TREESHAP_STORED":
        assert body["shap_factors"], "TREESHAP_STORED must carry factors"
        for f in body["shap_factors"]:
            assert {"feature", "label", "shap_value", "direction"} <= set(f)
            assert isinstance(f["shap_value"], (int, float))
    else:
        assert body["shap_factors"] == [], (
            "no attribution must mean an empty list, so the UI renders nothing "
            "rather than falling back to a placeholder"
        )


def test_shap_factors_differ_between_incidents(client, seeded_incidents):
    """Identical attribution on every target is a fabricated measurement.

    This is the regression test for the defect: the old dossier showed +0.72,
    +1.14 and -0.34 on every incident, hardcoded in the markup.
    """
    listing = client.get("/api/v1/alerts/active?limit=60").json()["incidents"]
    seen = set()
    for inc in listing[:25]:
        body = client.get(f"/api/v1/incident/{inc['id']}").json()
        if body["shap_factors"]:
            seen.add(tuple((f["feature"], f["shap_value"]) for f in body["shap_factors"]))
    if len(seen) < 2:
        pytest.skip(
            "Fewer than two incidents carry stored attribution; nothing to "
            "compare. This guards against identical SHAP on every incident, "
            "which needs at least two to be observable."
        )
    assert len(seen) > 1, "every incident returned the same attribution"


# --------------------------------------------------------------------------
# Recurrence evidence must come from the cell, not from a constant
# --------------------------------------------------------------------------

def test_incident_detail_exposes_real_recurrence(client, seeded_incidents):
    """The dossier printed `frp / 8.0` as "x baseline" -- a constant divided
    into the current reading, which knows nothing about what the cell does."""
    body = client.get(f"/api/v1/incident/{seeded_incidents[0]['id']}").json()

    rec = body["recurrence"]
    assert rec["status"] in {"OK", "NO_BASELINE"}
    if rec["status"] == "OK":
        assert {"detections_30d", "mean_frp_mw", "sigma_frp_mw",
                "persistence_threshold", "is_established"} <= set(rec)
        assert rec["is_established"] == (rec["detections_30d"] >= rec["persistence_threshold"])


def test_recurrence_is_not_a_function_of_frp_alone(client, seeded_incidents):
    """Two detections with similar FRP in different cells must not report the
    same baseline. That was exactly the defect of dividing FRP by a constant."""
    listing = client.get("/api/v1/alerts/active?limit=120").json()["incidents"]
    seen = {}
    for inc in listing[:40]:
        rec = client.get(f"/api/v1/incident/{inc['id']}").json()["recurrence"]
        if rec["status"] == "OK":
            seen[inc["id"]] = (rec["mean_frp_mw"], rec["detections_30d"])
    if len(set(seen.values())) < 2:
        pytest.skip(
            "Fewer than two incidents report an established baseline; nothing "
            "to compare."
        )
    assert len(set(seen.values())) > 1, "every cell reported an identical baseline"


# --------------------------------------------------------------------------
# Automatic refresh must be honest about whether it ran
# --------------------------------------------------------------------------

def test_refresh_status_is_reported(client):
    body = client.get("/api/v1/system/refresh").json()
    assert {"enabled", "last_status", "last_attempt_utc", "last_success_utc"} <= set(body)
    # Under pytest the scheduler must be off, or the suite starts depending on
    # NASA's uptime -- the same trap the weather provider set earlier.
    assert body["enabled"] is False
    assert body["last_status"] in {"DISABLED", "NEVER_RUN"}


def test_refresh_never_advances_success_on_failure(monkeypatch):
    """A failed pull must leave last_success_utc alone.

    The gap between last_attempt and last_success is the only honest measure of
    how long ingestion has been broken; advancing both on failure hides it.
    """
    from src.pipeline import auto_refresh

    before = auto_refresh.status()["last_success_utc"]
    monkeypatch.setattr(auto_refresh, "fetch_firms_window", lambda *a, **k: None)

    result = auto_refresh.refresh_once()

    assert result["status"] == "PULL_EMPTY"
    assert auto_refresh.status()["last_success_utc"] == before
    assert auto_refresh.status()["last_attempt_utc"] is not None


def test_refresh_survives_an_unreachable_firms(monkeypatch):
    """The scheduler must not die, and must not raise into the request."""
    from src.pipeline import auto_refresh

    def boom(*a, **k):
        raise ConnectionError("name resolution failed")

    monkeypatch.setattr(auto_refresh, "fetch_firms_window", boom)
    result = auto_refresh.refresh_once()

    assert result["status"] == "FAILED"
    assert "name resolution failed" in result["detail"]
    assert auto_refresh.status()["running"] is False


def test_refresh_is_disabled_under_pytest(monkeypatch):
    from src.pipeline import auto_refresh

    monkeypatch.setenv("AUTO_REFRESH_ENABLED", "true")
    assert auto_refresh.refresh_enabled() is False, (
        "PYTEST_CURRENT_TEST must override the flag"
    )


def test_refresh_interval_has_a_floor(monkeypatch):
    """FIRMS NRT publishes ~3 h behind the overpass; polling every minute just
    spends the map key's quota re-downloading what we already hold."""
    from src.pipeline import auto_refresh

    monkeypatch.setenv("AUTO_REFRESH_INTERVAL_MIN", "1")
    assert auto_refresh.refresh_interval_minutes() >= auto_refresh.MIN_INTERVAL_MINUTES
    monkeypatch.setenv("AUTO_REFRESH_INTERVAL_MIN", "not-a-number")
    assert auto_refresh.refresh_interval_minutes() == auto_refresh.DEFAULT_INTERVAL_MINUTES


def test_refresh_refuses_to_replace_an_archive_with_a_live_window(tmp_path):
    """A live NRT window must never overwrite a multi-month corpus.

    This shipped: the scheduled refresh wrote to PROCESSED_CORPUS, a deployment
    points that at the 12-month archive so the corpus-scale endpoints see all
    2,044,295 detections, and every three hours the refresh silently replaced it
    with ~1,100 rows. The only symptom was the archive layer reporting a
    thousand detections instead of two million.
    """
    import pandas as pd
    from src.pipeline.auto_refresh import ARCHIVE_ROW_FLOOR, would_shrink_corpus

    archive = tmp_path / "archive.parquet"
    pd.DataFrame({"latitude": [1.0] * (ARCHIVE_ROW_FLOOR + 1)}).to_parquet(archive)

    reason = would_shrink_corpus(archive, incoming_rows=1_100)
    assert reason and "Refusing to replace an archive" in reason


def test_guard_allows_a_normal_live_refresh(tmp_path):
    """It must not fire in the default configuration, where the live file
    legitimately IS the corpus -- a guard that refuses every safe write is
    worse than no guard, because it disables the feature silently."""
    import pandas as pd
    from src.pipeline.auto_refresh import would_shrink_corpus

    live = tmp_path / "live.parquet"
    pd.DataFrame({"latitude": [1.0] * 1_200}).to_parquet(live)
    assert would_shrink_corpus(live, incoming_rows=1_100) is None

    missing = tmp_path / "not-there.parquet"
    assert would_shrink_corpus(missing, incoming_rows=1_100) is None
