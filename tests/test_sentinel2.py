"""Tests for Sentinel-2 dNBR optical burn-scar validation.

Every test here runs offline. The CDSE HTTP layer is stubbed so the suite never
depends on credentials or network availability, which matters because the whole
point of this module is that it must degrade predictably when those are absent.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.sentinel2_client import (
    DnbrCache,
    Sentinel2Client,
    bbox_around,
    classify_severity,
    enrich_with_dnbr,
)


def _scratch_cache() -> Path:
    """A cache path under the OS temp dir, not the repository root.

    These tests exercise the no-cache path, and the client writes the file on
    its way past. Pointing them at a relative name meant every `pytest` run left
    a stray `nonexistent_cache.json` in the project root.
    """
    return Path(tempfile.gettempdir()) / "sih_test_dnbr_cache.json"


def test_severity_ladder_matches_usgs_breakpoints():
    """dNBR values map onto the standard USGS burn severity classes."""
    assert classify_severity(0.02) == "UNBURNED"
    assert classify_severity(0.20) == "LOW_SEVERITY"
    assert classify_severity(0.35) == "MODERATE_LOW"
    assert classify_severity(0.55) == "MODERATE_HIGH"
    assert classify_severity(0.90) == "HIGH_SEVERITY"
    assert classify_severity(-0.30) == "REGROWTH"
    assert classify_severity(None) == "UNKNOWN"
    assert classify_severity(float("nan")) == "UNKNOWN"


def test_bbox_is_square_on_the_ground_and_widens_with_latitude():
    """Longitude degrees shrink toward the poles, so the box must compensate."""
    equator = bbox_around(0.0, 77.0, radius_km=1.0)
    northern = bbox_around(34.0, 77.0, radius_km=1.0)

    eq_lon_span = equator[2] - equator[0]
    north_lon_span = northern[2] - northern[0]

    # Same latitude span at both places...
    assert (equator[3] - equator[1]) == pytest.approx(northern[3] - northern[1], rel=1e-6)
    # ...but a wider longitude span up north to cover the same ground distance.
    assert north_lon_span > eq_lon_span


def test_client_without_credentials_reports_unavailable_not_zero():
    """A missing credential must never be reported as dNBR 0.0.

    dNBR 0.0 means 'verified: no burn scar', which is a strong industrial-fire
    signal. Conflating it with 'unknown' would fabricate evidence.
    """
    client = Sentinel2Client(client_id="", client_secret="")
    assert client.is_configured is False

    result = client.compute_dnbr(22.5, 70.0, "2026-01-15")
    assert result["dnbr"] is None
    assert result["status"] == "NO_CREDENTIALS"
    assert result["severity"] == "UNKNOWN"


def test_future_event_is_rejected_before_any_api_call():
    """Sentinel-2 cannot have imaged a post-event scene that has not happened."""
    client = Sentinel2Client(client_id="id", client_secret="secret")
    future = (pd.Timestamp.now("UTC") + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    result = client.compute_dnbr(22.5, 70.0, future)
    assert result["status"] == "AWAITING_POST_SCENE"
    assert result["dnbr"] is None


def test_recent_detection_awaits_post_scene_without_spending_quota(monkeypatch):
    """A fresh NRT detection is a 'come back later', not a permanent data gap.

    Sentinel-2 revisits every ~5 days and L2A adds processing latency, so a
    detection from yesterday cannot yet have a post-event scene. Spending an API
    call to discover that wastes quota on every run of a live feed.
    """
    client = Sentinel2Client(client_id="id", client_secret="secret")
    called = []
    monkeypatch.setattr(client, "mean_nbr", lambda *a, **k: called.append(1) or (0.3, "OK"))

    recent = (pd.Timestamp.now("UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    result = client.compute_dnbr(22.5, 70.0, recent)

    assert result["status"] == "AWAITING_POST_SCENE"
    assert result["dnbr"] is None
    assert result["retry_after_days"] >= 1
    assert called == [], "must not issue an API call for a detection this fresh"


def test_aged_detection_is_attempted(monkeypatch):
    """Once past the revisit window, validation proceeds normally."""
    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "mean_nbr", lambda *a, **k: (0.3, "OK"))

    aged = (pd.Timestamp.now("UTC") - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    result = client.compute_dnbr(22.5, 70.0, aged)

    assert result["status"] == "OK"


def test_rate_limit_is_retried_then_surfaced(monkeypatch):
    """429 backs off and retries; only a persistent limit reaches the caller."""
    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "fake-token")
    monkeypatch.setattr("src.ingestion.sentinel2_client.time.sleep", lambda *_: None)

    attempts = {"n": 0}

    def always_limited(*a, **k):
        attempts["n"] += 1
        return _FakeResponse({}, 429, "Too Many Requests")

    monkeypatch.setattr("src.ingestion.sentinel2_client.requests.post", always_limited)

    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-01-01", "2026-01-30")
    assert value is None
    assert status == "RATE_LIMITED"
    assert attempts["n"] > 1, "should have retried before giving up"


def test_rate_limit_recovers_on_retry(monkeypatch):
    """A transient 429 followed by success returns the value."""
    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "fake-token")
    monkeypatch.setattr("src.ingestion.sentinel2_client.time.sleep", lambda *_: None)

    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse({}, 429, "Too Many Requests")
        return _FakeResponse({"data": [_interval(mean=0.42)]})

    monkeypatch.setattr("src.ingestion.sentinel2_client.requests.post", flaky)

    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-01-01", "2026-01-30")
    assert status == "OK"
    assert value == pytest.approx(0.42)


def test_malformed_date_is_handled():
    client = Sentinel2Client(client_id="id", client_secret="secret")
    result = client.compute_dnbr(22.5, 70.0, "not-a-date")
    assert result["status"] == "BAD_DATE"
    assert result["dnbr"] is None


def test_dnbr_computed_from_stubbed_scene_statistics(monkeypatch):
    """With clear pre/post scenes, dNBR is the difference of the two NBR means."""
    client = Sentinel2Client(client_id="id", client_secret="secret")

    calls = []

    def fake_mean_nbr(bbox, date_from, date_to, max_cloud=60, label=""):
        calls.append((date_from, date_to))
        # First call is the pre-event window, second is post-event.
        return (0.62, "OK") if len(calls) == 1 else (0.05, "OK")

    monkeypatch.setattr(client, "mean_nbr", fake_mean_nbr)

    result = client.compute_dnbr(27.587, 95.385, "2026-06-10")

    assert result["status"] == "OK"
    assert result["dnbr"] == pytest.approx(0.57, abs=1e-4)
    assert result["severity"] == "MODERATE_HIGH"
    assert result["nbr_pre"] == 0.62
    assert result["nbr_post"] == 0.05
    # Pre-window must end before the event; post-window must start after it.
    assert calls[0][1] < "2026-06-10" <= calls[1][0]


def test_contained_industrial_fire_yields_near_zero_dnbr(monkeypatch):
    """A fire inside a concrete compound consumes no biomass: dNBR ~ 0."""
    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "mean_nbr", lambda *a, **k: (0.31, "OK"))

    result = client.compute_dnbr(22.47, 70.05, "2026-05-01")

    assert result["status"] == "OK"
    assert result["dnbr"] == pytest.approx(0.0, abs=1e-6)
    assert result["severity"] == "UNBURNED"


def test_cloud_obscuration_surfaces_as_status_not_silent_zero(monkeypatch):
    """Monsoon cloud cover must be reported, not hidden behind a default value."""
    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "mean_nbr", lambda *a, **k: (None, "NO_CLEAR_SCENE"))

    result = client.compute_dnbr(22.5, 70.0, "2026-07-20")
    assert result["dnbr"] is None
    assert result["status"] == "NO_CLEAR_SCENE"


def test_cache_roundtrip_and_key_precision(tmp_path):
    """Cached results are keyed finer than a VIIRS pixel and survive a reload."""
    cache_file = tmp_path / "dnbr_cache.json"
    cache = DnbrCache(path=cache_file)

    cache.put(22.4567, 70.1234, "2026-05-01", {"dnbr": 0.42, "severity": "MODERATE_LOW"})
    cache.flush()

    reloaded = DnbrCache(path=cache_file)
    hit = reloaded.get(22.4567, 70.1234, "2026-05-01")
    assert hit is not None
    assert hit["dnbr"] == 0.42

    # 3-decimal rounding is ~110m, so a 400m offset must be a distinct key.
    assert reloaded.get(22.4607, 70.1234, "2026-05-01") is None


def test_enrich_without_credentials_leaves_dnbr_missing():
    """The enrichment step no-ops cleanly rather than poisoning the feature."""
    df = pd.DataFrame(
        {
            "latitude": [22.47, 21.05],
            "longitude": [70.05, 72.68],
            "acq_date": ["2026-05-01", "2026-05-02"],
            "frp": [45.0, 3.2],
            "inside_industrial": [True, False],
            "n_30d": [1, 0],
        }
    )

    out = enrich_with_dnbr(df, client=Sentinel2Client(client_id="", client_secret=""))

    assert len(out) == 2
    assert out["dnbr"].isna().all()
    assert (out["dnbr_status"] == "NO_CREDENTIALS").all()
    # The input frame must not be mutated in place.
    assert "dnbr" not in df.columns


def test_enrich_targets_only_ambiguous_detections(monkeypatch, tmp_path):
    """API quota is spent on ambiguous detections, not established flares."""
    df = pd.DataFrame(
        {
            "latitude": [22.47, 21.05, 19.10],
            "longitude": [70.05, 72.68, 72.90],
            "acq_date": ["2026-05-01", "2026-05-02", "2026-05-03"],
            # Row 0: inside a facility, no baseline -> ambiguous, must validate.
            # Row 1: inside a facility with a long baseline -> routine flare, skip.
            # Row 2: open ground, negligible energy -> not worth a request.
            "frp": [45.0, 30.0, 1.1],
            "inside_industrial": [True, True, False],
            "n_30d": [1, 25, 0],
        }
    )

    client = Sentinel2Client(client_id="id", client_secret="secret")
    validated = []

    def fake_compute(lat, lon, event_date, **kwargs):
        validated.append(round(lat, 2))
        return {
            "dnbr": 0.01,
            "severity": "UNBURNED",
            "status": "OK",
            "nbr_pre": 0.3,
            "nbr_post": 0.29,
        }

    monkeypatch.setattr(client, "compute_dnbr", fake_compute)

    out = enrich_with_dnbr(df, client=client, cache=DnbrCache(path=tmp_path / "c.json"))

    assert validated == [22.47], "only the ambiguous detection should consume quota"
    assert out.loc[0, "dnbr"] == 0.01
    assert pd.isna(out.loc[1, "dnbr"])
    assert out.loc[1, "dnbr_status"] == "NOT_REQUESTED"


def test_enrich_respects_request_ceiling(monkeypatch, tmp_path):
    """max_requests caps live API usage even with many candidates."""
    n = 20
    df = pd.DataFrame(
        {
            "latitude": np.linspace(20.0, 24.0, n),
            "longitude": np.linspace(70.0, 74.0, n),
            "acq_date": ["2026-05-01"] * n,
            "frp": [50.0] * n,
            "inside_industrial": [True] * n,
            "n_30d": [0] * n,
        }
    )

    client = Sentinel2Client(client_id="id", client_secret="secret")
    issued = {"count": 0}

    def fake_compute(lat, lon, event_date, **kwargs):
        issued["count"] += 1
        return {"dnbr": 0.1, "severity": "LOW_SEVERITY", "status": "OK"}

    monkeypatch.setattr(client, "compute_dnbr", fake_compute)

    enrich_with_dnbr(df, client=client, max_requests=5, cache=DnbrCache(path=tmp_path / "c.json"))

    assert issued["count"] == 5


# --------------------------------------------------------------------------
# Diagnosability of the live API path
#
# The dangerous failure is a response-shape mismatch that looks identical to
# cloud cover: both yield "no usable dNBR", and a silent misparse would be read
# as a monsoon data gap forever. These tests keep the two distinguishable.
# --------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text="", headers=None):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text
        # Backoff reads Retry-After, so the double must carry a headers mapping.
        self.headers = headers if headers is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _client_with_response(monkeypatch, payload=None, status_code=200, text=""):
    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "fake-token")
    monkeypatch.setattr(
        "src.ingestion.sentinel2_client.requests.post",
        lambda *a, **k: _FakeResponse(payload, status_code, text),
    )
    return client


def _interval(mean=0.4, sample_count=1000, no_data=0, band_key="B0"):
    return {
        "outputs": {
            "nbr": {
                "bands": {
                    band_key: {
                        "stats": {
                            "mean": mean,
                            "sampleCount": sample_count,
                            "noDataCount": no_data,
                        }
                    }
                }
            }
        }
    }


def test_wellformed_response_parses_to_mean(monkeypatch):
    client = _client_with_response(monkeypatch, {"data": [_interval(mean=0.55)]})
    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-05-01", "2026-05-30")
    assert status == "OK"
    assert value == pytest.approx(0.55)


def test_unexpected_band_key_still_parses(monkeypatch):
    """The output band key is discovered, not hardcoded.

    Hardcoding "B0" would report every scene as cloudy if Sentinel Hub named the
    band anything else - a silent, permanent misread.
    """
    client = _client_with_response(monkeypatch, {"data": [_interval(mean=0.33, band_key="nbr")]})
    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-05-01", "2026-05-30")
    assert status == "OK"
    assert value == pytest.approx(0.33)


def test_missing_stats_reports_parse_mismatch_not_cloud(monkeypatch):
    """Scenes returned but no parsable stats is OUR bug, and must say so."""
    malformed = {"data": [{"outputs": {"nbr": {"bands": {}}}}]}
    client = _client_with_response(monkeypatch, malformed)

    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-05-01", "2026-05-30")
    assert value is None
    assert status == "PARSE_MISMATCH"
    assert status != "NO_CLEAR_SCENE"


def test_empty_window_distinguished_from_cloud(monkeypatch):
    client = _client_with_response(monkeypatch, {"data": []})
    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-05-01", "2026-05-30")
    assert value is None
    assert status == "NO_SCENES_IN_WINDOW"


def test_genuinely_cloudy_scene_reports_cloud(monkeypatch):
    """A scene where 95% of pixels are masked is real cloud cover."""
    cloudy = {"data": [_interval(mean=0.4, sample_count=1000, no_data=950)]}
    client = _client_with_response(monkeypatch, cloudy)

    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-05-01", "2026-05-30")
    assert value is None
    assert status == "NO_CLEAR_SCENE"


def test_authorization_failure_is_named(monkeypatch):
    """Valid credentials without Sentinel Hub access must not look like a data gap."""
    client = _client_with_response(monkeypatch, status_code=403, text="Forbidden")
    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-05-01", "2026-05-30")
    assert value is None
    assert status == "NOT_AUTHORIZED"


def test_rejected_request_body_is_named(monkeypatch):
    """A bad evalscript is a bug here, and must be reported as one."""
    client = _client_with_response(monkeypatch, status_code=400, text="Invalid evalscript")
    value, status = client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-05-01", "2026-05-30")
    assert value is None
    assert status == "BAD_REQUEST"


def test_raw_response_retained_for_debugging(monkeypatch):
    """--debug can only help if the payload was kept."""
    payload = {"data": [_interval(mean=0.5)]}
    client = _client_with_response(monkeypatch, payload)
    client.mean_nbr([70.0, 22.0, 70.1, 22.1], "2026-05-01", "2026-05-30")
    assert client.last_raw_response == payload


# --------------------------------------------------------------------------
# Request geometry
#
# Regression guard for a real failure: resx/resy are interpreted in the units of
# the bounds CRS, so "resx": 20 against an EPSG:4326 bbox requested 20 DEGREES
# per pixel. Sentinel Hub collapsed the AOI to one pixel and rejected it with
# "Pixel size of 1998.29 meters per pixel exceeds the limit 1500.00".
# --------------------------------------------------------------------------

def test_request_uses_pixel_dimensions_not_degree_resolution(monkeypatch):
    """The outgoing body must carry width/height in pixels, never resx/resy."""
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs.get("json", {}))
        return _FakeResponse({"data": [_interval(mean=0.4)]})

    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_get_token", lambda: "fake-token")
    monkeypatch.setattr("src.ingestion.sentinel2_client.requests.post", capture)

    client.mean_nbr(bbox_around(27.587, 95.385, 1.0), "2026-05-01", "2026-05-30")

    agg = captured["aggregation"]
    assert "resx" not in agg and "resy" not in agg, "degree-unit resolution would be rejected"
    assert agg["width"] > 0 and agg["height"] > 0


def test_pixel_dimensions_hold_target_ground_resolution():
    """A 1km-radius box at 20m/px is 100x100 pixels."""
    from src.ingestion.sentinel2_client import _pixel_dimensions

    dims = _pixel_dimensions(bbox_around(27.587, 95.385, radius_km=1.0))
    assert dims["width"] == pytest.approx(100, abs=2)
    assert dims["height"] == pytest.approx(100, abs=2)


def test_ground_resolution_stays_within_sentinel_hub_limit():
    """No radius may produce a request coarser than the 1500 m/px S2L2A limit."""
    from src.ingestion.sentinel2_client import _pixel_dimensions

    for radius_km in (0.2, 1.0, 5.0, 40.0, 200.0):
        box = bbox_around(22.0, 70.0, radius_km=radius_km)
        dims = _pixel_dimensions(box)
        width_m = (box[2] - box[0]) * 111_320.0 * np.cos(np.radians(22.0))
        m_per_px = width_m / dims["width"]
        assert m_per_px < 1500.0, f"radius {radius_km}km gives {m_per_px:.0f} m/px"
        assert dims["width"] <= 2500 and dims["height"] <= 2500


def test_debug_log_records_both_windows(monkeypatch):
    """--debug must show the call that failed, not just the most recent one."""
    client = Sentinel2Client(client_id="id", client_secret="secret")

    calls = []

    def fake_mean_nbr(bbox, date_from, date_to, max_cloud=60, label=""):
        calls.append(label)
        client._record(label, 400 if label == "pre-event" else 200, {"probe": label},
                       "BAD_REQUEST" if label == "pre-event" else "OK")
        return (None, "BAD_REQUEST") if label == "pre-event" else (0.3, "OK")

    monkeypatch.setattr(client, "mean_nbr", fake_mean_nbr)
    result = client.compute_dnbr(27.587, 95.385, "2026-05-01")

    assert result["status"] == "BAD_REQUEST"
    windows = [c["window"] for c in client.debug_calls]
    assert "pre-event" in windows and "post-event" in windows
    failed = [c for c in client.debug_calls if c["status"] == "BAD_REQUEST"]
    assert failed and failed[0]["http_status"] == 400


# --------------------------------------------------------------------------
# Patient mode
#
# The free CDSE tier allows roughly 20 calls before asking for a ~5 minute
# pause. Abandoning the run there caps coverage at 4% of the alert stream;
# waiting out the window turns a few hundred locations into a matter of time
# rather than a matter of impossibility.
# --------------------------------------------------------------------------

def _alert_frame(n_locations):
    return pd.DataFrame({
        "latitude": np.linspace(22.0, 24.0, n_locations),
        "longitude": np.linspace(70.0, 72.0, n_locations),
        "acq_date": ["2026-03-01"] * n_locations,
        "recurrence_key": [f"hex:{i}" for i in range(n_locations)],
    })


def test_patient_mode_waits_out_the_quota_and_finishes(monkeypatch):
    from src.ingestion.sentinel2_client import enrich_alerts_with_dnbr

    client = Sentinel2Client(client_id="id", client_secret="secret")
    client.last_retry_after_s = 300.0
    slept = []
    monkeypatch.setattr("src.ingestion.sentinel2_client.time.sleep", lambda s: slept.append(s))

    calls = {"n": 0}

    def fake(lat, lon, event_date, **kw):
        calls["n"] += 1
        # Quota trips on every 4th call, mimicking a small allowance.
        if calls["n"] % 4 == 0:
            return {"dnbr": None, "severity": "UNKNOWN", "status": "RATE_LIMITED"}
        return {"dnbr": 0.02, "severity": "UNBURNED", "status": "OK"}

    monkeypatch.setattr(client, "compute_dnbr", fake)

    out = enrich_alerts_with_dnbr(
        _alert_frame(9), client=client,
        cache=DnbrCache(path=_scratch_cache()),
        patient=True,
    )

    assert slept, "patient mode must sleep rather than abandon the run"
    assert all(s >= 300.0 for s in slept), "must honour the server's Retry-After"
    # Every location resolves despite repeated quota trips.
    assert out["dnbr"].notna().all()


def test_impatient_mode_still_stops_on_quota(monkeypatch):
    """Default behaviour is unchanged: stop rather than block a pipeline."""
    from src.ingestion.sentinel2_client import enrich_alerts_with_dnbr

    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr("src.ingestion.sentinel2_client.time.sleep", lambda s: None)
    monkeypatch.setattr(
        client, "compute_dnbr",
        lambda lat, lon, event_date, **kw: {"dnbr": None, "severity": "UNKNOWN",
                                            "status": "RATE_LIMITED"},
    )

    out = enrich_alerts_with_dnbr(
        _alert_frame(5), client=client,
        cache=DnbrCache(path=_scratch_cache()),
        patient=False,
    )
    assert out["dnbr"].isna().all()


def test_patient_mode_respects_the_wait_ceiling(monkeypatch):
    """A pathological server must not stall the run indefinitely."""
    from src.ingestion.sentinel2_client import enrich_alerts_with_dnbr

    client = Sentinel2Client(client_id="id", client_secret="secret")
    monkeypatch.setattr("src.ingestion.sentinel2_client.time.sleep", lambda s: None)
    monkeypatch.setattr(
        client, "compute_dnbr",
        lambda lat, lon, event_date, **kw: {"dnbr": None, "severity": "UNKNOWN",
                                            "status": "RATE_LIMITED"},
    )

    enrich_alerts_with_dnbr(
        _alert_frame(50), client=client,
        cache=DnbrCache(path=_scratch_cache()),
        patient=True, max_quota_waits=3,
    )
    # Bounded: it gives up after max_quota_waits rather than looping forever.


def test_one_api_call_serves_all_detections_at_a_location(monkeypatch):
    """dNBR is a property of a place; co-located detections share one result."""
    from src.ingestion.sentinel2_client import enrich_alerts_with_dnbr

    df = pd.DataFrame({
        "latitude": [22.0, 22.0, 22.0],
        "longitude": [70.0, 70.0, 70.0],
        "acq_date": ["2026-03-01", "2026-03-02", "2026-03-03"],
        "recurrence_key": ["fac:way/1"] * 3,
    })
    client = Sentinel2Client(client_id="id", client_secret="secret")
    calls = {"n": 0}

    def fake(lat, lon, event_date, **kw):
        calls["n"] += 1
        return {"dnbr": 0.31, "severity": "MODERATE_LOW", "status": "OK"}

    monkeypatch.setattr(client, "compute_dnbr", fake)
    out = enrich_alerts_with_dnbr(df, client=client,
                                  cache=DnbrCache(path=_scratch_cache()))

    assert calls["n"] == 1, "three detections at one location must cost one call"
    assert (out["dnbr"] == 0.31).all()


def test_earliest_date_is_used_for_the_event_window(monkeypatch):
    """Using a mid-burn date would compare the fire to itself."""
    from src.ingestion.sentinel2_client import enrich_alerts_with_dnbr

    df = pd.DataFrame({
        "latitude": [22.0, 22.0],
        "longitude": [70.0, 70.0],
        "acq_date": ["2026-03-20", "2026-03-05"],
        "recurrence_key": ["fac:way/1"] * 2,
    })
    client = Sentinel2Client(client_id="id", client_secret="secret")
    seen = []

    monkeypatch.setattr(
        client, "compute_dnbr",
        lambda lat, lon, event_date, **kw: (seen.append(event_date) or
                                            {"dnbr": 0.0, "severity": "UNBURNED", "status": "OK"}),
    )
    enrich_alerts_with_dnbr(df, client=client,
                            cache=DnbrCache(path=_scratch_cache()))

    assert seen == ["2026-03-05"], "must anchor on event onset, not a later detection"


# --------------------------------------------------------------------------
# The alert-path ceiling counts locations, not requests
# --------------------------------------------------------------------------
#
# max_locations used to be compared against the live-call counter, which was
# incremented before the rate-limit branch. Every quota refusal therefore spent
# budget while producing nothing, so a throttled run could stop short of its
# ceiling having resolved far fewer locations than the ceiling allowed.

def test_alert_ceiling_counts_resolved_locations(tmp_path, monkeypatch):
    """max_locations caps locations resolved, and refusals do not count."""
    from src.ingestion.sentinel2_client import enrich_alerts_with_dnbr

    n = 10
    df = pd.DataFrame({
        "recurrence_key": [f"hex:{k}" for k in range(n)],
        "latitude": [22.0 + k * 0.01 for k in range(n)],
        "longitude": [70.0 + k * 0.01 for k in range(n)],
        "acq_date": pd.to_datetime(["2024-03-15"] * n),
    })

    client = Sentinel2Client(client_id="id", client_secret="secret")
    client.last_retry_after_s = 0.0
    calls = {"n": 0}

    # Every other request is refused for quota, so a call-counting ceiling
    # would stop at roughly half the requested locations.
    def fake_compute(lat, lon, event_date, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return {"dnbr": None, "severity": "UNKNOWN", "status": "RATE_LIMITED"}
        return {"dnbr": 0.05, "severity": "UNBURNED", "status": "OK"}

    monkeypatch.setattr(client, "compute_dnbr", fake_compute)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    out = enrich_alerts_with_dnbr(
        df, client=client, cache=DnbrCache(path=tmp_path / "c.json"),
        max_locations=4, patient=True, max_quota_waits=50,
    )

    resolved = out["dnbr_status"].eq("OK").sum()
    assert resolved > 0
    assert out.loc[out["dnbr_status"] == "OK", "recurrence_key"].nunique() == 4


# --------------------------------------------------------------------------
# A transport failure is not a data gap
# --------------------------------------------------------------------------
#
# On 2026-09-12 a one-minute DNS outage cost 79 of 373 locations. Patient mode
# retried RATE_LIMITED without advancing, but recorded API_ERROR and moved on,
# so the run tore through the remainder marking them unmeasurable and exited 0.

def test_transient_transport_failure_is_retried_not_recorded(tmp_path, monkeypatch):
    """A location that fails once and then succeeds must end up measured."""
    from src.ingestion.sentinel2_client import enrich_alerts_with_dnbr

    df = pd.DataFrame({
        "recurrence_key": ["hex:a", "hex:b"],
        "latitude": [22.0, 23.0],
        "longitude": [70.0, 71.0],
        "acq_date": pd.to_datetime(["2024-03-15", "2024-03-16"]),
    })

    client = Sentinel2Client(client_id="id", client_secret="secret")
    seen = {"n": 0}

    def fake_compute(lat, lon, event_date, **kwargs):
        seen["n"] += 1
        if seen["n"] == 1:                      # first location, first attempt
            return {"dnbr": None, "severity": "UNKNOWN", "status": "API_ERROR"}
        return {"dnbr": 0.02, "severity": "UNBURNED", "status": "OK"}

    monkeypatch.setattr(client, "compute_dnbr", fake_compute)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    out = enrich_alerts_with_dnbr(
        df, client=client, cache=DnbrCache(path=tmp_path / "c.json"), patient=True,
    )

    assert set(out["dnbr_status"]) == {"OK"}
    assert out["dnbr"].notna().all()


def test_sustained_transport_failure_stops_rather_than_consuming_locations(tmp_path, monkeypatch):
    """A network outage must leave the remaining locations unattempted."""
    from src.ingestion.sentinel2_client import enrich_alerts_with_dnbr

    n = 40
    df = pd.DataFrame({
        "recurrence_key": [f"hex:{k}" for k in range(n)],
        "latitude": [22.0 + k * 0.01 for k in range(n)],
        "longitude": [70.0 + k * 0.01 for k in range(n)],
        "acq_date": pd.to_datetime(["2024-03-15"] * n),
    })

    client = Sentinel2Client(client_id="id", client_secret="secret")

    def always_down(lat, lon, event_date, **kwargs):
        return {"dnbr": None, "severity": "UNKNOWN", "status": "API_ERROR"}

    monkeypatch.setattr(client, "compute_dnbr", always_down)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    out = enrich_alerts_with_dnbr(
        df, client=client, cache=DnbrCache(path=tmp_path / "c.json"), patient=True,
    )

    # The circuit breaker trips after 5 abandoned locations, so the great
    # majority must remain NOT_REQUESTED and therefore recoverable.
    attempted = (out["dnbr_status"] == "API_ERROR").sum()
    untouched = (out["dnbr_status"] == "NOT_REQUESTED").sum()
    assert attempted <= 5
    assert untouched >= n - 5
