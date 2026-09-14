"""Tests for historical FIRMS archive ingestion.

These exist because the project shipped with a client that could only ever fetch
the most recent day. Everything downstream is temporal -- the recurrence state
machine scores each detection against a 30-day baseline, and Sentinel-2 dNBR
needs detections older than the satellite revisit cycle -- so a single-day
snapshot left `n_30d` near zero, PERSISTENT_BASELINE never firing, and routine
refinery flares indistinguishable from genuine accidents.

All network access is stubbed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import firms_client as fc

SAMPLE_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "22.400,70.050,340.1,0.4,0.4,{date},0512,N,VIIRS,n,2.0NRT,295.0,12.5,D\n"
)


def _response(text="", status=200):
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    """Remove inter-request pacing so tests do not sleep."""
    monkeypatch.setattr(fc.time, "sleep", lambda *_: None)


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------

def test_day_range_cap_is_five():
    """Regression: the API rejects >5 days with 'Invalid day range. Expects [1..5]'.

    The original implementation assumed 10, so every archive window failed with
    HTTP 400 and the whole pull silently produced nothing.
    """
    assert fc.MAX_DAYS_PER_REQUEST == 5


@patch("src.ingestion.firms_client.requests.get")
def test_oversized_day_range_is_clamped_not_sent(mock_get):
    mock_get.return_value = _response(SAMPLE_CSV.format(date="2026-07-15"))

    fc.fetch_firms_nrt(map_key="k", day_range=30, bbox="68,6,97,35")

    url = mock_get.call_args[0][0]
    assert url.rstrip("/").endswith("/5"), f"day range should be clamped to 5, got {url}"


@patch("src.ingestion.firms_client.requests.get")
def test_start_date_is_appended_to_url(mock_get):
    """Without a start date the API only ever returns the present."""
    mock_get.return_value = _response(SAMPLE_CSV.format(date="2026-07-15"))

    fc.fetch_firms_nrt(map_key="k", day_range=5, bbox="68,6,97,35", start_date="2026-07-15")

    assert mock_get.call_args[0][0].endswith("/5/2026-07-15")


@patch("src.ingestion.firms_client.requests.get")
def test_omitting_start_date_leaves_url_unchanged(mock_get):
    mock_get.return_value = _response(SAMPLE_CSV.format(date="2026-09-12"))

    fc.fetch_firms_nrt(map_key="k", day_range=3, bbox="68,6,97,35")

    assert mock_get.call_args[0][0].endswith("/3")


# --------------------------------------------------------------------------
# Failure must not masquerade as "no fires"
# --------------------------------------------------------------------------

@patch("src.ingestion.firms_client.requests.get")
def test_http_failure_reports_request_failed_not_empty(mock_get):
    """An HTTP 400 and a fire-free window both yield zero rows.

    Conflating them is how a systematic request error reads as "there were no
    fires" -- exactly the misdiagnosis that hid the day-range bug.
    """
    mock_get.return_value = _response("Invalid day range. Expects [1..5].", status=400)

    gdf, status = fc.fetch_firms_window(
        satellite="MODIS_NRT", day_range=5, start_date="2026-07-15",
        map_key="k", bbox="68,6,97,35", timeout=10,
    )
    assert gdf.empty
    assert status == "REQUEST_FAILED"


@patch("src.ingestion.firms_client.requests.get")
def test_successful_but_empty_window_reports_empty(mock_get):
    header = SAMPLE_CSV.split("\n")[0]
    mock_get.return_value = _response(header)

    gdf, status = fc.fetch_firms_window(
        satellite="VIIRS_SNPP_NRT", day_range=5, start_date="2026-07-15",
        map_key="k", bbox="68,6,97,35", timeout=10,
    )
    assert gdf.empty
    assert status == "EMPTY"


@patch("src.ingestion.firms_client.requests.get")
def test_window_with_rows_reports_ok(mock_get):
    mock_get.return_value = _response(SAMPLE_CSV.format(date="2026-07-15"))

    gdf, status = fc.fetch_firms_window(
        satellite="VIIRS_SNPP_NRT", day_range=5, start_date="2026-07-15",
        map_key="k", bbox="68,6,97,35", timeout=10,
    )
    assert status == "OK"
    assert len(gdf) == 1


# --------------------------------------------------------------------------
# Paging
# --------------------------------------------------------------------------

@patch("src.ingestion.firms_client.requests.get")
def test_archive_pages_the_full_span(mock_get):
    """A 30-day span must be covered by ceil(30/5) windows per source."""
    mock_get.return_value = _response(SAMPLE_CSV.format(date="2026-08-01"))

    fc.fetch_firms_archive(
        start_date="2026-08-01", end_date="2026-08-30",
        satellites=["VIIRS_SNPP_NRT"], map_key="k", bbox="68,6,97,35",
        auto_source=False,
    )

    assert mock_get.call_count == 6, "30 days / 5-day windows = 6 requests"

    # Windows must tile the span without gaps.
    starts = [c[0][0].rsplit("/", 1)[-1] for c in mock_get.call_args_list]
    assert starts == [
        "2026-08-01", "2026-08-06", "2026-08-11",
        "2026-08-16", "2026-08-21", "2026-08-26",
    ]


@patch("src.ingestion.firms_client.requests.get")
def test_archive_queries_every_source(mock_get):
    mock_get.return_value = _response(SAMPLE_CSV.format(date="2026-08-01"))

    fc.fetch_firms_archive(
        start_date="2026-08-01", end_date="2026-08-05",
        satellites=["VIIRS_SNPP_NRT", "MODIS_NRT"], map_key="k", bbox="68,6,97,35",
        auto_source=False,
    )

    sources = {c[0][0].split("/")[-4] for c in mock_get.call_args_list}
    assert sources == {"VIIRS_SNPP_NRT", "MODIS_NRT"}


@patch("src.ingestion.firms_client.requests.get")
def test_archive_deduplicates_repeated_detections(mock_get):
    """Overlapping windows and multiple sensors can restate the same detection."""
    mock_get.return_value = _response(SAMPLE_CSV.format(date="2026-08-01"))

    out = fc.fetch_firms_archive(
        start_date="2026-08-01", end_date="2026-08-15",
        satellites=["VIIRS_SNPP_NRT"], map_key="k", bbox="68,6,97,35",
        auto_source=False,
    )

    # Three identical windows collapse to one unique detection.
    assert len(out) == 1


@patch("src.ingestion.firms_client.requests.get")
def test_archive_rejects_inverted_date_range(mock_get):
    out = fc.fetch_firms_archive(
        start_date="2026-08-30", end_date="2026-08-01",
        satellites=["VIIRS_SNPP_NRT"], map_key="k", bbox="68,6,97,35",
    )
    assert out.empty
    assert mock_get.call_count == 0, "must not issue requests for an invalid range"


def test_archive_without_key_returns_empty_schema():
    out = fc.fetch_firms_archive(
        start_date="2026-08-01", end_date="2026-08-05",
        map_key="your_nasa_firms_key_here",
    )
    assert out.empty
    assert "latitude" in out.columns


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------

@patch("src.ingestion.firms_client.requests.get")
def test_availability_table_is_parsed(mock_get):
    """Source windows are queried, not assumed.

    Asking for a date outside a source's window returns an empty CSV rather than
    an error, so the availability table is the only reliable way to choose
    between an NRT stream and its standard-processing archive.
    """
    mock_get.return_value = _response(
        "data_id,min_date,max_date\n"
        "VIIRS_SNPP_NRT,2026-04-28,2026-09-12\n"
        "VIIRS_SNPP_SP,2012-01-20,2026-04-27\n"
    )

    df = fc.fetch_data_availability(map_key="k")

    assert list(df.data_id) == ["VIIRS_SNPP_NRT", "VIIRS_SNPP_SP"]
    assert df.loc[df.data_id == "VIIRS_SNPP_SP", "min_date"].item() == "2012-01-20"


def test_every_nrt_source_has_an_archive_counterpart():
    """The NRT->SP mapping must stay complete, or long pulls silently lose a sensor."""
    for nrt, sp in fc.NRT_TO_SP_SOURCE.items():
        assert nrt.endswith("_NRT")
        assert sp.endswith("_SP")
        assert nrt.replace("_NRT", "") == sp.replace("_SP", "")


# --------------------------------------------------------------------------
# Availability-driven source switching
#
# Each sensor is split between a near-real-time stream and a standard-processing
# archive, and the cutover date moves. Querying a source outside its window
# returns an empty CSV rather than an error, so a 12-month pull that assumes one
# variant loses whole months of a sensor and never says so.
# --------------------------------------------------------------------------

AVAIL = {
    "VIIRS_SNPP_NRT": (pd.Timestamp("2026-04-28").date(), pd.Timestamp("2026-09-12").date()),
    "VIIRS_SNPP_SP": (pd.Timestamp("2012-01-20").date(), pd.Timestamp("2026-04-27").date()),
}


def test_recent_window_keeps_the_nrt_source():
    got = fc.resolve_source_for_window("VIIRS_SNPP_NRT", pd.Timestamp("2026-07-01").date(), AVAIL)
    assert got == "VIIRS_SNPP_NRT"


def test_old_window_falls_back_to_the_archive_source():
    """A date before the NRT cutover must switch to standard processing."""
    got = fc.resolve_source_for_window("VIIRS_SNPP_NRT", pd.Timestamp("2025-11-01").date(), AVAIL)
    assert got == "VIIRS_SNPP_SP"


def test_window_covered_by_neither_variant_is_skipped():
    """Before the sensor existed, no source covers it - skip rather than waste a call."""
    got = fc.resolve_source_for_window("VIIRS_SNPP_NRT", pd.Timestamp("2005-01-01").date(), AVAIL)
    assert got is None


def test_missing_availability_table_falls_back_to_requested_source():
    """A failed availability lookup must not block the pull entirely."""
    got = fc.resolve_source_for_window("VIIRS_SNPP_NRT", pd.Timestamp("2026-07-01").date(), {})
    assert got == "VIIRS_SNPP_NRT"


@patch("src.ingestion.firms_client.requests.get")
def test_availability_index_parses_dates(mock_get):
    mock_get.return_value = _response(
        "data_id,min_date,max_date\n"
        "VIIRS_SNPP_NRT,2026-04-28,2026-09-12\n"
        "VIIRS_SNPP_SP,2012-01-20,2026-04-27\n"
    )
    idx = fc.build_availability_index(map_key="k")

    assert idx["VIIRS_SNPP_NRT"][0] == pd.Timestamp("2026-04-28").date()
    assert idx["VIIRS_SNPP_SP"][1] == pd.Timestamp("2026-04-27").date()


@patch("src.ingestion.firms_client.requests.get")
def test_archive_skips_windows_no_source_covers(mock_get):
    """Uncovered windows must not be requested at all."""
    avail_csv = (
        "data_id,min_date,max_date\n"
        "VIIRS_SNPP_NRT,2026-08-10,2026-09-12\n"
        "VIIRS_SNPP_SP,2026-08-01,2026-08-09\n"
    )
    calls = {"n": 0}

    def responder(url, **kw):
        calls["n"] += 1
        if "data_availability" in url:
            return _response(avail_csv)
        return _response(SAMPLE_CSV.format(date="2026-08-11"))

    mock_get.side_effect = responder

    fc.fetch_firms_archive(
        start_date="2025-01-01", end_date="2025-01-10",
        satellites=["VIIRS_SNPP_NRT"], map_key="k", bbox="68,6,97,35",
    )

    # Only the availability lookup should have been issued; 2025 is covered by
    # neither variant, so both windows are skipped.
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# Schema compatibility across NRT and SP sources
#
# Regression: a full-year pull mixes near-real-time and standard-processing
# sources, which disagree on the type of `version` -- NRT emits the string
# "2.0NRT", SP emits a number. Concatenated, that column holds two Python types
# and the entire parquet write fails with "Could not convert '2.0NRT' with type
# str: tried to convert to int64". The 60-day pull never hit it because it was
# all NRT.
# --------------------------------------------------------------------------

SP_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "23.100,71.200,338.4,0.5,0.5,2025-11-02,0430,N,VIIRS,60,2,293.0,8.1,N\n"
)


def test_version_is_normalised_to_string():
    """NRT and SP disagree on this field's type; both must parse as str."""
    nrt = fc.csv_to_gdf(SAMPLE_CSV.format(date="2026-07-15"))
    sp = fc.csv_to_gdf(SP_CSV)

    assert nrt["version"].map(type).eq(str).all()
    assert sp["version"].map(type).eq(str).all()


def test_nrt_and_sp_pages_concatenate_to_a_writable_frame(tmp_path):
    """The mixed-source frame must survive serialization."""
    nrt = fc.csv_to_gdf(SAMPLE_CSV.format(date="2026-07-15"))
    sp = fc.csv_to_gdf(SP_CSV)

    combined = pd.concat(
        [pd.DataFrame(f.drop(columns=["geometry"])) for f in (nrt, sp)],
        ignore_index=True,
    )

    for col in combined.columns:
        if combined[col].dtype == object:
            assert combined[col].map(type).nunique() == 1, (
                f"column {col!r} holds mixed Python types and will break parquet"
            )

    import geopandas as gpd_
    out = gpd_.GeoDataFrame(
        combined,
        geometry=gpd_.points_from_xy(combined["longitude"], combined["latitude"]),
        crs="EPSG:4326",
    )
    target = tmp_path / "mixed.parquet"
    out.to_parquet(target, index=False)
    assert target.exists()
    assert len(pd.read_parquet(target)) == 2


def test_other_string_fields_are_also_normalised():
    """satellite/instrument/daynight vary in form across sources too."""
    g = fc.csv_to_gdf(SP_CSV)
    for col in ("confidence", "satellite", "instrument", "daynight"):
        assert g[col].map(type).eq(str).all(), f"{col} should be normalised to str"
