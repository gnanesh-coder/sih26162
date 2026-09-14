"""Unit tests for NASA FIRMS NRT ingestion module."""

from unittest.mock import MagicMock, patch
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.ingestion.firms_client import (
    create_empty_firms_gdf,
    fetch_firms_nrt,
    parse_firms_timestamps,
)


def test_parse_firms_timestamps():
    """Verify conversion of acq_date and acq_time into UTC timestamps."""
    df = pd.DataFrame(
        {
            "acq_date": ["2026-09-07", "2026-09-07"],
            "acq_time": [530, 1445],  # 530 should pad to 0530
        }
    )
    result = parse_firms_timestamps(df)
    assert len(result) == 2
    assert str(result.iloc[0]) == "2026-09-07 05:30:00+00:00"
    assert str(result.iloc[1]) == "2026-09-07 14:45:00+00:00"


def test_create_empty_firms_gdf():
    """Verify empty GeoDataFrame has expected columns and EPSG:4326 CRS."""
    gdf = create_empty_firms_gdf()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.empty
    assert gdf.crs is None or gdf.crs == "EPSG:4326"
    assert "latitude" in gdf.columns
    assert "longitude" in gdf.columns
    assert "frp" in gdf.columns
    assert "timestamp_utc" in gdf.columns


def test_fetch_firms_placeholder_key():
    """Verify graceful handling when key is missing or placeholder."""
    gdf = fetch_firms_nrt(map_key="your_nasa_firms_key_here")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.empty
    assert gdf.crs is None or gdf.crs == "EPSG:4326"


@patch("src.ingestion.firms_client.requests.get")
def test_fetch_firms_nrt_mock_success(mock_get):
    """Verify parsing and GeoDataFrame construction from valid FIRMS CSV payload."""
    mock_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        "22.5726,88.3639,340.5,0.4,0.4,2026-09-07,0615,N,VIIRS,nominal,2.0NRT,295.1,18.4,D\n"
        "28.6139,77.2090,325.0,0.5,0.4,2026-09-07,0615,N,VIIRS,nominal,2.0NRT,290.0,7.2,D\n"
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = mock_csv
    mock_get.return_value = mock_response

    gdf = fetch_firms_nrt(map_key="mock_valid_key_12345", bbox="68.0,6.5,97.5,35.5")

    assert not gdf.empty
    assert len(gdf) == 2
    assert gdf.crs is None or gdf.crs == "EPSG:4326"
    assert "timestamp_utc" in gdf.columns
    assert isinstance(gdf.geometry.iloc[0], Point)
    assert gdf.geometry.iloc[0].x == 88.3639
    assert gdf.geometry.iloc[0].y == 22.5726
    assert gdf["frp"].iloc[0] == 18.4
