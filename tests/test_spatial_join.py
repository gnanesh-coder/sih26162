"""Unit tests for Spatial Join and Facility Matching Engine."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from src.pipeline.spatial_join import classify_facility_type, perform_spatial_join


@pytest.fixture
def sample_polygons():
    """Fixture containing two mock industrial polygons."""
    # Poly 1: Jamnagar Refinery (Box around 22.4, 70.0)
    p1 = Polygon([(69.95, 22.35), (70.05, 22.35), (70.05, 22.45), (69.95, 22.45)])
    # Poly 2: Steel Plant (Box around 23.0, 85.0)
    p2 = Polygon([(84.95, 22.95), (85.05, 22.95), (85.05, 23.05), (84.95, 23.05)])

    gdf = gpd.GeoDataFrame(
        {
            "osm_id": ["way/101", "way/202"],
            "name": ["Reliance Petroleum Refinery", "Integrated Steel Works"],
            "landuse": ["industrial", "industrial"],
            "man_made": [None, "works"],
            "industrial": ["refinery", "steelworks"],
            "other_tags": ['"operator"=>"RIL"', '"operator"=>"SAIL"'],
        },
        geometry=[p1, p2],
        crs="EPSG:4326",
    )
    return gdf


@pytest.fixture
def sample_fire_detections():
    """Fixture with test hotspots representing exact match, buffer match, and distant background."""
    points = [
        Point(70.00, 22.40),  # Pt 0: Exact center of Refinery
        Point(70.052, 22.40), # Pt 1: ~200m outside Refinery (Within 500m buffer)
        Point(71.50, 24.00),  # Pt 2: Far away in agricultural plains (~150 km)
    ]
    gdf = gpd.GeoDataFrame(
        {
            "latitude": [22.40, 22.40, 24.00],
            "longitude": [70.00, 70.052, 71.50],
            "frp": [25.0, 15.0, 5.0],
            "timestamp_utc": [
                pd.Timestamp("2026-09-08 05:00:00", tz="UTC"),
                pd.Timestamp("2026-09-08 05:15:00", tz="UTC"),
                pd.Timestamp("2026-09-08 06:00:00", tz="UTC"),
            ],
        },
        geometry=points,
        crs="EPSG:4326",
    )
    return gdf


def test_perform_spatial_join_matches(sample_fire_detections, sample_polygons):
    """Verify exact match, buffer match, and external detection handling."""
    joined = perform_spatial_join(
        firms_gdf=sample_fire_detections,
        industrial_gdf=sample_polygons,
        buffer_meters=500.0,
        compute_nearest=True,
    )

    assert len(joined) == 3

    # Pt 0: Exact Match
    assert joined.iloc[0]["inside_industrial"] is True or joined.iloc[0]["inside_industrial"] == 1
    assert joined.iloc[0]["is_exact_match"] is True or joined.iloc[0]["is_exact_match"] == 1
    assert joined.iloc[0]["facility_name"] == "Reliance Petroleum Refinery"
    assert joined.iloc[0]["facility_type"] == "petrochemical_refinery"
    assert joined.iloc[0]["dist_to_industrial_km"] == 0.0

    # Pt 1: Buffer Match (~200m outside boundary, caught by 500m buffer)
    assert joined.iloc[1]["inside_industrial"] is True or joined.iloc[1]["inside_industrial"] == 1
    assert joined.iloc[1]["is_exact_match"] is False or joined.iloc[1]["is_exact_match"] == 0
    assert joined.iloc[1]["facility_type"] == "petrochemical_refinery"

    # Pt 2: Distant External Agricultural Hotspot
    assert not joined.iloc[2]["inside_industrial"]
    assert joined.iloc[2]["facility_type"] == "non_industrial"
    assert joined.iloc[2]["dist_to_industrial_km"] > 50.0


def test_classify_facility_type():
    """Verify domain classification logic from OSM tags."""
    row_steel = pd.Series({"inside_industrial": True, "name": "Bhilai Steel Plant", "landuse": "industrial"})
    assert classify_facility_type(row_steel) == "steel_metallurgy"

    row_refinery = pd.Series({"inside_industrial": True, "name": "IOCL Panipat Refinery", "landuse": "industrial"})
    assert classify_facility_type(row_refinery) == "petrochemical_refinery"

    row_kiln = pd.Series({"inside_industrial": True, "name": "Punjab Brick Kiln #4", "man_made": "kiln"})
    assert classify_facility_type(row_kiln) == "brick_kiln"

    row_outside = pd.Series({"inside_industrial": False, "name": "Bhilai Steel Plant"})
    assert classify_facility_type(row_outside) == "non_industrial"


def test_parallax_buffer_recovers_displaced_edge_of_swath_detection(sample_polygons):
    """An off-nadir detection displaced outside the fence must still be associated.

    A plume lofted by an intense fire is projected laterally by the satellite's
    oblique viewing geometry, so the reported pixel lands outside the true
    facility perimeter. The old flat 500m buffer dropped these; the scan-aware
    buffer recovers them.
    """
    # ~1.1km east of the Jamnagar polygon's eastern edge (70.05).
    displaced = gpd.GeoDataFrame(
        {
            "latitude": [22.40],
            "longitude": [70.06],
            "frp": [55.0],
            # Edge-of-swath pixel: 1.6km along-scan footprint, far from nadir.
            "scan": [1.6],
            "track": [0.8],
        },
        geometry=[Point(70.06, 22.40)],
        crs="EPSG:4326",
    )

    tight = perform_spatial_join(displaced, sample_polygons, buffer_meters=200.0, adaptive_buffer=False)
    assert not tight.iloc[0]["inside_industrial"], "200m buffer should not reach the facility"

    wide = perform_spatial_join(displaced, sample_polygons, buffer_meters=1500.0, adaptive_buffer=True)
    assert wide.iloc[0]["inside_industrial"], "scan-aware buffer should recover the displaced pixel"
    # Recovered by proximity, not containment - the distinction must survive so
    # the model can weight a buffer match differently from a true containment.
    assert not wide.iloc[0]["is_exact_match"]


def test_nadir_detection_keeps_tighter_association(sample_polygons):
    """A nadir pixel gets a narrower tolerance than an edge-of-swath pixel."""
    far_field = gpd.GeoDataFrame(
        {
            "latitude": [22.40, 22.40],
            "longitude": [70.070, 70.070],
            "frp": [20.0, 20.0],
            "scan": [0.375, 6.0],   # nadir vs extreme off-nadir
            "track": [0.375, 1.0],
        },
        geometry=[Point(70.070, 22.40), Point(70.070, 22.40)],
        crs="EPSG:4326",
    )

    joined = perform_spatial_join(far_field, sample_polygons, buffer_meters=1500.0, adaptive_buffer=True)

    # Same location, different pixel footprints: the wide-swath pixel earns the
    # larger tolerance, the nadir pixel does not.
    assert not joined.iloc[0]["inside_industrial"]
    assert joined.iloc[1]["inside_industrial"]


# --------------------------------------------------------------------------
# Facility naming: a descriptor is not a name
#
# 78.4% of the 28,587 mapped industrial polygons carry no `name` tag, and only
# 124 of those carry `operator` or `name:en` -- so there is no hidden name to
# recover. Thousands do carry `industrial=`, `power=` or `description=`, and
# surfacing those took identified sites from 21.6% to 39.6%.
#
# facility_name reaches SitReps and one-segment SMS alerts, so the line these
# tests hold is that everything shown comes from a tag OSM actually carries.
# --------------------------------------------------------------------------

def _row(name=None, other_tags=None):
    return pd.DataFrame([{
        "name": name, "other_tags": other_tags, "inside_industrial": True,
    }])


def test_a_real_osm_name_always_wins():
    from src.pipeline.spatial_join import _derive_facility_name
    out = _derive_facility_name(_row(name="Reliance Refinery",
                                     other_tags='"industrial"=>"factory"'))
    assert out.iloc[0] == "Reliance Refinery"


def test_operator_is_used_when_there_is_no_name():
    from src.pipeline.spatial_join import _derive_facility_name
    out = _derive_facility_name(_row(other_tags='"operator"=>"NTPC Limited"'))
    assert out.iloc[0] == "NTPC Limited"


def test_a_burning_plant_is_described_as_fired():
    from src.pipeline.spatial_join import _derive_facility_name
    out = _derive_facility_name(
        _row(other_tags='"power"=>"plant","plant:source"=>"coal"'))
    assert out.iloc[0] == "Coal-fired power plant"


def test_a_solar_plant_is_never_described_as_fired():
    """The regression: plant:source=solar produced "Solar-fired power plant".

    A photovoltaic array burns nothing, and this project excludes exactly these
    sites from combustion reasoning -- describing one as fired would contradict
    `is_non_combustion_site` in the same module.
    """
    from src.pipeline.spatial_join import _derive_facility_name
    out = _derive_facility_name(
        _row(other_tags='"power"=>"plant","plant:source"=>"solar"'))
    assert out.iloc[0] == "Solar power plant"
    assert "fired" not in out.iloc[0].lower()


def test_nothing_known_stays_the_placeholder():
    """An absent name must not become an invented one."""
    from src.pipeline.spatial_join import (
        PLACEHOLDER_FACILITY_NAME, _derive_facility_name)
    assert _derive_facility_name(_row()).iloc[0] == PLACEHOLDER_FACILITY_NAME
    assert _derive_facility_name(
        _row(other_tags='"last_check"=>"2024-01-01"')).iloc[0] == PLACEHOLDER_FACILITY_NAME


def test_malformed_tags_do_not_raise():
    from src.pipeline.spatial_join import (
        PLACEHOLDER_FACILITY_NAME, _derive_facility_name)
    for junk in ("", "not hstore at all", '"unclosed=>'):
        assert _derive_facility_name(_row(other_tags=junk)).iloc[0] == PLACEHOLDER_FACILITY_NAME
