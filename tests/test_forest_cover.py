"""Tests for forest land-cover determination and the FOREST_FIRE served class.

The problem statement requires industrial fires be segregated from forest fires.
These tests guard the three ways that could go wrong:

  * it could fail closed and never fire (the layer missing, silently)
  * it could fire too widely and eat the crop-burn class, which is the larger
    and better-evidenced of the two
  * it could fire ahead of the guards that withhold unearned claims, asserting
    Indian land cover on another continent
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import forest_cover as fc
from src.models.train_classifier import apply_serving_guards


# A square of "forest" in the Western Ghats, and one detection inside it.
GHATS_LAT, GHATS_LON = 11.67, 76.45          # Bandipur / Nilgiris
PUNJAB_LAT, PUNJAB_LON = 30.50, 75.80        # the verified crop-burn window
TEXAS_LAT, TEXAS_LON = 29.732, -95.092       # ITC Deer Park


@pytest.fixture
def forest_layer(tmp_path, monkeypatch):
    """Writes a one-polygon forest layer and points the module at it."""
    import geopandas as gpd
    from shapely.geometry import box

    path = tmp_path / "forest.parquet"
    gpd.GeoDataFrame(
        {"name": ["Test Reserve Forest"]},
        geometry=[box(GHATS_LON - 0.05, GHATS_LAT - 0.05,
                      GHATS_LON + 0.05, GHATS_LAT + 0.05)],
        crs="EPSG:4326",
    ).to_parquet(path)

    monkeypatch.setattr(fc, "DEFAULT_FOREST_PARQUET", path)
    fc.load_forest_cover(path, force=True)
    yield path
    fc.load_forest_cover(force=True)  # reset the module cache for other tests


@pytest.fixture
def no_forest_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "DEFAULT_FOREST_PARQUET", tmp_path / "absent.parquet")
    fc.load_forest_cover(tmp_path / "absent.parquet", force=True)
    yield
    fc.load_forest_cover(force=True)


class TestDegradation:
    """A missing layer must disable the determination, not break classification."""

    def test_missing_layer_returns_none_not_error(self, no_forest_layer):
        assert fc.load_forest_cover() is None

    def test_missing_layer_reports_not_in_forest(self, no_forest_layer):
        assert fc.in_forest(GHATS_LAT, GHATS_LON) is False

    def test_missing_layer_leaves_classification_untouched(self, no_forest_layer):
        served, note = fc.apply_forest_cover("AGRICULTURAL_BURN", GHATS_LAT, GHATS_LON)
        assert served == "AGRICULTURAL_BURN"
        assert note is None

    def test_garbage_coordinates_do_not_raise(self, forest_layer):
        assert fc.in_forest(None, None) is False
        assert fc.in_forest("north", "east") is False


class TestContainment:
    def test_point_inside_forest_is_detected(self, forest_layer):
        assert fc.in_forest(GHATS_LAT, GHATS_LON) is True

    def test_point_outside_forest_is_not(self, forest_layer):
        assert fc.in_forest(PUNJAB_LAT, PUNJAB_LON) is False

    def test_lat_lon_are_not_transposed(self, forest_layer):
        """The classic geospatial bug: shapely takes (x, y) = (lon, lat)."""
        assert fc.in_forest(GHATS_LAT, GHATS_LON) is True
        assert fc.in_forest(GHATS_LON, GHATS_LAT) is False


class TestDetermination:
    def test_open_ground_burn_in_forest_becomes_forest_fire(self, forest_layer):
        served, note = fc.apply_forest_cover("AGRICULTURAL_BURN", GHATS_LAT, GHATS_LON)
        assert served == "FOREST_FIRE"
        assert note and "forest cover" in note

    def test_open_ground_burn_outside_forest_is_unchanged(self, forest_layer):
        served, note = fc.apply_forest_cover("AGRICULTURAL_BURN", PUNJAB_LAT, PUNJAB_LON)
        assert served == "AGRICULTURAL_BURN"
        assert note is None

    @pytest.mark.parametrize("cls", ["PERSISTENT_BASELINE", "ACCIDENTAL_FIRE",
                                     "TRANSIENT_HOTSPOT"])
    def test_only_open_ground_burns_are_refined(self, forest_layer, cls):
        """A brick kiln in a forest clearing is still a brick kiln.

        Relabelling a persistent source because it sits inside forest cover would
        discard the recurrence evidence that identified it -- and an accidental
        fire relabelled FOREST_FIRE would be downgraded out of dispatch, which is
        the most expensive error this system can make.
        """
        served, note = fc.apply_forest_cover(cls, GHATS_LAT, GHATS_LON)
        assert served == cls
        assert note is None

    def test_an_override_is_never_silent(self, forest_layer):
        _, note = fc.apply_forest_cover("AGRICULTURAL_BURN", GHATS_LAT, GHATS_LON)
        assert note is not None and len(note) > 40


class TestGuardOrdering:
    """Forest must run behind the guards that withhold unearned claims."""

    def test_out_of_domain_burn_is_withheld_before_forest_is_consulted(self, forest_layer):
        """Deer Park is outside the calibrated region; the harvest guard wins.

        If forest ran first it could assert Indian land cover in Texas -- the
        exact error guard 2 exists to prevent.
        """
        served, note = apply_serving_guards("AGRICULTURAL_BURN", TEXAS_LAT, TEXAS_LON)
        assert served == "TRANSIENT_HOTSPOT"
        assert note and "outside" in note

    def test_non_combustion_land_use_still_wins(self, forest_layer):
        served, note = apply_serving_guards(
            "AGRICULTURAL_BURN", GHATS_LAT, GHATS_LON,
            facility_type="renewable_non_thermal",
        )
        assert served == "TRANSIENT_HOTSPOT"
        assert note and "non-combustion" in note

    def test_in_domain_forest_burn_is_served_as_forest_fire(self, forest_layer):
        served, note = apply_serving_guards("AGRICULTURAL_BURN", GHATS_LAT, GHATS_LON)
        assert served == "FOREST_FIRE"
        assert note and "FOREST_FIRE" in note

    def test_punjab_crop_burning_is_not_swallowed(self, forest_layer):
        """The crop-burn class is larger and better evidenced; it must survive."""
        served, note = apply_serving_guards("AGRICULTURAL_BURN", PUNJAB_LAT, PUNJAB_LON)
        assert served == "AGRICULTURAL_BURN"
        assert note is None
