"""Forest land-cover lookup, and the FOREST_FIRE determination built on it.

WHY THIS IS NOT A FIFTH MODEL CLASS
-----------------------------------
The problem statement requires industrial fires be "explicitly segregated from
forest fires and natural thermal events". Until this module, they were not: a
forest fire reached the classifier as sustained open-ground combustion with no
facility history, which is precisely what a crop fire looks like, and it was
served as AGRICULTURAL_BURN.

The obvious fix is a fifth class. It is the wrong one, for the same reason
`apply_serving_guards` exists at all: **the model is coordinate-free on purpose.**
It cannot see where it is, so it can never learn that one sustained open-ground
fire is in the Western Ghats and another is in a Punjab wheat field. The
radiometry is genuinely the same. Adding a class the feature vector cannot
separate would ask the model to guess, and it would guess from whatever spurious
correlation the corpus happened to contain.

What separates them is *land cover*, which is a fact about the map. So the model
keeps classifying combustion behaviour, and this module answers "what was
burning" from the same OpenStreetMap extract the industrial layer comes from.
That split is also what makes the result honest: when the map has no forest
polygon covering a point, the answer is "not established", never "not forest".

THE ASYMMETRY THAT MATTERS
--------------------------
Absence of a forest polygon is not absence of forest -- the same argument this
project already makes about industry and Baghjan. So the determination only ever
*adds* specificity to an open-ground burn; it never removes it. A detection
inside forest cover that the model called AGRICULTURAL_BURN becomes FOREST_FIRE.
A detection outside forest cover is left exactly as the model found it.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("forest_cover")

DEFAULT_FOREST_PARQUET = Path(
    os.getenv("FOREST_COVER", "data/reference/osm_india_forest_from_pbf.parquet")
)

# The class a detection is reported as when it burned on forested land. It is a
# *served* class, not one of CLASS_NAMES: the model never predicts it, and the
# evaluation harness never scores it, because no training label carries it.
FOREST_FIRE_CLASS = "FOREST_FIRE"

# Only an open-ground burn can become a forest fire. A persistent industrial
# baseline inside a forest is a facility in a clearing -- a brick kiln, a mine --
# and relabelling it would undo the recurrence evidence that identified it.
UPGRADEABLE_CLASSES = frozenset({"AGRICULTURAL_BURN"})

_lock = threading.Lock()
_index = None          # cached GeoDataFrame + sindex
_load_attempted = False


class ForestCover:
    """An R-tree over forest polygons, answering point containment.

    Mirrors `spatial_join.py`: geopandas' `sindex` is a GEOS R-tree, so the
    bounding-box candidates come back in microseconds and only those few
    candidates get an exact geometry test.
    """

    def __init__(self, gdf) -> None:
        self._gdf = gdf
        # Touch the index once at construction so the first query is not the one
        # that pays to build it.
        _ = self._gdf.sindex
        self.n_polygons = len(gdf)

    def contains(self, lat: float, lon: float) -> bool:
        """True only when the point falls inside a mapped forest polygon."""
        from shapely.geometry import Point

        try:
            point = Point(float(lon), float(lat))
        except (TypeError, ValueError):
            return False

        candidates = list(self._gdf.sindex.query(point, predicate="intersects"))
        if not candidates:
            return False
        return bool(self._gdf.geometry.iloc[candidates].intersects(point).any())


def load_forest_cover(path: Optional[Path] = None, force: bool = False) -> Optional[ForestCover]:
    """Loads the forest layer once and caches it.

    Returns None when the layer is absent, and says so once rather than on every
    call. A missing layer is a degraded state, not an error: every other part of
    the system keeps working and forest determination is simply withheld.
    """
    global _index, _load_attempted

    with _lock:
        if force:
            _index, _load_attempted = None, False
        if _load_attempted:
            return _index
        _load_attempted = True

        target = Path(path) if path is not None else DEFAULT_FOREST_PARQUET
        if not target.is_absolute():
            target = PROJECT_ROOT / target

        if not target.exists():
            logger.warning(
                "No forest layer at %s. FOREST_FIRE determination is disabled; "
                "open-ground burns will be reported as the model classified them. "
                "Build it with: python -m src.ingestion.pbf_extractor --layer forest",
                target,
            )
            return None

        try:
            import geopandas as gpd

            gdf = gpd.read_parquet(target)
            gdf = gdf[gdf.geometry.notna()]
            if gdf.empty:
                logger.warning("Forest layer at %s is empty.", target)
                return None
            _index = ForestCover(gdf)
            logger.info("Loaded %d forest polygons from %s.", _index.n_polygons, target)
        except Exception as e:  # pragma: no cover - corrupt file path
            logger.warning("Could not load forest layer %s: %s", target, e)
            _index = None

        return _index


def in_forest(lat: float, lon: float) -> bool:
    """Whether this coordinate falls inside mapped forest cover.

    False means "not established", not "not forest" -- OSM forest coverage is
    incomplete, so this is evidence in one direction only.
    """
    cover = load_forest_cover()
    if cover is None:
        return False
    return cover.contains(lat, lon)


def apply_forest_cover(predicted_class: str, lat: float, lon: float):
    """Refines an open-ground burn into FOREST_FIRE when the land cover says so.

    Returns `(served_class, note)`, matching the shape of
    `apply_serving_guards` so the two compose cleanly. `note` is None when
    nothing was changed, so an override is always visible to the caller and
    never silent.
    """
    if predicted_class not in UPGRADEABLE_CLASSES:
        return predicted_class, None

    if not in_forest(lat, lon):
        return predicted_class, None

    return FOREST_FIRE_CLASS, (
        f"Reported as {FOREST_FIRE_CLASS}: {lat:.3f}, {lon:.3f} falls inside mapped "
        "forest cover, so this open-ground burn is a forest fire rather than crop "
        "residue. The classifier is coordinate-free and cannot make this "
        "distinction; the land-cover layer can."
    )
