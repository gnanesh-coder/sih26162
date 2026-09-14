"""OSM .osm.pbf Boundary Extractor.

Extracts multipolygons from raw OpenStreetMap Protocolbuffer (.osm.pbf) files using
pyogrio's high-performance GDAL OSM driver with attribute pushdown filtering.

TWO LAYERS, TWO PURPOSES
------------------------
**Industrial** polygons answer "did this burn inside a facility?" and feed the
spatial join, the recurrence key and `inside_industrial`.

**Forest** polygons answer a different question, and one the problem statement
asks directly: it requires industrial fires be "explicitly segregated from forest
fires and natural thermal events". The classifier is deliberately coordinate-free,
so it can describe how something is burning but never where -- a forest fire and a
crop fire look identical to it, because both are sustained open-ground combustion
with no facility history.

Land cover is what separates them, and it is a property of the map rather than of
the model. Hence a second extraction here rather than a fifth model class: the
model classifies combustion *behaviour*, and the map says what is burning.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pyogrio

logger = logging.getLogger("pbf_extractor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Tag filters pushed down to the GDAL C++ driver. Doing this in SQL rather than in
# pandas matters: the India extract is 1.8 GB and an in-memory filter of the whole
# multipolygons layer exhausts memory on an ordinary laptop.
INDUSTRIAL_WHERE = (
    "landuse = 'industrial' OR man_made = 'works' "
    "OR other_tags LIKE '%\"industrial\"%'"
)

# `landuse=forest` is managed forestry; `natural=wood` is unmanaged tree cover.
# Both are forest for the purpose of this project -- the distinction matters to a
# forester and not to a fire. `natural=scrub` is deliberately excluded: scrubland
# abuts agricultural land across most of India and including it would pull crop
# burning into the forest class, which is the exact confusion this is meant to
# resolve.
FOREST_WHERE = (
    "landuse = 'forest' OR natural = 'wood' "
    "OR other_tags LIKE '%\"landuse\"=>\"forest\"%' "
    "OR other_tags LIKE '%\"natural\"=>\"wood\"%'"
)


def _extract_from_pbf(
    pbf_path: Path,
    output_parquet: Path,
    sql_where: str,
    fallback,
    label: str,
) -> gpd.GeoDataFrame:
    """Reads one filtered slice of the multipolygons layer and writes GeoParquet.

    `fallback` is applied only if the pushdown filter is refused by the driver --
    some GDAL builds reject `other_tags LIKE` -- in which case the layer is read
    whole and filtered in pandas. That path is slow and memory-hungry by nature,
    so it warns rather than failing silently.
    """
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found at: {pbf_path}")

    file_size_mb = pbf_path.stat().st_size / (1024 * 1024)
    logger.info("Found PBF file %s (%.1f MB).", pbf_path, file_size_mb)
    logger.info("Extracting 'multipolygons' layer with %s filter...", label)
    logger.info(
        "NOTE: GDAL creates a temporary SQLite node-cache on first pass. "
        "This may take several minutes for a large country file."
    )

    try:
        df = pyogrio.read_dataframe(pbf_path, layer="multipolygons", where=sql_where)
    except Exception as e:
        logger.warning(
            "Pushdown filter error (%s). Falling back to reading the whole "
            "multipolygons layer and filtering in memory...", e
        )
        df = pyogrio.read_dataframe(pbf_path, layer="multipolygons")
        df = df[fallback(df)].copy()

    logger.info("Extracted %d %s polygon features from PBF!", len(df), label)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Cast object columns to strings for safe Parquet serialization
    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == "object":
            gdf[col] = gdf[col].astype(str)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_parquet, index=False)
    logger.info("Saved %s polygons to: %s", label, output_parquet)

    return gdf


def extract_industrial_from_pbf(
    pbf_path: Path,
    output_parquet: Path = Path("data/reference/osm_india_industrial_from_pbf.parquet"),
) -> gpd.GeoDataFrame:
    """Extracts industrial polygons from a .osm.pbf file."""
    def _fallback(df):
        return (
            (df.get("landuse") == "industrial")
            | (df.get("man_made") == "works")
            | (df.get("other_tags").fillna("").astype(str)
               .str.contains('"industrial"', regex=False))
        )

    return _extract_from_pbf(
        pbf_path, output_parquet, INDUSTRIAL_WHERE, _fallback, "industrial"
    )


def extract_forest_from_pbf(
    pbf_path: Path,
    output_parquet: Path = Path("data/reference/osm_india_forest_from_pbf.parquet"),
) -> gpd.GeoDataFrame:
    """Extracts forest and natural-wood polygons from a .osm.pbf file.

    These are the land-cover layer behind `FOREST_FIRE`. See the module docstring
    for why this is a map question rather than a model one.
    """
    def _fallback(df):
        return (
            (df.get("landuse") == "forest")
            | (df.get("natural") == "wood")
            | (df.get("other_tags").fillna("").astype(str)
               .str.contains('"forest"', regex=False))
        )

    return _extract_from_pbf(
        pbf_path, output_parquet, FOREST_WHERE, _fallback, "forest"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract polygons from .osm.pbf")
    parser.add_argument(
        "--pbf",
        type=str,
        default="data/reference/india-260907-internal.osm.pbf",
        help="Path to .osm.pbf file",
    )
    parser.add_argument(
        "--layer",
        choices=["industrial", "forest"],
        default="industrial",
        help="Which land use to extract",
    )
    parser.add_argument("--output", type=str, default=None, help="Output GeoParquet path")
    args = parser.parse_args()

    pbf_file = Path(args.pbf)
    if args.layer == "forest":
        out = Path(args.output or "data/reference/osm_india_forest_from_pbf.parquet")
        extract_forest_from_pbf(pbf_file, out)
    else:
        out = Path(args.output or "data/reference/osm_india_industrial_from_pbf.parquet")
        extract_industrial_from_pbf(pbf_file, out)
