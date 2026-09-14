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
import pandas as pd
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


# Emergency response infrastructure. Unlike the other two layers this is
# mostly POINTS, not polygons: OSM maps a fire station as a node far more often
# than as a building footprint, so both layers have to be read and merged.
#
# Three categories are kept and labelled rather than pooled, because they are
# not interchangeable to an incident commander. A fire station is the response;
# a hospital is where casualties go; police handle cordon and evacuation. The
# SitRep already recommends an evacuation cordon, and recommending one without
# being able to say who enforces it is half an answer.
# Two filters, because the two layers do not share a schema. `multipolygons`
# exposes `amenity` as a real column; `points` does not -- it carries only
# osm_id, name, barrier, highway, ref, address, is_in, place, man_made and
# other_tags, so the amenity tag has to be matched inside the hstore.
#
# The first version used the multipolygons filter for both and GDAL rejected it
# on points as invalid SQL. The extractor logged the refusal and carried on with
# polygons alone, yielding 411 fire stations for the whole of India -- a plainly
# wrong number that would have gone unnoticed if the warning had not been read.
RESPONDER_WHERE_POLY = (
    "amenity IN ('fire_station', 'hospital', 'police') "
    "OR other_tags LIKE '%\"amenity\"=>\"fire_station\"%' "
    "OR other_tags LIKE '%\"emergency\"=>\"fire_station\"%'"
)

RESPONDER_WHERE_POINTS = (
    "other_tags LIKE '%\"amenity\"=>\"fire_station\"%' "
    "OR other_tags LIKE '%\"amenity\"=>\"hospital\"%' "
    "OR other_tags LIKE '%\"amenity\"=>\"police\"%' "
    "OR other_tags LIKE '%\"emergency\"=>\"fire_station\"%'"
)

RESPONDER_KINDS = {
    "fire_station": "FIRE",
    "hospital": "HOSPITAL",
    "police": "POLICE",
}


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


def extract_responders_from_pbf(
    pbf_path: Path,
    output_parquet: Path = Path("data/reference/osm_india_responders_from_pbf.parquet"),
) -> gpd.GeoDataFrame:
    """Extracts emergency response infrastructure as POINTS.

    Reads both the `points` and `multipolygons` layers and merges them: a fire
    station tagged as a node and one tagged as a building footprint are the same
    facility for dispatch purposes, and taking the representative point of the
    polygon puts them in one coordinate space.

    `representative_point()` rather than `centroid`: a centroid can fall outside
    a concave footprint, which would place a fire station in the car park next
    door. The difference is metres and irrelevant to an ETA, but a coordinate
    that is not on the thing it names is wrong for no reason.
    """
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found at: {pbf_path}")

    frames = []
    for layer, where in (
        ("points", RESPONDER_WHERE_POINTS),
        ("multipolygons", RESPONDER_WHERE_POLY),
    ):
        try:
            df = pyogrio.read_dataframe(pbf_path, layer=layer, where=where)
        except Exception as e:
            # Loud, and fatal for this layer. A silently skipped layer produced
            # 411 fire stations for a country of 1.4 billion people.
            raise RuntimeError(
                f"Pushdown filter refused on layer '{layer}': {e}. "
                "Refusing to write a responder layer that is missing a source."
            ) from e
        if len(df) == 0:
            continue
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        if layer == "multipolygons":
            gdf["geometry"] = gdf.geometry.representative_point()
        gdf["source_layer"] = layer
        frames.append(gdf)
        logger.info("Extracted %d responder features from '%s'.", len(gdf), layer)

    if not frames:
        raise RuntimeError("No responder features extracted from either layer.")

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )

    # Normalise the kind. `amenity` is a real column in the points layer but can
    # be absent from multipolygons, where the tag lives inside other_tags.
    other = merged.get("other_tags")
    other = other.fillna("").astype(str) if other is not None else ""
    amenity = merged.get("amenity")
    amenity = amenity.fillna("").astype(str) if amenity is not None else ""

    kind = []
    for a, o in zip(amenity, other):
        matched = RESPONDER_KINDS.get(a)
        if matched is None:
            for tag, label in RESPONDER_KINDS.items():
                if f'"{tag}"' in o:
                    matched = label
                    break
        kind.append(matched or "UNKNOWN")
    merged["responder_kind"] = kind

    # A feature whose category could not be resolved is dropped rather than
    # dispatched to as "UNKNOWN": sending an incident commander to something
    # that might be a hospital is worse than not listing it.
    before = len(merged)
    merged = merged[merged["responder_kind"] != "UNKNOWN"].copy()
    if before != len(merged):
        logger.info("Dropped %d features with an unresolved category.", before - len(merged))

    merged["responder_name"] = (
        merged.get("name").fillna("").astype(str) if "name" in merged.columns else ""
    )
    merged["latitude"] = merged.geometry.y
    merged["longitude"] = merged.geometry.x

    keep = ["responder_kind", "responder_name", "latitude", "longitude",
            "source_layer", "geometry"]
    merged = merged[[c for c in keep if c in merged.columns]]

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    logger.info(
        "Saved %d responders to %s (%s).",
        len(merged), output_parquet,
        merged["responder_kind"].value_counts().to_dict(),
    )
    return merged


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
        choices=["industrial", "forest", "responders"],
        default="industrial",
        help="Which land use to extract",
    )
    parser.add_argument("--output", type=str, default=None, help="Output GeoParquet path")
    args = parser.parse_args()

    pbf_file = Path(args.pbf)
    if args.layer == "responders":
        out = Path(args.output or "data/reference/osm_india_responders_from_pbf.parquet")
        extract_responders_from_pbf(pbf_file, out)
    elif args.layer == "forest":
        out = Path(args.output or "data/reference/osm_india_forest_from_pbf.parquet")
        extract_forest_from_pbf(pbf_file, out)
    else:
        out = Path(args.output or "data/reference/osm_india_industrial_from_pbf.parquet")
        extract_industrial_from_pbf(pbf_file, out)
