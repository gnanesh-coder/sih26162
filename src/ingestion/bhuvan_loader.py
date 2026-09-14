"""ISRO Bhuvan Thematic Ingestion & Spatial Fusion Module.

Processes ISRO Bhuvan Land Use / Land Cover (LULC 50K / 250K) vector shapefiles,
extracts Class 1.2 (Industrial) and Class 1.3 (Mining/Industrial Waste),
and merges them with OpenStreetMap polygons into a consolidated reference layer.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd

logger = logging.getLogger("bhuvan_loader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BHUVAN_DIR = Path("data/reference/bhuvan")
OSM_PARQUET = Path("data/reference/osm_india_industrial_from_pbf.parquet")
MERGED_OUTPUT = Path("data/reference/industrial_boundaries_merged.parquet")


def load_bhuvan_shapefiles(bhuvan_dir: Path = BHUVAN_DIR) -> gpd.GeoDataFrame:
    """Scans data/reference/bhuvan/ for .zip, .shp, .geojson, or .gpkg files and filters for industrial classes.

    Bhuvan LULC Standard Class Codes:
      - '1.2' / 'Industrial' / 'Commercial/Industrial'
      - '1.3' / 'Mining' / 'Quarry' / 'Industrial Waste'
    """
    if not bhuvan_dir.exists():
        bhuvan_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s. Place your downloaded Bhuvan shapefiles or ZIPs here.", bhuvan_dir)
        return gpd.GeoDataFrame(columns=["name", "facility_type", "geometry"], crs="EPSG:4326")

    # 1. Auto-extract any ZIP files dropped by user
    import zipfile
    for zip_path in list(bhuvan_dir.glob("*.zip")):
        target_dir = bhuvan_dir / zip_path.stem
        if not target_dir.exists():
            logger.info("Extracting Bhuvan ZIP archive %s to %s...", zip_path.name, target_dir)
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(target_dir)
            except Exception as e:
                logger.error("Failed extracting ZIP %s: %s", zip_path, e)

    # 2. Find all GIS vector files recursively (including inside extracted subfolders)
    shp_files = (
        list(bhuvan_dir.rglob("*.shp"))
        + list(bhuvan_dir.rglob("*.geojson"))
        + list(bhuvan_dir.rglob("*.gpkg"))
    )
    if not shp_files:
        logger.warning("No Bhuvan shapefiles, GeoJSON, or GPKG found in %s.", bhuvan_dir)
        return gpd.GeoDataFrame(columns=["name", "facility_type", "geometry"], crs="EPSG:4326")

    logger.info("Found %d Bhuvan GIS files across %s.", len(shp_files), bhuvan_dir)
    gdfs: List[gpd.GeoDataFrame] = []

    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            logger.info("Loaded %s (%d records).", f.name, len(gdf))

            # Search candidate description columns for industrial/mining keywords
            pattern = r"(?i)(?:industr|factory|refin|steel|power|mining|quarry|kiln|smelter|industrial\s*waste)"

            desc_cols = [
                c for c in gdf.columns
                if c != "geometry" and any(k in c.lower() for k in ["lulc", "class", "desc", "level", "category", "name", "type"])
            ]
            if not desc_cols:
                desc_cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == "object"]

            mask = pd.Series(False, index=gdf.index)
            for col in desc_cols:
                col_mask = gdf[col].astype(str).str.contains(pattern, na=False)
                mask = mask | col_mask

            filtered = gdf[mask].copy()
            logger.info("-> Extracted %d industrial/mining polygons (searched columns: %s).", len(filtered), desc_cols)

            if not filtered.empty:
                if filtered.crs != "EPSG:4326":
                    filtered = filtered.to_crs("EPSG:4326")

                # Standardize attributes to align cleanly with reference schema
                if "name" not in filtered.columns:
                    if "DIST_OFF" in filtered.columns and "DESCR_2" in filtered.columns:
                        filtered["name"] = filtered["DIST_OFF"].astype(str) + " - " + filtered["DESCR_2"].astype(str)
                    elif "DIST_OFF" in filtered.columns:
                        filtered["name"] = filtered["DIST_OFF"].astype(str) + " Industrial Zone"
                    else:
                        filtered["name"] = "Bhuvan Industrial/Mining Site"

                if "facility_type" not in filtered.columns:
                    if "DESCR_2" in filtered.columns:
                        filtered["facility_type"] = filtered["DESCR_2"].astype(str).str.lower()
                    elif "DESCR_1" in filtered.columns:
                        filtered["facility_type"] = filtered["DESCR_1"].astype(str).str.lower()
                    else:
                        filtered["facility_type"] = "industrial"

                filtered["source"] = "BHUVAN"
                gdfs.append(filtered)
        except Exception as e:
            logger.error("Error reading %s: %s", f, e)

    if not gdfs:
        return gpd.GeoDataFrame(columns=["name", "facility_type", "geometry"], crs="EPSG:4326")

    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    return combined


def create_unified_reference_layer(
    osm_path: Path = OSM_PARQUET,
    bhuvan_dir: Path = BHUVAN_DIR,
    output_path: Path = MERGED_OUTPUT,
) -> gpd.GeoDataFrame:
    """Fuses OSM and Bhuvan polygons into a single high-performance reference layer."""
    records = []

    # 1. Load OSM Polygons
    if osm_path.exists():
        logger.info("Loading OSM reference layer from %s...", osm_path)
        osm_gdf = gpd.read_parquet(osm_path)
        osm_gdf["source"] = "OSM"
        records.append(osm_gdf)
        logger.info("-> Loaded %d OSM polygons.", len(osm_gdf))
    else:
        logger.warning("OSM file %s not found.", osm_path)

    # 2. Load Bhuvan Polygons
    bhuvan_gdf = load_bhuvan_shapefiles(bhuvan_dir)
    if not bhuvan_gdf.empty:
        records.append(bhuvan_gdf)
        logger.info("-> Loaded %d Bhuvan polygons.", len(bhuvan_gdf))

    if not records:
        logger.error("No reference polygons available from either source.")
        return gpd.GeoDataFrame(crs="EPSG:4326")

    unified = gpd.GeoDataFrame(pd.concat(records, ignore_index=True), crs="EPSG:4326")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean object columns for safe Parquet export
    for c in unified.columns:
        if c != "geometry" and unified[c].dtype == "object":
            unified[c] = unified[c].astype(str)

    unified.to_parquet(output_path, index=False)
    logger.info("Unified reference layer saved to %s (%d total polygons).", output_path, len(unified))
    return unified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse Bhuvan and OSM industrial reference layers")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Generate unified reference layer (OSM + Bhuvan)",
    )
    args = parser.parse_args()
    create_unified_reference_layer()
