"""All-India OpenStreetMap (OSM) Ingestion via Geofabrik.

Automates downloading, selective unzipping, and filtering of the
Geofabrik India shapefile extract to build a clean nationwide industrial
polygon reference dataset saved in high-performance GeoParquet.
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

logger = logging.getLogger("geofabrik_osm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

GEOFABRIK_SHP_URL = "https://download.geofabrik.de/asia/india-latest-free.shp.zip"
TARGET_LAYER_PREFIX = "gis_osm_landuse_a_free_1"


def download_geofabrik_zip(dest_path: Path) -> bool:
    """Streams the Geofabrik all-India shapefile package with progress."""
    logger.info("Starting download of Geofabrik India shapefile package (~800 MB)...")
    logger.info("URL: %s", GEOFABRIK_SHP_URL)
    try:
        with requests.get(GEOFABRIK_SHP_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            total_length = int(r.headers.get("content-length", 0))
            downloaded = 0
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_length > 0:
                            pct = (downloaded / total_length) * 100
                            mb = downloaded / (1024 * 1024)
                            sys.stdout.write(f"\rProgress: [{mb:.1f} MB / {total_length / (1024*1024):.1f} MB] ({pct:.1f}%)")
                            sys.stdout.flush()
            sys.stdout.write("\n")
        logger.info("Download completed: %s", dest_path)
        return True
    except Exception as e:
        logger.error("Download failed: %s", e)
        return False


def extract_and_filter_industrial(
    zip_path: Path,
    output_parquet: Path = Path("data/reference/osm_india_industrial_all.parquet"),
) -> gpd.GeoDataFrame:
    """Extracts only the landuse shapefile from the Geofabrik zip, filters for industrial polygons, and exports to GeoParquet."""
    extract_dir = zip_path.parent / "temp_geofabrik_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Inspecting zip archive %s...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if m.startswith(TARGET_LAYER_PREFIX)]
        if not members:
            raise FileNotFoundError(f"Could not find landuse layer '{TARGET_LAYER_PREFIX}' inside zip archive.")
        logger.info("Extracting %d landuse layer components...", len(members))
        for m in members:
            z.extract(m, extract_dir)

    shp_path = extract_dir / f"{TARGET_LAYER_PREFIX}.shp"
    logger.info("Reading shapefile %s with GeoPandas...", shp_path)
    gdf = gpd.read_file(shp_path)
    logger.info("Total landuse features in India: %d", len(gdf))

    logger.info("Filtering for fclass == 'industrial'...")
    industrial_gdf = gdf[gdf["fclass"] == "industrial"].copy()
    logger.info("Extracted %d nationwide industrial polygons!", len(industrial_gdf))

    # Standardize CRS to EPSG:4326 if needed
    if industrial_gdf.crs != "EPSG:4326":
        industrial_gdf = industrial_gdf.to_crs("EPSG:4326")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    industrial_gdf.to_parquet(output_parquet, index=False)
    logger.info("Saved all-India industrial polygons to %s", output_parquet)

    # Cleanup temporary shapefile components
    for f in extract_dir.glob(f"{TARGET_LAYER_PREFIX}.*"):
        f.unlink(missing_ok=True)
    extract_dir.rmdir()

    return industrial_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract Geofabrik India industrial polygons")
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Path to an existing downloaded 'india-latest-free.shp.zip'",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the shapefile zip automatically from Geofabrik",
    )
    args = parser.parse_args()

    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_path:
        zip_file = Path(args.zip_path)
    else:
        zip_file = ref_dir / "india-latest-free.shp.zip"

    if args.download or not zip_file.exists():
        if not zip_file.exists():
            logger.info("Zip file %s not found locally. Initiating download...", zip_file)
            success = download_geofabrik_zip(zip_file)
            if not success:
                sys.exit(1)

    extract_and_filter_industrial(zip_file)
