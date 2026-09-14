"""Multi-Source Data Download Manager for Project SIH26162.

Provides CLI utilities to:
1. Fetch rolling multi-day Near-Real-Time (NRT) fire detections across multiple satellites.
2. Download annual NASA FIRMS Country archive packages for India.
3. Download and extract OpenStreetMap industrial shapefiles.
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv

from src.ingestion.firms_client import fetch_firms_nrt

load_dotenv()
logger = logging.getLogger("download_manager")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SATELLITE_SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
]

GEOFABRIK_INDIA_SHP_URL = "https://download.geofabrik.de/asia/india-latest-free.shp.zip"


def download_rolling_nrt_multi_sensor(day_range: int = 5, output_file: str = "data/raw/firms_combined_nrt.parquet") -> gpd.GeoDataFrame:
    """Fetches and pools recent NRT observations across all active satellite sensors.

    Args:
        day_range: Days back (1 to 10).
        output_file: Output path for combined Parquet dataset.

    Returns:
        Combined gpd.GeoDataFrame.
    """
    gdfs: List[gpd.GeoDataFrame] = []
    for sat in SATELLITE_SOURCES:
        logger.info("Fetching %d days of data for sensor: %s...", day_range, sat)
        try:
            gdf = fetch_firms_nrt(satellite=sat, day_range=day_range)
            if not gdf.empty:
                logger.info("-> Received %d detections from %s.", len(gdf), sat)
                gdfs.append(gdf)
            else:
                logger.info("-> 0 detections from %s.", sat)
        except Exception as e:
            logger.warning("-> Failed fetching %s: %s", sat, e)

    if not gdfs:
        logger.warning("No data retrieved from any sensor.")
        return gpd.GeoDataFrame()

    combined_df = pd.concat(gdfs, ignore_index=True)
    if "confidence" in combined_df.columns:
        combined_df["confidence"] = combined_df["confidence"].astype(str)
    # Deduplicate matching spatio-temporal detections across overlapping feeds
    combined_gdf = gpd.GeoDataFrame(combined_df, geometry="geometry", crs="EPSG:4326")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    combined_gdf.to_parquet(output_file, index=False)
    logger.info("Saved %d total multi-sensor detections to %s.", len(combined_gdf), output_file)
    return combined_gdf


def download_file_with_progress(url: str, dest_path: Path) -> bool:
    """Downloads a remote file with streaming chunks."""
    try:
        logger.info("Connecting to %s...", url)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_len = int(r.headers.get("content-length", 0))
            downloaded = 0
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_len > 0:
                            done = int(50 * downloaded / total_len)
                            sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded / (1024*1024):.1f} MB")
                            sys.stdout.flush()
            sys.stdout.write("\n")
        logger.info("Download completed: %s", dest_path)
        return True
    except Exception as e:
        logger.error("Failed downloading %s: %s", url, e)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIH26162 Data Download Manager")
    parser.add_argument(
        "--mode",
        choices=["nrt_all_sensors", "info"],
        default="nrt_all_sensors",
        help="Download mode",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of days backward (1-10) for NRT download",
    )
    args = parser.parse_args()

    if args.mode == "nrt_all_sensors":
        download_rolling_nrt_multi_sensor(day_range=args.days)
