"""Adds an `in_forest` column to a processed corpus.

WHY THIS IS A COLUMN AND NOT A LOOKUP AT INFERENCE
--------------------------------------------------
`forest_cover.in_forest()` answers one point at a time, which is right for the
serving guard and hopeless for 2,044,295 of them. This does the same question as
one vectorised spatial join so the labelling rule and the feature vector can both
read it.

WHAT IT IS FOR
--------------
Measured on the verified set: 2,225 detections whose cited truth is
AGRICULTURAL_BURN are classified TRANSIENT_HOTSPOT, and **2,117 of them (95%) are
the three forest fires** -- Similipal 1,365, Uttarakhand 672, Bandipur 80. Their
FRP median is 1.07 MW against a harvest artifact floor of 1.50 MW, and 2,174 of
the 2,225 carry nominal rather than low detection confidence. They are not junk;
they are ordinary smouldering forest fire, and one threshold is discarding them.

That threshold encodes a prior -- *low-energy open-ground detections are mostly
specular glint* -- which is true over bare ground and water and wrong over tree
canopy, which is dark and diffuse. Land cover is what tells the two apart, and
the map knows it: 80% of India's forest area is mapped in OpenStreetMap, against
1.5% of its cropland, which is why this is a forest column and not a land-cover
one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd

logger = logging.getLogger("add_forest_column")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

FOREST_PARQUET = Path("data/reference/osm_india_forest_from_pbf.parquet")


def add_forest_column(corpus_path: Path, forest_path: Path = FOREST_PARQUET,
                      out_path: Path | None = None) -> int:
    """Writes `in_forest` into the corpus. Returns the number of detections inside."""
    corpus_path = corpus_path if corpus_path.is_absolute() else PROJECT_ROOT / corpus_path
    forest_path = forest_path if forest_path.is_absolute() else PROJECT_ROOT / forest_path
    out_path = out_path or corpus_path

    if not forest_path.exists():
        raise FileNotFoundError(
            f"{forest_path} not found. Build it with:\n"
            "  python -m src.ingestion.pbf_extractor --layer forest"
        )

    logger.info("Reading %s ...", corpus_path)
    df = pd.read_parquet(corpus_path)

    logger.info("Reading forest layer ...")
    forest = gpd.read_parquet(forest_path)
    forest = forest[forest.geometry.notna()].reset_index(drop=True)[["geometry"]]
    forest["__forest"] = True
    logger.info("  %d forest polygons", len(forest))

    pts = gpd.GeoDataFrame(
        df[["latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )

    logger.info("Spatial join over %d detections ...", len(pts))
    joined = gpd.sjoin(pts, forest, predicate="intersects", how="left")
    # A detection touching two overlapping polygons appears twice; the question
    # is boolean, so the first hit settles it.
    joined = joined[~joined.index.duplicated(keep="first")]

    df["in_forest"] = joined["__forest"].notna().to_numpy()
    n = int(df["in_forest"].sum())
    logger.info("in_forest: %d of %d (%.2f%%)", n, len(df), 100.0 * n / max(len(df), 1))

    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add in_forest to a processed corpus")
    ap.add_argument("corpus", type=str, help="Processed parquet to annotate")
    ap.add_argument("--forest", type=str, default=str(FOREST_PARQUET))
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    add_forest_column(Path(a.corpus), Path(a.forest),
                      Path(a.out) if a.out else None)
