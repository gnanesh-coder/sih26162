"""Spatial Join and Facility Matching Engine for Project SIH26162.

Intersects NASA FIRMS thermal hotspot detections with OpenStreetMap / Bhuvan
industrial boundary polygons using R-Tree spatial indexing (GEOS/sindex).
Computes proximity distances, identifies facility classifications, and feeds
enriched records into the Recurrence State Machine.
"""

import argparse
import logging
import re
import sys
import warnings
import os
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.alerting.state_machine import H3RecurrenceTracker, evaluate_dataframe

logger = logging.getLogger("spatial_join")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# The archive is preferred over any NRT snapshot. Every temporal mechanism
# downstream needs history: `n_30d` is a 30-day recurrence count, so on a 2-day
# snapshot it stays near zero, PERSISTENT_BASELINE never fires, and routine
# refinery flares cannot be told apart from genuine accidents.
DEFAULT_FIRMS_PARQUET = Path("data/raw/firms_archive.parquet")
ARCHIVE_60D_PARQUET = Path("data/raw/firms_archive_60d.parquet")
NRT_COMBINED_PARQUET = Path("data/raw/firms_combined_nrt.parquet")
FALLBACK_FIRMS_PARQUET = Path("data/raw/firms_latest.parquet")

# Tried in order; the first that exists wins.
FIRMS_INPUT_PRECEDENCE = [
    DEFAULT_FIRMS_PARQUET,
    ARCHIVE_60D_PARQUET,
    NRT_COMBINED_PARQUET,
    FALLBACK_FIRMS_PARQUET,
]
DEFAULT_MERGED_PARQUET = Path("data/reference/industrial_boundaries_merged.parquet")
DEFAULT_OSM_PARQUET = Path("data/reference/osm_india_industrial_from_pbf.parquet")
# The corpus the API and compliance register read. Overridable so a deployment
# can point at the full national archive rather than whatever a quick pipeline
# run last wrote: the default 2-day file yields 465 routine detections against
# the 12-month archive's 491,016, and a compliance register built from the
# former would under-report every site in the country.
OUTPUT_PROCESSED_PARQUET = Path(
    os.getenv("PROCESSED_CORPUS", "data/processed/firms_industrial_joined.parquet")
)


# Land uses that are tagged industrial in OSM but contain no combustion process
# whatsoever. Photovoltaic arrays are the textbook sun-glint false positive the
# problem statement research names explicitly -- "solar arrays can reflect
# intense solar radiation directly into the sensor optics" -- and because the
# polygons carry landuse=industrial, facility-level recurrence pooling drove
# them to n_30d of 67 and reported them as persistent industrial heat sources:
# 97% of detections at Khavda and 83% at Pavagada were misclassified.
#
# These sites are still recorded (they are real infrastructure) but are excluded
# from combustion-based reasoning.
NON_COMBUSTION_KEYWORDS = [
    "solar", "photovoltaic", " pv ", "renewable energy park",
    "wind farm", "wind park", "windfarm",
]


def is_non_combustion_site(row: pd.Series) -> bool:
    """True when the containing polygon cannot host industrial combustion."""
    text = " ".join([
        str(row.get("name", "") or ""),
        str(row.get("landuse", "") or ""),
        str(row.get("man_made", "") or ""),
        str(row.get("other_tags", "") or ""),
        str(row.get("industrial", "") or ""),
    ]).lower()
    if "solar thermal" in text or "concentrated solar" in text:
        return False  # CSP genuinely concentrates heat.
    return any(k in text for k in NON_COMBUSTION_KEYWORDS)


PLACEHOLDER_FACILITY_NAME = "Unnamed Industrial Site"

# Tag values that describe what a facility *is* when nobody has given it a name.
# Read straight from OpenStreetMap, never inferred: 78.4% of the 28,587 mapped
# industrial polygons carry no `name` tag at all, and only 124 of those 22,416
# carry `operator` or `name:en`. So there is no hidden name to recover -- but
# thousands do carry `industrial=`, `power=`, `plant:source=` or `description=`,
# and "Coal-fired power plant" tells an operator far more than "Unnamed site".
#
# This is a DESCRIPTOR, not a name. It says what OSM knows about the place. The
# distinction matters because facility_name reaches SitReps and SMS alerts, and
# an invented name in a dispatch would be exactly the fabricated detail this
# project refuses elsewhere.
_HSTORE = re.compile(r'"([a-z_:]+)"=>"([^"]*)"')


def _osm_tags(raw) -> dict:
    """Parses GDAL's hstore-style `other_tags` blob into a dict."""
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    return dict(_HSTORE.findall(str(raw)))


def _descriptor_from_tags(tags: dict) -> Optional[str]:
    """What OSM says this place is, when it does not say what it is called."""
    # A real identity, where one exists at all.
    for key in ("name:en", "operator"):
        value = (tags.get(key) or "").strip()
        if value:
            return value

    # "-fired" is a claim about combustion, so it is reserved for fuels that
    # actually burn. A photovoltaic array described as "Solar-fired power plant"
    # would contradict the non-combustion handling three functions above, which
    # exists precisely because a solar park is not a heat source.
    source = (tags.get("plant:source") or "").strip().lower()
    if (tags.get("power") or "").strip() in ("plant", "generator"):
        if source in ("coal", "gas", "oil", "diesel", "biomass", "biofuel", "waste"):
            return f"{source.capitalize()}-fired power plant"
        if source:
            return f"{source.capitalize()} power plant"
        return "Power plant"

    industrial = (tags.get("industrial") or "").strip()
    if industrial:
        return industrial.replace("_", " ").capitalize()

    description = (tags.get("description") or "").strip()
    if description:
        return description[:60]

    return None


def _derive_facility_name(df: pd.DataFrame) -> pd.Series:
    """Facility name per row: the OSM name, else a tag descriptor, else the placeholder."""
    name = df["name"].replace("None", np.nan) if "name" in df.columns else pd.Series(np.nan, index=df.index)
    name = name.astype(object).where(name.notna() & (name.astype(str).str.strip() != ""))

    if "other_tags" not in df.columns:
        return name.fillna(PLACEHOLDER_FACILITY_NAME)

    missing = name.isna()
    if missing.any():
        derived = df.loc[missing, "other_tags"].map(
            lambda raw: _descriptor_from_tags(_osm_tags(raw))
        )
        name.loc[missing] = derived

    return name.fillna(PLACEHOLDER_FACILITY_NAME)


def classify_facility_type(row: pd.Series) -> str:
    """Categorizes the industrial facility into standardized operational domain.

    Categories:
      - 'steel_metallurgy': Steel plants, blast furnaces, smelters, rolling mills
      - 'petrochemical_refinery': Refineries, oil storage, chemical, fertilizer
      - 'brick_kiln': Brickworks, kilns, refractory
      - 'power_thermal': Power stations, thermal energy
      - 'mining': Open-cast mines, quarries, mineral extraction zones
      - 'general_industrial': General estates, factories, SEZs
      - 'non_industrial': Background / external
    """
    if not row.get("inside_industrial", False):
        return "non_industrial"

    # A photovoltaic or wind site is infrastructure, but not a heat source.
    if is_non_combustion_site(row):
        return "renewable_non_thermal"

    # Pre-classified from ISRO Bhuvan or specific source tags
    existing_type = str(row.get("facility_type", "") or "").lower()
    if existing_type in ["mining", "quarry"]:
        return "mining"
    elif existing_type in ["steel_metallurgy", "petrochemical_refinery", "brick_kiln", "power_thermal"]:
        return existing_type

    text = " ".join([
        str(row.get("name", "") or ""),
        str(row.get("landuse", "") or ""),
        str(row.get("man_made", "") or ""),
        str(row.get("other_tags", "") or ""),
        str(row.get("industrial", "") or ""),
        existing_type,
    ]).lower()

    if any(k in text for k in ["steel", "iron", "smelter", "blast furnace", "metallurg", "foundry", "rolling mill"]):
        return "steel_metallurgy"
    elif any(k in text for k in ["mining", "quarry", "mineral", "pit"]):
        return "mining"
    elif any(k in text for k in ["refinery", "petro", "oil", "gas", "chemical", "fertilizer", "polymer", "terminal"]):
        return "petrochemical_refinery"
    elif any(k in text for k in ["kiln", "brick", "bhatta"]):
        return "brick_kiln"
    elif any(k in text for k in ["power", "thermal", "energy", "generator", "turbine"]):
        return "power_thermal"
    else:
        return "general_industrial"


def perform_spatial_join(
    firms_gdf: gpd.GeoDataFrame,
    industrial_gdf: gpd.GeoDataFrame,
    buffer_meters: float = 1500.0,
    compute_nearest: bool = True,
    adaptive_buffer: bool = True,
) -> gpd.GeoDataFrame:
    """Performs spatial join between hotspot points and industrial polygons.

    Uses R-Tree spatial index for sub-second Point-in-Polygon (PIP) testing,
    accounts for satellite pixel resolution footprint with a configurable buffer,
    and computes distances to nearest industrial perimeter.

    The default 1500m buffer follows remote-sensing practice for associating
    thermal detections with ground infrastructure. Two physical effects displace
    a reported detection centroid from the true source:

      1. Off-nadir pixel growth. A VIIRS I-band pixel is 375m at nadir but grows
         toward the edge of the orbital swath; the reported `scan`/`track` fields
         carry the actual along-scan/along-track footprint in km.
      2. Plume parallax. An intense fire lofts a super-heated plume to altitude;
         at high scan angles the satellite projects that elevated thermal
         signature laterally onto the ground plane, so the detection lands
         outside the true facility perimeter.

    A flat 500m buffer under-recovers detections at swath edge. With
    `adaptive_buffer` the tolerance becomes `buffer_meters + half the along-scan
    pixel footprint`, so nadir detections keep a tight association while
    edge-of-swath detections (scan up to ~1.6km) widen appropriately.

    Args:
        firms_gdf: GeoDataFrame of fire detections in EPSG:4326.
        industrial_gdf: GeoDataFrame of industrial polygons in EPSG:4326.
        buffer_meters: Base tolerance buffer around polygon perimeters (default: 1500m).
        compute_nearest: Whether to calculate nearest industrial distance for points outside.
        adaptive_buffer: Widen the buffer per-detection by half the reported
            along-scan pixel footprint to absorb off-nadir growth and parallax.

    Returns:
        Augmented GeoDataFrame with 'inside_industrial', 'facility_name', 'facility_type',
        and 'dist_to_industrial_km'.
    """
    if firms_gdf.empty:
        logger.warning("Empty FIRMS GeoDataFrame provided for spatial join.")
        return firms_gdf.copy()

    if industrial_gdf.empty:
        logger.warning("Empty Industrial GeoDataFrame provided. Marking all inside_industrial=False.")
        out = firms_gdf.copy()
        out["inside_industrial"] = False
        out["is_exact_match"] = False
        out["dist_to_industrial_km"] = 999.0
        out["facility_name"] = None
        out["facility_type"] = "non_industrial"
        return out

    # Ensure CRS alignment (handling environments without native pyproj)
    try:
        if firms_gdf.crs is not None and str(firms_gdf.crs).upper() not in ("EPSG:4326", "OGC:CRS84", "WGS 84"):
            firms_gdf = firms_gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    try:
        if industrial_gdf.crs is not None and str(industrial_gdf.crs).upper() not in ("EPSG:4326", "OGC:CRS84", "WGS 84"):
            industrial_gdf = industrial_gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    # Keep relevant polygon metadata attributes
    poly_cols = ["geometry"]
    for c in ["osm_id", "name", "landuse", "man_made", "industrial", "facility_type", "source", "other_tags"]:
        if c in industrial_gdf.columns:
            poly_cols.append(c)

    clean_polys = industrial_gdf[poly_cols].copy()

    logger.info("Executing Primary Point-in-Polygon spatial join (exact intersection)...")
    exact_joined = gpd.sjoin(firms_gdf, clean_polys, how="left", predicate="intersects")
    # Drop potential spatial duplicates if a point touches multiple overlapping polygons
    exact_joined = exact_joined[~exact_joined.index.duplicated(keep="first")].copy()

    # Hotspot is inside industrial if it intersected any polygon (indicated by non-null index_right)
    exact_joined["inside_industrial"] = exact_joined["index_right"].notnull()
    exact_joined["is_exact_match"] = exact_joined["inside_industrial"]
    exact_matched_count = exact_joined["inside_industrial"].sum()
    logger.info("-> Exact polygon matches: %d / %d detections.", exact_matched_count, len(exact_joined))

    # Buffer Proximity Matching: (e.g. 1500m buffer ~0.0135 degrees in WGS84)
    # Catches tank farms, scrap yards, flare stacks on the outer perimeter fence,
    # and plume-parallax-displaced pixels at the edge of the orbital swath.
    unmatched_mask = ~exact_joined["inside_industrial"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*geographic CRS.*")
        if buffer_meters > 0 and unmatched_mask.sum() > 0:
            unmatched_pts = firms_gdf.loc[unmatched_mask].copy()

            # Per-detection tolerance: base buffer plus half the along-scan pixel
            # footprint, which absorbs off-nadir pixel growth and plume parallax.
            if adaptive_buffer and "scan" in unmatched_pts.columns:
                scan_km = pd.to_numeric(unmatched_pts["scan"], errors="coerce").fillna(0.375).clip(lower=0.375, upper=6.0)
                per_point_m = buffer_meters + (scan_km * 1000.0) / 2.0
                logger.info(
                    "Evaluating adaptive buffer proximity (%.0f-%.0f meters, base %.0fm) for %d unmatched hotspots...",
                    per_point_m.min(), per_point_m.max(), buffer_meters, unmatched_mask.sum(),
                )
            else:
                per_point_m = pd.Series(buffer_meters, index=unmatched_pts.index, dtype=float)
                logger.info(
                    "Evaluating fixed buffer proximity (%.1f meters) for %d unmatched hotspots...",
                    buffer_meters, unmatched_mask.sum(),
                )

            buffer_deg = per_point_m / 111320.0  # approximate meters to degrees at equator

            # Buffer points and intersect against industrial polygons (much faster than buffering 28k complex polygons)
            unmatched_buffered = unmatched_pts.copy()
            unmatched_buffered["geometry"] = unmatched_pts.geometry.buffer(buffer_deg)

            buf_joined = gpd.sjoin(unmatched_buffered, clean_polys, how="left", predicate="intersects")
            buf_joined = buf_joined[~buf_joined.index.duplicated(keep="first")]

            buf_matches = buf_joined[buf_joined["index_right"].notnull()]
            if not buf_matches.empty:
                logger.info("-> Recovered %d additional hotspots within the %dm+ facility perimeter buffer!", len(buf_matches), int(buffer_meters))

                # Bulk assignment, deliberately. The previous implementation
                # looped per matched row and per column doing scalar `.loc`
                # writes. Each such write costs time proportional to the size of
                # the whole frame, and the number of matches also grows with the
                # frame, so the loop was quadratic: 4s at 50k detections, 28s at
                # 200k (7x for 4x the data), extrapolating to roughly 49 minutes
                # at national scale -- around 80% of total pipeline runtime for
                # what is logically a simple column copy.
                copy_cols = [
                    c for c in poly_cols
                    if c != "geometry" and c in buf_matches.columns
                ]
                idx = buf_matches.index
                exact_joined.loc[idx, "inside_industrial"] = True
                if copy_cols:
                    exact_joined.loc[idx, copy_cols] = buf_matches.loc[idx, copy_cols].to_numpy()

        # Calculate distance to nearest industrial facility (in kilometers)
        if compute_nearest:
            logger.info("Computing nearest industrial proximity distances...")
            exact_joined["dist_to_industrial_km"] = np.where(exact_joined["inside_industrial"], 0.0, np.nan)

            outside_mask = ~exact_joined["inside_industrial"]
            if outside_mask.sum() > 0:
                outside_pts = exact_joined.loc[outside_mask, ["geometry"]].copy()
                try:
                    nearest = gpd.sjoin_nearest(
                        outside_pts,
                        clean_polys[["geometry"]],
                        how="left",
                        distance_col="dist_deg",
                        max_distance=1.0,
                    )
                    nearest = nearest[~nearest.index.duplicated(keep="first")]
                    dist_km = nearest["dist_deg"] * 111.32
                    exact_joined.loc[outside_mask, "dist_to_industrial_km"] = dist_km.fillna(999.0)
                except Exception as e:
                    logger.warning("Nearest distance calculation warning (%s). Defaulting to 999.0 km.", e)
                    exact_joined.loc[outside_mask, "dist_to_industrial_km"] = 999.0
        exact_joined["dist_to_industrial_km"] = exact_joined["dist_to_industrial_km"].fillna(999.0).round(2)

    # Standardize facility names and types
    exact_joined["facility_name"] = _derive_facility_name(exact_joined)
    exact_joined.loc[~exact_joined["inside_industrial"], "facility_name"] = None
    exact_joined["facility_type"] = exact_joined.apply(classify_facility_type, axis=1)

    # Non-combustion sites keep their identity but are withdrawn from
    # combustion reasoning: `inside_industrial` drives both the weak-labelling
    # rule and facility-level recurrence pooling, and leaving a solar park in
    # that population is what produced a 97% false-positive rate there.
    non_combustion = exact_joined["facility_type"] == "renewable_non_thermal"
    n_nc = int(non_combustion.sum())
    if n_nc:
        logger.info(
            "-> %d detections fall on non-combustion sites (solar/wind); excluded "
            "from industrial combustion classification.", n_nc,
        )
        exact_joined.loc[non_combustion, "inside_industrial"] = False
        exact_joined.loc[non_combustion, "is_exact_match"] = False

    logger.info(
        "Spatial join complete! Total inside industrial: %d (%.1f%% of detections).",
        exact_joined["inside_industrial"].sum(),
        (exact_joined["inside_industrial"].sum() / len(exact_joined)) * 100.0,
    )

    return exact_joined


def process_live_pipeline(
    firms_file: Optional[Path] = None,
    osm_file: Optional[Path] = None,
    output_file: Path = OUTPUT_PROCESSED_PARQUET,
    buffer_meters: float = 1500.0,
    optical_validation: bool = True,
    max_optical_requests: int = 50,
) -> gpd.GeoDataFrame:
    """End-to-end pipeline: FIRMS -> Spatial Join -> State Machine -> Optical Validation.

    Args:
        firms_file: Path to FIRMS Parquet file. Defaults to firms_combined_nrt or firms_latest.
        osm_file: Path to OSM industrial Parquet file. Defaults to DEFAULT_OSM_PARQUET.
        output_file: Output path for final enriched Parquet file.
        buffer_meters: Base buffer distance in meters (widened per-detection by scan angle).
        optical_validation: Run Sentinel-2 dNBR burn-scar validation on ambiguous
            detections. Requires CDSE credentials; silently skipped without them.
        max_optical_requests: Ceiling on live Sentinel-2 API calls per run.

    Returns:
        GeoDataFrame with spatial join features, alert decisions, and dNBR where available.
    """
    # 1. Resolve FIRMS input file
    if firms_file and firms_file.exists():
        input_firms = firms_file
    else:
        input_firms = next((c for c in FIRMS_INPUT_PRECEDENCE if c.exists()), None)
        if input_firms is None:
            raise FileNotFoundError(
                "No FIRMS parquet found. Tried: "
                + ", ".join(str(c) for c in FIRMS_INPUT_PRECEDENCE)
                + ". Build an archive with: python -m src.ingestion.firms_client "
                "--start YYYY-MM-DD --out data/raw/firms_archive.parquet"
            )

    # 2. Resolve Industrial Reference File (Merged OSM + Bhuvan preferred)
    if osm_file and osm_file.exists():
        input_ref = osm_file
    elif DEFAULT_MERGED_PARQUET.exists():
        input_ref = DEFAULT_MERGED_PARQUET
    elif DEFAULT_OSM_PARQUET.exists():
        input_ref = DEFAULT_OSM_PARQUET
    else:
        raise FileNotFoundError(f"No industrial reference file found at {DEFAULT_MERGED_PARQUET} or {DEFAULT_OSM_PARQUET}")

    logger.info("Loading FIRMS detections from %s...", input_firms)
    firms_gdf = gpd.read_parquet(input_firms)

    logger.info("Loading industrial reference boundaries from %s...", input_ref)
    ref_gdf = gpd.read_parquet(input_ref)

    logger.info("Executing spatial join (Fires: %d, Polygons: %d)...", len(firms_gdf), len(ref_gdf))
    joined_gdf = perform_spatial_join(firms_gdf, ref_gdf, buffer_meters=buffer_meters)

    # Guard against the failure that made every temporal mechanism inert: a corpus
    # shorter than the recurrence window. It produces no error, just silently
    # degenerate `n_30d`, no PERSISTENT_BASELINE, and no suppression.
    if "acq_date" in firms_gdf.columns and not firms_gdf.empty:
        span_days = pd.to_datetime(firms_gdf["acq_date"], errors="coerce").dt.date.nunique()
        if span_days < 30:
            logger.warning(
                "=" * 70
            )
            logger.warning(
                "CORPUS SPANS ONLY %d DISTINCT DAY(S). The recurrence state machine "
                "scores against a 30-day baseline, so `n_30d` cannot reach its "
                "thresholds: PERSISTENT_BASELINE will not fire, nothing will be "
                "suppressed, and routine industrial flares will be indistinguishable "
                "from accidental fires.", span_days,
            )
            logger.warning(
                "Build a proper archive first:  python -m src.ingestion.firms_client "
                "--start YYYY-MM-DD --out data/raw/firms_archive.parquet"
            )
            logger.warning("=" * 70)

    logger.info("Feeding spatially-joined detections into Recurrence State Machine...")
    evaluated_df = evaluate_dataframe(
        df=joined_gdf,
        lat_col="latitude",
        lon_col="longitude",
        frp_col="frp",
        time_col="timestamp_utc",
        industrial_col="inside_industrial",
    )

    # Sentinel-2 optical burn-scar validation. Runs after the state machine so
    # that recurrence (n_30d) is available to pick out the genuinely ambiguous
    # detections worth spending API quota on. No-ops without CDSE credentials.
    if optical_validation:
        try:
            from src.ingestion.sentinel2_client import enrich_with_dnbr

            logger.info("Running Sentinel-2 dNBR optical validation...")
            evaluated_df = enrich_with_dnbr(evaluated_df, max_requests=max_optical_requests)
        except Exception as e:
            logger.warning("Optical validation step skipped (%s). Continuing without dNBR.", e)
            evaluated_df["dnbr"] = np.nan
            evaluated_df["dnbr_severity"] = "UNKNOWN"
            evaluated_df["dnbr_status"] = "STEP_FAILED"
    else:
        evaluated_df["dnbr"] = np.nan
        evaluated_df["dnbr_severity"] = "UNKNOWN"
        evaluated_df["dnbr_status"] = "DISABLED"

    # Reconstruct GeoDataFrame
    result_gdf = gpd.GeoDataFrame(evaluated_df, geometry=joined_gdf.geometry, crs="EPSG:4326")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Ensure all string/object columns are cleanly typed for Parquet
    for col in result_gdf.columns:
        if col != "geometry" and result_gdf[col].dtype == "object":
            result_gdf[col] = result_gdf[col].astype(str)

    result_gdf.to_parquet(output_file, index=False)
    logger.info("Enriched and classified dataset saved to %s (%d records).", output_file, len(result_gdf))

    # Log operational alert summary
    logger.info("=== OPERATIONAL ALERT SUMMARY FOR INDIA ===")
    logger.info("Alert Priorities Breakdown:\n%s", result_gdf["priority"].value_counts().to_string())
    logger.info("Alert States Breakdown:\n%s", result_gdf["state"].value_counts().to_string())
    logger.info("Facility Types Breakdown:\n%s", result_gdf["facility_type"].value_counts().to_string())

    return result_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial join & alert pipeline for Project SIH26162")
    parser.add_argument(
        "--firms",
        type=str,
        default=None,
        help="Path to FIRMS Parquet file",
    )
    parser.add_argument(
        "--osm",
        type=str,
        default=None,
        help="Path to OSM industrial Parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/firms_industrial_joined.parquet",
        help="Path to output Parquet file",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=1500.0,
        help="Base tolerance buffer around industrial boundaries in meters (default: 1500m, "
             "widened per-detection by half the along-scan pixel footprint)",
    )
    parser.add_argument(
        "--no-optical",
        action="store_true",
        help="Skip Sentinel-2 dNBR optical validation even if CDSE credentials are set",
    )
    parser.add_argument(
        "--max-optical-requests",
        type=int,
        default=50,
        help="Ceiling on live Sentinel-2 API calls for this run (default: 50)",
    )
    args = parser.parse_args()

    f_file = Path(args.firms) if args.firms else None
    o_file = Path(args.osm) if args.osm else None
    out_file = Path(args.output)

    process_live_pipeline(
        firms_file=f_file,
        osm_file=o_file,
        output_file=out_file,
        buffer_meters=args.buffer,
        optical_validation=not args.no_optical,
        max_optical_requests=args.max_optical_requests,
    )
