"""OpenStreetMap (OSM) Industrial Boundary Ingestion Client.

Queries the Overpass API to extract industrial landuse polygons,
manufacturing works, refineries, and thermal industrial sites across India.
Converts the geometries to GeoPandas GeoDataFrames in EPSG:4326.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("osm_client")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Overpass API endpoints with automatic failover
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Curated High-Risk Industrial Clusters in India (Refineries, Kilns, Petrochemicals, Steel)
PREDEFINED_HUBS: Dict[str, Tuple[float, float, float, float]] = {
    "jamnagar": (22.30, 69.80, 22.55, 70.15),           # Jamnagar Refinery Complex (Reliance & Nayara)
    "dahej_ankleshwar": (21.55, 72.85, 21.75, 73.10),   # Dahej-Ankleshwar PCPIR Chemical & Petroleum Belt
    "korba_chhattisgarh": (22.25, 82.60, 22.45, 82.80), # Korba Thermal Power & Aluminum Belt
    "durgapur_asansol": (23.45, 86.85, 23.65, 87.35),   # Durgapur-Asansol Steel & Heavy Metallurgy Belt
    "ncr_brick_kilns": (28.30, 76.70, 28.75, 77.40),    # Greater NCR Brick Kiln & Industrial Belt
    "mumbai_thane_belapur": (19.00, 72.95, 19.20, 73.10),# Navi Mumbai TTC & Petrochemical Belt
    "visakhapatnam": (17.60, 83.15, 17.75, 83.35),       # Vizag Steel & Petroleum Corridor
}


def build_overpass_query(
    south: float,
    west: float,
    north: float,
    east: float,
    timeout_sec: int = 60,
) -> str:
    """Constructs Overpass QL query for industrial polygons within a bounding box."""
    bbox_str = f"{south},{west},{north},{east}"
    query = f"""
    [out:json][timeout:{timeout_sec}];
    (
      way["landuse"="industrial"]({bbox_str});
      relation["landuse"="industrial"]({bbox_str});
      way["man_made"="works"]({bbox_str});
      relation["man_made"="works"]({bbox_str});
      way["industrial"]({bbox_str});
      relation["industrial"]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    return query


def _parse_overpass_elements(elements: List[Dict[str, Any]]) -> gpd.GeoDataFrame:
    """Converts raw Overpass JSON elements (nodes, ways, relations) into polygon GeoDataFrame."""
    nodes = {}
    ways = {}
    polygons = []
    properties = []

    for el in elements:
        if el.get("type") == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    for el in elements:
        if el.get("type") == "way":
            way_nodes = [nodes[n] for n in el.get("nodes", []) if n in nodes]
            ways[el["id"]] = way_nodes
            tags = el.get("tags", {})
            if len(way_nodes) >= 4 and way_nodes[0] == way_nodes[-1] and tags:
                try:
                    poly = Polygon(way_nodes)
                    if poly.is_valid and not poly.is_empty:
                        polygons.append(poly)
                        tags["osm_id"] = f"way/{el['id']}"
                        tags["osm_type"] = "way"
                        properties.append(tags)
                except Exception as err:
                    logger.debug("Skipping invalid way %s: %s", el.get("id"), err)

    for el in elements:
        if el.get("type") == "relation":
            tags = el.get("tags", {})
            if tags.get("type") == "multipolygon" or "landuse" in tags or "industrial" in tags:
                outer_rings = []
                for member in el.get("members", []):
                    if member.get("type") == "way" and member.get("role") in ("outer", ""):
                        way_pts = ways.get(member.get("ref"), [])
                        if len(way_pts) >= 4 and way_pts[0] == way_pts[-1]:
                            outer_rings.append(Polygon(way_pts))
                if outer_rings:
                    try:
                        multi = MultiPolygon(outer_rings) if len(outer_rings) > 1 else outer_rings[0]
                        if multi.is_valid and not multi.is_empty:
                            polygons.append(multi)
                            tags["osm_id"] = f"relation/{el['id']}"
                            tags["osm_type"] = "relation"
                            properties.append(tags)
                    except Exception as err:
                        logger.debug("Skipping invalid relation %s: %s", el.get("id"), err)

    if not polygons:
        return gpd.GeoDataFrame(
            columns=["osm_id", "osm_type", "name", "landuse", "industrial", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(properties, geometry=polygons, crs="EPSG:4326")
    # Cast all object columns to str to ensure clean Parquet serialization
    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == "object":
            gdf[col] = gdf[col].astype(str)
    return gdf


def fetch_osm_industrial_polygons(
    bbox: Tuple[float, float, float, float],
    timeout_sec: int = 90,
) -> gpd.GeoDataFrame:
    """Queries Overpass API for industrial polygons in a bounding box (south, west, north, east)."""
    south, west, north, east = bbox
    query = build_overpass_query(south, west, north, east, timeout_sec=timeout_sec)

    response = None
    for server_url in OVERPASS_SERVERS:
        try:
            logger.info("Querying Overpass API at %s for bbox [%.3f, %.3f, %.3f, %.3f]...", server_url, south, west, north, east)
            resp = requests.post(server_url, data={"data": query}, timeout=timeout_sec + 15)
            if resp.status_code == 200:
                response = resp
                break
            else:
                logger.warning("Overpass server %s returned status %d. Trying next mirror...", server_url, resp.status_code)
        except requests.RequestException as e:
            logger.warning("Failed connecting to %s: %s. Trying next mirror...", server_url, e)

    if response is None or response.status_code != 200:
        logger.error("All Overpass API mirrors failed or timed out.")
        return gpd.GeoDataFrame(columns=["osm_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    try:
        data = response.json()
    except Exception as e:
        logger.error("Failed parsing Overpass JSON response: %s", e)
        return gpd.GeoDataFrame(columns=["osm_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    elements = data.get("elements", [])
    logger.info("Received %d raw OSM elements from Overpass.", len(elements))

    gdf = _parse_overpass_elements(elements)
    logger.info("Successfully extracted %d industrial polygon features.", len(gdf))
    return gdf


def save_industrial_polygons(gdf: gpd.GeoDataFrame, output_stem: str) -> None:
    """Saves GeoDataFrame to both GeoJSON and Parquet in data/reference/."""
    out_dir = Path("data/reference")
    out_dir.mkdir(parents=True, exist_ok=True)

    geojson_path = out_dir / f"{output_stem}.geojson"
    parquet_path = out_dir / f"{output_stem}.parquet"

    gdf.to_file(geojson_path, driver="GeoJSON")
    gdf.to_parquet(parquet_path, index=False)
    logger.info("Saved %d polygons to:\n  -> %s\n  -> %s", len(gdf), geojson_path, parquet_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract OSM industrial polygons via Overpass API")
    parser.add_argument(
        "--hub",
        choices=list(PREDEFINED_HUBS.keys()) + ["all"],
        default="jamnagar",
        help="Predefined industrial corridor name (or 'all')",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Custom bounding box as 'south,west,north,east' (e.g. '22.3,69.8,22.5,70.1')",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output filename stem (without extension)",
    )
    args = parser.parse_args()

    if args.bbox:
        parts = [float(x.strip()) for x in args.bbox.split(",")]
        if len(parts) != 4:
            logger.error("BBOX must contain exactly 4 comma-separated values: south,west,north,east")
            sys.exit(1)
        bbox = (parts[0], parts[1], parts[2], parts[3])
        stem = args.output_name or "osm_industrial_custom_bbox"
        gdf = fetch_osm_industrial_polygons(bbox)
        if not gdf.empty:
            save_industrial_polygons(gdf, stem)
    elif args.hub == "all":
        logger.info("Extracting all %d predefined industrial hubs...", len(PREDEFINED_HUBS))
        all_gdfs = []
        for name, bbox in PREDEFINED_HUBS.items():
            logger.info("--- Hub: %s ---", name)
            h_gdf = fetch_osm_industrial_polygons(bbox)
            if not h_gdf.empty:
                h_gdf["hub_name"] = name
                all_gdfs.append(h_gdf)
        if all_gdfs:
            combined = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
            save_industrial_polygons(combined, "osm_industrial_all_hubs")
    else:
        bbox = PREDEFINED_HUBS[args.hub]
        stem = args.output_name or f"osm_industrial_{args.hub}"
        logger.info("Extracting hub: %s with bbox %s...", args.hub, bbox)
        gdf = fetch_osm_industrial_polygons(bbox)
        if not gdf.empty:
            save_industrial_polygons(gdf, stem)
