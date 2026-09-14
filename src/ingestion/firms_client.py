"""NASA FIRMS Ingestion Client (near-real-time and historical archive).

Fetches active fire detection CSVs from the NASA FIRMS area API, parses them into
GeoPandas GeoDataFrames with EPSG:4326 geometries, and standardizes acquisition
timestamps to UTC.

Two access patterns are supported:

  * `fetch_firms_nrt`     - the most recent N days, or N days from a start date.
  * `fetch_firms_archive` - an arbitrary historical span, paged in 5-day windows.

The archive path exists because the analytics downstream are fundamentally
temporal. The recurrence state machine scores every detection against a 30-day
baseline (`n_30d`), and Sentinel-2 dNBR needs detections older than the satellite
revisit cycle. A single-day NRT snapshot satisfies neither: `n_30d` stays near
zero, no location ever reaches PERSISTENT_BASELINE, and routine refinery flares
are therefore indistinguishable from genuine accidents.
"""

import io
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from shapely.geometry import Point

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("firms_client")

# Load environment variables
load_dotenv()

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
AVAILABILITY_URL = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv"
DEFAULT_INDIA_BBOX = "68.0,6.5,97.5,35.5"

# The area endpoint enforces a maximum of 5 days per request (it rejects larger
# spans with "Invalid day range. Expects [1..5]"), so any longer span is paged.
MAX_DAYS_PER_REQUEST = 5

# FIRMS splits each sensor into a near-real-time stream and a standard-processing
# archive. NRT covers roughly the last two months; anything older lives in SP.
# Requesting the wrong one for a date returns an empty CSV rather than an error,
# so the source is checked against the availability table rather than assumed.
NRT_TO_SP_SOURCE = {
    "VIIRS_SNPP_NRT": "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_NRT": "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_NRT": "VIIRS_NOAA21_SP",
    "MODIS_NRT": "MODIS_SP",
}

# Courtesy pacing. The documented ceiling is 5000 transactions per 10 minutes,
# far above what paging needs, but a burst of rapid requests is still impolite.
MIN_REQUEST_INTERVAL_S = 0.4


def parse_firms_timestamps(df: pd.DataFrame) -> pd.Series:
    """Combines acq_date (YYYY-MM-DD) and acq_time (HHMM) into UTC pd.Timestamp.

    Args:
        df: DataFrame containing 'acq_date' and 'acq_time' columns.

    Returns:
        pd.Series of timezone-aware UTC datetime objects.
    """
    date_series = df["acq_date"].astype(str).str.strip()
    time_series = df["acq_time"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
    datetime_str = date_series + " " + time_series
    return pd.to_datetime(datetime_str, format="%Y-%m-%d %H%M", utc=True, errors="coerce")


def create_empty_firms_gdf() -> gpd.GeoDataFrame:
    """Creates an empty GeoDataFrame with standard FIRMS schema and EPSG:4326 CRS."""
    columns = [
        "latitude",
        "longitude",
        "bright_ti4",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "version",
        "bright_ti5",
        "frp",
        "daynight",
        "timestamp_utc",
        "geometry",
    ]
    return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:4326")


def csv_to_gdf(content: str) -> gpd.GeoDataFrame:
    """Parses a FIRMS CSV payload into a typed GeoDataFrame.

    Shared by the NRT and archive paths so both produce an identical schema; a
    divergence here would surface much later as a confusing feature mismatch.
    """
    content = content.strip()
    if not content or "Invalid MAP_KEY" in content or "Bad Request" in content:
        logger.error("FIRMS API returned invalid response: %s", content[:200])
        return create_empty_firms_gdf()

    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        logger.error("Failed to parse FIRMS CSV response: %s", e)
        return create_empty_firms_gdf()

    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return create_empty_firms_gdf()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    if "acq_date" in df.columns and "acq_time" in df.columns:
        df["timestamp_utc"] = parse_firms_timestamps(df)
    else:
        df["timestamp_utc"] = pd.NaT

    for num_col in ["bright_ti4", "bright_ti5", "frp", "scan", "track", "brightness", "bright_t31"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # VIIRS reports confidence categorically ('l'/'n'/'h'), MODIS numerically
    # (0-100). Keeping it a string preserves both; the feature pipeline
    # reconciles them onto a single scale.
    #
    # `version` needs the same treatment for a different reason: NRT sources emit
    # a string like "2.0NRT" while the standard-processing archives emit a
    # number. Concatenating both in one archive produces a mixed object column
    # that pyarrow cannot type, and the whole write fails with "Could not convert
    # '2.0NRT' with type str: tried to convert to int64". Normalising at parse
    # time keeps every page schema-compatible with every other.
    for str_col in ("confidence", "version", "satellite", "instrument", "daynight"):
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str)

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def _resolve_key(map_key: Optional[str]) -> Optional[str]:
    """Returns a usable FIRMS map key, or None with a clear explanation."""
    key = map_key or os.getenv("FIRMS_MAP_KEY", "")
    if not key or key == "your_nasa_firms_key_here":
        logger.warning(
            "FIRMS_MAP_KEY is missing or contains placeholder. "
            "Please register at https://firms.modaps.eosdis.nasa.gov/api/map_key/ "
            "and set FIRMS_MAP_KEY in .env."
        )
        return None
    return key


_last_request_at = {"t": 0.0}


def _get(url: str, masked: str, timeout: int) -> Optional[str]:
    """Issues a paced GET and returns the body, or None on any failure."""
    since = time.time() - _last_request_at["t"]
    if since < MIN_REQUEST_INTERVAL_S:
        time.sleep(MIN_REQUEST_INTERVAL_S - since)

    try:
        response = requests.get(url, timeout=timeout)
    except requests.exceptions.Timeout:
        logger.error("Request to FIRMS API timed out after %d seconds.", timeout)
        return None
    except requests.exceptions.RequestException as e:
        logger.error("Network error connecting to FIRMS API: %s", e)
        return None
    finally:
        _last_request_at["t"] = time.time()

    if response.status_code != 200:
        logger.error(
            "FIRMS API returned HTTP %d for %s: %s",
            response.status_code, masked, response.text[:200].strip(),
        )
        return None

    return response.text


def fetch_data_availability(
    map_key: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Returns the date range FIRMS actually serves for each sensor source.

    This is queried rather than assumed. Asking for a date outside a source's
    window returns an empty CSV, not an error, so without checking availability
    an archive pull can silently produce nothing and be misread as "there were no
    fires" instead of "the wrong source was queried".
    """
    key = _resolve_key(map_key)
    if key is None:
        return pd.DataFrame()

    body = _get(
        f"{AVAILABILITY_URL}/{key}/ALL",
        f"{AVAILABILITY_URL}/***KEY***/ALL",
        timeout,
    )
    if body is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(body.strip()))
    except Exception as e:
        logger.error("Could not parse availability table: %s", e)
        return pd.DataFrame()

    logger.info("FIRMS data availability:\n%s", df.to_string(index=False))
    return df


def fetch_firms_nrt(
    satellite: str = "VIIRS_SNPP_NRT",
    day_range: int = 1,
    map_key: Optional[str] = None,
    bbox: Optional[str] = None,
    timeout: int = 30,
    start_date: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Fetch fire detections from the NASA FIRMS area API.

    Args:
        satellite: Source name, e.g. 'VIIRS_SNPP_NRT' or 'VIIRS_SNPP_SP'.
        day_range: Days of data to retrieve (1 to 10).
        map_key: NASA FIRMS Map Key. Defaults to FIRMS_MAP_KEY env var.
        bbox: 'min_lon,min_lat,max_lon,max_lat'. Defaults to INDIA_BBOX env var.
        timeout: Request timeout in seconds.
        start_date: Optional 'YYYY-MM-DD'. The window then runs forward from this
            date for day_range days. Omitted, the API returns the most recent
            day_range days -- which is why the original client could only ever
            see the present and never accumulate a recurrence baseline.

    Returns:
        gpd.GeoDataFrame with point geometries in EPSG:4326 and UTC timestamps.
    """
    key = _resolve_key(map_key)
    if key is None:
        return create_empty_firms_gdf()

    box = bbox or os.getenv("INDIA_BBOX", DEFAULT_INDIA_BBOX)
    day_range = max(1, min(int(day_range), MAX_DAYS_PER_REQUEST))

    suffix = f"/{start_date}" if start_date else ""
    url = f"{FIRMS_BASE_URL}/{key}/{satellite}/{box}/{day_range}{suffix}"
    masked = f"{FIRMS_BASE_URL}/***KEY***/{satellite}/{box}/{day_range}{suffix}"
    logger.info(
        "Fetching FIRMS data (source=%s, days=%d, from=%s)",
        satellite, day_range, start_date or "most recent",
    )

    body = _get(url, masked, timeout)
    if body is None:
        return create_empty_firms_gdf()

    gdf = csv_to_gdf(body)
    logger.info("-> %d detections.", len(gdf))
    return gdf


def fetch_firms_window(
    satellite: str,
    day_range: int,
    start_date: Optional[str],
    map_key: str,
    bbox: str,
    timeout: int,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Fetches one window and reports WHY it is empty when it is.

    An HTTP failure and a genuinely fire-free window both yield zero rows. Left
    undistinguished, a systematic request error (a bad day range, an out-of-range
    date) reads as "there were no fires", which is how an entire archive pull can
    silently return nothing and be misdiagnosed.

    Returns:
        (gdf, status) with status in OK / EMPTY / REQUEST_FAILED.
    """
    day_range = max(1, min(int(day_range), MAX_DAYS_PER_REQUEST))
    suffix = f"/{start_date}" if start_date else ""
    url = f"{FIRMS_BASE_URL}/{map_key}/{satellite}/{bbox}/{day_range}{suffix}"
    masked = f"{FIRMS_BASE_URL}/***KEY***/{satellite}/{bbox}/{day_range}{suffix}"

    body = _get(url, masked, timeout)
    if body is None:
        return create_empty_firms_gdf(), "REQUEST_FAILED"

    gdf = csv_to_gdf(body)
    return gdf, ("OK" if not gdf.empty else "EMPTY")


def build_availability_index(
    map_key: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Tuple[date, date]]:
    """Returns {source_id: (min_date, max_date)} from the live availability table."""
    df = fetch_data_availability(map_key=map_key, timeout=timeout)
    index: Dict[str, Tuple[date, date]] = {}
    if df.empty:
        return index

    for _, row in df.iterrows():
        try:
            index[str(row["data_id"]).strip()] = (
                datetime.strptime(str(row["min_date"]).strip(), "%Y-%m-%d").date(),
                datetime.strptime(str(row["max_date"]).strip(), "%Y-%m-%d").date(),
            )
        except (ValueError, TypeError, KeyError):
            continue
    return index


def resolve_source_for_window(
    preferred: str,
    window_start: date,
    availability: Dict[str, Tuple[date, date]],
) -> Optional[str]:
    """Picks the sensor variant that actually covers a given date.

    Each sensor is split between a near-real-time stream and a standard-processing
    archive, and the boundary between them moves. Querying a source outside its
    window returns an empty CSV rather than an error, so a long pull that assumes
    one variant loses whole months of a sensor without ever saying so.

    Prefers the requested source, falls back to its counterpart, and returns None
    when neither covers the date.
    """
    if not availability:
        return preferred  # No table available; caller gets the documented default.

    counterpart = NRT_TO_SP_SOURCE.get(preferred)
    if counterpart is None:
        counterpart = next(
            (n for n, sp in NRT_TO_SP_SOURCE.items() if sp == preferred), None
        )

    for candidate in (preferred, counterpart):
        if candidate is None or candidate not in availability:
            continue
        lo, hi = availability[candidate]
        if lo <= window_start <= hi:
            return candidate
    return None


def fetch_firms_archive(
    start_date: str,
    end_date: Optional[str] = None,
    satellites: Optional[List[str]] = None,
    map_key: Optional[str] = None,
    bbox: Optional[str] = None,
    timeout: int = 60,
    auto_source: bool = True,
) -> gpd.GeoDataFrame:
    """Builds a historical archive by paging the area API in 5-day windows.

    Args:
        start_date: Earliest date to retrieve, 'YYYY-MM-DD'.
        end_date: Latest date, 'YYYY-MM-DD'. Defaults to today.
        satellites: Source names. Defaults to VIIRS S-NPP + NOAA-20 + MODIS.
        map_key: NASA FIRMS Map Key.
        bbox: Area of interest.
        timeout: Per-request timeout.
        auto_source: Consult the live availability table and switch each sensor
            between its NRT stream and its standard-processing archive per window.
            Without this, a long pull silently loses whole months of a sensor,
            because an out-of-range date returns an empty CSV rather than an error.

    Returns:
        Deduplicated GeoDataFrame spanning the requested period.
    """
    key = _resolve_key(map_key)
    if key is None:
        return create_empty_firms_gdf()

    satellites = satellites or ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]
    box = bbox or os.getenv("INDIA_BBOX", DEFAULT_INDIA_BBOX)

    d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
    d1 = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()
    if d0 > d1:
        logger.error("start_date %s is after end_date %s.", d0, d1)
        return create_empty_firms_gdf()

    total_days = (d1 - d0).days + 1
    n_windows = (total_days + MAX_DAYS_PER_REQUEST - 1) // MAX_DAYS_PER_REQUEST
    logger.info(
        "Archive pull: %s -> %s (%d days, %d windows x %d source(s) = %d requests)",
        d0, d1, total_days, n_windows, len(satellites), n_windows * len(satellites),
    )

    availability = build_availability_index(map_key=key, timeout=timeout) if auto_source else {}

    frames: List[gpd.GeoDataFrame] = []
    tally: Dict[str, Dict[str, int]] = {
        s: {"OK": 0, "EMPTY": 0, "REQUEST_FAILED": 0, "SKIPPED_UNAVAILABLE": 0}
        for s in satellites
    }
    substitutions: Dict[str, int] = {}

    for sat in satellites:
        cursor = d0
        while cursor <= d1:
            span = min(MAX_DAYS_PER_REQUEST, (d1 - cursor).days + 1)

            source = resolve_source_for_window(sat, cursor, availability) if auto_source else sat
            if source is None:
                tally[sat]["SKIPPED_UNAVAILABLE"] += 1
                cursor += timedelta(days=span)
                continue
            if source != sat:
                substitutions[f"{sat}->{source}"] = substitutions.get(f"{sat}->{source}", 0) + 1

            page, status = fetch_firms_window(
                satellite=source,
                day_range=span,
                start_date=cursor.isoformat(),
                map_key=key,
                bbox=box,
                timeout=timeout,
            )
            tally[sat][status] += 1
            if status == "OK":
                frames.append(page)
            cursor += timedelta(days=span)

        counts = tally[sat]
        logger.info(
            "Source %-18s %d with data, %d empty, %d failed, %d skipped (no coverage).",
            sat, counts["OK"], counts["EMPTY"], counts["REQUEST_FAILED"],
            counts["SKIPPED_UNAVAILABLE"],
        )

    for swap, n in sorted(substitutions.items()):
        logger.info("Auto-switched %s for %d window(s) outside the NRT coverage range.", swap, n)

    for sat, counts in tally.items():
        total = sum(counts.values())
        if counts["REQUEST_FAILED"] == total and total:
            logger.warning(
                "Source %s: EVERY request failed. This is a request problem, not a "
                "data gap - check the day range and that the dates fall inside the "
                "source's availability window (fetch_data_availability()).", sat,
            )
        elif counts["OK"] == 0 and counts["EMPTY"] == total and total:
            logger.warning(
                "Source %s: every request succeeded but returned no rows. The dates "
                "are likely outside this source's availability window - consider the "
                "%s archive source.",
                sat, NRT_TO_SP_SOURCE.get(sat, "standard-processing"),
            )

    if not frames:
        logger.error("Archive pull produced no data across all sources.")
        return create_empty_firms_gdf()

    # Drop geometry BEFORE concatenating. Each page carries one Shapely Point per
    # row, and a year of national coverage is ~2M rows; holding that many live
    # geometry objects through a concat plus a drop_duplicates is what exhausted
    # memory on the first full-year attempt. Coordinates are already in the
    # latitude/longitude columns, so geometry is rebuilt once at the end.
    plain = [
        pd.DataFrame(f.drop(columns=["geometry"], errors="ignore")) for f in frames
    ]
    frames.clear()

    combined = pd.concat(plain, ignore_index=True, copy=False)
    plain.clear()

    # Overlapping windows and multiple sensors can restate the same detection.
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"],
        keep="first",
    ).reset_index(drop=True)
    logger.info(
        "Archive assembled: %d detections (%d duplicates removed).",
        len(combined), before - len(combined),
    )

    logger.info("Rebuilding point geometry for %d rows...", len(combined))
    gdf = gpd.GeoDataFrame(
        combined,
        geometry=gpd.points_from_xy(combined["longitude"], combined["latitude"]),
        crs="EPSG:4326",
    )

    if "acq_date" in gdf.columns and not gdf.empty:
        logger.info(
            "Coverage: %s -> %s across %d distinct days.",
            gdf.acq_date.min(), gdf.acq_date.max(), gdf.acq_date.nunique(),
        )
    return gdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NASA FIRMS ingestion (SIH26162)")
    parser.add_argument("--days", type=int, default=1,
                        help="Days of history to fetch. Above 10 the archive pager is used.")
    parser.add_argument("--start", type=str, default=None,
                        help="Earliest date YYYY-MM-DD. Overrides --days as the window start.")
    parser.add_argument("--end", type=str, default=None,
                        help="Latest date YYYY-MM-DD (default: today).")
    parser.add_argument("--satellites", type=str,
                        default="VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT,MODIS_NRT",
                        help="Comma-separated FIRMS source names.")
    parser.add_argument("--out", type=str, default="data/raw/firms_latest.parquet",
                        help="Output parquet path.")
    parser.add_argument("--availability", action="store_true",
                        help="Print the FIRMS availability table and exit.")
    args = parser.parse_args()

    if args.availability:
        fetch_data_availability()
        sys.exit(0)

    sats = [s.strip() for s in args.satellites.split(",") if s.strip()]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.start or args.days > MAX_DAYS_PER_REQUEST:
        start = args.start or (date.today() - timedelta(days=args.days - 1)).isoformat()
        logger.info("Executing FIRMS archive ingestion run...")
        data = fetch_firms_archive(start_date=start, end_date=args.end, satellites=sats)
    else:
        logger.info("Executing standalone FIRMS ingestion run...")
        frames = [fetch_firms_nrt(satellite=s, day_range=args.days) for s in sats]
        frames = [f for f in frames if not f.empty]
        data = (
            gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
            if frames else create_empty_firms_gdf()
        )

    if data.empty:
        logger.warning(
            "Ingestion returned 0 records (check FIRMS_MAP_KEY in .env). "
            "Writing empty GeoDataFrame schema to %s for downstream pipeline stability.",
            output_path,
        )
    else:
        logger.info("Fetched %d detections. Writing to %s...", len(data), output_path)

    try:
        data.to_parquet(output_path, index=False)
    except Exception as e:
        logger.error("Failed to write %s: %s", output_path, e)
        logger.error("The fetch succeeded (%d detections) but serialization did not.", len(data))
        mixed = [
            c for c in data.columns
            if c != "geometry" and data[c].dtype == object
            and data[c].map(type).nunique() > 1
        ]
        if mixed:
            logger.error(
                "Columns holding more than one Python type: %s. This happens when "
                "NRT and standard-processing sources disagree on a field's type; "
                "normalise them in csv_to_gdf().", mixed,
            )
        else:
            logger.error("Retry with a shorter --start/--end span, or a narrower bbox.")
        sys.exit(1)

    logger.info("FIRMS ingestion completed. Output saved to %s (%d detections).",
                output_path, len(data))
