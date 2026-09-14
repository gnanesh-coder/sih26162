"""Sentinel-2 Optical Burn-Scar Validation for Project SIH26162.

Provides the third data modality required by the problem statement: independent
optical confirmation of what a thermal anomaly actually did to the ground.

NASA FIRMS tells us *something hot happened here*. OpenStreetMap tells us
*there is a refinery here*. Neither answers the decisive question: did the
combustion consume surrounding biomass, or was it contained inside a concrete
and steel facility?

The Normalized Burn Ratio answers it:

    NBR   = (B08 - B12) / (B08 + B12)          # NIR vs SWIR, 20m resolution
    dNBR  = NBR_pre_event - NBR_post_event

Healthy vegetation reflects strongly in NIR (B08) and weakly in SWIR (B12), so
it carries a high NBR. Charred biomass inverts that relationship. A wildfire or
crop burn therefore leaves a large positive dNBR. An industrial tank fire burning
inside a bunded concrete compound leaves dNBR close to 0 -- there was no
vegetation to consume. This is the failsafe that separates the two classes when
thermal radiometry alone is ambiguous.

Data source: Copernicus Data Space Ecosystem (CDSE) Sentinel Hub Statistical API.
Free, but requires OAuth2 client credentials. Register at
https://dataspace.copernicus.eu and set CDSE_CLIENT_ID / CDSE_CLIENT_SECRET.

Every entry point degrades gracefully: with no credentials, no network, or full
cloud cover, dNBR is reported as None with an explicit status string. It is
never silently coerced to 0.0, because 0.0 is itself a strong classification
signal ("verified no burn scar") and must not be confused with "unknown".
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Credentials live in .env alongside FIRMS_MAP_KEY. Without this the CLI and the
# pipeline would read an empty environment and report NO_CREDENTIALS even when
# .env is filled in correctly - a false negative that looks exactly like a
# missing registration.
load_dotenv()

logger = logging.getLogger("sentinel2_client")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STATISTICS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

CACHE_PATH = Path("data/processed/dnbr_cache.json")

# Native resolution of Sentinel-2 B12 (SWIR), the coarser of the two NBR bands.
TARGET_RESOLUTION_M = 20.0
# Sentinel Hub rejects S2L2A requests coarser than 1500 m/px, and caps output
# dimensions. Both bounds are enforced when converting a bbox to pixel counts.
MIN_DIMENSION_PX = 16
MAX_DIMENSION_PX = 2500

# Sentinel-2 A+B revisit is ~5 days at these latitudes, plus 1-2 days of L2A
# processing latency. Below this age a post-event scene simply cannot exist yet,
# so dNBR is unavailable for genuinely near-real-time detections. This is a
# property of the orbit, not a defect in the pipeline.
MIN_DAYS_FOR_POST_SCENE = 7

# Politeness/backoff for the Sentinel Hub rate limiter.
MIN_REQUEST_INTERVAL_S = 0.35
MAX_RATE_LIMIT_RETRIES = 3

# Upper bound on any single backoff sleep. CDSE answers an exhausted quota with
# a Retry-After measured in minutes (843s observed), and obeying that verbatim
# blocks a batch pipeline for the better part of an hour with zero CPU, looking
# indistinguishable from a hang. Past this cap the request is abandoned and
# reported as RATE_LIMITED so the caller can decide what to do.
MAX_BACKOFF_S = 45.0

# A transport failure is not a data gap. DNS loss, a dropped link or a gateway
# error says nothing about whether a scene exists, so the location must be
# retried rather than recorded as unmeasurable. A one-minute DNS outage on
# 2026-09-12 burned 79 of 373 locations precisely because the batch loop
# advanced past API_ERROR the way it advances past a genuine result.
TRANSPORT_BACKOFF_S = 5.0
MAX_TRANSPORT_BACKOFF_S = 60.0
MAX_TRANSPORT_RETRIES = 3
# If this many locations fail transport back to back, the network is down rather
# than flaky: stop, so the remaining locations stay unattempted and recoverable
# instead of being consumed at the rate the failures arrive.
MAX_CONSECUTIVE_TRANSPORT_FAILURES = 5

# Every status compute_dnbr can report. Only OK carries a usable dNBR.
# PARSE_MISMATCH, NOT_AUTHORIZED and BAD_REQUEST indicate a fault on our side or
# in configuration; the rest are genuine data gaps.
VALID_STATUSES = frozenset({
    "OK",
    "NO_CREDENTIALS",
    "NOT_AUTHORIZED",
    "BAD_REQUEST",
    "PARSE_MISMATCH",
    "API_ERROR",
    "RATE_LIMITED",
    "NO_CLEAR_SCENE",
    "NO_SCENES_IN_WINDOW",
    "EVENT_TOO_RECENT",
    "AWAITING_POST_SCENE",
    "BAD_DATE",
    "UNKNOWN",
})

# USGS / FIREMON standard dNBR burn severity breakpoints.
SEVERITY_BREAKS: List[Tuple[float, str]] = [
    (0.10, "UNBURNED"),
    (0.27, "LOW_SEVERITY"),
    (0.44, "MODERATE_LOW"),
    (0.66, "MODERATE_HIGH"),
    (float("inf"), "HIGH_SEVERITY"),
]

# Evalscript computing per-pixel NBR with Scene Classification Layer masking.
# SCL classes excluded: 0 no-data, 1 saturated/defective, 3 cloud shadow,
# 8 cloud medium probability, 9 cloud high probability, 10 thin cirrus, 11 snow.
NBR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B08", "B12", "SCL", "dataMask"] }],
    output: [
      { id: "nbr", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  var blocked = [0, 1, 3, 8, 9, 10, 11];
  var valid = s.dataMask;
  if (blocked.indexOf(s.SCL) >= 0) { valid = 0; }
  var denom = s.B08 + s.B12;
  var nbr = (denom === 0) ? 0 : (s.B08 - s.B12) / denom;
  return { nbr: [nbr], dataMask: [valid] };
}
"""


def classify_severity(dnbr: Optional[float]) -> str:
    """Maps a dNBR value onto the USGS burn severity ladder."""
    if dnbr is None or (isinstance(dnbr, float) and np.isnan(dnbr)):
        return "UNKNOWN"
    if dnbr < -0.10:
        return "REGROWTH"
    for threshold, label in SEVERITY_BREAKS:
        if dnbr < threshold:
            return label
    return "HIGH_SEVERITY"


def bbox_around(lat: float, lon: float, radius_km: float = 1.0) -> List[float]:
    """Builds a WGS84 bounding box of the given radius around a point."""
    dlat = radius_km / 111.32
    cos_lat = max(np.cos(np.radians(lat)), 1e-6)
    dlon = radius_km / (111.32 * cos_lat)
    return [
        round(lon - dlon, 6),
        round(lat - dlat, 6),
        round(lon + dlon, 6),
        round(lat + dlat, 6),
    ]


def _window_days(date_from: str, date_to: str) -> int:
    """Days spanned by a query window, sized so one aggregation interval fits.

    The request covers date_from T00:00:00Z through date_to T23:59:59Z, i.e. one
    second under (date_to - date_from + 1) days. Returning the un-incremented
    difference keeps the interval strictly inside the range, which is what
    Sentinel Hub requires before it will emit an interval at all.
    """
    try:
        d0 = datetime.strptime(date_from, "%Y-%m-%d")
        d1 = datetime.strptime(date_to, "%Y-%m-%d")
    except (ValueError, TypeError):
        return 1
    return max((d1 - d0).days, 1)


def _pixel_dimensions(
    bbox: List[float],
    target_res_m: float = TARGET_RESOLUTION_M,
) -> Dict[str, int]:
    """Converts a WGS84 bbox into pixel width/height at a target ground resolution.

    Sentinel Hub accepts either resx/resy (in the units of the bounds CRS) or
    width/height (in pixels). Because the bbox here is in degrees, resx/resy
    would have to be expressed in degrees too -- a conversion that is easy to get
    silently wrong. Pixel counts avoid the ambiguity entirely.

    20m is chosen as the target because it is the native resolution of Sentinel-2
    band B12 (SWIR), one of the two bands in the NBR; sampling finer would only
    interpolate.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mid = (lat_min + lat_max) / 2.0

    height_m = (lat_max - lat_min) * 111_320.0
    width_m = (lon_max - lon_min) * 111_320.0 * max(np.cos(np.radians(lat_mid)), 1e-6)

    def _clamp(meters: float) -> int:
        px = int(round(meters / target_res_m))
        # Floor keeps statistics meaningful on a tiny AOI; ceiling respects the
        # Statistical API's maximum output dimension.
        return max(MIN_DIMENSION_PX, min(px, MAX_DIMENSION_PX))

    return {"width": _clamp(width_m), "height": _clamp(height_m)}


class DnbrCache:
    """Disk-backed cache of computed dNBR results.

    Sentinel-2 scenes for a past date never change, so a result is permanently
    reusable. This keeps repeat pipeline runs and the live dashboard off the
    CDSE API, which matters both for rate limits and for demo reliability.
    """

    def __init__(self, path: Path = CACHE_PATH):
        self.path = Path(path)
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info("Loaded %d cached dNBR results from %s", len(self._data), self.path)
            except Exception as e:
                logger.warning("Could not read dNBR cache (%s). Starting empty.", e)
                self._data = {}

    @staticmethod
    def make_key(lat: float, lon: float, event_date: str) -> str:
        # 3 decimal places is roughly 110m, finer than the 375m VIIRS pixel.
        return f"{round(float(lat), 3)}:{round(float(lon), 3)}:{event_date}"

    def get(self, lat: float, lon: float, event_date: str) -> Optional[Dict]:
        return self._data.get(self.make_key(lat, lon, event_date))

    def put(self, lat: float, lon: float, event_date: str, result: Dict) -> None:
        self._data[self.make_key(lat, lon, event_date)] = result

    def flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            logger.info("Persisted %d dNBR results to %s", len(self._data), self.path)
        except Exception as e:
            logger.warning("Could not persist dNBR cache: %s", e)


class Sentinel2Client:
    """Thin client over the CDSE Sentinel Hub Statistical API.

    The Statistical API is used rather than the Process API deliberately: it
    returns aggregated band statistics as JSON, so no GeoTIFF download or raster
    decoding is needed for a single scalar per detection.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        timeout: int = 45,
    ):
        # `None` means "read the environment"; an explicit "" means "no
        # credentials". Using `or` here would conflate the two, so an explicit
        # empty string would silently pick up whatever is in .env -- which makes
        # it impossible to construct a deliberately unconfigured client.
        self.client_id = (
            os.getenv("CDSE_CLIENT_ID", "") if client_id is None else client_id
        ).strip()
        self.client_secret = (
            os.getenv("CDSE_CLIENT_SECRET", "") if client_secret is None else client_secret
        ).strip()
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        # Raw payload of the most recent statistics call, for --debug inspection.
        self.last_raw_response: Optional[Dict] = None
        # Every statistics call from the most recent compute_dnbr, labelled by
        # window, so --debug shows the call that actually failed.
        self.debug_calls: List[Dict] = []
        # Seconds the server last asked us to wait before retrying. Exposed so a
        # batch caller can sleep out a quota window rather than give up.
        self.last_retry_after_s: float = 0.0
        self._last_request_at: float = 0.0

    def _post_with_backoff(self, body: Dict, token: str):
        """POSTs to the Statistics API, throttled and retrying on HTTP 429.

        Sentinel Hub meters both request rate and processing units. Firing a
        batch of requests back-to-back trips the limiter almost immediately, and
        a bare RATE_LIMITED result is indistinguishable to the caller from a
        genuine data gap, so the retry happens here rather than being surfaced.
        """
        delay = 1.0
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            # Space out consecutive calls regardless of outcome.
            since = time.time() - self._last_request_at
            if since < MIN_REQUEST_INTERVAL_S:
                time.sleep(MIN_REQUEST_INTERVAL_S - since)

            resp = requests.post(
                STATISTICS_URL,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
            self._last_request_at = time.time()

            if resp.status_code != 429:
                return resp

            if attempt == MAX_RATE_LIMIT_RETRIES:
                logger.warning("Rate limited by CDSE after %d retries; giving up on this request.",
                               MAX_RATE_LIMIT_RETRIES)
                return None

            requested = float(resp.headers.get("Retry-After", delay) or delay)
            self.last_retry_after_s = requested
            if requested > MAX_BACKOFF_S:
                logger.warning(
                    "CDSE asked for a %.0fs backoff, which exceeds the %.0fs cap. "
                    "The processing-unit quota is likely exhausted; abandoning this "
                    "request rather than blocking the pipeline.",
                    requested, MAX_BACKOFF_S,
                )
                return None

            wait = min(requested, MAX_BACKOFF_S)
            logger.info("Rate limited by CDSE; backing off %.1fs (attempt %d/%d).",
                        wait, attempt + 1, MAX_RATE_LIMIT_RETRIES)
            time.sleep(wait)
            delay *= 2

        return None

    def _record(self, label: str, http_status: int, payload: Dict, status: str) -> None:
        """Retains one statistics call for --debug inspection."""
        self.last_raw_response = payload
        self.debug_calls.append(
            {"window": label or "(unlabelled)", "http_status": http_status,
             "status": status, "payload": payload}
        )

    @property
    def is_configured(self) -> bool:
        """True when OAuth2 credentials are present."""
        return bool(self.client_id and self.client_secret)

    def _get_token(self) -> Optional[str]:
        """Fetches (and caches) an OAuth2 access token."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        if not self.is_configured:
            return None

        try:
            resp = requests.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            self._token_expiry = time.time() + float(payload.get("expires_in", 600))
            logger.info("Acquired CDSE access token (valid %ss).", payload.get("expires_in"))
            return self._token
        except Exception as e:
            logger.warning("CDSE authentication failed: %s", e)
            return None

    def mean_nbr(
        self,
        bbox: List[float],
        date_from: str,
        date_to: str,
        max_cloud: int = 60,
        label: str = "",
    ) -> Tuple[Optional[float], str]:
        """Returns the spatial mean NBR over a bbox and date window.

        Returns:
            (mean_nbr, status). mean_nbr is None whenever no usable observation
            exists, with the reason carried in status.
        """
        token = self._get_token()
        if token is None:
            return None, "NO_CREDENTIALS"

        body = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {"maxCloudCoverage": max_cloud},
                    }
                ],
            },
            "aggregation": {
                "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                # One interval spanning the whole window, so the result is a
                # single mean NBR for the period.
                #
                # A fixed "P30D" silently returns NO intervals whenever the
                # requested window is shorter than 30 days -- and the window runs
                # T00:00:00Z to T23:59:59Z, so a nominal 30-day window is one
                # second short of P30D and yields an empty `data` array. Sizing
                # the interval from the actual span avoids that cliff entirely.
                "aggregationInterval": {"of": f"P{_window_days(date_from, date_to)}D"},
                "evalscript": NBR_EVALSCRIPT,
                # Dimensions are given in PIXELS, not resx/resy.
                #
                # resx/resy are interpreted in the units of the bounds CRS. Our
                # bbox is EPSG:4326, so "resx": 20 requests 20 DEGREES per pixel,
                # which collapses the AOI to a single pixel and is rejected with
                # "Pixel size of 1998.29 meters per pixel exceeds the limit
                # 1500.00 meters per pixel for collection S2L2A".
                #
                # Pixel counts are CRS-independent and say exactly what we mean.
                **_pixel_dimensions(bbox),
            },
            "calculations": {"nbr": {"statistics": {"default": {}}}},
        }

        try:
            resp = self._post_with_backoff(body, token)
            if resp is None:
                self._record(label, 429, {"note": "rate limited after retries"}, "RATE_LIMITED")
                return None, "RATE_LIMITED"
            if resp.status_code in (401, 403):
                logger.warning(
                    "CDSE rejected the request (HTTP %s). The credentials are valid for "
                    "authentication but this account may not have Sentinel Hub API access "
                    "enabled. Response: %s",
                    resp.status_code, resp.text[:400],
                )
                self._record(label, resp.status_code, {"body": resp.text[:2000]}, "NOT_AUTHORIZED")
                return None, "NOT_AUTHORIZED"
            if resp.status_code >= 400:
                # The request body itself was rejected. This is the signature of an
                # evalscript or schema problem, which is a bug here, not a data gap.
                logger.warning("CDSE rejected the request body (HTTP %s): %s",
                               resp.status_code, resp.text[:600])
                self._record(label, resp.status_code, {"body": resp.text[:2000]}, "BAD_REQUEST")
                return None, "BAD_REQUEST"
            data = resp.json()
        except Exception as e:
            logger.warning("CDSE statistics request failed: %s", e)
            return None, "API_ERROR"

        # Retained so --debug can show exactly what came back. A parsing mistake and
        # genuine cloud cover both produce "no usable mean", and without the raw
        # payload those two are indistinguishable from the outside.
        self.last_raw_response = data

        intervals = data.get("data", [])
        if not intervals:
            self._record(label, 200, data, "NO_SCENES_IN_WINDOW")
            return None, "NO_SCENES_IN_WINDOW"

        means: List[float] = []
        saw_stats = False
        rejected_for_cloud = 0

        for interval in intervals:
            outputs = interval.get("outputs", {})
            bands = outputs.get("nbr", {}).get("bands", {})

            # The single output band is conventionally keyed "B0", but the key is
            # not worth hardcoding: if Sentinel Hub names it otherwise, hardcoding
            # would silently report every scene as cloudy. Take whatever is there.
            if not bands:
                continue
            band_stats = bands.get("B0") or next(iter(bands.values()), {})
            stats = band_stats.get("stats", {})
            if not stats:
                continue

            saw_stats = True
            sample_count = stats.get("sampleCount", 0) or 0
            no_data = stats.get("noDataCount", 0) or 0

            # Require at least 20% valid (non-cloud) pixels to trust the mean.
            if sample_count > 0 and (sample_count - no_data) / sample_count >= 0.20:
                mean_val = stats.get("mean")
                if mean_val is not None:
                    means.append(float(mean_val))
                    continue
            rejected_for_cloud += 1

        if means:
            self._record(label, 200, data, "OK")
            return float(np.mean(means)), "OK"

        if not saw_stats:
            # Scenes came back, but no statistics were found where expected. That is
            # a parsing/response-shape mismatch on our side, NOT a data gap, and it
            # must not be reported as cloud cover.
            logger.warning(
                "CDSE returned %d interval(s) but no parsable statistics. This is a "
                "response-shape mismatch, not cloud cover. Re-run with --debug to dump "
                "the raw payload.", len(intervals),
            )
            self._record(label, 200, data, "PARSE_MISMATCH")
            return None, "PARSE_MISMATCH"

        logger.info("All %d scene(s) rejected: insufficient clear pixels.", rejected_for_cloud)
        self._record(label, 200, data, "NO_CLEAR_SCENE")
        return None, "NO_CLEAR_SCENE"

    def compute_dnbr(
        self,
        lat: float,
        lon: float,
        event_date: str,
        radius_km: float = 1.0,
        pre_window_days: int = 45,
        post_window_days: int = 30,
        cache: Optional[DnbrCache] = None,
    ) -> Dict:
        """Computes dNBR for a single thermal detection.

        Args:
            lat, lon: Detection coordinates in WGS84.
            event_date: Detection date as 'YYYY-MM-DD'.
            radius_km: Half-width of the analysis box around the detection.
            pre_window_days: How far back to search for a clear pre-event scene.
            post_window_days: How far forward to search for a clear post-event scene.
            cache: Optional DnbrCache to read from and write to.

        Returns:
            Dict with dnbr, nbr_pre, nbr_post, severity, status, and the
            geometry/time window used. dnbr is None when unavailable.
        """
        if cache is not None:
            hit = cache.get(lat, lon, event_date)
            if hit is not None:
                return {**hit, "cached": True}

        result = {
            "lat": float(lat),
            "lon": float(lon),
            "event_date": event_date,
            "dnbr": None,
            "nbr_pre": None,
            "nbr_post": None,
            "severity": "UNKNOWN",
            "status": "UNKNOWN",
            "radius_km": radius_km,
            "cached": False,
        }

        if not self.is_configured:
            result["status"] = "NO_CREDENTIALS"
            return result

        try:
            event_dt = datetime.strptime(event_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            result["status"] = "BAD_DATE"
            return result

        # A post-event scene cannot exist until the satellite has revisited and
        # the L2A product has been published. Reporting this distinctly matters:
        # it is a "come back later" condition, not a permanent data gap, and it
        # is the normal state for near-real-time FIRMS detections.
        age_days = (datetime.now(timezone.utc) - event_dt).days
        if age_days < MIN_DAYS_FOR_POST_SCENE:
            result["status"] = "AWAITING_POST_SCENE"
            result["age_days"] = age_days
            result["retry_after_days"] = MIN_DAYS_FOR_POST_SCENE - age_days
            return result

        box = bbox_around(lat, lon, radius_km)

        pre_from = (event_dt - timedelta(days=pre_window_days)).strftime("%Y-%m-%d")
        pre_to = (event_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        post_from = (event_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        post_to = (event_dt + timedelta(days=post_window_days)).strftime("%Y-%m-%d")

        self.debug_calls = []
        nbr_pre, status_pre = self.mean_nbr(box, pre_from, pre_to, label="pre-event")
        nbr_post, status_post = self.mean_nbr(box, post_from, post_to, label="post-event")

        result["bbox"] = box
        result["pre_window"] = [pre_from, pre_to]
        result["post_window"] = [post_from, post_to]
        result["nbr_pre"] = nbr_pre
        result["nbr_post"] = nbr_post

        if nbr_pre is None or nbr_post is None:
            # Report whichever leg failed; cloud obscuration is the usual cause
            # during the Indian monsoon and must be surfaced, not hidden.
            result["status"] = status_pre if nbr_pre is None else status_post
            return result

        dnbr = round(nbr_pre - nbr_post, 4)
        result["dnbr"] = dnbr
        result["severity"] = classify_severity(dnbr)
        result["status"] = "OK"

        if cache is not None:
            cache.put(lat, lon, event_date, result)

        return result


def enrich_alerts_with_dnbr(
    df: pd.DataFrame,
    location_col: str = "recurrence_key",
    client: Optional[Sentinel2Client] = None,
    cache: Optional[DnbrCache] = None,
    max_locations: int = 500,
    patient: bool = False,
    max_quota_waits: int = 40,
    pre_window_days: int = 45,
    post_window_days: int = 30,
) -> pd.DataFrame:
    """Computes dNBR once per LOCATION and broadcasts it to that location's rows.

    dNBR is a property of a place and a moment, not of an individual satellite
    pixel: every detection of the same event at the same site shares one burn
    scar. Computing it per detection therefore pays repeatedly for one answer.

    This matters because the arithmetic decides what is possible. Validating
    every ambiguous detection in a national year costs ~630,000 API calls, which
    no tier supports. Restricting to the P0 alerts an operator actually reviews
    -- roughly 500 detections across ~370 locations from 2M -- costs under 800
    calls and finishes in minutes. Optical validation belongs at the alert stage,
    as corroboration before someone dispatches a response, not as a bulk feature
    pass over the training corpus.

    For each location the EARLIEST detection date is used, so the pre-event
    window sits before the event began rather than mid-burn.

    Args:
        df: Detections to enrich, typically filtered to high-priority alerts.
        location_col: Column identifying a physical location.
        client: Sentinel2Client; constructed from environment if omitted.
        cache: DnbrCache; created if omitted.
        max_locations: Hard ceiling on locations validated in this run. It
            counts locations resolved, not requests sent, so quota refusals --
            which return no result -- never consume the budget.
        patient: On quota exhaustion, sleep out the window the server asks for
            and resume, rather than abandoning the run. The free CDSE tier
            allows roughly 20 calls before requesting a ~5 minute pause, so a
            few hundred locations is a matter of waiting rather than a matter of
            impossibility. Results are cached as they are produced, so an
            interrupted run resumes without repeating work.
        max_quota_waits: Ceiling on how many quota windows to wait through,
            so a pathological server response cannot stall indefinitely.
        pre_window_days: How far back to search for a clear pre-event scene.
        post_window_days: How far forward to search for a clear post-event
            scene. A burn scar persists for months, so widening this trades
            promptness for coverage rather than validity -- a location that
            returns NO_CLEAR_SCENE over 30 days may well be measurable over 90,
            because Sentinel-2's ~5-day revisit gives roughly three times the
            chances of an unclouded overpass.

    Transport failures are retried in place rather than recorded: an unreachable
    host says nothing about whether a scene exists. After
    MAX_TRANSPORT_RETRIES attempts the location is abandoned, and after
    MAX_CONSECUTIVE_TRANSPORT_FAILURES such locations the run stops, leaving the
    remainder unattempted so a later run can recover them.

    Returns:
        A copy of df with dnbr, dnbr_severity and dnbr_status populated.
    """
    out = df.copy()
    for col, default in (("dnbr", np.nan), ("dnbr_severity", "UNKNOWN"),
                         ("dnbr_status", "NOT_REQUESTED")):
        if col not in out.columns:
            out[col] = default

    if out.empty:
        return out

    client = client or Sentinel2Client()
    cache = cache or DnbrCache()

    if not client.is_configured:
        logger.warning("CDSE credentials absent - skipping alert optical validation.")
        out["dnbr_status"] = "NO_CREDENTIALS"
        return out

    if location_col not in out.columns:
        logger.warning("No %s column; falling back to per-detection enrichment.", location_col)
        return enrich_with_dnbr(out, client=client, cache=cache)

    dates = pd.to_datetime(out.get("acq_date"), errors="coerce")
    work = pd.DataFrame({
        "loc": out[location_col].astype(str),
        "lat": pd.to_numeric(out["latitude"], errors="coerce"),
        "lon": pd.to_numeric(out["longitude"], errors="coerce"),
        "date": dates,
    })
    # Earliest detection per location marks event onset.
    first = work.sort_values("date").groupby("loc", as_index=False).first()
    logger.info(
        "Alert optical validation: %d locations covering %d detections (cap %d).",
        len(first), len(out), max_locations,
    )

    results: Dict[str, Dict] = {}
    issued = 0
    quota_waits = 0
    transport_retries = 0
    consecutive_transport_failures = 0
    pending = [r for r in first.itertuples(index=False)
               if not (pd.isna(r.date) or pd.isna(r.lat) or pd.isna(r.lon))]
    total = len(pending)
    i = 0

    while i < len(pending):
        if len(results) >= max_locations:
            logger.info("Reached max_locations=%d; remaining locations unvalidated.", max_locations)
            break

        row = pending[i]
        event_date = row.date.strftime("%Y-%m-%d")
        was_cached = cache.get(row.lat, row.lon, event_date) is not None
        res = client.compute_dnbr(
            float(row.lat), float(row.lon), event_date, cache=cache,
            pre_window_days=pre_window_days, post_window_days=post_window_days,
        )

        if res["status"] == "RATE_LIMITED":
            if not patient:
                logger.warning(
                    "CDSE quota exhausted after %d live call(s). Stopping; %d location(s) "
                    "validated so far are retained in the cache.", issued, len(results),
                )
                break

            if quota_waits >= max_quota_waits:
                logger.warning("Hit max_quota_waits=%d; stopping with %d/%d locations done.",
                               max_quota_waits, len(results), total)
                break

            quota_waits += 1
            # Honour what the server asked for, plus a small margin, with a
            # floor in case no Retry-After was supplied.
            wait = max(client.last_retry_after_s, 60.0) + 15.0
            cache.flush()  # never lose completed work to an interrupted sleep
            logger.info(
                "Quota window %d exhausted at %d/%d locations (%d live calls). "
                "Sleeping %.0fs for the next window...",
                quota_waits, len(results), total, issued, wait,
            )
            time.sleep(wait)
            continue  # retry this same location; do NOT advance

        if res["status"] == "API_ERROR":
            if transport_retries < MAX_TRANSPORT_RETRIES:
                transport_retries += 1
                backoff = min(TRANSPORT_BACKOFF_S * 2 ** (transport_retries - 1),
                              MAX_TRANSPORT_BACKOFF_S)
                logger.warning(
                    "Transport failure at %s (attempt %d/%d); retrying in %.0fs. "
                    "The location is not recorded as unmeasurable.",
                    row.loc, transport_retries, MAX_TRANSPORT_RETRIES, backoff,
                )
                time.sleep(backoff)
                continue  # retry this same location; do NOT advance

            consecutive_transport_failures += 1
            transport_retries = 0
            logger.warning(
                "Giving up on %s after %d transport failures (%d location(s) failing "
                "back to back).", row.loc, MAX_TRANSPORT_RETRIES,
                consecutive_transport_failures,
            )
            if consecutive_transport_failures >= MAX_CONSECUTIVE_TRANSPORT_FAILURES:
                cache.flush()
                logger.error(
                    "%d locations failed transport consecutively; the network is down, "
                    "not flaky. Stopping at %d/%d so the remaining %d location(s) stay "
                    "unattempted and recoverable rather than being consumed.",
                    consecutive_transport_failures, len(results), total,
                    total - len(results),
                )
                break
        else:
            consecutive_transport_failures = 0

        transport_retries = 0

        # Counted here, not before the rate-limit branch: a refusal consumes no
        # processing units and yields no result, so counting it would both
        # overstate usage and let repeated refusals exhaust the budget.
        if not was_cached:
            issued += 1

        results[row.loc] = res
        i += 1

        if len(results) % 25 == 0:
            ok = sum(1 for r in results.values() if r["status"] == "OK")
            logger.info("  ... %d/%d locations done, %d usable dNBR, %d quota wait(s).",
                        len(results), total, ok, quota_waits)
            cache.flush()

    if quota_waits:
        logger.info("Completed after waiting through %d quota window(s).", quota_waits)

    cache.flush()

    loc_series = out[location_col].astype(str)
    out["dnbr"] = loc_series.map(
        lambda k: results[k]["dnbr"] if k in results and results[k]["dnbr"] is not None else np.nan
    )
    out["dnbr_severity"] = loc_series.map(
        lambda k: results[k]["severity"] if k in results else "UNKNOWN"
    )
    out["dnbr_status"] = loc_series.map(
        lambda k: results[k]["status"] if k in results else "NOT_REQUESTED"
    )

    usable = int(out["dnbr"].notna().sum())
    logger.info(
        "Alert validation complete: %d/%d detections carry a dNBR, from %d live call(s).",
        usable, len(out), issued,
    )
    if results:
        from collections import Counter
        logger.info("  location outcomes: %s",
                    dict(Counter(r["status"] for r in results.values())))
    return out


def enrich_with_dnbr(
    df: pd.DataFrame,
    client: Optional[Sentinel2Client] = None,
    max_requests: int = 50,
    only_candidates: bool = True,
    cache: Optional[DnbrCache] = None,
) -> pd.DataFrame:
    """Adds a `dnbr` column to a detections DataFrame.

    Optical validation is deliberately rationed. Running it for every detection
    would burn the CDSE quota on routine flares that need no confirmation, so by
    default only ambiguous candidates are validated: detections that are inside
    or near an industrial polygon and lack an established recurrence baseline,
    which is precisely the population where wildfire-versus-accident is in doubt.

    Args:
        df: Detections with latitude, longitude, and acq_date / timestamp_utc.
        client: Sentinel2Client; constructed from environment if omitted.
        max_requests: Hard ceiling on live API calls for this run.
        only_candidates: Restrict validation to ambiguous detections.
        cache: DnbrCache instance; created if omitted.

    Returns:
        A copy of df with `dnbr`, `dnbr_severity`, and `dnbr_status` columns.
    """
    out = df.copy()
    if "dnbr" not in out.columns:
        out["dnbr"] = np.nan
    if "dnbr_severity" not in out.columns:
        out["dnbr_severity"] = "UNKNOWN"
    if "dnbr_status" not in out.columns:
        out["dnbr_status"] = "NOT_REQUESTED"

    if out.empty:
        return out

    client = client or Sentinel2Client()
    cache = cache or DnbrCache()

    if not client.is_configured:
        logger.warning(
            "CDSE credentials absent - skipping optical validation. "
            "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to enable Sentinel-2 dNBR."
        )
        out["dnbr_status"] = "NO_CREDENTIALS"
        return out

    # Select the ambiguous population worth spending API quota on.
    if only_candidates:
        inside = out.get("inside_industrial", pd.Series(False, index=out.index))
        if inside.dtype == "object":
            inside = inside.astype(str).str.lower().isin(["true", "1"])
        else:
            inside = inside.astype(bool)
        n30 = pd.to_numeric(out.get("n_30d", 0), errors="coerce").fillna(0)
        frp = pd.to_numeric(out.get("frp", 0.0), errors="coerce").fillna(0.0)
        candidates = out[(inside | (frp >= 10.0)) & (n30 < 8)].index
    else:
        candidates = out.index

    logger.info("Optical validation queue: %d candidate detections (cap %d).", len(candidates), max_requests)

    issued = 0
    skipped_too_recent = 0
    for idx in candidates:
        if issued >= max_requests:
            logger.info("Reached max_requests=%d; remaining detections left unvalidated.", max_requests)
            break

        row = out.loc[idx]
        event_date = str(row.get("acq_date", "") or "")[:10]
        if not event_date:
            ts = pd.to_datetime(row.get("timestamp_utc"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            event_date = ts.strftime("%Y-%m-%d")

        # Cheap pre-filter: a detection younger than the Sentinel-2 revisit cycle
        # cannot have a post-event scene, so do not spend an API call discovering
        # that. This is the normal case for a live NRT feed.
        try:
            ev_dt = datetime.strptime(event_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - ev_dt).days < MIN_DAYS_FOR_POST_SCENE:
                out.loc[idx, "dnbr_status"] = "AWAITING_POST_SCENE"
                skipped_too_recent += 1
                continue
        except (ValueError, TypeError):
            pass

        was_cached = cache.get(row["latitude"], row["longitude"], event_date) is not None
        res = client.compute_dnbr(
            lat=float(row["latitude"]),
            lon=float(row["longitude"]),
            event_date=event_date,
            cache=cache,
        )
        if not was_cached:
            issued += 1

        out.loc[idx, "dnbr"] = res["dnbr"] if res["dnbr"] is not None else np.nan
        out.loc[idx, "dnbr_severity"] = res["severity"]
        out.loc[idx, "dnbr_status"] = res["status"]

        # Once the quota is exhausted every further call will fail the same way.
        # Continuing would spend the rest of the run relearning that.
        if res["status"] == "RATE_LIMITED":
            logger.warning(
                "CDSE quota exhausted after %d call(s). Stopping optical validation "
                "for this run; previously cached results are retained and the "
                "remaining detections keep dnbr=NaN.", issued,
            )
            break

    cache.flush()

    validated = int(out["dnbr"].notna().sum())
    logger.info(
        "Optical validation complete: %d/%d detections carry a usable dNBR "
        "(%d API calls issued, %d skipped as too recent for a post-event scene).",
        validated, len(out), issued, skipped_too_recent,
    )
    if skipped_too_recent and validated == 0:
        logger.info(
            "All candidates were younger than the %d-day Sentinel-2 revisit window. "
            "dNBR is a retrospective measure: re-run this step once the detections "
            "have aged past a post-event overpass.",
            MIN_DAYS_FOR_POST_SCENE,
        )
    return out


# Benchmark events used by --validate. These are a matched discriminative pair:
# one fire that consumed vegetation, one that did not. If dNBR cannot separate
# these two, the feature is not earning its place in the model.
VALIDATION_EVENTS = [
    {
        "name": "Punjab stubble burning",
        "lat": 30.500,
        "lon": 75.800,
        "date": "2023-11-05",
        "expect": "positive",
        "pre_window_days": 30,
        "post_window_days": 30,
        "why": "Peak Kharif residue burning; dry season, so skies are reliably clear",
    },
    {
        "name": "Baghjan blowout (Assam)",
        "lat": 27.587,
        "lon": 95.385,
        "date": "2020-05-27",
        "expect": "positive",
        # The blowout burned until November 2020. A standard 30-day post window
        # lands squarely in the Assam monsoon and returns NO_CLEAR_SCENE, so the
        # windows are widened to straddle it: a dry pre-monsoon baseline against
        # a post-monsoon observation of the burn scar.
        "pre_window_days": 100,
        "post_window_days": 220,
        "why": "Gas well blowout that burned surrounding wetland vegetation for months",
    },
    {
        "name": "Jamnagar refinery (Gujarat)",
        "lat": 22.400,
        "lon": 70.050,
        "date": "2024-03-15",
        "expect": "near_zero",
        "pre_window_days": 45,
        "post_window_days": 30,
        "why": "Routine flaring inside a refinery - no biomass available to burn",
    },
]


def run_validation(client: "Sentinel2Client", debug: bool = False) -> int:
    """Runs the benchmark pair and reports whether dNBR separates them.

    Returns a process exit code: 0 if the pair separated as expected.
    """
    cache = DnbrCache()
    results = []

    for ev in VALIDATION_EVENTS:
        logger.info("-" * 70)
        logger.info("Validating: %s", ev["name"])
        logger.info("  %s", ev["why"])
        logger.info("  Expectation: dNBR %s", ev["expect"].replace("_", " "))

        res = client.compute_dnbr(
            ev["lat"], ev["lon"], ev["date"],
            pre_window_days=ev.get("pre_window_days", 45),
            post_window_days=ev.get("post_window_days", 30),
            cache=cache,
        )
        results.append((ev, res))

        logger.info("  -> status=%s  dNBR=%s  severity=%s",
                    res["status"], res["dnbr"], res["severity"])

        if debug:
            for call in client.debug_calls:
                logger.info(
                    "  [%s] HTTP %s -> %s\n%s",
                    call["window"], call["http_status"], call["status"],
                    json.dumps(call["payload"], indent=2)[:2500],
                )

    cache.flush()

    print()
    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    usable = [(e, r) for e, r in results if r["dnbr"] is not None]
    if len(usable) < len(VALIDATION_EVENTS):
        print("INCONCLUSIVE - not every benchmark returned a usable dNBR.\n")
        for e, r in results:
            print(f"  {e['name']:34s} status={r['status']}")
        print()
        print("  PARSE_MISMATCH    -> response shape differs from what this client expects.")
        print("                       Re-run with --debug and send the raw payload.")
        print("  NOT_AUTHORIZED    -> credentials authenticate, but this account lacks")
        print("                       Sentinel Hub API access.")
        print("  BAD_REQUEST       -> the evalscript or request schema was rejected.")
        print("  NO_CLEAR_SCENE    -> genuine cloud cover; try a different date.")
        print("  NO_SCENES_IN_WINDOW -> no acquisitions at all in the search window.")
        return 2

    burns = [r["dnbr"] for e, r in usable if e["expect"] == "positive"]
    contained_vals = [r["dnbr"] for e, r in usable if e["expect"] == "near_zero"]

    if not burns or not contained_vals:
        print("INCONCLUSIVE - need at least one event of each type to compare.")
        return 2

    for e, r in usable:
        print(f"  {e['name']:32s} dNBR = {r['dnbr']:+.4f}  ({r['severity']})")
    print()

    burn = max(burns)
    contained = contained_vals[0]
    print(f"  Best vegetation-consuming event : {burn:+.4f}")
    print(f"  Contained industrial event      : {contained:+.4f}")
    print(f"  Separation                      : {burn - contained:+.4f}")
    print()

    if burn > 0.10 and abs(contained) < 0.10 and (burn - contained) > 0.15:
        print("  PASS - dNBR separates a biomass fire from a contained industrial fire.")
        print("  The feature is discriminative on real data. Safe to enable in the pipeline.")
        return 0

    print("  WEAK - the two events did not separate as expected.")
    print("  Do not trust dNBR as a classification feature until this is understood.")
    print("  Possible causes: wrong dates for the event, an AOI that misses the burn")
    print("  scar, or B08/B12 being read in the wrong order.")
    return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sentinel-2 dNBR optical validation (SIH26162)",
        epilog=(
            "Setup: register a free OAuth client at https://dataspace.copernicus.eu "
            "(Dashboard -> User Settings -> OAuth clients), then put CDSE_CLIENT_ID and "
            "CDSE_CLIENT_SECRET in .env. Run --validate first to confirm the integration "
            "works before enabling it across the pipeline."
        ),
    )
    parser.add_argument("--lat", type=float, help="Detection latitude")
    parser.add_argument("--lon", type=float, help="Detection longitude")
    parser.add_argument("--date", type=str, help="Event date YYYY-MM-DD")
    parser.add_argument("--radius", type=float, default=1.0, help="Analysis radius in km")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the benchmark event pair and report whether dNBR separates them",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump the raw CDSE response. Use this on the first live call: a response-shape "
             "mismatch and genuine cloud cover both yield 'no usable dNBR', and only the raw "
             "payload distinguishes them.",
    )
    args = parser.parse_args()

    c = Sentinel2Client()
    if not c.is_configured:
        logger.error(
            "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set.\n"
            "  1. Register free at https://dataspace.copernicus.eu\n"
            "  2. Dashboard -> User Settings -> OAuth clients -> Create new\n"
            "  3. Add both values to .env (never paste them into a chat or commit them)\n"
            "  4. Re-run: python -m src.ingestion.sentinel2_client --validate --debug"
        )
        sys.exit(1)

    if args.validate:
        sys.exit(run_validation(c, debug=args.debug))

    if args.lat is None or args.lon is None or args.date is None:
        parser.error("--lat, --lon and --date are required unless --validate is used")

    outcome = c.compute_dnbr(args.lat, args.lon, args.date, radius_km=args.radius, cache=DnbrCache())
    print(json.dumps(outcome, indent=2))

    if args.debug and c.last_raw_response is not None:
        print()
        print("--- RAW CDSE RESPONSE ---")
        print(json.dumps(c.last_raw_response, indent=2)[:8000])
