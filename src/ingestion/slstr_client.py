"""Sentinel-3 SLSTR fire-channel retrieval: the measured background for Dozier.

WHY THIS EXISTS
---------------
`src/pipeline/thermal_physics.py` implements the Dozier (1981) bi-spectral
retrieval correctly and is not wired into the pipeline, because when it was run
against the verified events every combustion regime collapsed into a 443-522 K
band -- a refinery flare, a steel furnace and a wheat-stubble fire all returned
roughly the same temperature, and a photovoltaic array returned 466 K where the
correct answer is that nothing is burning.

That module names the cause itself: the background temperature was *estimated*
from a low percentile of the thermal channel across a month/day-night stratum,
because the FIRMS active-fire product reports the fire pixel and nothing around
it. A mixture model with two unknowns cannot tolerate a guessed third term.

Sentinel-3 SLSTR supplies the missing term directly. It is a full-swath imager,
not a detection list, so the scene around a hot pixel is available and the
background can be *measured* on the same acquisition, through the same optics,
in the same atmosphere. It also carries dedicated fire channels:

    F1   3.74 um   mid-wave, does not saturate where the S7 channel does
    F2   10.85 um  thermal, dominated by background
    S7   3.74 um   the standard channel, clipped near 311 K by design

The saturation point matters as much as the background. VIIRS I-4 clips at
366.9 K and 3.0% of detections sit on that ceiling -- disproportionately the
large fires. F1 is designed for exactly those.

WHAT THIS MODULE CLAIMS, AND WHAT IT DOES NOT
---------------------------------------------
It claims to fetch F1/F2 brightness temperatures for a hot area and for an
annulus of undisturbed ground around it, and to hand both to the existing,
tested solver. It does not claim the retrieval succeeds: `--validate` runs the
same five sites that rejected the FIRMS-based attempt, so the two are directly
comparable, and the result is whatever it is.

One limitation is structural and worth stating before any number is read. SLSTR
resolves 1 km, against VIIRS at 375 m. A 10 m flare occupies about 1e-5 of an
SLSTR pixel and 7e-5 of a VIIRS pixel, so the sub-pixel dilution is roughly
seven times worse here. The measured background is a real improvement; the
resolution is a real regression. Which dominates is an empirical question, and
this module exists to answer it rather than assume it.

Credentials are the same CDSE OAuth2 client used for Sentinel-2 dNBR. No new
registration is required -- that was verified against a live token before this
module was written.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import requests
from dotenv import load_dotenv

# Load from the project root explicitly. `load_dotenv()` with no argument
# searches upward from the *calling frame's* file, so a helper script run from a
# scratch directory reads no .env at all and the token request comes back 401
# invalid_client -- indistinguishable from a revoked secret. That misdiagnosis
# cost this project an afternoon; the path is now pinned.
load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline.thermal_physics import (  # noqa: E402
    MAX_FIRE_TEMP_K,
    MIN_FIRE_TEMP_K,
    classify_combustion_regime,
    dozier_bispectral,
)

logger = logging.getLogger("slstr_client")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
STATISTICS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

COLLECTION = "sentinel-3-slstr"

# SLSTR fire-channel centre wavelengths, from the mission's band definitions.
F1_UM = 3.74
F2_UM = 10.85

# SLSTR's nadir thermal grid is 1 km. Requesting finer than the native grid
# resamples and invents no information, so the request resolution is pinned to
# it in degrees (1 km ~ 0.009 deg of latitude).
NATIVE_RES_DEG = 0.009

# The hot area: one to a few SLSTR pixels over the source.
DEFAULT_FIRE_RADIUS_KM = 1.5
# The background annulus: far enough out to exclude the plume and the facility's
# own warm infrastructure, close enough to share weather and surface type.
DEFAULT_BG_INNER_KM = 5.0
DEFAULT_BG_OUTER_KM = 15.0

# Background is taken as a low percentile of the annulus rather than its mean:
# the annulus can still contain roads, other plant and cloud edges, all of which
# bias the mean upward, and an inflated background suppresses the retrieval.
BACKGROUND_PERCENTILE = 25

MIN_REQUEST_INTERVAL_S = 0.4
MAX_RATE_LIMIT_RETRIES = 3
MAX_BACKOFF_S = 45.0

STATUS_OK = "OK"
STATUS_NO_CREDENTIALS = "NO_CREDENTIALS"
STATUS_NO_SCENE = "NO_SCENE"
STATUS_NO_SOLUTION = "NO_SOLUTION"
STATUS_API_ERROR = "API_ERROR"
STATUS_RATE_LIMITED = "RATE_LIMITED"

VALID_STATUSES = frozenset({
    STATUS_OK, STATUS_NO_CREDENTIALS, STATUS_NO_SCENE,
    STATUS_NO_SOLUTION, STATUS_API_ERROR, STATUS_RATE_LIMITED,
})

# dataMask is mandatory in the Statistical API's setup(); omitting it returns a
# 400 that names the output rather than the cause.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["F1", "F2", "S7", "dataMask"] }],
    output: [
      { id: "f1", bands: 1, sampleType: "FLOAT32" },
      { id: "f2", bands: 1, sampleType: "FLOAT32" },
      { id: "s7", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  return { f1: [s.F1], f2: [s.F2], s7: [s.S7], dataMask: [s.dataMask] };
}
"""


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _ring(lat: float, lon: float, radius_km: float, n: int = 48) -> List[List[float]]:
    """A closed lon/lat ring approximating a circle of the given radius."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    pts = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        pts.append([lon + dlon * math.cos(a), lat + dlat * math.sin(a)])
    pts.append(pts[0])
    return pts


def disc_geometry(lat: float, lon: float, radius_km: float) -> Dict[str, Any]:
    return {"type": "Polygon", "coordinates": [_ring(lat, lon, radius_km)]}


def annulus_geometry(lat: float, lon: float, inner_km: float, outer_km: float) -> Dict[str, Any]:
    """A ring with the source punched out.

    The hole is what makes this a background measurement rather than another
    observation of the fire. A plain disc centred on the source would include
    the hot pixels it is supposed to exclude, and the retrieval would then be
    solving for a fire against itself.
    """
    outer = _ring(lat, lon, outer_km)
    inner = list(reversed(_ring(lat, lon, inner_km)))  # opposite winding: a hole
    return {"type": "Polygon", "coordinates": [outer, inner]}


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class SlstrClient:
    """Thin client over the CDSE Statistical API for SLSTR fire channels."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        timeout: int = 90,
    ):
        # `None` means "read the environment"; "" means "deliberately no
        # credentials", so an unconfigured client can be constructed in tests.
        self.client_id = (
            os.getenv("CDSE_CLIENT_ID", "") if client_id is None else client_id
        ).strip()
        self.client_secret = (
            os.getenv("CDSE_CLIENT_SECRET", "") if client_secret is None else client_secret
        ).strip()
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._last_request_at: float = 0.0
        self.last_raw_response: Optional[Dict] = None

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _get_token(self) -> Optional[str]:
        if not self.is_configured():
            return None
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        try:
            resp = requests.post(TOKEN_URL, timeout=30, data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            })
        except requests.RequestException as exc:
            logger.error("CDSE token request failed: %s", exc)
            return None
        if resp.status_code != 200:
            logger.error("CDSE token request returned %s: %s",
                         resp.status_code, resp.text[:200])
            return None
        payload = resp.json()
        self._token = payload.get("access_token")
        self._token_expiry = time.time() + float(payload.get("expires_in", 600))
        return self._token

    def _post(self, body: Dict, token: str):
        delay = 1.0
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            since = time.time() - self._last_request_at
            if since < MIN_REQUEST_INTERVAL_S:
                time.sleep(MIN_REQUEST_INTERVAL_S - since)
            try:
                resp = requests.post(
                    STATISTICS_URL, json=body,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                logger.warning("SLSTR transport error: %s", exc)
                return None
            self._last_request_at = time.time()
            if resp.status_code != 429:
                return resp
            if attempt == MAX_RATE_LIMIT_RETRIES:
                return None
            wait = min(float(resp.headers.get("Retry-After", delay) or delay), MAX_BACKOFF_S)
            logger.info("Rate limited by CDSE; backing off %.1fs.", wait)
            time.sleep(wait)
            delay *= 2
        return None

    def _statistics(
        self,
        geometry: Dict[str, Any],
        date_from: str,
        date_to: str,
        token: str,
        interval: str = "P1D",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        body = {
            "input": {
                "bounds": {
                    "geometry": geometry,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [{"type": COLLECTION, "dataFilter": {}}],
            },
            "aggregation": {
                "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                "aggregationInterval": {"of": interval},
                "resx": NATIVE_RES_DEG, "resy": NATIVE_RES_DEG,
                "evalscript": EVALSCRIPT,
            },
            "calculations": {
                "default": {
                    "statistics": {
                        "default": {"percentiles": {"k": [10, 25, 50, 75, 90]}}
                    }
                }
            },
        }
        resp = self._post(body, token)
        if resp is None:
            return STATUS_RATE_LIMITED, []
        if resp.status_code != 200:
            logger.warning("SLSTR statistics %s: %s", resp.status_code, resp.text[:220])
            return STATUS_API_ERROR, []
        self.last_raw_response = resp.json()
        return STATUS_OK, self.last_raw_response.get("data", []) or []

    # ----------------------------------------------------------------------

    def retrieve_temperature(
        self,
        lat: float,
        lon: float,
        date_from: str,
        date_to: str,
        fire_radius_km: float = DEFAULT_FIRE_RADIUS_KM,
        bg_inner_km: float = DEFAULT_BG_INNER_KM,
        bg_outer_km: float = DEFAULT_BG_OUTER_KM,
    ) -> Dict[str, Any]:
        """Retrieves sub-pixel fire temperature from SLSTR F1/F2 over a window.

        The hottest single acquisition in the window is used, not the mean: a
        flare is intermittent and averaging over a month buries it under the
        days it was not burning.

        Returns a dict that always carries `status`. Temperature fields are
        present only when status is OK, and are never coerced to a number to
        keep the shape uniform -- a fabricated 0 K is worse than a missing key.
        """
        out: Dict[str, Any] = {
            "status": STATUS_NO_CREDENTIALS,
            "latitude": lat, "longitude": lon,
            "window": [date_from, date_to],
            "collection": COLLECTION,
            "background_basis": "MEASURED_ANNULUS",
        }
        token = self._get_token()
        if token is None:
            return out

        status, fire_data = self._statistics(
            disc_geometry(lat, lon, fire_radius_km), date_from, date_to, token)
        if status != STATUS_OK:
            out["status"] = status
            return out

        obs = _best_acquisition(fire_data)
        if obs is None:
            out["status"] = STATUS_NO_SCENE
            out["detail"] = "SLSTR returned no valid F1/F2 samples for this window."
            return out

        interval_from = obs["from"][:10]
        # The annulus is requested over the *same window* rather than the single
        # hot day. A from/to spanning one day returns zero intervals from this
        # API even where a month-long request returns that day populated, so a
        # narrow re-request would look like a data gap that is not there.
        status, bg_data = self._statistics(
            annulus_geometry(lat, lon, bg_inner_km, bg_outer_km),
            date_from, date_to, token)
        bg_obs = (_best_acquisition(bg_data, prefer="background", match_day=interval_from)
                  if status == STATUS_OK else None)
        if bg_obs is None:
            out["status"] = STATUS_NO_SCENE
            out["detail"] = ("The hot acquisition was found but its background annulus "
                             "returned no samples, so there is nothing to solve against.")
            return out

        t_bg = bg_obs["f2_background"]
        t_f1 = obs["f1_max"]
        t_f2 = obs["f2_max"]

        t_fire, area = dozier_bispectral(
            np.array([t_f1]), np.array([t_f2]), np.array([t_bg]), F1_UM, F2_UM)
        t_fire_v, area_v = float(t_fire[0]), float(area[0])

        out.update({
            "acquisition_date": interval_from,
            "f1_max_k": round(t_f1, 2),
            "f2_max_k": round(t_f2, 2),
            "s7_max_k": round(obs["s7_max"], 2) if obs.get("s7_max") is not None else None,
            "s7_f1_gap_k": (round(t_f1 - obs["s7_max"], 2)
                            if obs.get("s7_max") is not None else None),
            "background_k": round(t_bg, 2),
            "background_samples": bg_obs["n"],
            "excess_f1_over_background_k": round(t_f1 - t_bg, 2),
        })

        if not np.isfinite(t_fire_v) or not np.isfinite(area_v):
            out["status"] = STATUS_NO_SOLUTION
            out["detail"] = (
                "No admissible (T_fire, area) pair reproduces both channels within "
                f"{MIN_FIRE_TEMP_K:.0f}-{MAX_FIRE_TEMP_K:.0f} K. At 1 km the source may "
                "simply be too small to lift F1 measurably above the measured background."
            )
            return out

        out["status"] = STATUS_OK
        out["fire_temperature_k"] = round(t_fire_v, 1)
        out["area_fraction"] = area_v
        out["fire_area_m2"] = round(area_v * 1.0e6, 1)   # 1 km pixel
        out["combustion_regime"] = str(classify_combustion_regime(np.array([t_fire_v]))[0])
        return out


def _best_acquisition(
    data: List[Dict[str, Any]],
    prefer: str = "fire",
    match_day: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Picks the acquisition to solve from, and pulls the numbers out of it.

    For the source, the hottest F1 in the window: a flare is intermittent, and a
    window mean is dominated by the days it was not burning.

    For the background, `match_day` pins it to the *same* acquisition as the
    source. Solving a hot pixel from one day against ambient ground from another
    would silently mix weather, and the retrieval would attribute the difference
    to fire.
    """
    best = None
    for interval in data:
        outputs = interval.get("outputs", {})

        def stats(key: str) -> Dict[str, Any]:
            return ((outputs.get(key, {}) or {}).get("bands", {})
                    .get("B0", {}) or {}).get("stats", {}) or {}

        f1, f2, s7 = stats("f1"), stats("f2"), stats("s7")
        n = int(f1.get("sampleCount", 0)) - int(f1.get("noDataCount", 0))
        if n <= 0 or f1.get("max") is None or f2.get("max") is None:
            continue

        day = (interval.get("interval", {}) or {}).get("from", "")[:10]
        if match_day is not None and day != match_day:
            continue

        rec = {
            "from": (interval.get("interval", {}) or {}).get("from", ""),
            "n": n,
            "f1_max": float(f1["max"]),
            "f2_max": float(f2["max"]),
            "s7_max": float(s7["max"]) if s7.get("max") is not None else None,
            "f2_background": float(
                (f2.get("percentiles", {}) or {}).get(str(BACKGROUND_PERCENTILE))
                or (f2.get("percentiles", {}) or {}).get(f"{BACKGROUND_PERCENTILE}.0")
                or f2.get("mean", f2["max"])
            ),
        }
        if prefer == "background":
            return rec
        if best is None or rec["f1_max"] > best["f1_max"]:
            best = rec
    return best


# --------------------------------------------------------------------------
# Validation -- the same five sites that rejected the FIRMS-based retrieval
# --------------------------------------------------------------------------

VALIDATION_SITES = [
    # name, lat, lon, window, what a working retrieval should say
    ("Reliance Jamnagar flare", 22.3450, 69.8700, ("2026-03-01", "2026-03-31"),
     "1700-2000 K"),
    ("JSW Vijayanagar furnace", 15.1700, 76.6400, ("2026-03-01", "2026-03-31"),
     "1500-1900 K"),
    ("Visakhapatnam Steel", 17.6300, 83.1900, ("2026-03-01", "2026-03-31"),
     "1500-1900 K"),
    ("Punjab crop burning", 30.7500, 75.5000, ("2025-10-15", "2025-11-15"),
     "800-1000 K"),
    ("Khavda SOLAR PARK", 23.8000, 71.0800, ("2026-03-01", "2026-03-31"),
     "no combustion -- must not solve"),
]


def run_validation(client: SlstrClient) -> int:
    """Runs the retrieval over known regimes and prints the comparison table.

    Exit code is 0 whatever the numbers say. This is a measurement, not a test
    of the code, and a non-zero exit would invite someone to "fix" it by
    loosening the physics.
    """
    if not client.is_configured():
        print("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not set; nothing to validate.")
        return 0

    print()
    print(f"{'site':<28}{'F1max':>8}{'S7max':>8}{'bg':>8}{'excess':>8}"
          f"{'T_fire':>9}{'regime':>22}   expected")
    print("-" * 118)
    results = []
    for name, lat, lon, (d0, d1), expected in VALIDATION_SITES:
        res = client.retrieve_temperature(lat, lon, d0, d1)
        results.append((name, res, expected))
        if res["status"] in (STATUS_OK, STATUS_NO_SOLUTION):
            t = res.get("fire_temperature_k")
            print(f"{name[:27]:<28}"
                  f"{res.get('f1_max_k', float('nan')):>8.1f}"
                  f"{(res.get('s7_max_k') if res.get('s7_max_k') is not None else float('nan')):>8.1f}"
                  f"{res.get('background_k', float('nan')):>8.1f}"
                  f"{res.get('excess_f1_over_background_k', float('nan')):>8.2f}"
                  f"{(t if t is not None else float('nan')):>9.1f}"
                  f"{res.get('combustion_regime', res['status']):>22}   {expected}")
        else:
            print(f"{name[:27]:<28}{res['status']:>63}   {expected}")
    print("-" * 118)
    solved = [r for _, r, _ in results if r["status"] == STATUS_OK]
    print(f"{len(solved)} of {len(results)} sites produced a solution.")
    if solved:
        temps = [r["fire_temperature_k"] for r in solved]
        print(f"retrieved temperature spread: {min(temps):.0f} - {max(temps):.0f} K")
        print("A retrieval that cannot separate a gas flare from a wheat field is not "
              "a temperature measurement, whatever the numbers look like.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sentinel-3 SLSTR fire-channel retrieval.")
    parser.add_argument("--validate", action="store_true",
                        help="Run the five known-regime sites and print the table.")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--from", dest="date_from")
    parser.add_argument("--to", dest="date_to")
    args = parser.parse_args(argv)

    client = SlstrClient()
    if args.validate:
        return run_validation(client)
    if args.lat is None or args.lon is None or not args.date_from:
        parser.error("give --validate, or --lat/--lon/--from [--to]")
    print(json.dumps(
        client.retrieve_temperature(args.lat, args.lon, args.date_from,
                                    args.date_to or args.date_from),
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
