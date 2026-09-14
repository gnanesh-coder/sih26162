"""Tactical Situation Report (SitRep) Generator for Project SIH26162.

Provides defense-grade situation reports for tactical watchstanders, disaster response
authorities (NDRF, Fire & Emergency Services), and NTRO command elements.

Supports:
  - Dual-coordinate referencing (WGS84 + UTM / MGRS)
  - Atmospheric dispersion & chemical emission modeling (SO2, NOx, CO2, PM2.5)
  - Uber H3 recurrence baseline comparison
  - Human-in-the-loop audit trail compilation
  - Standalone, publication-quality A4 printable HTML with `@media print` styling
  - Programmatic machine-readable JSON exports
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

try:
    import pyproj
except (ImportError, OSError):
    pyproj = None

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.database import AuditLog, H3Baseline, Incident

logger = logging.getLogger("sitrep_generator")


# MGRS Latitude Band definitions (8 deg intervals from 80 deg S to 84 deg N)
MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"


def _latlon_to_utm_pure(lat: float, lon: float, zone: int, is_north: bool) -> tuple[float, float]:
    """Pure Python WGS84 Transverse Mercator (UTM) conversion without native C dependencies."""
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = 2 * f - f**2
    e_prime2 = e2 / (1 - e2)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    T = math.tan(lat_rad)**2
    C = e_prime2 * math.cos(lat_rad)**2
    A = math.cos(lat_rad) * (lon_rad - lon0)

    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_rad)
    )

    x = k0 * N * (
        A
        + (1 - T + C) * A**3 / 6
        + (5 - 18 * T + T**2 + 72 * C - 58 * e_prime2) * A**5 / 120
    ) + 500000.0

    y = k0 * (
        M
        + N * math.tan(lat_rad) * (
            A**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * e_prime2) * A**6 / 720
        )
    )

    if not is_north:
        y += 10000000.0

    return x, y


def wgs84_to_utm_mgrs(lat: float, lon: float) -> Dict[str, Any]:
    """Converts WGS84 coordinates to UTM Zone, Hemispheric Easting/Northing, and MGRS string."""
    zone = int((lon + 180) / 6) + 1
    is_north = lat >= 0
    hemisphere = "north" if is_north else "south"

    # Band letter
    band_idx = int((lat + 80.0) / 8.0)
    band_letter = MGRS_BANDS[min(max(band_idx, 0), len(MGRS_BANDS) - 1)]

    # Transform to UTM projection
    if pyproj is not None:
        try:
            crs_utm = pyproj.CRS(f"+proj=utm +zone={zone} +{hemisphere} +datum=WGS84")
            transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
            easting, northing = transformer.transform(lon, lat)
        except Exception:
            easting, northing = _latlon_to_utm_pure(lat, lon, zone, is_north)
    else:
        easting, northing = _latlon_to_utm_pure(lat, lon, zone, is_north)

    # 100km square approximate MGRS representation
    east_int = int(easting)
    north_int = int(northing)
    grid_e = (east_int % 100000) // 10
    grid_n = (north_int % 100000) // 10
    mgrs_str = f"{zone}{band_letter} {grid_e:04d} {grid_n:04d}"

    return {
        "utm_zone": zone,
        "hemisphere": "N" if lat >= 0 else "S",
        "latitude_band": band_letter,
        "easting_meters": round(easting, 1),
        "northing_meters": round(northing, 1),
        "utm_string": f"Zone {zone}{'N' if lat >= 0 else 'S'} E:{easting:.0f}m N:{northing:.0f}m",
        "mgrs_string": mgrs_str,
    }


EMISSIONS_DISCLAIMER = (
    "ORDER-OF-MAGNITUDE ESTIMATE. The scaling factors below are hand-chosen to "
    "rank facility types against one another, not traced to EPA AP-42 or any "
    "other published emission-factor table. Treat the ratios between facility "
    "types as meaningful and the absolute mass rates as indicative only."
)


def estimate_chemical_emissions(frp_mw: float, facility_type: str) -> Dict[str, Any]:
    """Order-of-magnitude emission rates scaled from Fire Radiative Power.

    These are NOT AP-42 emission factors. They are hand-chosen coefficients that
    put a refinery above a brick kiln and a coal plant above open biomass, with
    sub-linear exponents so mass rate does not scale linearly with FRP. They
    were never derived from a published table, and earlier documentation
    describing them as AP-42 was wrong.

    Replacing them with sourced factors is worthwhile; presenting them as
    sourced when they are not is the part that had to stop.
    """
    ftype = (facility_type or "non_industrial").lower()
    frp_safe = max(frp_mw, 0.1)

    # Hand-chosen scaling factors -- see EMISSIONS_DISCLAIMER.
    if "petro" in ftype or "refin" in ftype or "oil" in ftype:
        co2_factor = 320.0  # kg/h per MW
        so2_factor = 1.45   # kg/h per MW^0.8
        nox_factor = 0.95   # kg/h per MW^0.8
        pm25_factor = 2.80  # kg/h per MW^0.7
        primary_hazard = "Volatile Hydrocarbons, SO2, Flare Soot"
    elif "steel" in ftype or "metal" in ftype or "smelt" in ftype:
        co2_factor = 410.0
        so2_factor = 2.10
        nox_factor = 1.15
        pm25_factor = 3.60
        primary_hazard = "Metallic Particulates (PM2.5/PM10), SO2, Carbon Monoxide"
    elif "chem" in ftype:
        co2_factor = 290.0
        so2_factor = 1.80
        nox_factor = 1.30
        pm25_factor = 2.10
        primary_hazard = "Hazardous Combustion Byproducts, NOx, Chemical Particulates"
    elif "power" in ftype or "coal" in ftype:
        co2_factor = 450.0
        so2_factor = 2.40
        nox_factor = 1.20
        pm25_factor = 3.20
        primary_hazard = "Fly Ash, High SO2/NOx Flue Exhaust"
    else:
        # Non-industrial / biomass / open field
        co2_factor = 260.0
        so2_factor = 0.18
        nox_factor = 0.35
        pm25_factor = 4.20
        primary_hazard = "Fine Smoke Particulates (PM2.5), Carbon Monoxide"

    co2_kg_h = round(co2_factor * frp_safe, 1)
    so2_kg_h = round(so2_factor * (frp_safe ** 0.8), 2)
    nox_kg_h = round(nox_factor * (frp_safe ** 0.8), 2)
    pm25_kg_h = round(pm25_factor * (frp_safe ** 0.7), 2)

    return {
        "co2_kg_h": co2_kg_h,
        "so2_kg_h": so2_kg_h,
        "nox_kg_h": nox_kg_h,
        "pm25_kg_h": pm25_kg_h,
        "primary_hazard": primary_hazard,
        "estimate_basis": "HAND_TUNED_SCALING",
        "estimate_disclaimer": EMISSIONS_DISCLAIMER,
    }


# Every field this module derives from a coordinate rather than from an
# observation. A consumer that renders any of these must render the disclosure
# alongside it; a consumer that cannot must not render them at all.
SYNTHETIC_WEATHER_FIELDS = (
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "wind_direction_deg", "wind_compass", "downwind_deg", "downwind_compass",
)

WEATHER_DISCLAIMER = (
    "NOT A MEASUREMENT. Temperature, humidity and wind are generated from the "
    "incident coordinates by a fixed formula -- no weather service is queried. "
    "They are placeholders that demonstrate the dispersion model, and the plume "
    "and evacuation figures derived from them are illustrative only. Do not use "
    "for operational decisions."
)


OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_TIMEOUT_S = 15.0

# Weather for a past hour at a fixed place never changes, so it is fetched once.
# Without this every SitRep render and every dashboard refresh would make a
# blocking outbound request, putting a third party's latency on the critical
# path of an incident briefing.
_WEATHER_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
_WEATHER_CACHE_MAX = 2048
_COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def _compass_of(degrees: float) -> str:
    return _COMPASS[int((float(degrees) + 11.25) / 22.5) % 16]


def fetch_observed_weather(
    lat: float, lon: float, when: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """Retrieves observed weather for an incident from the Open-Meteo archive.

    Open-Meteo's historical endpoint is free and needs no key, which is why it
    is wired here rather than left as a hook: the alternative was a formula over
    the incident coordinates presented to an operator as measurement.

    Returns None on any failure -- no network, no data for that date, a
    malformed response -- and the caller then falls back to the clearly-labelled
    synthetic values. Returning None is the honest failure: a partial or guessed
    observation would reintroduce exactly the problem this replaces.

    Args:
        lat, lon: Incident coordinates.
        when: Incident time. Anything pandas can parse, or None for today.
    """
    try:
        ts = pd.to_datetime(when, utc=True, errors="coerce") if when is not None else None
        if ts is None or pd.isna(ts):
            ts = pd.Timestamp.now(tz="UTC")
        day = ts.strftime("%Y-%m-%d")

        key = f"{round(float(lat), 3)}:{round(float(lon), 3)}:{day}:{int(ts.hour)}"
        if key in _WEATHER_CACHE:
            return _WEATHER_CACHE[key]

        resp = requests.get(
            OPEN_METEO_ARCHIVE,
            params={
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "start_date": day,
                "end_date": day,
                "hourly": ("temperature_2m,relative_humidity_2m,"
                           "wind_speed_10m,wind_direction_10m"),
                "timezone": "UTC",
            },
            timeout=WEATHER_TIMEOUT_S,
        )
        if resp.status_code != 200:
            logger.info("Weather archive returned HTTP %s; using synthetic values.",
                        resp.status_code)
            return None  # transient: deliberately not cached

        hourly = (resp.json() or {}).get("hourly") or {}
        temps = hourly.get("temperature_2m") or []
        if not temps:
            logger.info("Weather archive has no data for %s at %.3f,%.3f; "
                        "using synthetic values.", day, lat, lon)
            return None

        # The hour of the detection, not a daily mean: a plume disperses on the
        # wind that was blowing when the fire was observed.
        hour = min(int(ts.hour), len(temps) - 1)
        if temps[hour] is None:
            hour = next((i for i, v in enumerate(temps) if v is not None), None)
            if hour is None:
                return None

        wind_deg = float((hourly.get("wind_direction_10m") or [0])[hour] or 0)
        downwind = (wind_deg + 180) % 360
        observation = {
            "temperature_c": round(float(temps[hour]), 1),
            "humidity_pct": int((hourly.get("relative_humidity_2m") or [0])[hour] or 0),
            "wind_speed_kmh": round(float((hourly.get("wind_speed_10m") or [0])[hour] or 0), 1),
            "wind_direction_deg": int(wind_deg),
            "wind_compass": _compass_of(wind_deg),
            "downwind_deg": int(downwind),
            "downwind_compass": _compass_of(downwind),
            "observed_at": f"{(hourly.get('time') or [day])[hour]}Z",
        }
        if len(_WEATHER_CACHE) < _WEATHER_CACHE_MAX:
            _WEATHER_CACHE[key] = observation
        return observation
    except Exception as exc:  # noqa: BLE001 - weather must never take a SitRep down
        logger.info("Weather lookup failed (%s); using synthetic values.", exc)
        return None


def compute_atmospheric_dispersion(
    lat: float, lon: float, frp: float, when: Optional[Any] = None
) -> Dict[str, Any]:
    """Plume geometry from FRP, over weather that is synthetic unless a provider is wired.

    The plume length and evacuation radius are a real function of Fire Radiative
    Power. The *direction* they point, and the atmospheric context beside them,
    come from ``fetch_observed_weather`` if a provider exists and from a fixed
    formula over the coordinates if one does not.

    The returned dict always carries ``weather_source``, ``weather_is_measured``
    and, when synthetic, ``weather_disclaimer``. Callers must surface these.
    """
    # The incident's own time, not now. Without it a March fire would be
    # briefed with September weather and stamped OBSERVED -- a mislabelled
    # measurement, which is worse than the honestly-flagged synthetic values
    # this replaced.
    observed = fetch_observed_weather(lat, lon, when)

    if observed is not None:
        weather = {k: observed[k] for k in SYNTHETIC_WEATHER_FIELDS}
        weather["weather_source"] = "OBSERVED"
        weather["weather_is_measured"] = True
        weather["weather_observed_at"] = observed.get("observed_at")
    else:
        # Fixed formula over the coordinates. Stable per location so a demo is
        # reproducible, and wrong everywhere, which is why it is labelled.
        lat_factor = (lat - 8.0) / 28.0

        temp_c = round(32.0 + 4.0 * np.sin(lat * 0.5) - 3.0 * lat_factor, 1)
        humidity = int(np.clip(45 + 30 * np.cos(lon * 0.3) + 10 * np.sin(lat * 0.4), 30, 88))
        wind_speed = round(14.0 + 10.0 * abs(np.sin(lat * 1.2 + lon * 0.8)), 1)
        wind_deg = int((240 + 70 * np.sin(lat * 0.7)) % 360)

        compass_dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        compass = compass_dirs[int((wind_deg + 11.25) / 22.5) % 16]
        downwind_deg = (wind_deg + 180) % 360

        weather = {
            "temperature_c": temp_c,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind_speed,
            "wind_direction_deg": wind_deg,
            "wind_compass": compass,
            "downwind_deg": downwind_deg,
            "downwind_compass": compass_dirs[int((downwind_deg + 11.25) / 22.5) % 16],
            "weather_source": "SYNTHETIC_FORMULA",
            "weather_is_measured": False,
            "weather_disclaimer": WEATHER_DISCLAIMER,
        }

    # Plume length is a genuine function of FRP; its bearing is not, so the
    # geometry inherits the weather's provenance.
    plume_len_deg = float(np.clip(0.02 + np.sqrt(max(frp, 1.0)) * 0.008, 0.03, 0.20))
    hazard_dist_km = round(plume_len_deg * 111.32, 1)

    weather["hazard_length_km"] = hazard_dist_km
    weather["recommended_evac_radius_km"] = max(0.5, round(hazard_dist_km * 0.35, 1))
    weather["dispersion_is_illustrative"] = not weather["weather_is_measured"]
    return weather


class SitRepGenerator:
    """Compiles and renders tactical situation report briefings."""

    @staticmethod
    def generate_sitrep_data(incident: Incident, db: Session) -> Dict[str, Any]:
        """Gathers all telemetry and intelligence layers for an incident."""
        now_utc = datetime.now(timezone.utc)
        dtg_str = now_utc.strftime("%d%H%MZ %b %y").upper()

        # 1. Geospatial & Coordinate References
        grid_info = wgs84_to_utm_mgrs(incident.latitude, incident.longitude)

        # 2. H3 Baseline Recurrence Query
        baseline = db.query(H3Baseline).filter(H3Baseline.h3_index == incident.h3_index).first() if incident.h3_index else None
        if baseline and baseline.n_30d > 0:
            n_30d = baseline.n_30d
            mu_frp = baseline.mu_frp
            std_frp = max(math.sqrt(baseline.var_frp), 0.5)
            z_score = round((incident.frp - mu_frp) / std_frp, 2)
            surge_ratio = round(incident.frp / max(mu_frp, 0.1), 2)
        else:
            n_30d = 1 if incident.inside_industrial else 0
            mu_frp = 8.0 if incident.inside_industrial else incident.frp
            z_score = round(max(0.0, (incident.frp - 8.0) / 4.0), 2) if incident.inside_industrial else 0.0
            surge_ratio = round(incident.frp / 8.0, 2) if incident.inside_industrial else 1.0

        # 3. Weather & Dispersion Cone
        dispersion = compute_atmospheric_dispersion(
            incident.latitude, incident.longitude, incident.frp,
            when=incident.timestamp_utc,
        )

        # 4. Chemical Emission Estimates
        emissions = estimate_chemical_emissions(incident.frp, incident.facility_type or "non_industrial")

        # 5. Audit Trail History
        audit_records = (
            db.query(AuditLog)
            .filter(AuditLog.incident_id == incident.id)
            .order_by(desc(AuditLog.timestamp_utc))
            .limit(10)
            .all()
        )
        audit_trail = [a.to_dict() for a in audit_records]

        # 6. Thermal Radiative Severity
        if incident.frp >= 200.0:
            severity = "EXTREME (Catastrophic Radiative Thermal Flux)"
            severity_code = "CRITICAL"
        elif incident.frp >= 75.0:
            severity = "HIGH (Major Flareup / Industrial Anomaly)"
            severity_code = "HIGH"
        elif incident.frp >= 25.0:
            severity = "MODERATE (Elevated Operational Combustion)"
            severity_code = "MODERATE"
        else:
            severity = "LOW (Baseline Operational Level)"
            severity_code = "LOW"

        sitrep_id = f"SITREP-2026-F{incident.id:05d}"

        return {
            "report_metadata": {
                "sitrep_id": sitrep_id,
                "dtg": dtg_str,
                "generated_at_utc": now_utc.isoformat(),
                "classification": "RESTRICTED // OPERATIONAL INTELLIGENCE // SIH26162",
                "issuing_authority": "National Technical Research Organisation (NTRO) / SIH26162 Thermal Surveillance Grid",
                "operation_name": "PROJECT SIH26162 // OPERATION IND-FLAMEWATCH",
            },
            "incident": {
                "id": incident.id,
                "detection_id": incident.detection_id,
                "satellite": incident.satellite or "VIIRS S-NPP",
                "timestamp_utc": incident.timestamp_utc.strftime("%Y-%m-%d %H:%M:%S UTC") if incident.timestamp_utc else "UNKNOWN",
                "alert_priority": incident.alert_priority,
                "alert_state": incident.alert_state,
                "status": incident.status,
                "predicted_class": incident.predicted_class,
                "confidence_pct": round(incident.confidence * 100.0, 1),
                "shap_rationale": incident.shap_rationale or "Autonomous inference via XGBoost + TreeSHAP feature attribution.",
                "operator_notes": incident.operator_notes or "None logged to date.",
            },
            "geospatial": {
                "latitude": round(incident.latitude, 5),
                "longitude": round(incident.longitude, 5),
                "coordinates_wgs84": f"{incident.latitude:.5f}°N, {incident.longitude:.5f}°E",
                "utm_string": grid_info["utm_string"],
                "mgrs_string": grid_info["mgrs_string"],
                "h3_index": incident.h3_index or "N/A",
                "facility_name": incident.facility_name or "Unattributed Site / General Terrain",
                "facility_type": incident.facility_type or "non_industrial",
                "inside_industrial": bool(incident.inside_industrial),
                "dist_to_industrial_km": round(incident.dist_to_industrial_km, 3),
            },
            "thermal_telemetry": {
                "frp_mw": round(incident.frp, 2),
                "bright_ti4_k": round(incident.bright_ti4, 1) if incident.bright_ti4 else None,
                "bright_ti5_k": round(incident.bright_ti5, 1) if incident.bright_ti5 else None,
                "severity_level": severity,
                "severity_code": severity_code,
            },
            "recurrence_metrics": {
                "n_30d": n_30d,
                "mu_frp_mw": round(mu_frp, 2),
                "z_score": z_score,
                "surge_ratio": surge_ratio,
                "interpretation": (
                    f"Acute thermal surge: FRP is {surge_ratio:.1f}x higher than 30-day baseline ({mu_frp:.1f} MW, Z={z_score:.2f})."
                    if surge_ratio > 2.0 else
                    f"Within normal historical baseline boundaries ({mu_frp:.1f} MW, Z={z_score:.2f})."
                ),
            },
            "atmospheric_hazard": dispersion,
            "chemical_emissions": emissions,
            "audit_trail": audit_trail,
        }

    @staticmethod
    def render_html_sitrep(data: Dict[str, Any]) -> str:
        """Generates an A4 print-optimized, defense-grade tactical HTML document."""
        meta = data["report_metadata"]
        inc = data["incident"]
        geo = data["geospatial"]
        therm = data["thermal_telemetry"]
        rec = data["recurrence_metrics"]
        disp = data["atmospheric_hazard"]
        chem = data["chemical_emissions"]
        audits = data["audit_trail"]

        # A reader must not be able to mistake a formula for an observation, so
        # the provenance travels with the numbers rather than sitting in a
        # footnote they can skip.
        if disp.get("weather_is_measured"):
            weather_tag = f"Observed {disp.get('weather_observed_at', '')}".strip()
            weather_banner = ""
        else:
            weather_tag = "SYNTHETIC - not observed"
            weather_banner = (
                '<div style="border:2px solid var(--crimson-accent);'
                'background:rgba(190,30,45,.08);padding:10px 14px;margin-bottom:12px;'
                'font-weight:700;line-height:1.45;">'
                '&#9888; SYNTHETIC WEATHER &mdash; NOT A MEASUREMENT.<br>'
                '<span style="font-weight:400;">'
                + disp.get("weather_disclaimer", WEATHER_DISCLAIMER) +
                '</span></div>'
            )
        emissions_disclaimer = chem.get("estimate_disclaimer", EMISSIONS_DISCLAIMER)

        prio = inc["alert_priority"]
        if "P0" in prio:
            prio_badge_class = "badge-p0"
            status_desc = "IMMEDIATE EMERGENCY ACTION REQUIRED"
        elif "P1" in prio:
            prio_badge_class = "badge-p1"
            status_desc = "TACTICAL VERIFICATION & AIR MONITORING"
        elif "P2" in prio:
            prio_badge_class = "badge-p2"
            status_desc = "ROUTINE FACILITY MONITORING"
        else:
            prio_badge_class = "badge-p3"
            status_desc = "BACKGROUND THERMAL DETECTION"

        audit_rows_html = ""
        if audits:
            for a in audits:
                ts = str(a.get("timestamp_utc", "N/A"))[:19].replace("T", " ")
                action = a.get("action", "UNKNOWN")
                notes = a.get("operator_notes") or "Logged by C2 operator"
                prev_c = a.get("previous_class") or "ORIGINAL"
                new_c = a.get("new_class") or prev_c
                audit_rows_html += f"""
                <tr>
                  <td class="font-mono" style="color: #94a3b8;">{ts} UTC</td>
                  <td style="font-weight: 700;">{action}</td>
                  <td class="font-mono">{prev_c} &rarr; {new_c}</td>
                  <td>{notes}</td>
                </tr>
                """
        else:
            audit_rows_html = """
            <tr>
              <td colspan="4" style="text-align: center; color: #64748b; padding: 12px; font-style: italic;" class="font-mono">
                No human-in-the-loop interventions logged to date. Autonomous AI surveillance active.
              </td>
            </tr>
            """

        critical_class = "critical" if "P0" in prio else ""
        ti4_str = f"{therm['bright_ti4_k']} K" if therm.get("bright_ti4_k") else "N/A"
        ti5_str = f"{therm['bright_ti5_k']} K" if therm.get("bright_ti5_k") else "N/A"
        dist_str = "0m (Inside Perimeter)" if geo["inside_industrial"] else f"{geo['dist_to_industrial_km']} km from boundary"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meta["sitrep_id"]} // TACTICAL SITUATION REPORT</title>

  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

  <!-- FontAwesome -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />

  <style>
    :root {{
      --bg-chassis: #07090e;
      --bg-card: #0e121c;
      --border-color: #1f2738;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --amber-accent: #f59e0b;
      --crimson-accent: #ff2a5f;
      --emerald-accent: #10b981;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Space Grotesk', -apple-system, sans-serif;
      background-color: var(--bg-chassis);
      color: var(--text-main);
      line-height: 1.5;
      font-size: 13px;
      padding: 24px;
    }}

    .font-mono {{
      font-family: 'IBM Plex Mono', monospace;
    }}

    .action-toolbar {{
      max-width: 920px;
      margin: 0 auto 20px auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #111726;
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 10px 18px;
    }}

    .btn {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 12px;
      font-weight: 600;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: none;
      transition: all 0.2s ease;
    }}

    .btn-primary {{
      background: #f59e0b;
      color: #07090e;
    }}

    .btn-primary:hover {{
      background: #fbbf24;
    }}

    .btn-outline {{
      background: transparent;
      color: #cbd5e1;
      border: 1px solid #334155;
    }}

    .btn-outline:hover {{
      background: #1e293b;
      color: #fff;
    }}

    .sitrep-sheet {{
      max-width: 920px;
      margin: 0 auto;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 36px 40px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
    }}

    .security-banner {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 2px;
      text-align: center;
      padding: 6px;
      background: rgba(245, 158, 11, 0.1);
      color: #f59e0b;
      border: 1px dashed rgba(245, 158, 11, 0.6);
      border-radius: 6px;
      margin-bottom: 24px;
      text-transform: uppercase;
    }}

    .doc-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 2px solid var(--border-color);
      padding-bottom: 18px;
      margin-bottom: 24px;
    }}

    .title-block h1 {{
      font-size: 21px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.5px;
    }}

    .title-block .subtitle {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    .meta-block {{
      text-align: right;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      line-height: 1.6;
    }}

    .meta-label {{
      color: var(--text-muted);
    }}

    .meta-val {{
      color: #f8fafc;
      font-weight: 600;
    }}

    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    .badge-p0 {{
      background: rgba(255, 42, 95, 0.15);
      color: #ff2a5f;
      border: 1px solid #ff2a5f;
    }}

    .badge-p1 {{
      background: rgba(245, 158, 11, 0.15);
      color: #f59e0b;
      border: 1px solid #f59e0b;
    }}

    .badge-p2 {{
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid #38bdf8;
    }}

    .badge-p3 {{
      background: rgba(100, 116, 139, 0.2);
      color: #94a3b8;
      border: 1px solid #475569;
    }}

    .sitrep-section {{
      margin-bottom: 24px;
    }}

    .section-title {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 12px;
      font-weight: 700;
      color: var(--amber-accent);
      text-transform: uppercase;
      letter-spacing: 1px;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 6px;
      margin-bottom: 14px;
    }}

    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }}

    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}

    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
    }}

    .data-box {{
      background: #07090e;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 14px;
    }}

    .data-box-label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }}

    .data-box-val {{
      font-size: 14px;
      font-weight: 700;
      color: #ffffff;
    }}

    .data-box-sub {{
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    .threat-banner {{
      background: #111726;
      border-left: 4px solid var(--amber-accent);
      padding: 14px 18px;
      border-radius: 0 8px 8px 0;
      margin-bottom: 16px;
    }}

    .threat-banner.critical {{
      border-left-color: var(--crimson-accent);
      background: #1e0d14;
    }}

    .threat-banner-title {{
      font-size: 13px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 4px;
    }}

    .threat-banner-desc {{
      font-size: 12px;
      color: #cbd5e1;
    }}

    .table-container {{
      overflow-x: auto;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      background: #07090e;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      text-align: left;
    }}

    th {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      background: #111726;
      color: var(--text-muted);
      padding: 8px 12px;
      border-bottom: 1px solid var(--border-color);
    }}

    td {{
      padding: 8px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e2e8f0;
    }}

    tr:last-child td {{
      border-bottom: none;
    }}

    .doc-footer {{
      margin-top: 30px;
      padding-top: 16px;
      border-top: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px;
      color: var(--text-muted);
    }}

    @media print {{
      body {{
        background: #ffffff !important;
        color: #000000 !important;
        padding: 0 !important;
      }}

      .action-toolbar {{
        display: none !important;
      }}

      .sitrep-sheet {{
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        max-width: 100% !important;
        background: #ffffff !important;
      }}

      .security-banner {{
        background: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px dashed #0f172a !important;
      }}

      .doc-header {{
        border-bottom: 2px solid #0f172a !important;
      }}

      .title-block h1 {{
        color: #000000 !important;
      }}

      .meta-val, .data-box-val {{
        color: #000000 !important;
      }}

      .section-title {{
        color: #000000 !important;
        border-bottom: 1px solid #0f172a !important;
      }}

      .data-box {{
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
      }}

      .data-box-label, .meta-label {{
        color: #475569 !important;
      }}

      .threat-banner {{
        background: #f8fafc !important;
        border-left: 4px solid #0f172a !important;
      }}

      .threat-banner-title {{
        color: #000000 !important;
      }}

      .threat-banner-desc {{
        color: #334155 !important;
      }}

      .table-container {{
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
      }}

      th {{
        background: #f1f5f9 !important;
        color: #0f172a !important;
        border-bottom: 1px solid #cbd5e1 !important;
      }}

      td {{
        color: #0f172a !important;
        border-bottom: 1px solid #e2e8f0 !important;
      }}

      .badge-p0 {{
        background: #ffe4e6 !important;
        color: #9f1239 !important;
        border: 1px solid #9f1239 !important;
      }}

      .badge-p1 {{
        background: #fef3c7 !important;
        color: #92400e !important;
        border: 1px solid #92400e !important;
      }}

      .badge-p2 {{
        background: #e0f2fe !important;
        color: #0369a1 !important;
        border: 1px solid #0369a1 !important;
      }}

      .badge-p3 {{
        background: #f1f5f9 !important;
        color: #334155 !important;
        border: 1px solid #334155 !important;
      }}

      .doc-footer {{
        border-top: 1px solid #cbd5e1 !important;
        color: #64748b !important;
      }}

      @page {{
        size: A4 portrait;
        margin: 14mm 12mm 14mm 12mm;
      }}
    }}
  </style>
</head>
<body>

  <!-- Printable Action Toolbar -->
  <div class="action-toolbar">
    <div style="display: flex; align-items: center; gap: 12px;">
      <span class="badge {prio_badge_class}">{prio}</span>
      <span class="font-mono text-xs text-slate-300"><strong>{meta["sitrep_id"]}</strong></span>
    </div>
    <div style="display: flex; gap: 10px;">
      <button onclick="window.print()" class="btn btn-primary">
        <i class="fa-solid fa-print"></i> Print / Save as PDF
      </button>
      <button onclick="window.close()" class="btn btn-outline">
        <i class="fa-solid fa-xmark"></i> Close
      </button>
    </div>
  </div>

  <!-- Tactical SitRep Document Sheet -->
  <div class="sitrep-sheet">

    <!-- Top Security Classification Stamp -->
    <div class="security-banner">
      &#9632; {meta["classification"]} &#9632;
    </div>

    <!-- Header Block -->
    <header class="doc-header">
      <div class="title-block">
        <h1>TACTICAL SITUATION REPORT (SITREP)</h1>
        <div class="subtitle">{meta["operation_name"]}</div>
        <div class="subtitle" style="margin-top: 6px; color: #38bdf8;">
          <i class="fa-solid fa-shield-halved"></i> {meta["issuing_authority"]}
        </div>
      </div>
      <div class="meta-block">
        <div><span class="meta-label">REPORT ID:</span> <span class="meta-val">{meta["sitrep_id"]}</span></div>
        <div><span class="meta-label">DTG (UTC):</span> <span class="meta-val">{meta["dtg"]}</span></div>
        <div><span class="meta-label">SOURCE:</span> <span class="meta-val">{inc["satellite"]}</span></div>
        <div><span class="meta-label">STATUS:</span> <span class="meta-val">{inc["status"]}</span></div>
      </div>
    </header>

    <!-- SECTION 1: EXECUTIVE THREAT ASSESSMENT -->
    <section class="sitrep-section">
      <div class="section-title">
        <i class="fa-solid fa-triangle-exclamation"></i>
        <span>1. Executive Threat Assessment & Classification</span>
      </div>

      <div class="threat-banner {critical_class}">
        <div class="threat-banner-title">
          <span class="badge {prio_badge_class}">{prio}</span>
          <span style="margin-left: 8px;">{status_desc} &mdash; {inc["alert_state"]}</span>
        </div>
        <div class="threat-banner-desc" style="margin-top: 6px;">
          <strong>SHAP Decision Attribution:</strong> {inc["shap_rationale"]}
        </div>
      </div>

      <div class="grid-4">
        <div class="data-box">
          <div class="data-box-label">Inference Class</div>
          <div class="data-box-val font-mono">{inc["predicted_class"]}</div>
          <div class="data-box-sub">Confidence: <strong>{inc["confidence_pct"]}%</strong></div>
        </div>
        <div class="data-box">
          <div class="data-box-label">Fire Radiative Power</div>
          <div class="data-box-val font-mono" style="color: var(--crimson-accent);">{therm["frp_mw"]} MW</div>
          <div class="data-box-sub">{therm["severity_level"].split()[0]} Intensity</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">30-Day Surge Multiplier</div>
          <div class="data-box-val font-mono" style="color: var(--amber-accent);">{rec["surge_ratio"]}x Base</div>
          <div class="data-box-sub">Z-Score: <strong>+{rec["z_score"]} &sigma;</strong></div>
        </div>
        <div class="data-box">
          <div class="data-box-label">Operator Status</div>
          <div class="data-box-val font-mono">{inc["status"]}</div>
          <div class="data-box-sub font-mono">ID: #{inc["id"]}</div>
        </div>
      </div>
    </section>

    <!-- SECTION 2: GEOSPATIAL & FACILITY CORRELATION -->
    <section class="sitrep-section">
      <div class="section-title">
        <i class="fa-solid fa-location-crosshairs"></i>
        <span>2. Geospatial Intelligence & Critical Asset Correlation</span>
      </div>

      <div class="grid-2" style="margin-bottom: 12px;">
        <div class="data-box">
          <div class="data-box-label">Correlated Industrial Facility</div>
          <div class="data-box-val" style="font-size: 15px;">{geo["facility_name"]}</div>
          <div class="data-box-sub">
            Type: <strong>{geo["facility_type"]}</strong> | 
            Distance: <strong>{dist_str}</strong>
          </div>
        </div>
        <div class="data-box">
          <div class="data-box-label">Spatial Grid & H3 Hexagonal Cell</div>
          <div class="data-box-val font-mono" style="font-size: 13px; color: #38bdf8;">{geo["h3_index"]}</div>
          <div class="data-box-sub">Resolution 9 Uber Hex Cell (~100m precision)</div>
        </div>
      </div>

      <div class="grid-3">
        <div class="data-box">
          <div class="data-box-label">WGS84 Coordinates</div>
          <div class="data-box-val font-mono" style="font-size: 13px;">{geo["coordinates_wgs84"]}</div>
          <div class="data-box-sub">Decimal Degrees</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">UTM Coordinates</div>
          <div class="data-box-val font-mono" style="font-size: 12px;">{geo["utm_string"]}</div>
          <div class="data-box-sub">Universal Transverse Mercator</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">MGRS Grid Reference</div>
          <div class="data-box-val font-mono" style="font-size: 13px; color: var(--amber-accent);">{geo["mgrs_string"]}</div>
          <div class="data-box-sub">Military Grid Reference System</div>
        </div>
      </div>
    </section>

    <!-- SECTION 3: SATELLITE SENSOR & THERMAL TELEMETRY -->
    <section class="sitrep-section">
      <div class="section-title">
        <i class="fa-solid fa-satellite"></i>
        <span>3. Satellite Sensor & Thermal Radiative Metrics</span>
      </div>

      <div class="grid-4">
        <div class="data-box">
          <div class="data-box-label">Sensor Platform</div>
          <div class="data-box-val font-mono">{inc["satellite"]}</div>
          <div class="data-box-sub">Overpass: {inc["timestamp_utc"]}</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">Radiative Flux (MW)</div>
          <div class="data-box-val font-mono">{therm["frp_mw"]} MW</div>
          <div class="data-box-sub">{therm["severity_level"]}</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">TI4 Brightness Temp</div>
          <div class="data-box-val font-mono">{ti4_str}</div>
          <div class="data-box-sub">Mid-IR (3.75 &mu;m) Channel</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">TI5 Brightness Temp</div>
          <div class="data-box-val font-mono">{ti5_str}</div>
          <div class="data-box-sub">Long-Wave IR (11 &mu;m) Channel</div>
        </div>
      </div>
    </section>

    <!-- SECTION 4: ATMOSPHERIC DISPERSION & TOXIC EMISSIONS PROJECTION -->
    <section class="sitrep-section">
      <div class="section-title">
        <i class="fa-solid fa-wind"></i>
        <span>4. Atmospheric Dispersion & Chemical Emissions Hazard</span>
      </div>

      {weather_banner}

      <div class="grid-4" style="margin-bottom: 12px;">
        <div class="data-box">
          <div class="data-box-label">Wind Vector</div>
          <div class="data-box-val font-mono">{disp["wind_speed_kmh"]} km/h {disp["wind_compass"]}</div>
          <div class="data-box-sub">Heading: {disp["wind_direction_deg"]}&deg; &middot; {weather_tag}</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">Plume Downwind Trajectory</div>
          <div class="data-box-val font-mono" style="color: var(--amber-accent);">{disp["downwind_compass"]} ({disp["downwind_deg"]}&deg;)</div>
          <div class="data-box-sub">Opposite wind arrival vector &middot; {weather_tag}</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">Toxic Plume Reach</div>
          <div class="data-box-val font-mono" style="color: var(--crimson-accent);">{disp["hazard_length_km"]} km</div>
          <div class="data-box-sub">Smoke Dispersion Distance &middot; {weather_tag}</div>
        </div>
        <div class="data-box">
          <div class="data-box-label">Evacuation Perimeter</div>
          <div class="data-box-val font-mono" style="color: var(--crimson-accent);">{disp["recommended_evac_radius_km"]} km</div>
          <div class="data-box-sub">Recommended Safety Cordon &middot; {weather_tag}</div>
        </div>
      </div>

      <!-- Chemical Emissions Sub-Table -->
      <div style="border-left:3px solid var(--amber-accent);padding:8px 12px;margin-bottom:10px;font-size:11px;line-height:1.45;">
        <strong>Order-of-magnitude estimate.</strong> {emissions_disclaimer}
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Hazardous Pollutant / Gas</th>
              <th>Mass Release Rate (indicative)</th>
              <th>Health & Environmental Impact</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Sulfur Dioxide (SO<sub>2</sub>)</strong></td>
              <td class="font-mono">{chem["so2_kg_h"]} kg/h</td>
              <td>Acute respiratory irritant, acid aerosol precursor.</td>
            </tr>
            <tr>
              <td><strong>Nitrogen Oxides (NO<sub>x</sub>)</strong></td>
              <td class="font-mono">{chem["nox_kg_h"]} kg/h</td>
              <td>Toxic combustion gas, photochemical ozone precursor.</td>
            </tr>
            <tr>
              <td><strong>Carbon Dioxide (CO<sub>2</sub>)</strong></td>
              <td class="font-mono">{chem["co2_kg_h"]} kg/h</td>
              <td>Combustion footprint and atmospheric thermal load.</td>
            </tr>
            <tr>
              <td><strong>Fine Particulate Matter (PM<sub>2.5</sub>)</strong></td>
              <td class="font-mono">{chem["pm25_kg_h"]} kg/h</td>
              <td>Severe inhalable smoke particulates, reduced visibility.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- SECTION 5: COMMAND ACTIONS & AUDIT CHRONOLOGY -->
    <section class="sitrep-section">
      <div class="section-title">
        <i class="fa-solid fa-clipboard-check"></i>
        <span>5. Command Actions & Human-in-the-Loop Audit Trail</span>
      </div>

      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Timestamp (UTC)</th>
              <th>Action Executed</th>
              <th>State Transition</th>
              <th>Operator Operational Notes</th>
            </tr>
          </thead>
          <tbody>
            {audit_rows_html}
          </tbody>
        </table>
      </div>
    </section>

    <!-- Document Footer -->
    <footer class="doc-footer">
      <div>
        DOCUMENT SECURITY: <strong>{meta["classification"]}</strong>
      </div>
      <div>
        SYSTEM IDENTIFIER: <strong>PROJECT SIH26162 // NTRO C2 GRID</strong>
      </div>
      <div>
        GENERATED: <strong>{meta["dtg"]}</strong>
      </div>
    </footer>

    <!-- Bottom Security Classification Stamp -->
    <div class="security-banner" style="margin-top: 20px; margin-bottom: 0;">
      &#9632; {meta["classification"]} &#9632;
    </div>

  </div>

</body>
</html>
"""
        return html
