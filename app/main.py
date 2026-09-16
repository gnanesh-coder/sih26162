"""FastAPI Serving Backend for Project SIH26162.

Provides high-throughput, low-latency REST endpoints for:
  - GIS Dashboard Web Serving (/ & /static)
  - System Health & Telemetry Status (/api/v1/health)
  - Active HOTSPOT and Incident Retrieval (/api/v1/alerts/active)
  - Operator Incident Lifecycle Management (/api/v1/incident/{id}/status)
  - Executive & Tactical KPI Analytics (/api/v1/stats)
  - Telemetry Pipeline Synchronization (/api/v1/sync)
  - Real-Time XGBoost + TreeSHAP HOTSPOT Inference (/api/v1/classify)
"""

import json
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from shapely.geometry import Point
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.database import (
    AuditLog,
    DATABASE_URL,
    OFFLINE_SQLITE,
    H3Baseline,
    Incident,
    IS_POSTGRES,
    get_db,
    incidents_within_km,
    init_db,
    postgis_version,
    SessionLocal,
)
from src.alerting.sitrep_generator import SitRepGenerator, compute_atmospheric_dispersion
from src.alerting.dispatcher import AlertDispatcher
from src.ingestion.slstr_client import SlstrClient
from src.pipeline import auto_refresh
from src.reporting.compliance_register import build_compliance_register
from src.reporting.hex_aggregator import (
    DEFAULT_RESOLUTION,
    MAX_CELLS,
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    aggregate_hexes,
)
from src.alerting.state_machine import H3RecurrenceTracker, latlng_to_h3
from src.models.explainability import FireExplainer, parse_rationale_factors
from src.models.train_classifier import CLASS_NAMES, METRICS_PATH, apply_serving_guards
from src.models.verified_labels import (
    CLASS_NAME_TO_INDEX,
    KNOWN_UNDETECTED_INCIDENTS,
    SERVED_ONLY_CLASSES,
    VERIFIED_EVENTS,
)
from src.pipeline import replay_mode
from src.pipeline.responders import (
    RESPONDER_KINDS,
    layer_summary as responder_layer_summary,
    nearest_responders,
)
from src.pipeline.spatial_join import (
    DEFAULT_MERGED_PARQUET,
    DEFAULT_OSM_PARQUET,
    OUTPUT_PROCESSED_PARQUET,
    classify_facility_type,
)

logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Ensure database tables exist immediately
init_db()

# Global in-memory singletons for sub-15ms response latencies
osm_gdf: Optional[gpd.GeoDataFrame] = None
explainer: Optional[FireExplainer] = None
recurrence_tracker: Optional[H3RecurrenceTracker] = None


def get_explainer() -> FireExplainer:
    """Returns singleton FireExplainer instance, initializing if needed."""
    global explainer
    if explainer is None:
        explainer = FireExplainer()
    return explainer


def get_osm_gdf() -> Optional[gpd.GeoDataFrame]:
    """Returns singleton industrial polygons GeoDataFrame (fused OSM + ISRO Bhuvan), loading if needed."""
    global osm_gdf
    if osm_gdf is None:
        if DEFAULT_MERGED_PARQUET.exists():
            logger.info("Loading unified OSM + Bhuvan reference layer (%s)...", DEFAULT_MERGED_PARQUET)
            osm_gdf = gpd.read_parquet(DEFAULT_MERGED_PARQUET)
        elif DEFAULT_OSM_PARQUET.exists():
            logger.info("Loading OSM reference layer (%s)...", DEFAULT_OSM_PARQUET)
            osm_gdf = gpd.read_parquet(DEFAULT_OSM_PARQUET)
    return osm_gdf


def get_recurrence_tracker() -> H3RecurrenceTracker:
    """Returns singleton H3RecurrenceTracker instance, initializing if needed."""
    global recurrence_tracker
    if recurrence_tracker is None:
        recurrence_tracker = H3RecurrenceTracker()
    return recurrence_tracker


# The authoritative optical-coverage record: every P0 alert with its dNBR
# result, including the ones that could not be measured.
P0_DNBR_PARQUET = Path("data/processed/p0_alerts_dnbr.parquet")


def resolve_audit_corpus() -> Path:
    """The corpus a long-window audit should read.

    Precedence: an explicit `PROCESSED_CORPUS` always wins -- if an operator
    named a file, serve that file. Otherwise take the largest processed corpus
    on disk, because the live pipeline's output is a 1-2 day slice and a
    compliance register built from it reports an empty page with status OK.

    This is deliberately *not* the global default. `seed_database_from_parquet`
    walks its input row by row and must keep reading the live file.
    """
    if os.getenv("PROCESSED_CORPUS", "").strip():
        return OUTPUT_PROCESSED_PARQUET

    candidates = sorted(
        (p for p in Path("data/processed").glob("firms_industrial_joined*.parquet")
         if p.exists()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not candidates:
        return OUTPUT_PROCESSED_PARQUET

    chosen = candidates[0]
    if chosen != OUTPUT_PROCESSED_PARQUET:
        logger.info(
            "Audit endpoints reading %s (%.0f MB) rather than the live %s -- "
            "a compliance register needs an observation window, not a snapshot. "
            "Set PROCESSED_CORPUS to override.",
            chosen, chosen.stat().st_size / 1e6, OUTPUT_PROCESSED_PARQUET,
        )
    return chosen


def _optional_float(value) -> Optional[float]:
    """A float, or None when the corpus does not carry the field.

    None is the point: a missing pixel footprint must stay missing rather than
    defaulting to a nominal size, because the whole reason to store it is that
    the real one varies with scan angle.
    """
    if value is None or pd.isnull(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def seed_database_from_parquet(db: Session, parquet_path: Path = OUTPUT_PROCESSED_PARQUET) -> int:
    """Populates Incident and H3Baseline tables from processed FIRMS parquet."""
    if not parquet_path.exists():
        logger.warning("Processed parquet file %s does not exist. Skipping seeding.", parquet_path)
        return 0

    logger.info("Reading processed detections from %s...", parquet_path)
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return 0

    # Ensure explainer is initialized for industrial explanations
    exp = get_explainer()

    existing_ids = {row[0] for row in db.query(Incident.detection_id).all()}
    records_to_insert = []
    h3_counts: Dict[str, List[float]] = {}

    for idx, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        sat = str(row.get("satellite", "VIIRS"))
        ts_val = row.get("timestamp_utc")
        if pd.isnull(ts_val):
            ts = datetime.now(timezone.utc)
        elif isinstance(ts_val, (pd.Timestamp, datetime)):
            ts = ts_val.to_pydatetime() if hasattr(ts_val, "to_pydatetime") else ts_val
        else:
            ts = pd.to_datetime(ts_val, utc=True).to_pydatetime()

        det_id = f"{sat}_{lat:.4f}_{lon:.4f}_{ts.strftime('%Y%m%d%H%M')}_{idx}"
        if det_id in existing_ids:
            continue

        frp = float(row.get("frp", 0.0))
        ti4 = float(row.get("bright_ti4", 0.0)) if pd.notnull(row.get("bright_ti4")) else None
        ti5 = float(row.get("bright_ti5", 0.0)) if pd.notnull(row.get("bright_ti5")) else None
        inside_ind = bool(row.get("inside_industrial", False))
        fac_name = str(row.get("facility_name")) if pd.notnull(row.get("facility_name")) else None
        fac_type = str(row.get("facility_type", "non_industrial"))
        dist_km = float(row.get("dist_to_industrial_km", 999.0))
        h3_idx = str(row.get("h3_index", ""))

        if h3_idx:
            if h3_idx not in h3_counts:
                h3_counts[h3_idx] = []
            h3_counts[h3_idx].append(frp)

        priority = str(row.get("priority", "NON_ALERT"))
        state = str(row.get("state", "TRANSIENT_SUSPICION"))

        # Every detection is classified.
        #
        # This used to be gated on `inside_ind`, so only detections inside a
        # mapped industrial polygon reached the model. The gate was a cost
        # optimisation and it had a consequence nobody had looked at: a crop
        # burn is by definition NOT inside an industrial polygon, so
        # AGRICULTURAL_BURN was filtered out before the classifier ever saw it.
        # The model holds 207,521 agricultural training examples and had no
        # live path to ever predict one. FOREST_FIRE was unreachable for the
        # same reason, being an upgrade applied to AGRICULTURAL_BURN.
        #
        # Two of five classes were structurally absent from the running system.
        #
        # What replaces the gate is a cheaper split rather than a cheaper
        # filter: predict everything, and spend TreeSHAP only on what becomes
        # an alert. Attribution costs ~30 ms and is worth it for a detection a
        # human will open; it is not worth it for background thermal activity
        # nobody will ever click.
        alerting = priority in ("P0_EMERGENCY", "P1_ALERT", "P2_ADVISORY")
        try:
            sample_dict = {
                "frp": frp,
                "bright_ti4": ti4 if ti4 else 350.0,
                "bright_ti5": ti5 if ti5 else 295.0,
                "scan": float(row.get("scan", 0.4)),
                "track": float(row.get("track", 0.4)),
                "daynight": str(row.get("daynight", "D")),
                "timestamp_utc": ts.isoformat(),
                "inside_industrial": inside_ind,
                "is_exact_match": bool(row.get("is_exact_match", False)),
                "dist_to_industrial_km": dist_km,
                "facility_type": fac_type,
                "n_30d": int(row.get("n_30d", 0)),
                "mu_frp": float(row.get("mu_frp", frp)),
                "z_frp": float(row.get("z_frp", 0.0)),
                "frp_ratio": float(row.get("frp_ratio", 1.0)),
            }
            explainer_inst = get_explainer()
            if alerting:
                exp = explainer_inst.explain_detection(sample_dict)
                rationale = exp["narrative_rationale"]
            else:
                exp = explainer_inst.predict_detection(sample_dict)
                rationale = str(row.get(
                    "rationale",
                    "Classified without SHAP attribution: suppressed as "
                    "routine, so the per-factor explanation is computed on "
                    "demand rather than for every background detection.",
                ))
            model_class = exp["predicted_class"]
            conf = exp["confidence_percent"] / 100.0

            # The guards matter far more now than they did behind the gate.
            # Classifying everything means solar farms, forest and cropland
            # all reach the model, and those are exactly the cases it
            # cannot judge for itself: it is coordinate-free by design and
            # its corpus is entirely Indian. An override is recorded in the
            # rationale rather than applied silently.
            pred_class, guard_note = apply_serving_guards(
                model_class, lat=lat, lon=lon, facility_type=fac_type,
            )
            if guard_note:
                rationale = f"{rationale} [serving guard] {guard_note}"

        except Exception as e:
            # A classifier that failed did not produce a class, and must not be
            # made to look as though it did. The previous version fell back to
            # a state-machine guess with a hard-coded 0.95 confidence -- the
            # same fabrication that CONTROLLED_PROCESS carried, in a rarer
            # branch where it would have been harder to notice.
            logger.warning(
                "Classification failed for detection %s (%s). Recording it as "
                "NOT_ASSESSED rather than guessing.", det_id, e,
            )
            pred_class = "NOT_ASSESSED"
            conf = None
            rationale = (
                f"Classification failed ({type(e).__name__}). No class and no "
                "confidence were computed for this detection."
            )

        incident = Incident(
            detection_id=det_id,
            latitude=lat,
            longitude=lon,
            frp=frp,
            bright_ti4=ti4,
            bright_ti5=ti5,
            timestamp_utc=ts,
            satellite=sat,
            # Pixel footprint, carried through so the map can draw where the
            # fire actually might be rather than a point it cannot justify.
            # Absent in a corpus built before these columns existed, and left
            # absent rather than guessed.
            scan_km=_optional_float(row.get("scan")),
            track_km=_optional_float(row.get("track")),
            inside_industrial=inside_ind,
            facility_name=fac_name,
            facility_type=fac_type,
            dist_to_industrial_km=dist_km,
            h3_index=h3_idx,
            predicted_class=pred_class,
            confidence=conf,
            alert_priority=priority,
            alert_state=state,
            shap_rationale=rationale,
            status="OPEN",
        )
        records_to_insert.append(incident)

    if records_to_insert:
        db.add_all(records_to_insert)
        db.commit()
        logger.info("Successfully committed %d incidents to database.", len(records_to_insert))

    # Update H3 baselines
    for h3_cell, frp_list in h3_counts.items():
        existing_b = db.query(H3Baseline).filter(H3Baseline.h3_index == h3_cell).first()
        if not existing_b:
            b = H3Baseline(
                h3_index=h3_cell,
                n_30d=len(frp_list),
                mu_frp=float(np.mean(frp_list)),
                var_frp=float(np.var(frp_list)) if len(frp_list) > 1 else 1.0,
            )
            db.add(b)
        else:
            existing_b.n_30d += len(frp_list)
            all_vals = [existing_b.mu_frp] * (existing_b.n_30d - len(frp_list)) + frp_list
            existing_b.mu_frp = float(np.mean(all_vals))
            existing_b.var_frp = float(np.var(all_vals)) if len(all_vals) > 1 else 1.0
    db.commit()

    return len(records_to_insert)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database schema, preloads GIS boundaries and ML model."""
    logger.info("--- STARTING PROJECT SIH26162 FASTAPI BACKEND ---")

    # 1. Initialize DB Schema
    init_db()

    # A quiet corpus mismatch is the most expensive configuration error here.
    # PROCESSED_CORPUS defaults to whatever a short pipeline run last wrote, and
    # a 2-day file serves every endpoint a plausible-looking answer built from
    # 0.07% of the data -- the compliance register reported 465 routine
    # detections instead of 485,781 this way, and said nothing. Say something.
    _bigger = [p for p in Path("data/processed").glob("firms_industrial_joined*.parquet")
               if p.exists() and p.stat().st_size > OUTPUT_PROCESSED_PARQUET.stat().st_size * 2] \
        if OUTPUT_PROCESSED_PARQUET.exists() else []
    if _bigger:
        largest = max(_bigger, key=lambda p: p.stat().st_size)
        logger.warning(
            "Serving from %s, but %s is %.0fx larger. Set PROCESSED_CORPUS in .env "
            "to point at the archive you mean to demonstrate.",
            OUTPUT_PROCESSED_PARQUET, largest,
            largest.stat().st_size / max(OUTPUT_PROCESSED_PARQUET.stat().st_size, 1),
        )

    # Keep the corpus current without anyone remembering to.
    #
    # The freshness badge degrades honestly to STALE when nothing is ingested,
    # which is right for a badge and wrong for a surveillance system. This
    # closes the loop. It runs on a worker thread and never blocks startup: if
    # FIRMS is unreachable the dashboard still opens, serving what it has, with
    # the badge saying how old that is.
    auto_refresh.start_scheduler()

    # 2. Warm up singletons
    get_explainer()
    get_osm_gdf()
    get_recurrence_tracker()

    # 3. Auto-seed if database is empty
    db = SessionLocal()
    try:
        count = db.query(Incident).count()
        if count == 0 and OUTPUT_PROCESSED_PARQUET.exists():
            logger.info("Incident database is empty. Auto-seeding from %s...", OUTPUT_PROCESSED_PARQUET)
            seeded = seed_database_from_parquet(db)
            logger.info("Auto-seeded %d incidents.", seeded)
    finally:
        db.close()

    yield
    logger.info("Shutting down Project SIH26162 FastAPI backend...")


# FastAPI Application Instance
app = FastAPI(
    title="Project SIH26162: Industrial Fire Classifier & Thermal Surveillance",
    description=(
        "Autonomous NTRO surveillance platform fusing NASA FIRMS telemetry, "
        "OpenStreetMap/Bhuvan industrial boundaries, Uber H3 recurrence tracking, "
        "and XGBoost + TreeSHAP classification for zero-false-alarm fire detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Templates & Static
TEMPLATES_DIR = Path("app/templates")
STATIC_DIR = Path("app/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# -------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# -------------------------------------------------------------------------
class HotspotInferenceRequest(BaseModel):
    latitude: float = Field(..., description="WGS84 Latitude", ge=-90.0, le=90.0)
    longitude: float = Field(..., description="WGS84 Longitude", ge=-180.0, le=180.0)
    frp: float = Field(..., description="Fire Radiative Power in MW", ge=0.0)
    bright_ti4: Optional[float] = Field(350.0, description="VIIRS I4 / MODIS B21/22 Brightness Temp (K)")
    bright_ti5: Optional[float] = Field(295.0, description="VIIRS I5 / MODIS B31 Brightness Temp (K)")
    scan: Optional[float] = Field(0.4, description="Satellite scan pixel size (km)")
    track: Optional[float] = Field(0.4, description="Satellite track pixel size (km)")
    daynight: Optional[str] = Field("N", description="'D' for Day, 'N' for Night")
    timestamp_utc: Optional[str] = Field(None, description="ISO-format UTC timestamp string")
    facility_type_override: Optional[str] = Field(None, description="Manual facility type override for scenario modeling")
    inside_industrial_override: Optional[bool] = Field(None, description="Manual industrial perimeter override")
    wind_speed_kmh: Optional[float] = Field(None, description="Manual wind speed in km/h for dispersion simulation")
    wind_direction_deg: Optional[float] = Field(None, description="Manual wind direction (0-360 deg) for dispersion simulation")
    recurrence_n_30d_override: Optional[int] = Field(None, description="Manual 30-day recurrence count override")


class StatusUpdateRequest(BaseModel):
    status: str = Field(..., description="New lifecycle status: OPEN, ACKNOWLEDGED, DISPATCHED, RESOLVED, MUTED_ROUTINE")
    operator_notes: Optional[str] = Field(None, description="Operational notes or dispatch reference ID")


# -------------------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["UI Dashboard"])
def serve_dashboard(request: Request):
    """Serves the Tactical Leaflet GIS Incident Dashboard."""
    google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"google_maps_api_key": google_maps_key},
    )


@app.get("/api/v1/health", tags=["System"])
def get_health(db: Session = Depends(get_db)):
    """Returns runtime health, model state, and database configuration."""
    incident_count = db.query(Incident).count()
    baseline_count = db.query(H3Baseline).count()
    polys = get_osm_gdf()
    exp = get_explainer()
    google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

    return {
        "status": "healthy",
        "service": "Project SIH26162 Thermal Surveillance",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "database_mode": "postgis" if IS_POSTGRES else "sqlite_offline",
        "database_url": DATABASE_URL.split("@")[-1],
        "postgis_version": postgis_version(),
        "spatial_index": "GiST on incidents.geom" if IS_POSTGRES else None,
        # SQLite is an explicit offline choice, never an automatic fallback --
        # see app/database.py. Reported so the mode is never invisible.
        "offline_mode": OFFLINE_SQLITE,
        "records_count": {
            "incidents": incident_count,
            "h3_baselines": baseline_count,
        },
        "model_loaded": exp is not None,
        "osm_polygons_loaded": len(polys) if polys is not None else 0,
        "has_google_maps_key": bool(google_maps_key),
    }


@app.get("/api/v1/alerts/active", tags=["Surveillance"])
def get_active_incidents(
    priority: Optional[str] = Query(None, description="Filter by priority (P0_EMERGENCY, P1_ALERT, P2_ADVISORY, NON_ALERT)"),
    industrial_only: bool = Query(False, description="Restrict to hotspots inside industrial boundaries"),
    status: Optional[str] = Query(None, description="Filter by lifecycle status (OPEN, ACKNOWLEDGED, etc.)"),
    limit: int = Query(2000, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    """Retrieves active fire incidents and detections for map rendering."""
    query = db.query(Incident)

    if priority:
        query = query.filter(Incident.alert_priority == priority.upper())
    if industrial_only:
        query = query.filter(Incident.inside_industrial.is_(True))
    if status:
        query = query.filter(Incident.status == status.upper())

    # Prioritize emergency and industrial alerts first, then most recent
    query = query.order_by(
        desc(Incident.inside_industrial),
        desc(Incident.frp),
        desc(Incident.timestamp_utc),
    ).limit(limit)

    incidents = [inc.to_dict() for inc in query.all()]
    return {
        "status": "success",
        "total": len(incidents),
        "incidents": incidents,
    }


@app.get("/api/v1/incident/{incident_id}", tags=["Surveillance"])
def get_incident_details(incident_id: int, db: Session = Depends(get_db)):
    """Detailed record for one incident, including its per-feature attribution.

    `shap_factors` is the stored rationale parsed back into structured
    contributions. It is an **empty list** for detections the state machine
    classified rather than the model, and a caller that draws bars must draw
    none of them in that case: the dashboard previously rendered three fixed
    values -- +0.72, +1.14, -0.34 -- for every incident, under a heading that
    said "TreeSHAP attribution". Identical numbers on every target is a
    fabricated measurement, not an explanation.
    """
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")
    payload = inc.to_dict()
    payload["shap_factors"] = parse_rationale_factors(inc.shap_rationale)
    payload["shap_basis"] = ("TREESHAP_STORED" if payload["shap_factors"]
                             else "NO_PER_FEATURE_ATTRIBUTION")
    payload["recurrence"] = _recurrence_evidence(inc, db)
    return payload


# Detections per 30-day window above which the labelling rule treats a source as
# an established baseline rather than an event. Mirrors train_classifier.
PERSISTENCE_THRESHOLD = 8


def _recurrence_evidence(inc: Incident, db: Session) -> Dict[str, Any]:
    """The observed history of the H3 cell this detection sits in.

    This is the evidence the suppression logic actually reasons from, and it was
    nowhere on the dashboard. The dossier instead printed `frp / 8.0` under the
    label "x baseline" -- a constant divided into the current reading, which is
    not a baseline comparison at all: it has no knowledge of what this cell
    normally does, so a 40 MW refinery flare and a 40 MW depot fire scored
    identically at 5.0x.

    The real figures are already stored per cell by the recurrence tracker.
    """
    baseline = (db.query(H3Baseline)
                .filter(H3Baseline.h3_index == inc.h3_index).first()) if inc.h3_index else None
    if baseline is None:
        return {
            "status": "NO_BASELINE",
            "detail": "No recurrence history for this cell; it has been observed once.",
        }

    mu = float(baseline.mu_frp or 0.0)
    var = float(baseline.var_frp or 0.0)
    sigma = math.sqrt(var) if var > 0 else 0.0
    frp = float(inc.frp or 0.0)
    z = (frp - mu) / sigma if sigma > 1e-6 else None

    return {
        "status": "OK",
        "h3_index": inc.h3_index,
        "detections_30d": int(baseline.n_30d or 0),
        "mean_frp_mw": round(mu, 2),
        "sigma_frp_mw": round(sigma, 2),
        "z_frp": round(z, 2) if z is not None else None,
        "persistence_threshold": PERSISTENCE_THRESHOLD,
        "is_established": int(baseline.n_30d or 0) >= PERSISTENCE_THRESHOLD,
        "basis": (
            "Observed history of this ~174 m H3 cell over the loaded corpus. "
            "The threshold is calibrated for a 30-day window; a corpus shorter "
            "than that cannot reach it, and no source here should be read as "
            "established on this evidence alone."
        ),
    }


@app.get("/api/v1/incident/{incident_id}/optical-validation", tags=["Surveillance", "Validation"])
def validate_incident_optically(incident_id: int, db: Session = Depends(get_db)):
    """Runs Sentinel-2 dNBR burn-scar validation for an incident.

    Answers the question thermal data alone cannot: did this fire consume the
    surrounding vegetation, or was it contained inside the facility? A near-zero
    dNBR at a high-FRP detection is strong corroboration of an industrial event;
    a large positive dNBR points to a wildfire or crop burn instead.
    """
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    from src.ingestion.sentinel2_client import DnbrCache, Sentinel2Client

    client = Sentinel2Client()
    if not client.is_configured:
        return {
            "incident_id": incident_id,
            "status": "NO_CREDENTIALS",
            "dnbr": None,
            "severity": "UNKNOWN",
            "detail": (
                "Sentinel-2 optical validation is not configured. Register a free "
                "OAuth client at https://dataspace.copernicus.eu and set "
                "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET in .env to enable it."
            ),
        }

    event_dt = inc.timestamp_utc or datetime.now(timezone.utc)
    result = client.compute_dnbr(
        lat=float(inc.latitude),
        lon=float(inc.longitude),
        event_date=event_dt.strftime("%Y-%m-%d"),
        cache=DnbrCache(),
    )

    interpretation = {
        "UNBURNED": "No burn scar detected - consistent with a fire contained inside the facility.",
        "LOW_SEVERITY": "Minimal vegetation loss - likely contained, with some perimeter scorching.",
        "MODERATE_LOW": "Moderate vegetation loss - fire has breached the facility perimeter.",
        "MODERATE_HIGH": "Substantial burn scar - open-ground combustion is dominant.",
        "HIGH_SEVERITY": "Severe burn scar - consistent with a wildfire or large crop burn.",
        "REGROWTH": "Negative dNBR indicates vegetation gain, not fire. Treat the thermal detection as suspect.",
        "UNKNOWN": "No usable Sentinel-2 observation for this location and date window.",
    }.get(result["severity"], "")

    return {
        "incident_id": incident_id,
        **result,
        "interpretation": interpretation,
    }


@app.get("/api/v1/incident/{incident_id}/sitrep", tags=["Surveillance", "Briefings"])
def get_incident_sitrep(
    incident_id: int,
    format: str = Query("html", description="Output format: 'html' for printable A4 view, 'json' for data payload"),
    db: Session = Depends(get_db),
):
    """Generates an official tactical Situation Report (SitRep) for an incident."""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    sitrep_data = SitRepGenerator.generate_sitrep_data(inc, db)
    if format.lower() == "json":
        return sitrep_data

    html_content = SitRepGenerator.render_html_sitrep(sitrep_data)
    return HTMLResponse(content=html_content, status_code=200)


# One dispatcher for the process, so its in-session dedupe actually spans
# requests: two clicks on the same incident must not page the same desk twice.
_DISPATCHER: Optional[AlertDispatcher] = None


def get_dispatcher() -> AlertDispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = AlertDispatcher()
    return _DISPATCHER


@app.get("/api/v1/compliance/register", tags=["Compliance"])
def compliance_register(
    min_detections: int = Query(10, ge=1, description="Omit sites below this many detections"),
    limit: int = Query(100, ge=1, le=1000, description="Max sites returned, largest total FRP first"),
):
    """Routine industrial flaring, logged for CPCB compliance auditing.

    The alerting table in the problem statement gives routine flaring its own
    row: passive logging rather than dispatch. This is that row.

    It deliberately reports the detections the pipeline **suppressed**. A
    refinery flare correctly withheld from an incident commander is exactly the
    record an auditor wants -- operationally unremarkable, and a continuous
    emission. Regulators have historically depended on operator self-reporting
    for this; satellite observation is independent of the operator, which is
    what makes it an audit.

    Sites are identified by OSM polygon, not by name: thousands of polygons
    share the placeholder "Unnamed Industrial Site", and grouping on the name
    would attribute one operator's emissions to another. Non-combustion land
    uses (solar, wind) are excluded -- they have nothing to emit.

    Detection counts, observed days and FRP come from the sensor. Fire Radiative
    Energy is derived with a stated assumption. **Flared gas volume and CO2 are
    not computed**, because no calibrated combustion efficiency or heating value
    has been established here, and an uncalibrated figure under a regulatory
    heading would be a fabricated measurement.
    """
    corpus = resolve_audit_corpus()
    if not corpus.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Processed corpus {corpus} not found; run the pipeline first.",
        )

    cols = ["state", "frp", "latitude", "longitude", "acq_date", "facility_name",
            "facility_type", "osm_id", "recurrence_key", "suppressed"]
    available = set(pq.ParquetFile(corpus).schema_arrow.names)
    df = pd.read_parquet(corpus, columns=[c for c in cols if c in available])
    register = build_compliance_register(df, min_detections=min_detections, limit=limit)
    # Name the corpus in the response. A register whose observation window is one
    # day is not a smaller register, it is a different claim, and the reader
    # should be able to see which corpus produced the figures.
    register["corpus"] = corpus.name
    return register


@app.get("/api/v1/incident/{incident_id}/thermal-probe",
         tags=["Surveillance", "Validation"])
def thermal_probe(incident_id: int, window_days: int = Query(14, ge=1, le=90),
                  db: Session = Depends(get_db)):
    """Sentinel-3 SLSTR fire channels over an incident, and what they support.

    This is a second, independent instrument: a full-swath imager rather than a
    detection list, carrying dedicated fire channels (F1 at 3.74um, F2 at
    10.85um) that do not saturate where VIIRS I-4 clips at 366.9K.

    Two things come back, and they are different in kind.

    **Measured.** F1 and F2 brightness temperature over the incident, the
    ambient background measured from an annulus of undisturbed ground on the
    *same* acquisition, and the gap between the standard S7 channel and F1 --
    S7 is clipped near 311K by design, so that gap is a direct reading of how
    far past the ordinary channel's ceiling the source is running.

    **Attempted, and usually refused.** A Dozier bi-spectral fire temperature.
    The retrieval is implemented and tested; it is not trusted, and the endpoint
    returns NO_SOLUTION rather than a number whenever the mixture has no
    admissible solve. Measured against known regimes at 1km, the retrieved
    temperatures do not separate a refinery flare from a wheat field. See
    PROJECT_DOCUMENTATION.md 5b.4d. Any temperature this returns is reported
    with that caveat attached, not as a measurement.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    when = incident.timestamp_utc or datetime.now(timezone.utc)
    start = (when - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (when + timedelta(days=1)).strftime("%Y-%m-%d")

    result = SlstrClient().retrieve_temperature(
        float(incident.latitude), float(incident.longitude), start, end)
    result["incident_id"] = incident_id
    result["interpretation"] = (
        "Brightness temperatures and the measured background are observations. "
        "`fire_temperature_k`, when present, is NOT: the same retrieval returned "
        "466K for a photovoltaic array with no combustion process, and it is "
        "reported here for transparency rather than for use."
    )
    return result


_HEX_COLUMNS = ["latitude", "longitude", "h3_index", "frp", "acq_date",
                "state", "priority", "suppressed"]

# The corpus is a static file, so both the frame and the aggregation are
# deterministic and worth holding. Without this every pan of the map re-reads
# 2M rows and re-runs the group-by, which takes 5-10 seconds and makes an
# aggregated map slower than the unaggregated one it replaced.
_hex_corpus_cache: Dict[str, Any] = {"mtime": None, "frame": None}
_hex_result_cache: Dict[tuple, Dict[str, Any]] = {}
_HEX_RESULT_CACHE_MAX = 32


def _hex_corpus() -> pd.DataFrame:
    """Loads the processed corpus for map aggregation, cached on file mtime."""
    stat = OUTPUT_PROCESSED_PARQUET.stat()
    if _hex_corpus_cache["mtime"] != stat.st_mtime or _hex_corpus_cache["frame"] is None:
        available = set(pq.ParquetFile(OUTPUT_PROCESSED_PARQUET).schema_arrow.names)
        _hex_corpus_cache["frame"] = pd.read_parquet(
            OUTPUT_PROCESSED_PARQUET,
            columns=[c for c in _HEX_COLUMNS if c in available],
        )
        _hex_corpus_cache["mtime"] = stat.st_mtime
        logger.info("Map corpus cached: %s rows from %s",
                    f"{len(_hex_corpus_cache['frame']):,}", OUTPUT_PROCESSED_PARQUET)
    return _hex_corpus_cache["frame"]


@app.get("/api/v1/map/optical-validation", tags=["Validation"])
def map_optical_validation(
    limit: int = Query(1000, ge=1, le=5000, description="Max points returned"),
):
    """Precomputed Sentinel-2 dNBR for every P0 alert, for the map overlay.

    Reports whether vegetation actually burned at each emergency-priority
    detection -- the one signal in this system independent of thermal
    radiometry. An industrial fire consumes stored fuel inside a compound and
    leaves the canopy intact; a crop or forest fire does not.

    **This refutes without confirming.** A contained anomaly could equally be
    routine flaring the state machine mislabelled; neither burns vegetation. It
    bounds the false-positive rate from one side only.
    """
    path = PROJECT_ROOT / P0_DNBR_PARQUET if not P0_DNBR_PARQUET.is_absolute() else P0_DNBR_PARQUET
    if not path.exists():
        return {
            "status": "NO_VALIDATION_RUN",
            "detail": (
                f"{P0_DNBR_PARQUET} not found. Optical validation has not been run "
                "over the alert set; the overlay has nothing to draw."
            ),
            "points": [],
        }

    df = pd.read_parquet(path)
    total = len(df)

    keep = ["latitude", "longitude", "frp", "acq_date", "facility_name",
            "facility_type", "dnbr", "dnbr_severity", "dnbr_status"]
    df = df[[c for c in keep if c in df.columns]].copy()

    measurable = df["dnbr_status"].eq("OK") if "dnbr_status" in df else pd.Series(False, index=df.index)

    points = []
    for _, r in df.head(limit).iterrows():
        ok = str(r.get("dnbr_status")) == "OK"
        dnbr = r.get("dnbr")
        points.append({
            "lat": round(float(r["latitude"]), 5),
            "lon": round(float(r["longitude"]), 5),
            "frp": round(float(r.get("frp") or 0.0), 2),
            "date": str(r.get("acq_date"))[:10],
            "facility": (str(r.get("facility_name")) or "")[:60],
            # None rather than 0.0 when unmeasured. A fabricated zero here would
            # read as "measured, nothing burned", which is the precise error
            # the whole optical layer exists to avoid.
            "dnbr": round(float(dnbr), 4) if ok and pd.notna(dnbr) else None,
            "severity": str(r.get("dnbr_severity")) if ok else None,
            "status": str(r.get("dnbr_status")),
            "measurable": bool(ok),
        })

    status_counts = (df["dnbr_status"].value_counts().to_dict()
                     if "dnbr_status" in df else {})
    sev_counts = (df.loc[measurable, "dnbr_severity"].value_counts().to_dict()
                  if "dnbr_severity" in df else {})

    return {
        "status": "OK",
        "alerts_total": int(total),
        "measurable": int(measurable.sum()),
        "unmeasurable": int(total - measurable.sum()),
        "returned": len(points),
        "truncated": total > limit,
        "status_breakdown": {str(k): int(v) for k, v in status_counts.items()},
        "severity_breakdown": {str(k): int(v) for k, v in sev_counts.items()},
        "caveat": (
            "dNBR refutes without confirming. A contained anomaly could equally be "
            "routine flaring the state machine mislabelled; neither burns vegetation. "
            "Unmeasurable points are cloud or no-scene, NOT evidence of no burn."
        ),
        "points": points,
    }


@app.get("/api/v1/counter-register", tags=["Validation"])
def get_counter_register():
    """Documented industrial accidents this system produced no signature for.

    The mirror of the verified-event register, and the half that systems built
    to impress usually omit. These are not failures of the classifier: in most
    cases there was nothing to classify, because no detection existed. They are
    the boundary of what a thermal classifier can be asked to do.

    Every accuracy figure this project publishes is conditional on the fire
    being visible to a polar-orbiting radiometer at the moment it passes
    overhead. This endpoint states what that condition costs, with a cited
    source and a measured control for each case -- the control matters, because
    "we found nothing" is only evidence if the retrieval that found nothing is
    known to have been working.

    Deliberately *not* verified events: there is nothing to label, so including
    them would inflate the event count without contributing one scorable row.
    """
    incidents = [
        {
            "name": inc.name,
            "date": inc.date,
            "latitude": inc.lat,
            "longitude": inc.lon,
            "source": inc.source,
            "why_missed": inc.why_missed,
            "evidence": inc.evidence,
        }
        for inc in KNOWN_UNDETECTED_INCIDENTS
    ]
    return {
        "status": "OK",
        "purpose": (
            "Documented industrial accidents that produced no usable FIRMS "
            "signature. These bound what every accuracy figure in this project "
            "may claim."
        ),
        "n_undetected": len(incidents),
        "n_verified_for_contrast": len(VERIFIED_EVENTS),
        "reading": (
            "A detection rate cannot be computed from these. They were found by "
            "searching news archives for documented industrial fires and then "
            "checking the corpus, which is a biased sample: it finds fires that "
            "were reported, not a random draw from fires that occurred. What "
            "they establish is that the blind spots are real and have named "
            "mechanisms, not that they are rare."
        ),
        "incidents": incidents,
    }


@app.get("/api/v1/incident/{incident_id}/responders", tags=["Surveillance", "Dispatch"])
def get_incident_responders(
    incident_id: int,
    per_kind: int = Query(1, ge=1, le=5, description="How many of each category to return"),
    max_km: float = Query(100.0, gt=0, le=500, description="Search radius in kilometres"),
    db: Session = Depends(get_db),
):
    """Nearest fire station, hospital and police station to an incident.

    Grouped by category rather than returned as a flat nearest-N list: the three
    are not interchangeable. A flat list of the five closest facilities in a city
    centre would be five hospitals and no fire station, which is the one category
    an incident commander actually needs.

    **Distance is geodesic, and travel time is explicitly not routed.** The
    payload carries the circuity factor and assumed speed that produced the
    indicative minutes, so the figure cannot travel without its assumptions. A
    straight line divided by an assumed speed, presented as a "real-time ETA",
    would be a fabricated measurement of the same family as the synthetic
    weather this project removed.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    result = nearest_responders(
        lat=float(incident.latitude),
        lon=float(incident.longitude),
        per_kind=per_kind,
        max_km=max_km,
    )
    result["incident"] = {
        "id": incident.id,
        "latitude": float(incident.latitude),
        "longitude": float(incident.longitude),
        "predicted_class": incident.predicted_class,
        "alert_priority": incident.alert_priority,
        "facility_name": incident.facility_name,
    }
    return result


@app.get("/api/v1/responders/coverage", tags=["Dispatch"])
def get_responder_coverage():
    """What the responder layer contains, and what it does not.

    Reported as its own endpoint because the count is a caveat as much as a
    capability: OSM maps 741 fire stations for the whole of India, which is
    certainly an undercount. A facility absent here is absent from the map, not
    from the ground, and a dispatch surface that did not say so would imply a
    completeness it does not have.
    """
    summary = responder_layer_summary()
    summary["kinds"] = list(RESPONDER_KINDS)
    summary["build_command"] = "python -m src.ingestion.pbf_extractor --layer responders"
    summary["caveat"] = (
        "OpenStreetMap community data, not an official register. Coverage is "
        "uneven: hospitals are mapped densely and fire stations sparsely, so the "
        "nearest mapped fire station can be far further than the nearest real "
        "one. Distances are geodesic; no road graph is consulted."
    )
    return summary


@app.get("/api/v1/incidents/near", tags=["Surveillance", "Spatial"])
def get_incidents_near(
    lat: float = Query(..., ge=-90, le=90, description="WGS84 latitude of the query point"),
    lon: float = Query(..., ge=-180, le=180, description="WGS84 longitude of the query point"),
    radius_km: float = Query(10.0, gt=0, le=500, description="Search radius in kilometres"),
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    """Incidents within a true-metre radius of a point, nearest first.

    Served by PostGIS `ST_DWithin` over `geography` against the GiST index on
    `incidents.geom`. Distance is measured in metres on the spheroid rather
    than in degrees, which matters across this corpus: a degree of longitude
    is 111 km at Kanyakumari and 96 km at Srinagar, so a latitude/longitude
    bounding box is a different real distance in Kerala than in Kashmir.

    The response reports which engine answered, because the SQLite fallback
    returns the same rows by table scan and the difference should not be
    invisible to the caller.
    """
    pairs = incidents_within_km(db, lat=lat, lon=lon, radius_km=radius_km, limit=limit)
    if not pairs:
        return {
            "status": "success",
            "query": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "engine": "postgis_st_dwithin" if IS_POSTGRES else "sqlite_scan_haversine",
            "total": 0,
            "incidents": [],
        }

    by_id = {
        inc.id: inc
        for inc in db.query(Incident).filter(Incident.id.in_([i for i, _ in pairs])).all()
    }
    out = []
    for inc_id, distance_km in pairs:
        inc = by_id.get(inc_id)
        if inc is None:
            continue
        row = inc.to_dict()
        row["distance_km"] = round(distance_km, 4)
        out.append(row)

    return {
        "status": "success",
        "query": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "engine": "postgis_st_dwithin" if IS_POSTGRES else "sqlite_scan_haversine",
        "total": len(out),
        "incidents": out,
    }


@app.get("/api/v1/map/hexes", tags=["Analytics", "Surveillance"])
def map_hexes(
    resolution: int = Query(DEFAULT_RESOLUTION, ge=MIN_RESOLUTION, le=MAX_RESOLUTION,
                            description="H3 resolution; 6 is about 3km across"),
    bbox: Optional[str] = Query(None, description="west,south,east,north in degrees"),
    state: Optional[str] = Query(None, description="Comma-separated pipeline states"),
    min_detections: int = Query(1, ge=1),
    limit: int = Query(5000, ge=1, le=MAX_CELLS),
):
    """Detections aggregated into H3 cells, for a map that can draw the corpus.

    The dashboard's heat layer cannot render the 12-month archive: 2,044,295
    detections is roughly 200 MB of JSON before a marker is drawn. The fix is
    not a faster renderer, it is not sending the points.

    Every detection already carries `h3_index` at resolution 8, because the
    recurrence tracker needs it. Rolling those up is an index operation, so the
    server answers with a few thousand cells instead of two million points --
    and deck.gl's H3HexagonLayer draws them from the cell id alone, so no
    geometry crosses the wire either.

    Cells are ordered by the highest alert priority they contain, then by
    volume. Sorting on count alone would bury a single P0 under the dense
    agricultural regions, which is the failure the alerting tier exists to
    prevent.
    """
    if not OUTPUT_PROCESSED_PARQUET.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Processed corpus {OUTPUT_PROCESSED_PARQUET} not found; run the pipeline first.",
        )

    bounds = None
    if bbox:
        parts = [p.strip() for p in bbox.split(",") if p.strip()]
        if len(parts) != 4:
            raise HTTPException(status_code=422,
                                detail="bbox must be 'west,south,east,north'.")
        try:
            bounds = [float(p) for p in parts]
        except ValueError:
            raise HTTPException(status_code=422, detail="bbox values must be numbers.")

    states = [s.strip() for s in state.split(",")] if state else None

    key = (resolution, tuple(bounds) if bounds else None,
           tuple(states) if states else None, min_detections, limit,
           OUTPUT_PROCESSED_PARQUET.stat().st_mtime)
    if key in _hex_result_cache:
        return _hex_result_cache[key]

    result = aggregate_hexes(
        _hex_corpus(), resolution=resolution, bbox=bounds,
        states=states, min_detections=min_detections, limit=limit,
    )
    if len(_hex_result_cache) >= _HEX_RESULT_CACHE_MAX:
        _hex_result_cache.clear()
    _hex_result_cache[key] = result
    return result


@app.post("/api/v1/incident/{incident_id}/dispatch", tags=["Surveillance", "Dispatch"])
def dispatch_incident(incident_id: int, db: Session = Depends(get_db)):
    """Routes an incident to responders over the channels its priority calls for.

    **Dispatch is off unless `ALERT_DISPATCH_ENABLED=true`.** Unset -- the
    default, and the right setting for demos and pipeline replays -- every
    channel builds and validates its payload, reports `DRY_RUN`, and sends
    nothing. The response always states `dispatch_enabled` so a caller can never
    mistake a rehearsal for a delivery.

    Routing follows the alerting table in the problem statement: a P0 accidental
    fire goes to every channel, a P1 anomaly to email and webhook, a P2 advisory
    to webhook only. `SUPPRESSED` and `NON_ALERT` incidents are refused --
    routine industrial operation must not reach an operator, and that mandate is
    enforced here as well as upstream, so calling this endpoint directly cannot
    bypass it.

    Every attempt is written to the audit log, delivered or not.
    """
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    alert = {
        "incident_id": inc.id,
        "detection_id": inc.detection_id,
        "priority": inc.alert_priority,
        "state": inc.alert_state,
        "latitude": inc.latitude,
        "longitude": inc.longitude,
        "frp": inc.frp,
        "timestamp_utc": str(inc.timestamp_utc),
        "satellite": inc.satellite,
        "facility_name": inc.facility_name,
        "facility_type": inc.facility_type,
        "predicted_class": inc.predicted_class,
        "confidence": inc.confidence,
        "rationale": inc.shap_rationale,
    }

    report = get_dispatcher().dispatch(alert)

    db.add(AuditLog(
        incident_id=inc.id,
        detection_id=inc.detection_id,
        action="DISPATCH",
        previous_class=inc.status,
        new_class=("DISPATCHED" if report["any_delivered"] else inc.status),
        operator_notes="; ".join(
            f"{c['channel']}={c['status']}" for c in report["channels"]
        ),
    ))
    # Only a real delivery advances the incident. A dry run must not leave an
    # operator believing responders were notified.
    if report["any_delivered"]:
        inc.status = "DISPATCHED"
    db.commit()

    return report


@app.post("/api/v1/incident/{incident_id}/status", tags=["Surveillance"])
def update_incident_status(
    incident_id: int,
    payload: StatusUpdateRequest,
    db: Session = Depends(get_db),
):
    """Updates the operator lifecycle status and dispatch notes for an incident."""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    valid_statuses = {"OPEN", "ACKNOWLEDGED", "DISPATCHED", "RESOLVED", "MUTED_ROUTINE"}
    new_status = payload.status.upper().strip()
    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{payload.status}'. Valid choices: {sorted(list(valid_statuses))}",
        )

    inc.status = new_status
    if payload.operator_notes:
        inc.operator_notes = payload.operator_notes

    db.commit()
    db.refresh(inc)
    return {
        "status": "success",
        "message": f"Incident {incident_id} status updated to {new_status}.",
        "incident": inc.to_dict(),
    }


@app.get("/api/v1/model/provenance", tags=["Analytics"])
def model_provenance():
    """What model is actually on disk, read from its own metrics artifact.

    The dashboard used to carry these figures as literals in the markup and they
    drifted a whole rule version behind the model they described. Anything a
    reader will take for a measurement has to come from the measurement.
    """
    path = METRICS_PATH if METRICS_PATH.is_absolute() else PROJECT_ROOT / METRICS_PATH
    if not path.exists():
        return {"status": "NO_METRICS",
                "detail": f"{METRICS_PATH} not found; train the classifier first."}

    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "UNREADABLE", "detail": str(e)}

    run = metrics.get("run", {})
    cv = metrics.get("spatial_block_cv", {})
    circ = metrics.get("circularity_audit", {})
    prov = metrics.get("label_provenance", {})

    return {
        "status": "OK",
        "run_id": run.get("run_id"),
        "trained_at_utc": run.get("trained_at_utc"),
        "labelling_rule_version": run.get("labelling_rule_version"),
        "n_training_rows": prov.get("real_detections_heuristic_labelled"),
        "n_synthetic_rows": prov.get("synthetic_augmented"),
        "n_features": len(metrics.get("feature_names") or []),
        "spatial_block_cv_macro_f1": cv.get("mean_macro_f1"),
        "spatial_block_cv_std": cv.get("std_macro_f1"),
        "spatial_block_cv_worst_fold": cv.get("min_macro_f1"),
        "circularity_full_macro_f1": circ.get("full_feature_macro_f1"),
        "circularity_ablated_macro_f1": circ.get("ablated_macro_f1"),
        "circularity_delta": circ.get("delta"),
        "caveat": (
            "Spatial-block CV is measured against rule-derived labels. The only "
            "independent figure is the verified-label evaluation."
        ),
    }


@app.get("/api/v1/stats", tags=["Analytics"])
def get_statistics(db: Session = Depends(get_db)):
    """Returns tactical and executive KPI metrics."""
    total = db.query(Incident).count()
    p0 = db.query(Incident).filter(Incident.alert_priority == "P0_EMERGENCY").count()
    p1 = db.query(Incident).filter(Incident.alert_priority == "P1_ALERT").count()
    p2 = db.query(Incident).filter(Incident.alert_priority == "P2_ADVISORY").count()
    non_alert = db.query(Incident).filter(Incident.alert_priority == "NON_ALERT").count()
    industrial_total = db.query(Incident).filter(Incident.inside_industrial.is_(True)).count()

    # Alert Fatigue Reduction: % of total detections suppressed from emergency siren
    suppressed_count = total - (p0 + p1)
    fatigue_reduction = (suppressed_count / total * 100.0) if total > 0 else 100.0

    # Status breakdown
    status_counts = (
        db.query(Incident.status, func.count(Incident.id))
        .group_by(Incident.status)
        .all()
    )
    status_dict = {s: cnt for s, cnt in status_counts}

    return {
        "total_detections": total,
        "p0_emergencies": p0,
        "p1_alerts": p1,
        "p2_advisories": p2,
        "suppressed": suppressed_count,
        "fatigue_reduction_pct": round(fatigue_reduction, 2),
        "industrial_incidents": industrial_total,
        "status_breakdown": status_dict,
    }


@app.get("/api/v1/system/replay", tags=["System"])
def replay_status():
    """Replay-mode state plus a preflight of what survives without a network.

    The preflight is the useful half. Knowing the mode exists is worth less than
    knowing which panels keep working when the venue wifi fails, and it is meant
    to be read *before* a demonstration rather than during one.
    """
    return replay_mode.preflight()


@app.post("/api/v1/system/replay", tags=["System"])
def set_replay(
    enabled: bool = Query(..., description="True to serve from the local archive"),
    reason: str = Query("", description="Optional note recorded with the switch"),
):
    """Switches Historical Replay Mode on or off.

    On: the FIRMS refresh loop stops attempting the network and the freshness
    badge reads REPLAY rather than STALE. Nothing is reseeded and nothing is
    fabricated -- every archive-backed surface already reads from disk, and the
    genuinely live panels keep reporting their own degraded states.

    The distinction the badge carries is the point. STALE means nobody has
    ingested anything and you should worry. REPLAY means ingestion is
    deliberately frozen and you are looking at the archive.
    """
    state = replay_mode.set_enabled(enabled, reason=reason)
    logger.info("Replay mode %s%s", "ENABLED" if enabled else "DISABLED",
                f" ({reason})" if reason else "")
    if not enabled and auto_refresh.refresh_enabled():
        # Leaving replay resumes ingestion, otherwise the badge would sit at
        # REPLAY forever with nothing behind it.
        auto_refresh.start_scheduler()
    return {"replay": state, "preflight": replay_mode.preflight()}


@app.get("/api/v1/system/refresh", tags=["Pipeline", "System"])
def refresh_status():
    """State of the automatic FIRMS refresh.

    `last_success_utc` only advances on a run that completed all three stages.
    A failed pull leaves it untouched, so the gap between it and
    `last_attempt_utc` is the honest measure of how long ingestion has been
    broken -- which a single "last refreshed" field would hide.
    """
    return auto_refresh.status()


@app.post("/api/v1/system/refresh", tags=["Pipeline", "System"])
def trigger_refresh(day_range: int = Query(2, ge=1, le=7)):
    """Runs the full ingest now: FIRMS pull, spatial join, reseed.

    This is what `POST /api/v1/sync` is commonly mistaken for. `sync` re-imports
    the processed corpus already on disk and never contacts FIRMS; this fetches
    new detections.
    """
    return auto_refresh.refresh_once(day_range=day_range)


@app.post("/api/v1/sync", tags=["Pipeline"])
def sync_detections(db: Session = Depends(get_db)):
    """Re-imports the processed corpus on disk into the incident database.

    This does **not** contact NASA FIRMS. It reseeds from whatever
    `PROCESSED_CORPUS` currently points at, so on a stale file it will report a
    successful sync and change nothing about how old the data is. Use
    `POST /api/v1/system/refresh` to actually ingest.
    """
    if not OUTPUT_PROCESSED_PARQUET.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Processed parquet file not found at {OUTPUT_PROCESSED_PARQUET}. Run pipeline first.",
        )

    t0 = time.time()
    synced = seed_database_from_parquet(db, OUTPUT_PROCESSED_PARQUET)
    elapsed = time.time() - t0

    return {
        "status": "success",
        "records_synced": synced,
        "elapsed_seconds": round(elapsed, 3),
        "total_in_db": db.query(Incident).count(),
    }


@app.post("/api/v1/classify", tags=["Machine Learning"])
def classify_custom_hotspot(
    payload: HotspotInferenceRequest,
    db: Session = Depends(get_db),
):
    """Executes live spatial matching, recurrence evaluation, XGBoost inference, and TreeSHAP."""
    explainer_inst = get_explainer()
    polys = get_osm_gdf()

    lat = payload.latitude
    lon = payload.longitude
    frp = payload.frp
    pt = Point(lon, lat)

    # 1. Spatial Join PIP & Proximity Test
    inside_ind = False
    is_exact = False
    fac_name = None
    fac_type = "non_industrial"
    dist_km = 999.0

    if polys is not None and not polys.empty:
        # Query spatial index
        candidate_idxs = list(polys.sindex.query(pt))
        for idx in candidate_idxs:
            geom = polys.geometry.iloc[idx]
            if geom.contains(pt):
                inside_ind = True
                is_exact = True
                dist_km = 0.0
                row_dict = polys.iloc[idx].to_dict()
                row_dict["inside_industrial"] = True
                fac_name = str(row_dict.get("name")) if pd.notnull(row_dict.get("name")) else "Unnamed Industrial Site"
                fac_type = classify_facility_type(pd.Series(row_dict))
                break

        # 500m buffer check if not exact match (~0.0045 deg)
        if not inside_ind:
            buf_pt = pt.buffer(500.0 / 111320.0)
            buf_cands = list(polys.sindex.query(buf_pt))
            for idx in buf_cands:
                geom = polys.geometry.iloc[idx]
                if geom.intersects(buf_pt):
                    inside_ind = True
                    is_exact = False
                    dist_km = 0.25
                    row_dict = polys.iloc[idx].to_dict()
                    row_dict["inside_industrial"] = True
                    fac_name = str(row_dict.get("name")) if pd.notnull(row_dict.get("name")) else "Industrial Perimeter Site"
                    fac_type = classify_facility_type(pd.Series(row_dict))
                    break

    # 1B. Apply Simulation Scenario Overrides if specified
    if payload.inside_industrial_override is not None:
        inside_ind = payload.inside_industrial_override
        if inside_ind and dist_km > 0.5:
            dist_km = 0.0
            is_exact = True
            if not fac_name:
                fac_name = "Simulated Industrial Facility"
    if payload.facility_type_override:
        fac_type = payload.facility_type_override

    # 2. Uber H3 Cell Binning & Baseline Retrieval
    h3_cell = latlng_to_h3(lat, lon, 9)
    baseline = db.query(H3Baseline).filter(H3Baseline.h3_index == h3_cell).first()

    if baseline and baseline.n_30d > 0:
        n_30d = baseline.n_30d
        mu_frp = baseline.mu_frp
        std_frp = max(np.sqrt(baseline.var_frp), 0.5)
        z_frp = (frp - mu_frp) / std_frp
        frp_ratio = frp / max(mu_frp, 0.1)
    else:
        if inside_ind:
            n_30d = 20 if frp <= 25.0 else 1
            mu_frp = 8.0
            z_frp = max(0.0, (frp - 8.0) / 4.0)
            frp_ratio = frp / 8.0
        else:
            n_30d = 0
            mu_frp = frp
            z_frp = 0.0
            frp_ratio = 1.0

    if payload.recurrence_n_30d_override is not None:
        n_30d = payload.recurrence_n_30d_override
        if n_30d > 0 and mu_frp <= 0.1:
            mu_frp = 8.0
            z_frp = max(0.0, (frp - 8.0) / 4.0)
            frp_ratio = frp / 8.0

    # 3. Assemble Inference Payload
    ts_now = payload.timestamp_utc or datetime.now(timezone.utc).isoformat()
    sample_dict = {
        "frp": frp,
        "bright_ti4": payload.bright_ti4 or 350.0,
        "bright_ti5": payload.bright_ti5 or 295.0,
        "scan": payload.scan or 0.4,
        "track": payload.track or 0.4,
        "daynight": (payload.daynight or "N").upper(),
        "timestamp_utc": ts_now,
        "inside_industrial": inside_ind,
        "is_exact_match": is_exact,
        "dist_to_industrial_km": dist_km,
        "facility_type": fac_type,
        "n_30d": n_30d,
        "mu_frp": mu_frp,
        "z_frp": z_frp,
        "frp_ratio": frp_ratio,
    }

    # 4. Run TreeSHAP Explanation & Classification
    t0 = time.time()
    explanation = explainer_inst.explain_detection(sample_dict)
    inference_time_ms = round((time.time() - t0) * 1000.0, 2)

    pred_class = explanation["predicted_class"]

    # 4b. Guard rails the model cannot learn.
    #
    # The labelling rule generalises further than the trained model does, and
    # that gap is structural rather than a training shortfall. Rule v5 declines
    # to apply India's harvest calendar outside the region where it was
    # calibrated; the model decides from a coordinate-free feature vector -- on
    # purpose, because a model handed coordinates memorises where refineries are
    # -- and its training corpus is entirely Indian, so it has never seen an
    # out-of-domain example and never will. Adding an "in domain" flag would not
    # help: the flag is constant across every training row.
    #
    # So the rule is applied here instead, over the model's output, exactly as
    # the non-combustion override works. A detection in the Chihuahuan Desert is
    # no longer reported as an Indian crop burn.
    guarded, guard_note = apply_serving_guards(pred_class, lat=lat, lon=lon,
                                               facility_type=fac_type)
    overridden = guarded != pred_class
    pred_class = guarded

    # The served class carries its OWN index, and a served-only class has none.
    #
    # This was a real defect. `predicted_class_id` was taken straight from the
    # explainer, which is pre-guard, so the two fields disagreed whenever a
    # guard fired: Bandipur returned {"predicted_class": "FOREST_FIRE",
    # "predicted_class_id": 2} -- and 2 is AGRICULTURAL_BURN. A consumer
    # keying on the id got the un-guarded answer, which defeats the entire
    # point of the guard layer: an override is supposed to be visible, not
    # silent. It was silent for anyone reading JSON rather than prose.
    #
    # FOREST_FIRE resolves to None rather than to a stand-in integer, because
    # it is decided by land cover and the model has no index for it. Inventing
    # one would re-introduce exactly the confusion this fixes.
    served_class_id = CLASS_NAME_TO_INDEX.get(pred_class)

    # 5. Alert Priority & State Formulation
    if pred_class == "ACCIDENTAL_FIRE":
        priority = "P0_EMERGENCY" if inside_ind else "P1_ALERT"
        state = "UNCONTAINED_EMERGENCY"
    elif pred_class == "PERSISTENT_BASELINE":
        priority = "P2_ADVISORY" if inside_ind else "NON_ALERT"
        state = "PERSISTENT_BASELINE"
    elif pred_class == "FOREST_FIRE":
        # Segregated, not discarded. The problem statement asks for forest fires
        # to be separated from industrial ones; separating them means routing
        # them elsewhere, not dropping them. Paging a plant response team to a
        # reserve forest is the alert fatigue this project exists to remove --
        # and so is telling a forestry department nothing.
        priority = "P2_ADVISORY"
        state = "FOREST_FIRE"
    else:
        priority = "NON_ALERT"
        state = "TRANSIENT_SUSPICION"

    # 6. Environmental Dispersion Simulation
    dispersion_data = get_weather_and_dispersion(
        lat=lat,
        lon=lon,
        frp=frp,
        wind_speed=payload.wind_speed_kmh,
        wind_deg=payload.wind_direction_deg,
    )

    return {
        "predicted_class": pred_class,
        # None for a served-only class such as FOREST_FIRE, which the model
        # has no index for. Pairs with `predicted_class`, never with the
        # model's raw answer.
        "predicted_class_id": served_class_id,
        "served_only_class": pred_class in SERVED_ONLY_CLASSES,
        "confidence_percent": explanation["confidence_percent"],
        # Which class the confidence figure actually describes. When a guard
        # fires, the model's confidence belongs to the class it predicted, not
        # to the one being served -- Bandipur reported 100% confidence in a
        # FOREST_FIRE the model never scored. The number is unchanged; what is
        # added is what it refers to.
        "confidence_describes": (
            "model_predicted_class" if overridden else "predicted_class"
        ),
        "model_predicted_class": explanation["predicted_class"],
        "model_predicted_class_id": explanation["predicted_class_id"],
        "serving_guard_applied": overridden,
        "serving_guard_note": guard_note,
        "alert_priority": priority,
        "alert_state": state,
        "inference_time_ms": inference_time_ms,
        "spatial_enrichment": {
            "inside_industrial": inside_ind,
            "is_exact_match": is_exact,
            "facility_name": fac_name,
            "facility_type": fac_type,
            "dist_to_industrial_km": dist_km,
        },
        "h3_metrics": {
            "h3_index": h3_cell,
            "n_30d": n_30d,
            "mu_frp": round(mu_frp, 2),
            "z_frp": round(z_frp, 2),
            "frp_ratio": round(frp_ratio, 2),
        },
        "class_probabilities": explanation["class_probabilities"],
        "top_driving_factors": explanation["top_driving_factors"],
        "narrative_rationale": explanation["narrative_rationale"],
        "atmospheric_dispersion": dispersion_data,
    }


class AnalystActionPayload(BaseModel):
    action: str = Field(..., description="Action to execute: CONFIRM, RECLASSIFY, FALSE_POSITIVE, DISPATCH")
    new_class: Optional[str] = Field(None, description="New predicted class name if reclassifying")
    operator_notes: Optional[str] = Field(None, description="Analyst verification rationale")


@app.post("/api/v1/incident/{incident_id}/action", tags=["Analyst Review"])
def execute_analyst_action(
    incident_id: int,
    payload: AnalystActionPayload,
    db: Session = Depends(get_db),
):
    """Executes human-in-the-loop analyst triage action with audit trail logging."""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident with ID {incident_id} not found.")

    prev_class = inc.predicted_class
    new_class = payload.new_class or prev_class
    action_type = payload.action.upper()

    if action_type == "CONFIRM":
        inc.status = "DISPATCHED"
        if inc.alert_priority == "P2_ADVISORY":
            inc.alert_priority = "P0_EMERGENCY"
        inc.alert_state = "CONFIRMED_DISPATCH"
    elif action_type == "RECLASSIFY":
        if payload.new_class:
            inc.predicted_class = payload.new_class
            inc.status = "ACKNOWLEDGED"
            if payload.new_class == "ACCIDENTAL_FIRE":
                inc.alert_priority = "P0_EMERGENCY"
                inc.alert_state = "UNCONTAINED_EMERGENCY"
            elif payload.new_class == "PERSISTENT_BASELINE":
                inc.alert_priority = "P2_ADVISORY"
                inc.alert_state = "PERSISTENT_BASELINE"
            else:
                inc.alert_priority = "NON_ALERT"
                inc.alert_state = "TRANSIENT_SUSPICION"
    elif action_type == "FALSE_POSITIVE":
        inc.status = "MUTED_ROUTINE"
        inc.alert_priority = "NON_ALERT"
        inc.alert_state = "SUPPRESSED_FALSE_ALARM"
    elif action_type == "DISPATCH":
        inc.status = "DISPATCHED"
        inc.alert_state = "EMERGENCY_DISPATCHED"
    else:
        inc.status = "ACKNOWLEDGED"

    if payload.operator_notes:
        inc.operator_notes = payload.operator_notes

    # Create immutable audit log entry
    audit = AuditLog(
        incident_id=inc.id,
        detection_id=inc.detection_id,
        action=action_type,
        previous_class=prev_class,
        new_class=new_class,
        operator_notes=payload.operator_notes,
    )
    db.add(audit)
    db.commit()
    db.refresh(inc)

    return {
        "status": "success",
        "action": action_type,
        "message": f"Incident {incident_id} successfully updated with action {action_type}.",
        "incident": inc.to_dict(),
        "audit_log": audit.to_dict(),
    }


@app.get("/api/v1/audit/logs", tags=["Analyst Review"])
def get_audit_logs(limit: int = 100, db: Session = Depends(get_db)):
    """Returns human-in-the-loop analyst decisions and audit history."""
    logs = db.query(AuditLog).order_by(desc(AuditLog.timestamp_utc)).limit(limit).all()
    return [l.to_dict() for l in logs]


@app.get("/api/v1/analytics/charts", tags=["Analytics"])
def get_analytics_charts(db: Session = Depends(get_db)):
    """Provides pre-aggregated data distributions for Chart.js tactical visualizations."""
    incidents = db.query(Incident).all()
    if not incidents:
        return {
            "frp_distribution": {"labels": [], "counts": []},
            "facility_types": {"labels": [], "counts": []},
            "alert_priorities": {"labels": [], "counts": []},
            "alert_states": {"labels": [], "counts": []},
            "top_hotspots": [],
        }

    # 1. FRP Intensity Histogram
    frp_bins = [0, 10, 25, 50, 100, 250, 99999]
    frp_labels = ["0-10 MW", "10-25 MW", "25-50 MW", "50-100 MW", "100-250 MW", "250+ MW"]
    frp_counts = [0] * len(frp_labels)

    facility_counts: Dict[str, int] = {}
    priority_counts: Dict[str, int] = {}
    state_counts: Dict[str, int] = {}

    for inc in incidents:
        # FRP binning
        val = inc.frp
        for i in range(len(frp_bins) - 1):
            if frp_bins[i] <= val < frp_bins[i + 1]:
                frp_counts[i] += 1
                break

        # Facility type count
        ftype = inc.facility_type or "non_industrial"
        facility_counts[ftype] = facility_counts.get(ftype, 0) + 1

        # Priority count
        prio = inc.alert_priority or "NON_ALERT"
        priority_counts[prio] = priority_counts.get(prio, 0) + 1

        # State count
        st = inc.alert_state or "TRANSIENT_SUSPICION"
        state_counts[st] = state_counts.get(st, 0) + 1

    # Top 10 critical hotspots by FRP
    sorted_incidents = sorted(incidents, key=lambda x: x.frp, reverse=True)[:10]
    top_hotspots = [
        {
            "id": x.id,
            "detection_id": x.detection_id,
            "facility_name": x.facility_name or "Unnamed Site",
            "facility_type": x.facility_type,
            "latitude": x.latitude,
            "longitude": x.longitude,
            "frp": x.frp,
            "alert_priority": x.alert_priority,
            "predicted_class": x.predicted_class,
            "confidence": x.confidence,
            "status": x.status,
        }
        for x in sorted_incidents
    ]

    return {
        "frp_distribution": {
            "labels": frp_labels,
            "counts": frp_counts,
        },
        "facility_types": {
            "labels": list(facility_counts.keys()),
            "counts": list(facility_counts.values()),
        },
        "alert_priorities": {
            "labels": list(priority_counts.keys()),
            "counts": list(priority_counts.values()),
        },
        "alert_states": {
            "labels": list(state_counts.keys()),
            "counts": list(state_counts.values()),
        },
        "top_hotspots": top_hotspots,
    }


@app.get("/api/v1/weather/context", tags=["Weather & Dispersion"])
def get_weather_and_dispersion(
    lat: float,
    lon: float,
    frp: float = 15.0,
    at: Optional[str] = Query(None, description="Incident time (ISO8601). Defaults to now."),
    wind_speed: Optional[float] = Query(None, description="Override wind speed in km/h"),
    wind_deg: Optional[float] = Query(None, description="Override wind heading in degrees (0-360)"),
):
    """Smoke plume cone over weather that is SYNTHETIC unless a provider is wired.

    **The weather in this response is not a measurement.** No weather service is
    queried. Temperature, humidity and wind are produced by a fixed formula over
    the coordinates, and the dispersion cone derived from them is illustrative.
    Every response carries `weather_source`, `weather_is_measured` and, when
    synthetic, `weather_disclaimer`. Callers must surface those to anyone who
    sees the numbers.

    Pass `wind_speed` / `wind_deg` to substitute values from a real forecast; the
    response then reports `weather_source="CALLER_SUPPLIED"` for the overridden
    fields.
    """
    # One implementation, imported rather than copied: this endpoint and the
    # SitRep previously carried the same formula in two files, so a correction
    # to one would have silently missed the other.
    ctx = compute_atmospheric_dispersion(lat, lon, frp, when=at)

    overridden = []
    if wind_speed is not None:
        ctx["wind_speed_kmh"] = round(float(wind_speed), 1)
        overridden.append("wind_speed_kmh")
    if wind_deg is not None:
        deg = int(wind_deg) % 360
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        ctx["wind_direction_deg"] = deg
        ctx["wind_compass"] = dirs[int((deg + 11.25) / 22.5) % 16]
        ctx["downwind_deg"] = (deg + 180) % 360
        ctx["downwind_compass"] = dirs[int(((deg + 180) % 360 + 11.25) / 22.5) % 16]
        overridden.append("wind_direction_deg")

    downwind_rad = np.radians(ctx["downwind_deg"])
    plume_len_deg = float(np.clip(0.02 + np.sqrt(max(frp, 1.0)) * 0.008, 0.03, 0.20))
    half_angle_rad = np.radians(max(12.0, 26.0 - ctx["wind_speed_kmh"] * 0.4))

    rad_left = downwind_rad - half_angle_rad
    rad_right = downwind_rad + half_angle_rad
    p0 = [lon, lat]
    p1 = [round(lon + plume_len_deg * np.sin(rad_left), 5), round(lat + plume_len_deg * np.cos(rad_left), 5)]
    p2 = [round(lon + (plume_len_deg * 1.15) * np.sin(downwind_rad), 5),
          round(lat + (plume_len_deg * 1.15) * np.cos(downwind_rad), 5)]
    p3 = [round(lon + plume_len_deg * np.sin(rad_right), 5), round(lat + plume_len_deg * np.cos(rad_right), 5)]

    return {
        "coordinates": {"latitude": lat, "longitude": lon},
        "weather_source": ("CALLER_SUPPLIED" if overridden else ctx["weather_source"]),
        "weather_is_measured": ctx["weather_is_measured"],
        "weather_disclaimer": ctx.get("weather_disclaimer"),
        "caller_supplied_fields": overridden,
        "temperature_c": ctx["temperature_c"],
        "humidity_pct": ctx["humidity_pct"],
        "wind": {
            "speed_kmh": ctx["wind_speed_kmh"],
            "direction_deg": ctx["wind_direction_deg"],
            "direction_compass": ctx["wind_compass"],
            "downwind_deg": ctx["downwind_deg"],
        },
        "smoke_dispersion_cone": {
            "type": "Polygon",
            "coordinates": [[p0, p1, p2, p3, p0]],
            "hazard_length_km": round(plume_len_deg * 111.32, 1),
            "is_illustrative": not ctx["weather_is_measured"],
            "risk_assessment": (
                "CRITICAL: High-temperature toxic combustion plume extends downwind. Recommended evacuation alert."
                if frp > 50 else
                "MODERATE: Visible smoke plume downwind. Monitor air quality indices (PM2.5 / VOCs)."
            ),
        },
    }


@app.post("/api/v1/simulation/sitrep", tags=["Surveillance", "Briefings", "Simulation"])
def generate_simulated_sitrep(
    payload: HotspotInferenceRequest,
    format: str = Query("html", description="Output format: 'html' for printable A4 view, 'json' for data payload"),
    db: Session = Depends(get_db),
):
    """Generates an official tactical Situation Report (SitRep) for a simulated scenario."""
    # Execute full inference pipeline with overrides
    inf = classify_custom_hotspot(payload, db)

    # Synthesize simulated incident instance
    sim_incident = Incident(
        id=99999,
        detection_id=f"SIM_{payload.latitude:.4f}_{payload.longitude:.4f}_{int(payload.frp)}MW",
        latitude=payload.latitude,
        longitude=payload.longitude,
        frp=payload.frp,
        bright_ti4=payload.bright_ti4,
        bright_ti5=payload.bright_ti5,
        timestamp_utc=datetime.now(timezone.utc),
        satellite="SIMULATED (What-If Scenario)",
        inside_industrial=inf["spatial_enrichment"]["inside_industrial"],
        facility_name=inf["spatial_enrichment"]["facility_name"] or "Simulated Facility Target",
        facility_type=inf["spatial_enrichment"]["facility_type"],
        dist_to_industrial_km=inf["spatial_enrichment"]["dist_to_industrial_km"],
        h3_index=inf["h3_metrics"]["h3_index"],
        predicted_class=inf["predicted_class"],
        confidence=inf["confidence_percent"] / 100.0,
        alert_priority=inf["alert_priority"],
        alert_state=inf["alert_state"],
        shap_rationale=f"[SIMULATION EXERCISE] {inf['narrative_rationale']}",
        status="SIMULATED",
        operator_notes="Tactical what-if simulation run by watchstander.",
    )

    sitrep_data = SitRepGenerator.generate_sitrep_data(sim_incident, db)
    sitrep_data["report_metadata"]["sitrep_id"] = f"SITREP-SIM-{int(payload.frp)}MW"
    sitrep_data["report_metadata"]["classification"] = "SIMULATION EXERCISE // UNCLASSIFIED // SIH26162"
    sitrep_data["report_metadata"]["operation_name"] = "PROJECT SIH26162 // WHAT-IF TACTICAL SIMULATION"

    if format.lower() == "json":
        return sitrep_data

    html_content = SitRepGenerator.render_html_sitrep(sitrep_data)
    return HTMLResponse(content=html_content, status_code=200)

