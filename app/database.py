"""Database and ORM Layer for Project SIH26162.

**PostGIS is the primary datastore.** Incident geometry is stored as a real
`geometry(Point, 4326)` column with a GiST index, and proximity queries are
served by `ST_DWithin` rather than by scanning the table and filtering in
Python.

Why the migration away from SQLite happened, recorded so it is not undone:

  1. `src/pipeline/auto_refresh.py` runs a daemon thread that opens its own
     session and writes incidents while the API is serving analyst writes --
     dispatch, status changes, audit rows. That is two concurrent writers.
     SQLite serialises every write behind a single file lock and surfaces the
     contention as `database is locked`, intermittently, under exactly the
     load a demonstration does not reproduce.
  2. The audit log is regulatory evidence for the CPCB register. Evidence that
     can be lost to a half-completed file copy is not evidence. PostgreSQL has
     WAL archiving and point-in-time recovery; SQLite has a file.
  3. A file-backed database cannot serve more than one API worker safely, so
     `uvicorn --workers N` was never available.

What did **not** move into PostGIS, deliberately:

  - **Bulk point-in-polygon enrichment stays in GeoPandas.** Measured on this
    corpus: 2,044,295 detections against 28,587 polygons joins in 1.22 s with
    an in-memory R-tree. A per-row SQL join would be slower, so PostGIS is
    used for *query-time* geometry, not batch enrichment.
  - **The analytical corpus stays in Parquet.** 2.04M rows read columnar in
    0.05 s. Postgres is the wrong store for a scan-everything workload.

SQLite remains reachable, for two cases only:

  - `pytest`, which must not require a running database server;
  - Historical Replay Mode on a disconnected machine.

Both are opt-in by setting `DATABASE_URL` to an explicit `sqlite://` URL.

**The mode is decided by configuration, never by connectivity.** An earlier
draft fell back to SQLite when PostGIS was unreachable, which was wrong in a
way worth recording: the geometry column is declared only in PostGIS mode, so
a transient connection failure at import time would have silently built a
*different schema* -- an application running without `incidents.geom`, and
therefore answering every proximity query with nothing, while reporting
itself healthy. A configured-but-unreachable PostGIS now raises at startup
and says what to fix. Running offline is a decision an operator makes, not an
accident a dropped connection makes for them.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
logger = logging.getLogger("database")

DEFAULT_SQLITE_URL = "sqlite:///data/fire_db.sqlite"

# Placeholder values shipped in .env.example. Treating them as "unset" stops a
# copied-but-unedited .env from being read as a real configuration.
_PLACEHOLDER_PREFIXES = (
    "postgresql://user:password",
    "postgresql+psycopg://user:password",
)


# These must stay identical to the defaults in docker-compose.yml. They are
# local-development values, not secrets: the compose file declares the same
# ones in plain text and the server they reach listens only on localhost.
#
# They are duplicated here deliberately rather than left blank. An earlier
# version defaulted to user "postgres" with no password while compose
# defaulted to "sih"/"sih_local_dev", so `docker compose up -d db` with no
# .env produced a server the application could not authenticate against --
# the exact drift this function's docstring claimed to prevent. A deployment
# overrides all of them via .env; a laptop should not need to.
_COMPOSE_DEFAULTS = {
    "POSTGRES_USER": "sih",
    "POSTGRES_PASSWORD": "sih_local_dev",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "sih26162",
}


def _compose_postgres_url() -> str:
    """Builds the PostGIS URL from the same variables docker-compose reads.

    One set of names *and* one set of defaults, so `docker compose up -d db`
    and the application cannot drift apart in configuration.
    """
    user = os.getenv("POSTGRES_USER") or _COMPOSE_DEFAULTS["POSTGRES_USER"]
    password = os.getenv("POSTGRES_PASSWORD") or _COMPOSE_DEFAULTS["POSTGRES_PASSWORD"]
    host = os.getenv("POSTGRES_HOST") or _COMPOSE_DEFAULTS["POSTGRES_HOST"]
    port = os.getenv("POSTGRES_PORT") or _COMPOSE_DEFAULTS["POSTGRES_PORT"]
    db = os.getenv("POSTGRES_DB") or _COMPOSE_DEFAULTS["POSTGRES_DB"]
    credentials = f"{user}:{password}" if password else user
    return f"postgresql+psycopg://{credentials}@{host}:{port}/{db}"


def _normalise_driver(url: str) -> str:
    """Pins the psycopg 3 driver.

    A bare `postgresql://` URL makes SQLAlchemy look for psycopg2, which this
    project does not install. Rewriting it here means a connection string
    copied from any other tool works unchanged.
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


env_db_url = os.getenv("DATABASE_URL", "").strip()
if env_db_url and env_db_url.startswith(_PLACEHOLDER_PREFIXES):
    env_db_url = ""

DATABASE_URL: str = _normalise_driver(env_db_url) if env_db_url else _compose_postgres_url()
IS_POSTGRES: bool = DATABASE_URL.startswith("postgresql")

# True only when the operator explicitly asked for SQLite. There is no code
# path that sets this because something failed.
OFFLINE_SQLITE: bool = not IS_POSTGRES


def _build_engine(url: str):
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False}, echo=False)

    # connect_timeout bounds the startup probe. Without it an unreachable host
    # -- a container that is down, a firewalled port, a wrong hostname -- makes
    # the process hang on import instead of failing, and a server that never
    # finishes starting is harder to diagnose than one that refuses to.
    return create_engine(
        url,
        connect_args={"connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))},
        # pool_pre_ping costs one round trip per checkout and removes the
        # stale-connection failure that otherwise appears after the container
        # restarts underneath a long-running API process.
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )


class PostGISUnavailable(RuntimeError):
    """Raised when PostGIS is configured but cannot be reached.

    Deliberately fatal. See the module docstring: falling back would change
    the schema underneath a running application.
    """


engine = _build_engine(DATABASE_URL)

if IS_POSTGRES:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connected to PostgreSQL/PostGIS: %s", DATABASE_URL.split("@")[-1])
    except SQLAlchemyError as exc:
        detail = str(exc).splitlines()[0][:300]
        raise PostGISUnavailable(
            f"PostGIS is configured but unreachable at {DATABASE_URL.split('@')[-1]}.\n"
            f"  {detail}\n\n"
            "Fix one of these:\n"
            "  - start the server:      docker compose up -d db\n"
            "  - set the credentials:   POSTGRES_PASSWORD / DATABASE_URL in .env\n"
            "  - run offline on purpose: DATABASE_URL=sqlite:///data/fire_db.sqlite\n\n"
            "The fallback is not automatic by design: the geometry column exists "
            "only in PostGIS mode, so degrading silently would leave every "
            "proximity query returning nothing from a schema that looks healthy."
        ) from exc
else:
    Path("data").mkdir(parents=True, exist_ok=True)
    logger.warning(
        "Running on SQLite (%s). This is offline/test mode: concurrent writes "
        "serialise behind a single file lock and proximity queries run without "
        "a GiST index.",
        DATABASE_URL,
    )


if OFFLINE_SQLITE:
    # WAL lets readers proceed during a write. Without it the auto-refresh
    # thread's commit blocks every dashboard request for its duration. It does
    # not make SQLite safe for concurrent writers -- nothing does -- but it
    # removes the reader stall in the fallback path.
    @event.listens_for(engine, "connect")
    def _enable_sqlite_wal(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


if IS_POSTGRES:
    from geoalchemy2 import Geometry
else:  # pragma: no cover - exercised only in the SQLite fallback
    Geometry = None  # type: ignore[assignment]

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Incident(Base):
    """Stores classified fire incidents, facility attribution, and dispatch lifecycle status."""
    __tablename__ = "incidents"

    # No index=True on the primary key: PostgreSQL already builds a unique
    # index for it, and the extra declaration produced a second, redundant
    # B-tree (ix_incidents_id alongside incidents_pkey) that every insert had
    # to maintain for nothing. Index bytes had grown past heap bytes.
    id = Column(Integer, primary_key=True, autoincrement=True)
    detection_id = Column(String(64), unique=True, index=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    frp = Column(Float, nullable=False)
    bright_ti4 = Column(Float, nullable=True)
    bright_ti5 = Column(Float, nullable=True)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    satellite = Column(String(32), default="VIIRS")

    if IS_POSTGRES:
        # The reason PostGIS is here rather than a pair of indexed floats.
        # A bounding-box filter on latitude/longitude is wrong at range: a
        # degree of longitude is 111 km at the equator and 96 km at Srinagar,
        # so a "within 5 km" box is a different distance in Kerala than in
        # Kashmir. ST_DWithin over geography measures metres.
        #
        # GeoAlchemy2 creates the GiST index for this column automatically.
        geom = Column(Geometry(geometry_type="POINT", srid=4326), nullable=True)

    # Spatial & Facility Enrichment
    inside_industrial = Column(Boolean, default=False, index=True)
    facility_name = Column(String(255), nullable=True)
    facility_type = Column(String(64), default="non_industrial")
    dist_to_industrial_km = Column(Float, default=999.0)
    h3_index = Column(String(32), index=True)

    # Machine Learning & State Machine Decision
    predicted_class = Column(String(64), nullable=False)
    # Nullable, and that is the point. Detections outside any industrial
    # polygon are never sent to the model -- the pipeline deliberately does
    # not spend inference on background thermal activity. Those rows used to
    # carry a hard-coded 0.99, which the dashboard then rendered as
    # "THERMOSCOPE AI classifier assigned 99% confidence", crediting a model
    # that had not run, on 74% of the table. A confidence that was never
    # computed is now absent rather than invented.
    confidence = Column(Float, nullable=True)
    alert_priority = Column(String(32), nullable=False, index=True)
    alert_state = Column(String(32), nullable=False)
    shap_rationale = Column(Text, nullable=True)

    # Operator Lifecycle Management
    status = Column(String(32), default="OPEN", index=True)  # OPEN, ACKNOWLEDGED, DISPATCHED, RESOLVED
    operator_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # The dashboard's default view is "open incidents, worst first, newest
    # first". On SQLite that was three separate single-column indexes and a
    # sort; one composite index serves the whole query.
    _base_indexes = (
        Index("ix_incidents_triage", "status", "alert_priority", "timestamp_utc"),
        Index("ix_incidents_lat_lon", "latitude", "longitude"),
    )

    if IS_POSTGRES:
        # Two spatial indexes, because they answer different questions and the
        # first one alone is a trap.
        #
        # GeoAlchemy2 creates `idx_incidents_geom`, a GiST index on the
        # geometry column. That index serves geometry predicates -- map-extent
        # queries, ST_Intersects against a facility polygon.
        #
        # It does NOT serve `ST_DWithin(geom::geography, ...)`, which is what
        # a true-metre radius search compiles to. Measured on 200,004 rows:
        # with only the geometry index the planner chose a Parallel Seq Scan
        # at 268 ms; with this functional index on the cast expression it uses
        # a Bitmap Index Scan at 4.6 ms. The claim "GiST-indexed proximity
        # search" is false without this line, and the EXPLAIN is what proved
        # it rather than the assumption that one spatial index covers both.
        __table_args__ = _base_indexes + (
            Index(
                "idx_incidents_geog",
                text("(geom::geography)"),
                postgresql_using="gist",
            ),
        )
    else:
        __table_args__ = _base_indexes

    def to_dict(self):
        return {
            "id": self.id,
            "detection_id": self.detection_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "frp": self.frp,
            "bright_ti4": self.bright_ti4,
            "bright_ti5": self.bright_ti5,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
            "satellite": self.satellite,
            "inside_industrial": self.inside_industrial,
            "facility_name": self.facility_name,
            "facility_type": self.facility_type,
            "dist_to_industrial_km": self.dist_to_industrial_km,
            "h3_index": self.h3_index,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "alert_priority": self.alert_priority,
            "alert_state": self.alert_state,
            "shap_rationale": self.shap_rationale,
            "status": self.status,
            "operator_notes": self.operator_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


if IS_POSTGRES:
    @event.listens_for(Incident, "before_insert")
    def _populate_geom(_mapper, _connection, target):
        """Derives `geom` from latitude/longitude on every ORM insert.

        Attached as an event rather than set at each call site so that no
        write path can create a row whose geometry is missing -- a NULL geom
        is invisible to every ST_DWithin query, which would be a silent
        omission rather than an error.
        """
        if target.latitude is not None and target.longitude is not None:
            target.geom = f"SRID=4326;POINT({target.longitude} {target.latitude})"


class H3Baseline(Base):
    """Tracks 30-day thermal recurrence baselines per H3 hexagon cell."""
    __tablename__ = "h3_baselines"

    h3_index = Column(String(32), primary_key=True)  # PK already indexed
    n_30d = Column(Integer, default=0)
    mu_frp = Column(Float, default=0.0)
    var_frp = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "h3_index": self.h3_index,
            "n_30d": self.n_30d,
            "mu_frp": self.mu_frp,
            "var_frp": self.var_frp,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class AuditLog(Base):
    """Stores human-in-the-loop analyst decisions, verifications, and dispatch history."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # The audit trail is the evidence behind the CPCB register, so the link to
    # the incident it describes is enforced by the database rather than by
    # convention. ondelete=SET NULL keeps the row: an audit entry whose
    # incident was removed is still a record that the action happened, and
    # deleting it to satisfy a constraint would be destroying evidence.
    incident_id = Column(
        Integer,
        ForeignKey("incidents.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    detection_id = Column(String(64), index=True, nullable=False)
    action = Column(String(32), nullable=False)  # CONFIRM, RECLASSIFY, FALSE_POSITIVE, DISPATCH
    previous_class = Column(String(64), nullable=True)
    new_class = Column(String(64), nullable=True)
    operator_notes = Column(Text, nullable=True)
    timestamp_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "detection_id": self.detection_id,
            "action": self.action,
            "previous_class": self.previous_class,
            "new_class": self.new_class,
            "operator_notes": self.operator_notes,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
        }


def postgis_version() -> Optional[str]:
    """Returns the server's PostGIS version, or None when not on PostGIS."""
    if not IS_POSTGRES:
        return None
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT PostGIS_Version()")).scalar()
    except SQLAlchemyError:
        return None


def init_db():
    """Creates the PostGIS extension and all tables."""
    if IS_POSTGRES:
        # CREATE EXTENSION is idempotent and needs to run before create_all,
        # because the geometry type must exist before a column can declare it.
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        logger.info("PostGIS extension present: %s", postgis_version())

    Base.metadata.create_all(bind=engine)
    logger.info(
        "Initialized %s tables: incidents, h3_baselines, audit_logs.",
        "PostGIS" if IS_POSTGRES else "SQLite",
    )


def incidents_within_km(db, lat: float, lon: float, radius_km: float, limit: int = 200):
    """Returns incidents within `radius_km` of a point, nearest first.

    On PostGIS this is `ST_DWithin` over `geography`, which measures true
    metres and uses the GiST index. On the SQLite fallback it degrades to a
    bounding-box prefilter followed by an exact haversine sort -- correct, but
    a table scan, and the reason the fallback is documented as degraded.
    """
    if IS_POSTGRES:
        stmt = text(
            """
            SELECT id, ST_Distance(geom::geography,
                                   ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                   ) / 1000.0 AS distance_km
            FROM incidents
            WHERE geom IS NOT NULL
              AND ST_DWithin(geom::geography,
                             ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                             :radius_m)
            ORDER BY distance_km ASC
            LIMIT :limit
            """
        )
        rows = db.execute(
            stmt,
            {"lat": lat, "lon": lon, "radius_m": radius_km * 1000.0, "limit": limit},
        ).all()
        return [(int(r[0]), float(r[1])) for r in rows]

    # SQLite fallback: degrees-to-km is latitude dependent, so the box is
    # widened by 1/cos(lat) in longitude and the exact distance is computed
    # afterwards. Deliberately generous -- a box that is too small silently
    # drops real neighbours.
    import math

    lat_pad = radius_km / 111.32
    lon_pad = radius_km / max(1e-6, 111.32 * math.cos(math.radians(lat)))
    candidates = (
        db.query(Incident)
        .filter(Incident.latitude.between(lat - lat_pad, lat + lat_pad))
        .filter(Incident.longitude.between(lon - lon_pad, lon + lon_pad))
        .all()
    )
    out = []
    for inc in candidates:
        dlat = math.radians(inc.latitude - lat)
        dlon = math.radians(inc.longitude - lon)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat)) * math.cos(math.radians(inc.latitude)) * math.sin(dlon / 2) ** 2
        )
        d = 6371.0 * 2 * math.asin(math.sqrt(a))
        if d <= radius_km:
            out.append((inc.id, d))
    out.sort(key=lambda t: t[1])
    return out[:limit]


def get_db() -> Generator:
    """FastAPI dependency yielding database sessions with safe commit/rollback."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
