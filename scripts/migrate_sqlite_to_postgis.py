"""Moves an existing SQLite incident database into PostGIS.

The analytical corpus is not touched: detections live in Parquet and are read
columnar. What moves here is the transactional state that SQLite held --
incidents, H3 recurrence baselines, and the audit log, which is the half that
matters most because it is the regulatory evidence trail behind the CPCB
register.

Usage:

    docker compose up -d db
    python scripts/migrate_sqlite_to_postgis.py                   # migrate
    python scripts/migrate_sqlite_to_postgis.py --dry-run         # count only
    python scripts/migrate_sqlite_to_postgis.py --source path.db

The migration is idempotent on `detection_id`: rows already present in PostGIS
are skipped rather than duplicated, so an interrupted run can simply be
repeated. Nothing is deleted from the SQLite file -- it is left intact as the
offline-replay database.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate")

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "fire_db.sqlite"

BATCH = 2000


def _ts_key(value) -> str:
    """Canonical string for a timestamp coming from either database.

    SQLite hands back `timestamp_utc` as a string; PostgreSQL hands back a
    `datetime`. Comparing the two directly never matches, which silently broke
    the audit log's idempotency check: a second migration run re-inserted
    every audit row, and 31 evidence entries became 61. Duplicated evidence is
    corrupt evidence, so both sides are normalised to microsecond-precision
    ISO text before the tuples are compared.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text_value = value.strip().replace("T", " ")
        # SQLite writes '...12:00:00' or '...12:00:00.123456'; pad so both
        # forms produce the same key.
        if "." not in text_value and len(text_value) == 19:
            text_value += ".000000"
        return text_value
    iso = value.replace(tzinfo=None).isoformat(sep=" ", timespec="microseconds")
    return iso


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="SQLite file to migrate from")
    ap.add_argument("--dry-run", action="store_true", help="Report counts without writing")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        logger.error("Source database not found: %s", source)
        return 1

    # Import after argument parsing so a --help run does not need a live server.
    from app.database import (  # noqa: E402
        AuditLog,
        H3Baseline,
        Incident,
        IS_POSTGRES,
        SessionLocal,
        init_db,
    )

    if not IS_POSTGRES:
        logger.error(
            "Destination is not PostGIS. Start the server (docker compose up -d db) "
            "and set DATABASE_URL or POSTGRES_* in .env, then re-run."
        )
        return 2

    init_db()

    src_engine = create_engine(f"sqlite:///{source.as_posix()}")
    SrcSession = sessionmaker(bind=src_engine)

    # Read the source through raw SQL rather than the ORM: the ORM models now
    # declare a geometry column that the SQLite file does not have, and
    # reflecting the old schema directly avoids having to keep a second set of
    # legacy model classes alive purely for the migration.
    moved = {"incidents": 0, "h3_baselines": 0, "audit_logs": 0}
    skipped = {"incidents": 0, "audit_logs": 0}

    with SrcSession() as src, SessionLocal() as dst:
        src_tables = {
            r[0] for r in src.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).all()
        }

        # ---- incidents -----------------------------------------------------
        if "incidents" in src_tables:
            existing = {r[0] for r in dst.execute(text("SELECT detection_id FROM incidents")).all()}
            rows = src.execute(text("SELECT * FROM incidents")).mappings().all()
            logger.info("incidents: %d in source, %d already in PostGIS", len(rows), len(existing))

            batch = []
            for r in rows:
                if r["detection_id"] in existing:
                    skipped["incidents"] += 1
                    continue
                # geom is not copied: the before_insert event derives it from
                # latitude/longitude, so there is exactly one code path that
                # decides what a row's geometry is.
                batch.append(
                    Incident(
                        detection_id=r["detection_id"],
                        latitude=r["latitude"],
                        longitude=r["longitude"],
                        frp=r["frp"],
                        bright_ti4=r["bright_ti4"],
                        bright_ti5=r["bright_ti5"],
                        timestamp_utc=r["timestamp_utc"],
                        satellite=r["satellite"],
                        inside_industrial=bool(r["inside_industrial"]),
                        facility_name=r["facility_name"],
                        facility_type=r["facility_type"],
                        dist_to_industrial_km=r["dist_to_industrial_km"],
                        h3_index=r["h3_index"],
                        predicted_class=r["predicted_class"],
                        confidence=r["confidence"],
                        alert_priority=r["alert_priority"],
                        alert_state=r["alert_state"],
                        shap_rationale=r["shap_rationale"],
                        status=r["status"],
                        operator_notes=r["operator_notes"],
                        created_at=r["created_at"],
                    )
                )
                if len(batch) >= BATCH and not args.dry_run:
                    dst.add_all(batch)
                    dst.commit()
                    moved["incidents"] += len(batch)
                    batch = []
            if batch and not args.dry_run:
                dst.add_all(batch)
                dst.commit()
                moved["incidents"] += len(batch)
            elif batch:
                moved["incidents"] += len(batch)

        # ---- h3 baselines --------------------------------------------------
        if "h3_baselines" in src_tables:
            existing_h3 = {r[0] for r in dst.execute(text("SELECT h3_index FROM h3_baselines")).all()}
            rows = src.execute(text("SELECT * FROM h3_baselines")).mappings().all()
            batch = [
                H3Baseline(
                    h3_index=r["h3_index"],
                    n_30d=r["n_30d"],
                    mu_frp=r["mu_frp"],
                    var_frp=r["var_frp"],
                    last_updated=r["last_updated"],
                )
                for r in rows
                if r["h3_index"] not in existing_h3
            ]
            moved["h3_baselines"] = len(batch)
            if batch and not args.dry_run:
                dst.add_all(batch)
                dst.commit()

        # ---- audit log -----------------------------------------------------
        if "audit_logs" in src_tables:
            # Remap incident_id through detection_id.
            #
            # This was a real bug: the first version copied incident_id
            # verbatim from SQLite while PostgreSQL assigned brand-new ids
            # from the SERIAL sequence. The two happened to agree on the
            # first run, because rows were inserted in rowid order into an
            # empty table -- luck, not correctness. Re-running against a
            # partly-populated destination, or migrating a second SQLite
            # file, would have silently pointed audit rows at the WRONG
            # incidents. For a trail described as regulatory evidence, a
            # silently wrong link is worse than a crash.
            id_by_detection = {
                r[0]: r[1]
                for r in dst.execute(text("SELECT detection_id, id FROM incidents")).all()
            }
            src_detection_by_id = {
                r[0]: r[1]
                for r in src.execute(text("SELECT id, detection_id FROM incidents")).all()
            }
            # The audit trail is append-only evidence. Deduplicate on the whole
            # tuple rather than on an id, because ids are reassigned by the
            # destination sequence and would collide meaninglessly.
            existing_audit = {
                (r[0], r[1], _ts_key(r[2]))
                for r in dst.execute(
                    text("SELECT detection_id, action, timestamp_utc FROM audit_logs")
                ).all()
            }
            rows = src.execute(text("SELECT * FROM audit_logs")).mappings().all()
            batch = []
            for r in rows:
                key = (r["detection_id"], r["action"], _ts_key(r["timestamp_utc"]))
                if key in existing_audit:
                    skipped["audit_logs"] += 1
                    continue
                # Resolve the destination id via the detection_id the audit
                # row carries; fall back to the source incident table when the
                # audit row's own detection_id is absent.
                detection = r["detection_id"] or src_detection_by_id.get(r["incident_id"])
                remapped = id_by_detection.get(detection)
                if r["incident_id"] is not None and remapped is None:
                    logger.warning(
                        "audit row for detection %s has no matching incident in "
                        "PostGIS; storing it with a null incident_id rather than "
                        "a wrong one.", detection,
                    )
                batch.append(
                    AuditLog(
                        incident_id=remapped,
                        detection_id=r["detection_id"],
                        action=r["action"],
                        previous_class=r["previous_class"],
                        new_class=r["new_class"],
                        operator_notes=r["operator_notes"],
                        timestamp_utc=r["timestamp_utc"],
                    )
                )
            moved["audit_logs"] = len(batch)
            if batch and not args.dry_run:
                dst.add_all(batch)
                dst.commit()

        # ---- verification --------------------------------------------------
        if not args.dry_run:
            total = dst.execute(text("SELECT COUNT(*) FROM incidents")).scalar()
            with_geom = dst.execute(
                text("SELECT COUNT(*) FROM incidents WHERE geom IS NOT NULL")
            ).scalar()
            logger.info("PostGIS now holds %d incidents, %d with geometry", total, with_geom)
            dupes = dst.execute(text(
                "SELECT count(*) - count(DISTINCT (detection_id, action, timestamp_utc)) "
                "FROM audit_logs"
            )).scalar()
            if dupes:
                logger.error(
                    "%d duplicate audit rows present. The evidence trail must not "
                    "contain repeats; de-duplicate before relying on it.", dupes,
                )
                return 5

            crossed = dst.execute(text(
                "SELECT count(*) FROM audit_logs a JOIN incidents i "
                "ON i.id = a.incident_id WHERE i.detection_id <> a.detection_id"
            )).scalar()
            if crossed:
                logger.error(
                    "%d audit rows point at an incident with a different "
                    "detection_id -- the evidence trail is misaligned.", crossed,
                )
                return 4

            if total != with_geom:
                # A NULL geometry is invisible to every ST_DWithin query, so a
                # partial population is a silent data loss rather than an error.
                logger.error(
                    "%d incidents have no geometry and will not appear in any "
                    "proximity query.", total - with_geom,
                )
                return 3

    verb = "would migrate" if args.dry_run else "migrated"
    logger.info(
        "%s: %d incidents, %d h3 baselines, %d audit rows "
        "(skipped as already present: %d incidents, %d audit rows)",
        verb, moved["incidents"], moved["h3_baselines"], moved["audit_logs"],
        skipped["incidents"], skipped["audit_logs"],
    )
    logger.info("Source left intact at %s (offline-replay database).", source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
