# SIH26162 — Industrial Fire & Persistent Thermal Source Classification

**Problem Statement 26162 (NTRO)** — AI-Based Detection and Classification of Industrial
Fires and Persistent Thermal Sources Using NASA FIRMS, OSM & Satellite Data.

A thermal surveillance pipeline that ingests NASA FIRMS satellite detections, grounds them
against OpenStreetMap infrastructure, classifies them with XGBoost, and routes only what an
incident commander should see — narrowing **2,044,295 detections over 365 days** to **530
alerts a year**, a suppression ratio of roughly 1 in 3,900.

> **Read [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) before quoting any accuracy
> figure from this project.** Section 5b explains why the headline number and the honest
> number are different, and what each one means.

---

## The one thing to understand first

No public per-detection ground truth exists for Indian industrial thermal anomalies. The
2,044,295 training labels were therefore produced by a **deterministic rule**, and the model
is trained to reproduce that rule.

| Figure | Value | What it actually says |
| :--- | ---: | :--- |
| Spatial-block CV macro F1 | **0.9961** | The model reproduces *its own labelling rule* almost perfectly across unseen geography. **Not** field accuracy |
| **Verified-label macro F1** | **0.5544** | Measured against **21 externally cited events**. The only figure independent of the rule |
| Circularity delta | **0.2631** over 14 columns | How much of the headline is the model reciting its own rule back |

**Quote the 0.5544.** The gap between the two is the subject of the documentation, not an
embarrassment to be hidden.

---

## Quickstart

Requires **Python 3.14** and, optionally, Docker.

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
cp .env.example .env                                # then fill in your own keys
python -m uvicorn app.main:app --port 8000
```

Dashboard at **http://127.0.0.1:8000/**, Swagger at `/docs`, ReDoc at `/redoc`.

### ⚠️ A fresh clone will not work until you fetch the data

**The derived reference layers ARE committed** (88 MB: 28,587 industrial polygons, 76,186
forest, 61,001 responders), so a fresh clone classifies correctly out of the box. What is
*not* committed is the 1.8 GB OSM extract they came from and the FIRMS corpora, which are
gitignored. Without a corpus the system starts and `POST /api/v1/classify` works, but the
incident list is empty and the compliance register and H3 map report no data.

| Asset | Size | How to get it |
| :--- | ---: | :--- |
| `data/reference/india-*.osm.pbf` | 1.8 GB | [Geofabrik India extract](https://download.geofabrik.de/asia/india.html) |
| `osm_india_industrial_from_pbf.parquet` | 7 MB | `python -m src.ingestion.pbf_extractor --layer industrial` |
| `osm_india_forest_from_pbf.parquet` | 127 MB | `python -m src.ingestion.pbf_extractor --layer forest` |
| `osm_india_responders_from_pbf.parquet` | 3 MB | `python -m src.ingestion.pbf_extractor --layer responders` |
| `firms_industrial_joined_12m.parquet` | 144 MB | FIRMS archive pull — needs `FIRMS_MAP_KEY` |
| `data/fire_db.sqlite` | 1.6 MB | Created on first run; seeded from the processed corpus |

Each extractor run takes several minutes: GDAL builds a temporary node cache on first pass.

**Ask a teammate for the built layers rather than rebuilding them** — they are deterministic
outputs, and copying 137 MB is faster than a 1.8 GB download plus three extractions.

### Database

`.env` ships with the **offline default**:

```
DATABASE_URL=sqlite:///data/fire_db.sqlite
```

No server, no Docker. `/api/v1/health` reports `"database_mode": "sqlite_offline"` and
`"offline_mode": true` — SQLite is an explicit choice here, never an automatic fallback.

**PostGIS is the deployment target.** To use it:

```bash
docker compose up -d db                              # PostGIS 16-3.4, healthchecked
# then in .env, swap to:
# DATABASE_URL=postgresql://sih:sih_local_dev@localhost:5432/sih26162
python scripts/migrate_sqlite_to_postgis.py
```

`docker-compose.yml` and `app/database.py` read the same `POSTGRES_*` names **and the same
defaults**, so this needs no further configuration. A test parses the compose file and asserts
they match.

**PostGIS does not fall back to SQLite on its own.** If it is configured and unreachable the
app raises at startup and tells you the three ways to fix it — because the geometry column
exists only in PostGIS mode, so degrading silently would leave every proximity query returning
nothing from a database reporting itself healthy.

---

## What it does

```
NASA FIRMS (VIIRS 375m + MODIS 1km, NRT + SP)
        ↓
OSM spatial join — 28,587 industrial polygons, 76,186 forest, 61,001 responders
        ↓
H3 recurrence state machine (res 9 index, res 8 persistence, res 6 neighbourhood)
        ↓
XGBoost — 33 deliberately coordinate-free features, labelling rule v8
        ↓
apply_serving_guards() — non-combustion · out-of-domain · forest cover
        ↓
FastAPI dashboard · TreeSHAP attribution · dispatch
```

### Events, not pixels

Detections are grouped into **events** before anything reasons about them — one fire is one
row, not one row per overpass. `src/pipeline/event_builder.py` groups by recurrence key,
splits on gaps over 48 h, and links detections within 1 km (the same parallax budget
`persistence_resolution` was chosen for).

That unlocks features the labelling rule structurally cannot see, because they are properties
of a *group*: `duration_h`, `centroid_drift_km`, `extent_km`, `frp_trend_mw_per_day`. Drift is
the one worth arguing for — a flare stack is bolted down and a fire front is not, and that
distinction owes nothing to OpenStreetMap coverage. Measured on synthetic sources, a facility
drifts 0.041 km/day against a moving front's 0.432.

`evaluate_events_against_verified()` then scores the register **one citation at a time**.
This matters more than it sounds: a model answering `PERSISTENT_BASELINE` for everything
scores 0.9785 detection-weighted and 0.4000 event-weighted, because Jharia's 23,633 pixels
drown Buncefield's 5.

Whether these features carry real signal is still a hypothesis — the circularity delta after
retraining is what decides it, and that number does not exist yet.

**Four trained classes plus one served-only class:** `PERSISTENT_BASELINE`,
`ACCIDENTAL_FIRE`, `AGRICULTURAL_BURN`, `TRANSIENT_HOTSPOT`, and `FOREST_FIRE` — which the
model is *never trained on*, because a forest fire is radiometrically identical to a crop
fire and only land cover can separate them.

**Coordinates are never features.** Industrial facilities are static, so a model given
latitude and longitude memorises where refineries are and collapses on unseen geography. A
test enforces the exclusion permanently; the cost is paid at serving time instead.

### Three engines, each doing what it is good at

| Workload | Engine | Why |
| :--- | :--- | :--- |
| Incident state, audit log, query-time geometry | **PostgreSQL + PostGIS** | Concurrent writers, durability, GiST-indexed `ST_DWithin` |
| Bulk point-in-polygon enrichment | **GeoPandas R-tree** | 2.04M × 28,587 in **1.22 s**; per-row SQL would be slower |
| Analytical corpus | **Parquet** | 2.04M rows read columnar in **0.05 s** |

---

## What makes this different

**The counter-register.** Six documented industrial accidents that produced **no usable FIRMS
signature** — fires that burned between overpasses, heat confined inside a boiler, a blast at
a plant that is always hot, and a toxic release that was never thermal at all. Each measured
with a control proving the retrieval worked. Every accuracy figure here is conditional on the
fire being visible to a polar-orbiting radiometer at the moment it passes overhead, and
section 5c states exactly what that costs.

**The circularity audit.** Retrains with all 14 label-rule columns ablated and publishes the
drop: `0.9976 → 0.7345`. Read the *ablation count* beside the delta — it moved 0.3024 → 0.2938
→ 0.2458 → 0.2502 → 0.2629 → 0.2631 across rule versions 3 to 8 while the ablated set grew
from 7 columns to 14. A rise can mean the audit got stricter, not that the model got worse,
and only the count tells you which.

**Rejection on measurement.** Of five candidate rule changes, **two were adopted and three
rejected**: an inside-polygon surge test (+0.0040, and it degraded every persistent anchor),
a coarse onset key (it sent long-lag crop burns from 2.17% to 56.04% — it taught the rule that
Punjab agriculture began in October), and an OSM farmland feature (27,148 km² mapped, and
**0 of 2,000** verified crop burns fell inside a polygon).

**Refusals.** CO₂ liability is not computed under a regulatory heading, enforced by test.
Sub-pixel fire temperature was implemented, validated on two instruments, and rejected both
times. Travel time to responders is explicitly *not routed*, and the payload carries the two
assumptions that produced it.

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest -q
```

**468 tests, 20 skipped.** The suite sets its own SQLite URL before anything imports, so it
never touches the live database and needs no server.

Notable guards: no coordinate can reach the feature space; the weak-labelling rule cannot file
an established refinery flare as an accident; `LABEL_RULE_FEATURES` stays in sync with the rule
that reads it; the metrics file always carries its provenance and caveats; the generated
register in §5c still contains every event and citation, so a stale evidence table fails the
build instead of reading as authoritative.

---

## Layout

```
app/          FastAPI service — database.py (PostGIS ORM), main.py (29 endpoints), templates/
src/
  ingestion/  FIRMS, OSM PBF extraction, Sentinel-2, Sentinel-3 SLSTR
  pipeline/   spatial join, event builder, event features, forest cover, responders, replay
  models/     XGBoost trainer, labelling rule v8, verified labels, explainability
  alerting/   H3 state machine, SitRep generator, dispatcher
  reporting/  H3 aggregation for the map
scripts/      corpus utilities, register generation, PostGIS migration
tests/        468 tests
```

---

## Known limitations

Stated here rather than discovered by a reader. Full detail in §5b.5.

- **`ACCIDENTAL_FIRE` recall is 0.396** — and **443 of the 490 misses are Baghjan**, a
  five-month blowout that became its own baseline so recurrence reports it as routine. The fix
  is hysteresis, it is architectural rather than a threshold, and it has not been attempted.
- **Rumaila flare field scores 0.065.** Recurrence is keyed on facility identity where a
  polygon exists and a ~900 m cell where none does; a dispersed 80 km flare field never
  reaches the threshold. The notion of "persistent" is borrowed from a map, and the map stops
  at the border.
- **Deer Park scores 0.182.** The labelling rule generalises further than the trained model:
  the rule declines to apply India's harvest calendar abroad, and the model cannot, because it
  has no coordinates and its corpus is entirely Indian.
- **21 verified events is enough to expose defects and reject fixes. It is not enough to
  certify accuracy.**
- **Responder coverage is uneven.** OSM maps 55,714 hospitals and only **741 fire stations**
  nationally. An absent facility means absent from the map, not from the ground.
- **Timestamps are `timestamp without time zone`.** Every writer stamps UTC so the data is
  correct, but the database does not enforce it.
- **Basemap tiles are the one unresolved offline gap** in Historical Replay Mode. Eight of
  eight other offline capabilities have their assets on disk.
- **INSAT-3D/3DS and VNP14IMG need registrations this project does not hold.**

---

## Configuration

All optional except `FIRMS_MAP_KEY`. See [`.env.example`](.env.example); `.env` is gitignored
and never committed.

| Variable | Purpose |
| :--- | :--- |
| `FIRMS_MAP_KEY` | NASA FIRMS — required for live ingestion |
| `DATABASE_URL` | Defaults to offline SQLite; PostGIS for deployment |
| `CDSE_CLIENT_ID` / `_SECRET` | Copernicus — Sentinel-2 dNBR validation |
| `ALERT_DISPATCH_ENABLED` | **`false` by default, deliberately** — a replay over a year of archived detections must not be able to page a control room |
| `ALERT_WEBHOOK_URL` / `_FORMAT` | `text` mode delivers a readable alert to a phone via ntfy/Gotify with no account |

---

*NTRO · Smart India Hackathon 2026 · Problem Statement 26162*
