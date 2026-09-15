# Project SIH26162: AI-Based Detection & Classification of Industrial Fires and Persistent Thermal Sources

> **Sponsored by:** National Technical Research Organisation (NTRO) / Government of India  
> **Problem Statement ID:** SIH26162  
> **Core Objective:** Autonomous, high-precision detection, recurrence tracking, and classification of industrial thermal anomalies using NASA FIRMS NRT satellite feeds, OpenStreetMap (OSM) infrastructure, and ISRO Bhuvan land use data.  
> **Repository:** `d:\industrial-fire-classifier`  
> **Runtime Environment:** Python 3.14.7 | FastAPI | XGBoost | Leaflet.js + deck.gl | **PostgreSQL + PostGIS**

---

## 1. Executive Summary & Problem Formulation

In industrial belts, chemical hubs, and petroleum refineries (such as Dahej PCPIR, Ankleshwar GIDC, and Jamnagar), satellite-based thermal sensors frequently register high-intensity heat anomalies. Standard remote sensing pipelines flag all high-temperature detections identically, leading to severe **alert fatigue** caused by:
1. Routine, controlled operational heat sources:
   - Refinery flare stacks (continuous or intermittent pressure relief)
   - Steel plant blast furnaces and basic oxygen converters
   - Thermal power plant cooling towers and boiler exhaust
   - Brick kilns and metallurgical smelting furnaces
2. Agricultural residue burning and seasonal forest fires adjacent to industrial corridors.

When a genuine catastrophic emergency occurs (such as a boiler explosion, chemical storage tank breach, or crude distillation unit fire), response units need **immediate tactical intelligence**, not ambiguous satellite points.

### The Solution: Project SIH26162
This project provides a full-stack, tactical Command and Control (C2) intelligence platform that:
- **Ingests Real-Time Satellite Data**: NASA FIRMS NRT (VIIRS 375m & MODIS 1km), re-ingested on a background interval (3 hours by default, floored at 30 minutes), with each sensor routed between its near-real-time stream and its standard-processing archive per window. An incremental write that would shrink the archive is refused rather than applied.
- **Fuses Geospatial Layers**: Combines OpenStreetMap (OSM) industrial polygons with ISRO Bhuvan 1:50,000 Land Use / Land Cover (LULC) vector data.
- **Validates Optically Against Sentinel-2**: Computes the Normalized Burn Ratio delta (dNBR) to establish whether a fire consumed surrounding biomass or was contained inside the facility — the one signal in the feature vector independent of thermal radiometry. It is now **drawn**: a burn-scar overlay renders all 530 P0 alerts, **425 measurable** (342 contained, 60 perimeter scorching, 21 vegetation gain, 2 breached) and **105 that could not be observed**, the latter as hollow rings, because cloud is an absence of observation rather than evidence that nothing burned. *(Requires free CDSE credentials; the pipeline degrades cleanly without them.)*
- **Applies R-Tree Spatial Intersections**: Microsecond point-in-polygon containment and nearest-perimeter distance calculations, with a scan-angle-adaptive buffer that absorbs off-nadir pixel growth and plume parallax.
- **Maintains an Uber H3 Recurrence State Machine**: Tracks 30-day thermal baselines to separate normal operational flaring from an abnormal surge. Recurrence is keyed on **facility identity** where a detection lands inside a mapped industrial polygon, and on a resolution-8 cell (~900 m) where it does not. Keying purely on a resolution-9 cell (~174 m) was measured wrong: 43 detections at the Jamnagar complex scattered across 18 cells, never reached the persistence threshold, and 42 of them were misclassified (section 5b.2). Outside mapped infrastructure the cell remains the key, which is correct for wildfires -- a spreading front *should* fragment, because it genuinely is moving.
- **Executes XGBoost Multi-Class Classification**: Predicts whether a detection is a *Persistent Baseline* source, an *Accidental Industrial Fire*, an *Agricultural Burn*, or a *Transient Hotspot*, from **33 deliberately coordinate-free features** -- a model given latitude and longitude memorises where refineries are and collapses on unseen regions, and a test enforces the exclusion permanently. Evaluated under spatial-block cross-validation with a published circularity audit (section 5b), and constrained at serving time by `apply_serving_guards()`, which withholds claims the coordinate-free model has no way to withhold for itself.
- **Generates Game-Theoretic TreeSHAP Explanations**: Attributes each prediction to the features that drove it, computed through native XGBoost `pred_contribs` with a pure-Python fallback, and surfaced as ranked contributing factors per incident. The dashboard renders the factors the model actually used -- an earlier build showed a fixed illustrative set identical for every incident.
- **Produces Automated Situation Reports (SitRep)**: Formats MGRS/UTM coordinates, Gaussian plume dispersion under Pasquill-Gifford stability classes, indicative $\text{SO}_2 / \text{NO}_x / \text{PM}_{2.5}$ emission rates and hazard radii for incident commanders. Weather is read from the Open-Meteo archive at the *incident's own hour*; when it cannot be, a flagged synthetic fallback returns and four separate surfaces declare it. **The emission factors are hand-tuned scaling coefficients, not US EPA AP-42** -- the ratios between facility types are meaningful, the absolute mass rates are indicative, and every response carries `estimate_basis: HAND_TUNED_SCALING` (section 4).
- **Features an Interactive "What-If" Simulation Sandbox**: Allows safety engineers to drag simulation reticles, adjust FRP, temperature, and wind vectors, and view dynamically rotated atmospheric plumes in real time.
- **Surfaces the Compliance Register**: the mandate gives routine flaring its own row -- passive logging for CPCB auditing rather than dispatch -- and it now has a screen. The register reports the detections the pipeline **suppressed**: **485,781** routine detections across a **364-day** window and 25 sites, Rourkela Steel Plant continuous on **257** of them. Gas volume and CO2 are deliberately absent and the panel states why, rather than leaving a reader to assume an oversight.
- **Dispatches, and Shows Its Own Safety**: the dossier carries a dispatch control wired to `POST /api/v1/incident/{id}/dispatch`. It renders every channel's status verbatim rather than a success message -- with `ALERT_DISPATCH_ENABLED` unset a P0 reports `sms NOT_IMPLEMENTED`, `email NOT_CONFIGURED`, `webhook NOT_CONFIGURED` and says plainly that nothing was sent, and a `NON_ALERT` incident is refused at the router with the reason. A button reading "Sent" over a dry run would be the fabricated delivery receipt the dispatcher's own docstring exists to prevent.
- **Probes a Second Instrument**: the dossier queries Sentinel-3 SLSTR per incident. It separates what was **measured** -- F1, S7, the annulus background and the excess over it -- from what was merely **computed**: `fire_temperature_k` is rendered struck through under "rejected on validation", because the same retrieval returned 466 K for a photovoltaic array. Deleting it would hide a real result; printing it plainly would launder one.
- **Delivers a Tactical C2 Dashboard**: an operator console on unwatermarked Esri World Imagery, in light or dark, with one rule running through the palette -- **colour is data and the chrome carries no hue**. Every panel, border and active state is a neutral; the only saturated pixels are the map, the four alert tiers and Fire Radiative Power. An earlier build used amber chrome, and amber is also a P2 advisory, so the interface shouted at exactly the frequency of its own alerts. A deck.gl H3 layer draws the entire 2.04M-detection archive as 2,828 cells.
- **Asks Whether The Neighbourhood Was Already Burning**: an onset -- a source that was quiet and then started -- is the strongest accident signature available, and outside mapped infrastructure it misfires badly. A coal-seam front migrating through unmapped terrain lights a succession of cells, each individually new, and every one reads as a fresh ignition: **1,404 of 1,709 false accidents were Jharia**. Rule **v8** gates the branch on `neighbourhood_active_keys`, the count of *other* sources active nearby in the trailing 30 days. A migrating front is surrounded by its own recent history; an isolated blowout is not, even five months in. `ACCIDENTAL_FIRE` precision **0.142 → 0.183** with recall unchanged.
- **Conditions Its Artifact Floor On Land Cover**: the energy floor below which an open-ground detection is treated as an artifact encodes a prior -- *low-energy detections are mostly specular glint* -- which holds over bare ground, water and metal and **fails over tree canopy**, which is dark and diffuse. Rule **v7** gives forested ground its own floor (0.82 MW against 1.59 harvest, 3.00 normal). Found by the verified forest fires: 2,117 of the 2,225 detections wrongly called artifacts were theirs, at a median 1.07 MW, and 2,174 of 2,225 carried *nominal* rather than low detection confidence. Verified macro F1 **0.4967 → 0.5425**.
- **Segregates Forest Fires From Industrial Ones**: the mandate asks for industrial fires to be *"explicitly segregated from forest fires and natural thermal events"*, and radiometry alone cannot do it -- a forest fire and a crop fire are both sustained open-ground combustion with no facility history, and the classifier is coordinate-free by design. **Land cover decides it instead.** **76,186** forest and natural-wood polygons are extracted from the same national OSM file the industrial layer comes from, and the class is scored against three cited fires -- Bandipur 2019, Similipal 2021 and the 2016 Uttarakhand fires, and `apply_forest_cover()` refines an open-ground burn inside mapped forest into `FOREST_FIRE` at serving time. It is routed to forestry rather than industrial response: segregated, not discarded.
- **Survives the Venue's Network**: every live dependency here is somebody else's uptime -- FIRMS, Copernicus, Open-Meteo and the basemap tiles. **Historical Replay Mode** pauses ingestion and serves the local archive, and the freshness badge reads `REPLAY` rather than `STALE`, which are different claims: STALE means nobody has ingested anything and you should worry, REPLAY means ingestion is deliberately frozen. A preflight reports what survives per capability -- currently **8 of 8 offline capabilities ready, 5 network-dependent** -- and names the one unresolved gap, basemap tiles, rather than hiding it. Verified by serving ten archive-backed endpoints with ingestion paused.
- **Records What It Cannot See**: alongside the verified-event register, the project keeps a **counter-register of six documented industrial accidents that produced no usable signature** -- fires that burned between satellite overpasses, heat confined inside boilers, a blast at a plant that is always hot, and a toxic release that was never thermal at all. Every accuracy figure in this project is conditional on the event being visible to a polar-orbiting radiometer at the moment it passes overhead, and section 5c states exactly what that costs.

---

## 2. End-to-End System Architecture

```
                +--------------------------------------------------------------------+
                |                   REMOTE SENSING & REFERENCE DATA                  |
                +--------------------------------------------------------------------+
       |                  |                  |                  |                  |
  NASA FIRMS      ISRO Bhuvan LULC   Geofabrik OSM PBF   Copernicus S-2    Copernicus S-3
 VIIRS 375m NRT    1:50k shapefiles   industrial tags     L2A B08/B12       SLSTR F1/F2
 MODIS 1km NRT/SP                                          @ 20m          3.74 / 10.85 um
       |                  |                  |                  |                  |
       v                  v                  v                  v                  v
 firms_client.py   bhuvan_loader.py   pbf_extractor.py  sentinel2_client   slstr_client.py
 auto_refresh.py           \                 / \                 |                  |
       |                    v               v   v                |                  |
       |          industrial_boundaries_merged   osm_india_forest |                  |
       |             (28,587 polygons)          (land cover)      |                  |
       |                          |                  |           |                  |
       |                          |                             |                  |
       +------------+-------------+                             |                  |
                    v                                           |                  |
          spatial_join.py  (R-tree)                             |                  |
   [inside polygon | exact match | distance to boundary         |                  |
    scan-angle parallax buffer: 1500m + scan/2                  |                  |
    non-combustion land use (solar, wind) withdrawn]            |                  |
                    |                                           |                  |
                    v                                           |                  |
          state_machine.py  (Uber H3)                           |                  |
   [30-day baselines keyed on FACILITY inside mapped polygons,  |                  |
    on a res-8 cell (~900m) outside | onset lag | burst rate]   |                  |
                    |                                           |                  |
                    |   dNBR burn-scar corroboration of         |                  |
                    +<-- ambiguous detections only -------------+                  |
                    |      (rationed, cached, optional)                            |
                    |                                                              |
                    |   sub-pixel thermal probe, on demand                         |
                    +<-------------------------------------------------------------+
                    v
          feature_engineering.py
   [33 features: radiometry, diurnal ratio, geometry, sensor confidence,
    recurrence, burst rate, neighbourhood density, land cover,
    dNBR + availability flag.
    NO coordinates -- spatial leakage guard, permanently test-enforced]
                    |
                    v
          train_classifier.py  (XGBoost)
   [labelling rule v8 | spatial-block CV | circularity audit]
                    |                                    ^
                    |                                    |  scored independently by
                    |                                    +--- verified_labels.py
                    v                                         (18 cited events,
          apply_serving_guards()                               6 undetected incidents)
   [1 non-combustion land use | 2 out-of-domain harvest calendar
    3 forest cover -> FOREST_FIRE  (forest_cover.py + the land-cover layer)]
                    |
                    v
          explainability.py  (TreeSHAP)
                    |
      +-------------+--------------+---------------------+--------------------+
      v                            v                     v                    v
 sitrep_generator.py        dispatcher.py      compliance_register.py   FastAPI backend
 MGRS/UTM, Gaussian         webhook + SMTP     CPCB flaring log from    app/main.py
 plume, hand-tuned          routed by          SUPPRESSED detections;   23 REST endpoints
 emission scaling           priority;          no CO2 or gas volume     hex_aggregator.py
 (NOT AP-42)                OFF by default     (uncalibrated)           (2.04M -> 2,828)
      |                            |                     |                    |
      +-------------+--------------+---------------------+--------------------+
                    |
                    v
          Tactical C2 dashboard
   [achromatic chrome -- colour reserved for the four alert tiers and FRP
    Leaflet + deck.gl H3 layer | Esri World Imagery | light and dark]
```

---

## 3. Directory Structure & Key Files

```
d:\industrial-fire-classifier\
│
├── .env                              # Environment variables (API keys, DB URLs, configs)
├── .env.example                      # Template for configuration variables
├── docker-compose.yml                # PostGIS service; same POSTGRES_* names the app reads
├── requirements.txt                  # Python dependencies
├── PROJECT_DOCUMENTATION.md          # Master comprehensive project documentation
│
├── app/                              # FastAPI Service & Frontend
│   ├── __init__.py
│   ├── database.py                   # PostGIS ORM layer: geometry(Point,4326), GiST, ST_DWithin
│   ├── main.py                       # 28 REST API endpoints, routing, state managers
│   ├── templates/
│   │   └── index.html                # Sentinel Thermal C2 Dashboard & Simulation UI
│   └── static/                       # Static web assets
│
├── src/                              # Core Processing Engines
│   ├── __init__.py
│   │
│   ├── ingestion/                    # Satellite & GIS Data Harvesters
│   │   ├── firms_client.py           # NASA FIRMS API client, CSV/Parquet parser
│   │   ├── bhuvan_loader.py          # ISRO Bhuvan LULC shapefile converter
│   │   ├── osm_client.py             # Overpass API client for industrial polygons
│   │   ├── geofabrik_osm.py          # Geofabrik regional PBF downloader
│   │   ├── pbf_extractor.py          # Fast OSM .osm.pbf parser via pyrosm/shapely
│   │   ├── sentinel2_client.py       # Copernicus Sentinel-2 dNBR burn-scar validation
│   │   ├── slstr_client.py           # Sentinel-3 SLSTR fire channels & measured background
│   │   └── download_manager.py       # Orchestrated CLI downloader
│   │
│   ├── pipeline/                     # Spatial Join & Feature Engineering
│   │   ├── spatial_join.py           # GeoPandas R-tree spatial join & nearest distance
│   │   ├── thermal_physics.py        # Dozier retrieval -- validated, REJECTED, not wired in
│   │   └── feature_engineering.py    # Thermal ratios, z-scores, temporal deltas
│   │
│   ├── alerting/                     # State Machine & Intelligence Reporting
│   │   ├── state_machine.py          # Uber H3 Resolution 9 recurrence state machine
│   │   ├── dispatcher.py             # Outbound SMS/email/webhook routing, off by default
│   │   └── sitrep_generator.py       # MGRS/UTM coords, Gaussian plume & chemical emissions
│   │
│   ├── reporting/                    # Passive Audit & Corpus-Scale Rendering
│   │   ├── compliance_register.py    # CPCB flaring register from suppressed detections
│   │   └── hex_aggregator.py         # H3 roll-up so the map can draw 2.04M detections
│   │
│   └── models/                       # Machine Learning & Explainability
│       ├── train_classifier.py       # XGBoost trainer, labelling rule v8, circularity audit
│       ├── verified_labels.py        # 21 adjudicated events + 6 undetected incidents
│       ├── explainability.py         # TreeSHAP factor attribution engine (native C++ & shap)
│       ├── fire_classifier_xgb.json  # Trained multi-class XGBoost model weights
│       ├── feature_pipeline.joblib   # Fitted scikit-learn feature pipeline
│       └── model_evaluation_metrics.json # Confusion matrix, macro F1, and log-loss scores
│
├── scripts/                          # Maintenance utilities, not runtime code
│   ├── generate_verified_register.py # Writes section 5c from the live events + model
│   └── add_forest_column.py          # Vectorised in_forest join onto a processed corpus
│
├── data/                             # Data Assets (Local & Cached)
│   ├── fire_db.sqlite                # Offline/replay database only -- PostGIS is primary
│   ├── raw/                          # Raw satellite feeds
│   │   ├── firms_latest.parquet
│   │   └── firms_combined_nrt.parquet
│   ├── processed/                    # Feature-engineered Parquet datasets
│   │   ├── firms_industrial_joined.parquet
│   │   └── dnbr_cache.json           # Persistent Sentinel-2 dNBR result cache
│   └── reference/                    # Reference GIS Boundaries
│       ├── industrial_boundaries_merged.parquet # Master unified OSM + Bhuvan layer
│       ├── osm_industrial_dahej_ankleshwar.parquet
│       ├── osm_india_industrial_from_pbf.parquet
│       └── bhuvan/                   # ISRO Bhuvan 1:50k vector shapefiles
│
└── tests/                            # Pytest Test Suite (314 Tests)
    ├── test_evaluation_integrity.py  # 50 Leakage, labelling, provenance, thresholds & metric honesty
    ├── test_sentinel2.py             # 36 dNBR client, degradation, quota & transport recovery
    ├── test_dispatcher.py            # 27 Dispatch safety: dry-run default, honest status, suppression
    ├── test_hex_aggregator.py        # 18 H3 roll-up, bounded payload, priority ordering
    ├── test_slstr.py                 # 14 SLSTR client, background annulus, refused retrievals
    ├── test_compliance_register.py   # 9 Operator misattribution, solar farms as emitters, invented carbon figures
    ├── test_firms_archive.py         # 23 Archive paging, NRT/SP routing & dedup
    ├── test_verified_labels.py       # 28 Verified-event harness, register drift & undetected incidents
    ├── test_forest_cover.py          # 17 Forest land cover, FOREST_FIRE determination & guard ordering
    ├── test_replay_mode.py           # 13 Historical replay, refresh interaction & offline preflight
    ├── test_state_machine.py         # 15 H3 recurrence, facility keying, onset & suppression
    ├── test_api.py                   # 33 REST endpoint & contract tests
    ├── test_sitrep.py                # 15 MGRS, plume geometry & fabricated-value disclosure
    ├── test_spatial_join.py          # 4 Point-in-polygon, parallax buffer & distance
    ├── test_simulation.py            # 4 What-If simulation engine
    ├── test_models.py                # 4 XGBoost & TreeSHAP explainability
    └── test_firms_client.py          # 4 NASA FIRMS parser & client
```

---

## 4. Component-by-Component Deep Dive

### 4.1 Data Ingestion & Multi-Source Geospatial Fusion (`src/ingestion/`)
1. **NASA FIRMS NRT Client (`firms_client.py`)**:
   - Downloads Near Real-Time thermal anomalies from VIIRS (S-NPP, NOAA-20, NOAA-21 at 375m) and MODIS (Terra/Aqua at 1km).
   - Retrieves active fire attributes: Latitude, Longitude, FRP (Fire Radiative Power), Brightness Temperatures (`bright_ti4`, `bright_ti5`), Scan/Track pixel dimensions, Acquisition Date/Time, and Day/Night flag.
   - Cleans timestamps into standardized ISO UTC formats and yields spatial `Point` geometries in EPSG:4326.
2. **ISRO Bhuvan Loader (`bhuvan_loader.py`)**:
   - Parses official NRSC/ISRO 1:50,000 scale Land Use / Land Cover (LULC) vector shapefiles (e.g. Sundargarh industrial corridor, Odisha).
   - Extracts `Level-II` and `Level-III` industrial polygons, standardizes attribute naming, and repairs invalid geometries.
3. **OpenStreetMap Extractor (`pbf_extractor.py` & `geofabrik_osm.py`)**:
   - Parses regional `.osm.pbf` dumps using spatial filters targeting key industrial tags:
     - `landuse=industrial`
     - `man_made=works`, `man_made=petroleum_well`, `man_made=storage_tank`, `man_made=chimney`
     - `industrial=oil_refinery`, `industrial=chemical`, `industrial=steelwork`, `industrial=brick_kiln`
4. **Master Spatial Merge**:
   - Unifies ISRO Bhuvan and OSM vectors into `industrial_boundaries_merged.parquet`, establishing ground-truth boundaries for India's major industrial zones (Dahej PCPIR, Ankleshwar, Jamnagar, Sundargarh, etc.).

---

### 4.2 Spatial Join & Feature Engineering (`src/pipeline/`)
1. **R-Tree Point-in-Polygon Engine (`spatial_join.py`)**:
   - Uses GeoPandas spatial indexing (`sindex`) to perform sub-millisecond point-in-polygon intersections.
   - Flags:
     - `inside_industrial` (Boolean): Direct physical containment inside an industrial boundary polygon.
     - `is_exact_match` (Boolean): Direct polygon containment ONLY. A detection recovered via the proximity buffer sets `inside_industrial=True` but leaves `is_exact_match=False`, so the model can weight containment and proximity differently.
     - `dist_to_industrial_km` (Float): Geodesic distance in kilometers to the nearest facility boundary.
     - `facility_type`: Categorizes into `petrochemical_refinery`, `chemical_plant`, `steel_metallurgical`, `thermal_power_plant`, `brick_kiln`, or `general_industrial`.
2. **Engineered Radiometric & Recurrence Features (`feature_engineering.py`)**:
   - `frp`: Raw Fire Radiative Power (MW).
   - `bright_ti4` / `bright_ti5`: 375m high-temperature and split-window thermal infrared measurements.
   - `scan` / `track`: Off-nadir pixel growth corrections.
   - `n_30d`: Cumulative historical observation count in the local spatial cell over the past 30 days.
   - `mu_frp`: Historical moving average FRP for this spatial location.
   - `z_frp`: Energy Z-Score anomaly metric:
     $$Z_{FRP} = \frac{FRP - \mu_{FRP}}{\sigma_{FRP} + \epsilon}$$
   - `frp_ratio`: Surge ratio relative to baseline:
     $$R_{FRP} = \frac{FRP}{\max(\mu_{FRP}, 1.0)}$$

---

   - `detection_confidence`: FIRMS sensor confidence normalized onto a single 0-100 scale. VIIRS emits the categorical classes `l`/`n`/`h` while MODIS emits a 0-100 integer; both appear in the combined NRT feed, so they are reconciled into one column. Low confidence is the dominant signature of solar glint off metallic factory roofing, making this a direct discriminator for the `TRANSIENT_HOTSPOT` class.
   - `dnbr` / `dnbr_available`: Sentinel-2 optical burn-scar validation (see 4.2b). `dnbr` is `NaN` wherever validation could not run, and XGBoost routes the NaN natively. The companion `dnbr_available` flag is what lets the model separate *"verified: no burn scar"* (a strong contained-industrial-fire signal) from *"unknown"* -- conflating those two would fabricate evidence.

3. **Scan-Angle-Adaptive Parallax Buffer**:
   Two physical effects displace a reported detection centroid from the true combustion source:
   - **Off-nadir pixel growth**: a VIIRS I-band pixel is 375m at nadir but grows substantially toward the edge of the orbital swath. The reported `scan`/`track` fields carry the actual footprint in km.
   - **Plume parallax**: an intense fire lofts a super-heated plume to altitude, and at high scan angles the satellite projects that elevated thermal signature laterally onto the ground plane -- so the detection lands outside the true facility perimeter.

   The association tolerance is therefore `1500m + (scan * 1000) / 2`, evaluated per detection. A nadir pixel keeps a tight ~1.7km association while an edge-of-swath pixel widens to ~3.3km. On the 12-month corpus the evaluated tolerance spans 1,688-3,910m: 98,203 of 2,044,295 detections fall exactly inside a polygon, and the adaptive buffer recovers a further **103,652 hotspots** that a flat 500m buffer would have discarded.

---

### 4.2b Sentinel-2 Optical Burn-Scar Validation (`src/ingestion/sentinel2_client.py`)

Thermal data establishes that *something hot happened*; OSM establishes that *there is a refinery here*. Neither answers the decisive question: **did the combustion consume surrounding biomass, or was it contained inside a concrete and steel facility?** The Normalized Burn Ratio answers it directly.

1. **The Index**:

   $$NBR = \frac{B08 - B12}{B08 + B12} \qquad dNBR = NBR_{pre} - NBR_{post}$$

   Healthy vegetation reflects strongly in NIR (B08) and weakly in SWIR (B12), carrying a high NBR; charred biomass inverts that relationship. A wildfire or crop burn therefore leaves a large positive dNBR. **A tank fire burning inside a bunded compound leaves dNBR close to 0 -- there was no vegetation to consume.** This is the failsafe that separates the two classes when thermal radiometry alone is ambiguous.

2. **Data Path**: Copernicus Data Space Ecosystem (CDSE) Sentinel Hub **Statistical API**, chosen over the Process API deliberately -- it returns aggregated band statistics as JSON, so no GeoTIFF download or raster decoding is needed for a single scalar per detection. Cloud, cloud shadow, cirrus and snow pixels are masked via the Scene Classification Layer, and a scene is only trusted when at least 20% of its pixels survive the mask.

3. **USGS Severity Ladder**: `UNBURNED` (<0.10) -> `LOW_SEVERITY` (<0.27) -> `MODERATE_LOW` (<0.44) -> `MODERATE_HIGH` (<0.66) -> `HIGH_SEVERITY`. A negative dNBR returns `REGROWTH`, which flags the thermal detection itself as suspect.

4. **Rationed Execution**: Validating every detection would burn the CDSE quota on routine flares that need no confirmation. By default only the genuinely ambiguous population is validated -- detections inside or near an industrial polygon that lack an established recurrence baseline -- with a hard per-run request ceiling. Results are cached permanently to disk, since a Sentinel-2 scene for a past date never changes.

5. **Graceful Degradation** *(required behaviour, covered by 28 tests)*: with no credentials, no network, a cloud-obscured scene, or an event too recent for L2A publication, the module reports `dnbr = None` with an explicit status string (`NO_CREDENTIALS`, `NO_CLEAR_SCENE`, `EVENT_TOO_RECENT`, `API_ERROR`, `RATE_LIMITED`). **It is never silently coerced to 0.0.** The rest of the pipeline runs unchanged without it.

6. **Configuration**: free registration at <https://dataspace.copernicus.eu>, then Dashboard -> User Settings -> OAuth clients -> Create new. Put `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` in `.env` (never commit them). Without them the pipeline logs one warning and continues.

7. **Verify Before Trusting** — run the benchmark pair first:

   ```bash
   python -m src.ingestion.sentinel2_client --validate --debug
   ```

   This evaluates three real events against live CDSE data. A PASS requires a burn event above +0.10, the contained event within +/-0.10, and at least 0.15 separation between them.

   **Measured result (verified against live Sentinel-2 data):**

   | Event | Measured dNBR | Severity | Expectation | Outcome |
   | :--- | ---: | :--- | :--- | :--- |
   | Punjab stubble burning (30.500, 75.800) | **+0.2775** | MODERATE_LOW | positive | PASS |
   | Jamnagar refinery (22.400, 70.050) | **+0.0345** | UNBURNED | near zero | PASS |
   | Baghjan blowout (27.587, 95.385) | -0.0434 | UNBURNED | positive | **inconclusive** |
   | **Separation (Punjab vs Jamnagar)** | **+0.2430** | | >0.15 | **PASS** |

   **dNBR is confirmed discriminative on real data**: a crop burn that consumed biomass reads +0.28, while contained refinery flaring reads +0.03, an order of magnitude apart.

   Baghjan is recorded as inconclusive rather than quietly dropped. The blowout occurred 27 May 2020 and burned into November, entirely within the Assam monsoon; the standard 30-day post window returns `NO_CLEAR_SCENE`, and widening it to 220 days averages the burn scar together with subsequent vegetation regrowth, yielding a slightly negative value. This is the cloud-cover limitation the problem statement's own research anticipated, and it is precisely why Punjab -- a dry-season event -- is the load-bearing positive control.

8. **Distinguishable Failure Statuses**: the response parser never lets a bug masquerade as a data gap.

   | Status | Meaning | Whose problem |
   | :--- | :--- | :--- |
   | `OK` | Usable dNBR returned | -- |
   | `PARSE_MISMATCH` | Scenes returned, statistics not where expected | **Ours** (bug) |
   | `BAD_REQUEST` | Evalscript or request schema rejected | **Ours** (bug) |
   | `NOT_AUTHORIZED` | Credentials valid, account lacks Sentinel Hub access | Configuration |
   | `RATE_LIMITED` | Still limited after backoff and retries | Quota |
   | `NO_CLEAR_SCENE` | Scenes exist but are cloud-obscured | Genuine data gap |
   | `NO_SCENES_IN_WINDOW` | No acquisitions in the window at all | Genuine data gap |
   | `AWAITING_POST_SCENE` | Detection too fresh for a post-event overpass | Timing (retry later) |

   The output band key is discovered rather than hardcoded, since hardcoding it would silently report every scene as cloudy if Sentinel Hub named it differently.

9. **The Temporal Constraint (important)**: dNBR is inherently **retrospective**. It requires a post-event scene, and Sentinel-2's A+B revisit is ~5 days plus 1-2 days of L2A processing latency. A detection from the live NRT feed is typically 0-2 days old and therefore *cannot* have one yet.

   The pipeline pre-filters on detection age and reports `AWAITING_POST_SCENE` without spending an API call, since on a live feed that would otherwise waste the entire quota every run. On the original 2-day corpus **all 626 candidates were too recent, so dNBR coverage was 0 by orbital mechanics, not by failure.** The 12-month archive removed that constraint: its detections are aged, and optical validation now returns real measurements.

   The practical consequence: dNBR is valuable for **forensic and benchmark validation** of aged events -- which is exactly where this project needs it, for validating against confirmed incidents -- and not as a real-time feature for fresh detections. Re-running the optical step over a corpus older than a week populates it.

10. **Rate Limiting and the Free-Tier Quota Ceiling**: Sentinel Hub meters both request rate and processing units. Requests are spaced by a minimum interval and retried with exponential backoff; only a persistent limit surfaces as `RATE_LIMITED`.

    Backoff is **capped at 45 seconds**, and this cap is load-bearing. An exhausted quota is answered with a `Retry-After` measured in minutes -- **843 seconds was observed in practice** -- and obeying that verbatim blocked the pipeline for 14 minutes at zero CPU, indistinguishable from a hang. Past the cap the request is abandoned, and the first `RATE_LIMITED` aborts the remaining optical sweep rather than spending the rest of the run rediscovering the same limit.

    **Operational consequence**: the free CDSE tier cannot support bulk dNBR enrichment across a large corpus. It allows roughly 20 calls before demanding a ~5 minute pause, which makes 630,000 detections impossible but a few hundred *locations* merely slow. Two mechanisms make that practical:

    - **Location-level deduplication.** dNBR is computed once per `recurrence_key` and broadcast to every detection at that location, collapsing 530 P0 detections into 373 API calls.
    - **Patient mode** (`enrich_alerts_with_dnbr(patient=True)`). On `RATE_LIMITED` the client honours the server's `Retry-After`, flushes the cache so completed work survives an interruption, sleeps out the window, and retries the *same* location rather than skipping it. A `max_quota_waits` ceiling keeps a pathological response from stalling indefinitely.
    - **Transport failures are retried, not recorded.** An unreachable host says nothing about whether a scene exists, so `API_ERROR` retries in place with exponential backoff. After `MAX_TRANSPORT_RETRIES` the location is abandoned, and after `MAX_CONSECUTIVE_TRANSPORT_FAILURES` such locations the run stops entirely, leaving the remainder unattempted and recoverable. This exists because a one-minute DNS outage on 2026-09-12 consumed 79 of 373 locations: the loop advanced past each `API_ERROR` the way it advances past a genuine result, then exited with status 0 as though it had succeeded.

    Optical validation is therefore a **targeted forensic tool** -- run against the alert set, not the corpus. Enriching every detection remains out of reach on the free tier.

---

### 4.3 Uber H3 Recurrence & Suppression State Machine (`src/alerting/state_machine.py`)
1. **Spatial Binning via Uber H3 Resolution 9**:
   - Hexagonal global discrete grid cells with edge length of ~174 meters (area ~0.1 km²).
   - Perfect resolution for tracking specific refinery process units, flare stacks, or storage tanks.
   - Includes pure-Python deterministic hex fallback for environments where native C-extensions are restricted.
2. **Five Categorical Classification States**:
   - `PERSISTENT_BASELINE`: High observation count ($N_{30d} \ge 3$), stable energy ($Z_{FRP} < 2.5$). Represents routine flare stacks or blast furnaces.
   - `ESCALATED_FLAREUP`: Known persistent source with a sudden anomalous energy surge ($Z_{FRP} \ge 2.5$ or $R_{FRP} > 3.0$). Represents an operational failure or excessive flaring event.
   - `ACCIDENTAL_FIRE`: Inside or immediately adjacent to industrial facility, no historical baseline ($N_{30d} \le 2$), elevated FRP. Indicates a sudden catastrophic fire.
   - `TRANSIENT_SUSPICION`: Inside industrial zone, low/marginal energy, single occurrence. Requires monitoring.
   - `MONITORING`: Low-priority non-industrial or background thermal event.
3. **Dispatch Priority Levels**:
   - `P0_EMERGENCY`: Immediate incident commander dispatch and alert escalation.
   - `P1_ALERT`: Priority notification of operational anomaly or flare-up.
   - `P2_ADVISORY`: Developing thermal anomaly or transient event.
   - `SUPPRESSED`: Benign routine baseline; suppressed from disturbing tactical dispatchers.
   - `NON_ALERT`: Outside industrial perimeter.

---

### 4.4 Machine Learning & Explainability Engine (`src/models/`)
1. **XGBoost Multi-Class Classifier (`train_classifier.py`)**:
   - Trained on multi-spectral satellite telemetry, temporal deltas, and spatial join characteristics.
   - Target Classes:
     - Class 0: `PERSISTENT_BASELINE` — routine flare stack, blast furnace, brick kiln
     - Class 1: `ACCIDENTAL_FIRE` — chemical explosion, tank breach, plant structural fire
     - Class 2: `AGRICULTURAL_BURN` — crop residue and open biomass burning
     - Class 3: `TRANSIENT_HOTSPOT` — low-confidence artifacts, solar glint, marginal anomalies

     (Authoritative mapping: `CLASS_NAMES` in `src/models/train_classifier.py`.)
   - Yields calibrated probability distributions across all 4 categories.
2. **Game-Theoretic TreeSHAP Engine (`explainability.py`)**:
   - Computes exact Shapley Additive exPlanations for each feature.
   - Provides local per-incident factor breakdowns (e.g. +38% FRP Spike, +24% Industrial Proximity, -15% Multi-Day Recurrence).
   - Generates automated human-readable intelligence narratives for NTRO analysts.
   - Includes native XGBoost C++ `pred_contribs=True` execution ensuring zero runtime failures even under Windows Smart App Control restrictions.

---

### 4.5 Automated Situation Report (SitRep) Generator (`src/alerting/sitrep_generator.py`)
1. **Military & Emergency Coordinate Translation**:
   - Converts standard WGS84 coordinates into:
     - **MGRS (Military Grid Reference System)**: e.g. `43QDF 2827 0109` for joint-service tactical coordination.
     - **UTM Coordinates**: Zone, Hemispheric Easting & Northing meters.
     - **Google Plus Code**: For local civil defense dispatch.
2. **Gaussian Plume Atmospheric Dispersion Physics**:
   - Models downwind chemical smoke dispersion using Pasquill-Gifford atmospheric stability classification (Class A extremely unstable through Class F moderately stable).
   - Computes lateral spread ($\sigma_y$) and vertical spread ($\sigma_z$) along downwind distance ($x$).
**Weather is synthetic.** `compute_atmospheric_dispersion()` queries no weather service. Temperature, humidity and wind are produced by a fixed formula over the incident coordinates, and the plume bearing and evacuation radius derived from them are illustrative. This shipped for some time with the docstring "Calculates weather context" and no disclosure at any surface a reader would see, which made a formula look like an observation on a briefing that recommends an evacuation cordon.

Every path now declares it: `weather_source`, `weather_is_measured` and `weather_disclaimer` on the API response, a red banner at the top of the SitRep's dispersion panel, and a `SYNTHETIC - not observed` tag on each individual figure. `fetch_observed_weather()` is a deliberately unimplemented hook -- an empty provider is honest, a formula pretending to be one is not. Wiring a real archive API flips `weather_source` to `OBSERVED` and removes the banner automatically. Five tests in `test_sitrep.py` hold the disclosure in place.

Note that the plume *length* is a genuine function of Fire Radiative Power. It is the direction it points, and the atmospheric context beside it, that are invented.

3. **Quantitative Toxic Emission Estimates**:
   - Order-of-magnitude emission estimates for petrochemical, chemical, metallurgical, and power plants. **These are hand-chosen scaling factors, not US EPA AP-42 emission factors** -- earlier revisions of this document claimed AP-42 provenance they never had. The ratios between facility types are meaningful; the absolute mass rates are indicative. Every response carries `estimate_basis: HAND_TUNED_SCALING` and a disclaimer:
     - Sulfur Dioxide ($\text{SO}_2$) in kg/hr
     - Nitrogen Oxides ($\text{NO}_x$) in kg/hr
     - Particulate Matter ($\text{PM}_{2.5}$) in kg/hr
     - Volatile Organic Compounds (VOCs) in kg/hr
     - Carbon Monoxide (CO) in kg/hr
4. **Three Tactical Hazard Zones**:
   - **Zone 1 (Immediate Flash / IDLH Lethality)**: 100m – 500m radius.
   - **Zone 2 (Moderate Toxic Irritation)**: 500m – 1,500m radius.
   - **Zone 3 (Precautionary Shelter-in-Place)**: 1,500m – 5,000m downwind corridor.
5. **Standard Operating Procedures (SOP)**:
   - Dynamic tactical action checklists based on facility type and chemical hazards (foam concentrate requirements, blast perimeter isolation, mutual aid activation).
   - Formatted in clean `@media print` A4 printable executive briefing layouts.

---

### 4.6 Interactive "What-If" Simulation Sandbox
1. **Real-Time Simulation Engine (`POST /api/v1/classify`)**:
   - Allows operators to simulate hypothetical disasters anywhere in the world.
   - Dynamic parameters:
     - Latitude & Longitude (via draggable crosshair reticle on map)
     - Fire Radiative Power: 1 to 500 MW
     - Brightness Temperature: 300 to 500 K
     - Wind Speed: 0 to 60 km/h
     - Wind Direction: 0° to 360° compass bearing
     - Facility Type: Refinery, Chemical, Steel, Power Plant, Brick Kiln, Non-Industrial
2. **Sub-30ms Reactive Loop**:
   - Instantly triggers spatial containment verification, XGBoost classification, TreeSHAP attribution, toxic emissions calculation, and hazard zone modeling.
3. **Dynamic 360° Rotating Leaflet SVG Plume**:
   - Plume physically points and stretches in the downwind direction:
     $$\theta_{downwind} = (\theta_{wind} + 180^\circ) \pmod{360^\circ}$$
   - Dynamic gradient shading matching real-time wind bearing and thermal intensity.

---

### 4.7 Tactical C2 User Interface (`app/templates/index.html`)

1. **Layout: a grid shell, because the floating one could not hold.**

   The previous interface was a full-bleed map with every panel positioned
   `absolute` on top of it. At any real viewport those panels collided: the
   command capsule ran into the map toolstrip so the Triage tab was clipped
   behind the viewport selector, the evidence dossier overflowed the right edge
   of the window part-way through a 40-character sensor identifier, and the
   timeline scrubber sat on top of the SHAP attribution bars. No amount of
   restyling fixes that, because the cause is that nothing owned any space.

   The shell is now a CSS Grid with named areas:

   ```
   masthead  masthead  masthead      52px
   rail      stage     dossier       1fr
   status    status    status        26px
   340px     1fr       384px
   ```

   Every region owns a cell, so overlap is impossible by construction.
   Collapsing the rail or the dossier changes `grid-template-columns`, which
   means the map **reflows into the freed space** rather than being uncovered.
   Below 1280px the dossier collapses; below 900px the rail does too.

   Two consequences worth stating, because both were live defects:

   - Leaflet assigns its own panes `z-index` 400 to 700, and deck.gl's overlay
     canvas joins them. With the map container at `z-index: auto` those panes
     competed directly with the stage's controls, so switching on the archive
     layer painted the deck canvas straight over the map toolstrip and the
     timeline -- the controls were still there and still clickable at their
     coordinates, and completely invisible. `#map` now carries an explicit
     `z-index`, which contains every Leaflet and deck.gl pane inside one
     stacking context.
   - Leaflet caches its container size, so a grid cell that changes width when a
     panel opens leaves the map rendering a stale viewport. A `ResizeObserver`
     on the stage cell keeps them in step; a hand-maintained list of
     `setTimeout(invalidateSize)` calls at every toggle site eventually drifts
     out of sync with the CSS, and did.

2. **Palette: colour is data, the chrome carries no hue.**

   Sensor imagery is fundamentally grayscale; false colour is applied only where
   it means something. This interface works the same way.

   An earlier build used amber for the chrome -- borders, brand, icons, active
   tabs -- and amber is also the colour of a P2 advisory, so the interface was
   shouting at exactly the frequency of its own alerts. In a system whose pitch
   is alert-fatigue suppression that is an information-design defect, not a
   matter of taste. Swapping the accent for a different hue only moves the
   problem; removing hue from the chrome is what fixes it. Active and
   interactive states are therefore a **bright neutral**, which is also what
   current product interfaces do.

   The only saturated values on the interface are the four alert tiers, the
   archive layer, and the map:

   | Token | Value | Meaning |
   | :--- | :--- | :--- |
   | `--p0` | `#e5484d` | dispatch now |
   | `--p1` | `#f76b15` | alert |
   | `--p2` | `#ffb224` | advisory |
   | `--routine` | `#30a46c` | suppressed as routine |
   | `--archive` | `#0091ff` | H3 corpus layer |

   **Both themes are defined at token level**, and components read tokens only,
   never literals -- so a colour cannot exist in one theme and be missing from
   the other, which is the classic unreadable-dashboard bug. The alert tiers are
   deliberately **identical in both**: a P0 that shifts hue because someone
   flipped to light mode is a colour code that cannot be learned. The choice
   persists per browser and is applied by an inline script before first paint,
   so a reload never flashes the wrong theme.

   Facility-type icons used to be tinted orange, yellow and rose, which made a
   steel plant look more dangerous than a refinery for reasons of decoration.
   The icon shape carries the sector; the colour stays reserved.

   **Where the vividness lives, and why it is safe.** The rule is unchanged --
   warm colour means combustion and nothing may compete with a P0 -- but
   "achromatic chrome" was a stricter reading than the rule requires. The fire
   ramp occupies red through yellow and routine occupies green, which leaves
   the entire violet-to-cyan half of the wheel unused. A hue 120 degrees from
   anything on the alert ramp cannot be mistaken for one.

   So the chrome is vivid, and it is vivid *there*: an electric violet
   (`#6d3bff`) to cyan (`#00c2ff`) gradient carries the brand mark, active
   tabs, filter chips and focus rings, over a slow-drifting aurora ground and
   translucent glass surfaces with backdrop saturation. The alert tiers were
   turned up to match -- `#ff2d55`, `#ff7a00`, `#ffd60a`, `#00e6a0` -- so they
   still dominate a brighter interface.

   Two bugs this surfaced, both worth recording because neither was visible in
   the tokens:

   - **The aurora cannot be one layer for both themes.** Bright violet and cyan
     blobs read as luminous over a light ground and simply *whiten* a near-black
     one. The dark theme rendered as pale lavender with every token resolving
     correctly. Light paints the colour directly; dark screens a fainter version
     at half opacity, adding edge glow instead of lifting the ground.
   - **Never transition a background whose value comes from a custom property.**
     When the token changes the element keeps interpolating from the value it
     had, and in practice never arrives. The symptom was exact and misleading:
     `--surface` resolved to the dark value on `:root`, `.shell` (which has no
     background transition) painted dark, and *every element that did transition
     its background stayed light*. 910 of 972 nodes in the rail failed contrast
     while the stylesheet was correct. Transitions are now suppressed for the
     duration of a theme swap and restored on the next frame, so hover keeps its
     easing and the swap itself is instant.

   **Contrast is measured, not eyeballed.** Every text node in the masthead,
   rail, dossier, status bar, map toolstrip and all three full-screen panels was
   checked against *the surface it actually sits on* -- not against the page,
   which is the mistake that makes a dark chip on a light page look like a
   failure and hides a faint label on a raised card. 5,410 nodes per theme across four alternating passes,
   **0 below the 4.5:1 WCAG AA floor** in either.

   Getting there surfaced four distinct defects, all invisible until measured:

   - The ported panels used Tailwind's slate scale directly, which is calibrated
     for a dark ground. On white, `text-slate-400` headings washed out and
     `border-white/10` dividers vanished. They now read the token ladder.
   - The controller injects class strings at runtime, so a markup-only pass
     missed them: `text-white` in a triage table row is white-on-white at
     **1.08:1**, literally invisible.
   - Chart.js was configured with literal greys and a font this build no longer
     loads. Its colours now come from the tokens, and because Chart.js reads its
     config once at construction, the charts are **rebuilt on theme change** --
     restyling a live canvas is not something the library supports.
   - Tier *fills* and tier *text* need different values. `--p0` on white is
     3.7:1, which fails at body size. Each tier therefore has a paired
     `--p0-ink` that shifts lightness while holding the hue, so the colour code
     stays learnable and the glyphs stay legible. Fills are identical across
     themes; ink is not.

   A related class of bug worth recording: after the palette changed, several
   inline styles still referenced retired token names (`--ink`, `--ink-faint`,
   `--ground`) and hardcoded hex from the previous scheme. `var()` on an
   undefined token silently resolves to nothing, so the archive pill and the
   SitRep button simply stayed dark on a light ground with no error anywhere.
   A check for referenced-but-undefined tokens now catches that class outright.

3. **Typography and surface treatment: a product interface, not a HUD.**

   The console this replaced set **9px uppercase micro-labels with wide
   letter-spacing** on every field, drew a 1px hairline box around every
   surface, used a 4px radius throughout, and put monospace on all text. That is
   a tactical-HUD convention from the early 2010s, and it read as dated rather
   than as precise.

   | | Before | Now |
   | :--- | :--- | :--- |
   | Field labels | 9px caps, `.1em` tracking | 12px sentence case |
   | Body | 13px | 13.5px |
   | Row title | 13px | 15px / 600 |
   | Smallest text anywhere | 9px | 12px |
   | Separation | 1px hairline on everything | surface fill + elevation |
   | Radius | 4px | 12px cards, 8px controls, pill nav |
   | Monospace | all text | measured values only |
   | Rows | flush, 55px | elevated cards, 8px gutter, hover lift |

   **Plus Jakarta Sans** for the interface; **Geist Mono** reserved for values
   actually read digit by digit -- coordinates, radiometry, identifiers,
   timestamps -- set in tabular figures so they align down a column.

4. **Five modes, one shell**: Surveillance (map), Analytics, Triage, Audit and
   the What-If Sandbox. Mode panels fill the region between masthead and status
   bar; the sandbox HUD docks inside the stage beside the rail.

5. **The incident rail names a source by what distinguishes it.**

   Every unmapped detection previously rendered as the same string -- *"Rural /
   Non-Industrial Node"* in the rail and *"Open Field / Non-Industrial Node"* in
   the dossier, so the same detection was called two different things depending
   on where you read it. The first thing anyone saw was four identical rows from
   a system whose whole claim is telling thermal sources apart.

   An unmapped source has no name but it does have a position, and the position
   is its identity, so coordinates carry the headline. OSM placeholder names such
   as *"Unnamed Industrial Site"* are treated as unnamed for the same reason they
   broke the compliance register: thousands of distinct polygons share that one
   string. One `sourceTitle()` serves both surfaces. Each row carries a severity
   stripe on its left edge -- the only saturated thing in the rail, so priority
   reads before any text does -- and shows the **classification**, which is the
   actual product of this system and was not on the card at all.

6. **The archive layer has a first-class control.** `Archive ON/OFF` sits in the
   map toolstrip rather than being a checkbox three clicks inside a popover. It
   is the one view that draws all 2,044,295 detections at once (4.2e).

7. **The legend states encodings, not colours**: alert tier with the channels
   each one dispatches on; the archive layer's fill/outline/opacity split and why
   it is coloured by dominant rather than highest tier; and the FRP ramp with the
   caveat that radiative power is an observation rather than a classification --
   a 60 MW refinery flare is routine and a 4 MW depot fire is not.

8. **A status bar carries provenance.** Corpus size and span, labelling rule
   version, training run id, spatial-block CV, and the verified macro F1 with the
   words *"the only rule-independent number"* attached. Nothing in the previous
   build said which corpus, which rule or which run the classifications on screen
   came from.

9. **Freshness is computed, never asserted.**

   The status bar previously read a hardcoded `LIVE` while the newest detection
   on screen was **five days old**. A system whose pitch is near-real-time
   surveillance cannot print the word "live" as static markup; it has to derive
   it from the data it holds, or it will make a false statement the moment a
   sync is missed -- which is exactly what had happened.

   The badge now reads the newest loaded timestamp and degrades:

   | Age of newest detection | Badge | Colour |
   | :--- | :--- | :--- |
   | under 6 h | `LIVE` | routine green |
   | 6 h to 48 h | `RECENT · N h old` | P2 brass |
   | over 48 h | `STALE · N d old` | P0 red |

   Six hours is the honest ceiling: FIRMS NRT products publish roughly three
   hours behind the overpass, so nothing on this data source is fresher.

   The sync control in the masthead was also relabelled. It re-imports the
   processed corpus into the incident database and does **not** fetch new
   detections from FIRMS, which is not what "Synchronise telemetry" implies.

   **And the refresh is now automatic** (`src/pipeline/auto_refresh.py`). A
   badge that degrades honestly to STALE is right for a badge and wrong for a
   surveillance system: a near-real-time pipeline that only ingests when a human
   remembers a script is a manual import with a clock on it. The API starts a
   worker thread at boot that pulls FIRMS, runs the spatial join and recurrence
   state machine, and reseeds -- immediately, then every 3 hours.

   Three properties it holds to:

   - **It never blocks startup.** If FIRMS is slow or unreachable the dashboard
     still opens, serving what it has, with the badge saying how old that is.
   - **It never reports a refresh it did not perform.** A failed pull leaves
     `last_success_utc` untouched, so the gap between that and
     `last_attempt_utc` measures how long ingestion has been broken. One
     "last refreshed" field would hide exactly that.
   - **It is off under pytest**, regardless of configuration. A suite that
     reaches a third-party API depends on someone else's uptime -- the trap the
     weather provider set earlier in this project.

   **Live ingest and the analytical archive are different files.** They were
   the same one, and it destroyed data. The refresh wrote to
   `PROCESSED_CORPUS`; a deployment points that at the 12-month archive so
   `/map/hexes` and `/compliance/register` see all 2,044,295 detections; so
   every three hours the scheduler silently replaced that archive with a
   two-day NRT window. The only symptom was the archive layer reporting ~1,100
   detections instead of two million, and the compliance register reporting a
   one-day observation span. Nothing errored.

   The refresh now writes to a fixed live path, and `would_shrink_corpus()`
   refuses any write that would replace a corpus above 100,000 rows with a
   window an order of magnitude smaller -- guarding the *shape* of the write
   rather than comparing paths, because in the default configuration the live
   file legitimately is the corpus and a path comparison would refuse every
   safe refresh while catching no dangerous one.

   Optical validation is deliberately excluded from the scheduled run: Sentinel
   Hub is a metered free tier, and a job silently consuming quota every few
   hours is how it disappears before a demonstration.

   `GET /api/v1/system/refresh` reports the state; `POST` to the same path forces
   a run. The browser re-polls every 5 minutes and preserves the operator's
   selection, filter and search across a refresh rather than resetting them.
   `refresh_live_data.cmd` remains for a manual pull without the API running.

10. **Accessibility and motion**: visible keyboard focus rings, `prefers-reduced-motion`
   honoured (the P0 marker pulse and all transitions stop), and no text below
   9px.


## 5. Complete REST API Reference

All 28 endpoints run on `http://127.0.0.1:8000` with interactive Swagger docs at `/docs` and ReDoc at `/redoc`.

| Method | Endpoint | Tags | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | UI Dashboard | Serves the Sentinel Thermal C2 Leaflet GIS dashboard, including the corpus-scale H3 archive layer |
| `GET` | `/api/v1/health` | System | Runtime health, database record counts, GIS boundary & model status. Reports `database_mode`, `postgis_version` and `offline_mode`, so a degraded datastore is never invisible |
| `GET` | `/api/v1/incident/{incident_id}/responders` | Dispatch | Nearest fire station, hospital and police station. Grouped by category, because the three are not interchangeable. Distance is geodesic; travel time is explicitly **not routed** and carries the circuity factor and assumed speed that produced it |
| `GET` | `/api/v1/responders/coverage` | Dispatch | What the responder layer holds and what it does not -- 61,001 facilities, of which only 741 are fire stations, which is an undercount and is reported as one |
| `GET` | `/api/v1/incidents/near` | Spatial | Incidents within a true-metre radius of a point, nearest first. PostGIS `ST_DWithin` over `geography` against the GiST index; reports which engine answered |
| `GET` | `/api/v1/alerts/active` | Surveillance | Fetches active incidents with priority, state, FRP, and facility filtering |
| `GET` | `/api/v1/incident/{incident_id}` | Surveillance | Deep incident telemetry, coordinates, and local TreeSHAP factor attributions |
| `GET` | `/api/v1/incident/{incident_id}/optical-validation` | Validation | Runs Sentinel-2 dNBR burn-scar validation; reports severity and a plain-language interpretation. Returns `NO_CREDENTIALS` with guidance when CDSE is unconfigured |
| `POST` | `/api/v1/incident/{incident_id}/dispatch` | Dispatch | Routes an incident to responders over the channels its priority calls for. Off unless `ALERT_DISPATCH_ENABLED` is true; every attempt is audit-logged, delivered or not |
| `GET` | `/api/v1/compliance/register` | Compliance | Routine flaring logged for CPCB auditing. Reports the detections the pipeline suppressed; excludes non-combustion land use; does not compute gas volume or CO2 |
| `GET` | `/api/v1/incident/{incident_id}/sitrep` | Briefings | Generates full tactical Situation Report (MGRS/UTM, Gaussian plume, indicative emissions). Weather is synthetic unless a provider is wired -- the report carries a banner saying so |
| `POST` | `/api/v1/incident/{incident_id}/status` | Surveillance | Updates incident lifecycle status (`OPEN`, `ACKNOWLEDGED`, `DISPATCHED`, `RESOLVED`, `MUTED_ROUTINE`) |
| `GET` | `/api/v1/stats` | Analytics | Aggregates surveillance metrics (P0 emergency counts, P1 flare-ups, suppressed counts, avg FRP) |
| `POST` | `/api/v1/sync` | Pipeline | Triggers data pipeline synchronization (FIRMS ingestion, spatial join, baseline update) |
| `POST` | `/api/v1/classify` | Machine Learning | On-demand inference engine evaluating coordinates, FRP, temperatures, and facility overrides |
| `POST` | `/api/v1/incident/{incident_id}/action` | Analyst Review | Operator action to manually override priority or suppress routine baseline flares |
| `GET` | `/api/v1/audit/logs` | Analyst Review | Retrieves complete audit log trail of analyst overrides and operational actions |
| `GET` | `/api/v1/analytics/charts` | Analytics | Generates structured chart data for facility distributions and diurnal patterns |
| `GET` | `/api/v1/weather/context` | Weather & Dispersion | Fetches local meteorological context (wind speed, direction, temperature, atmospheric stability) |
| `POST` | `/api/v1/simulation/sitrep` | Simulation Briefings | Generates full Situation Report for What-If simulation scenarios with custom wind & FRP |
| `GET` | `/api/v1/system/refresh` | Pipeline, System | State of the automatic FIRMS ingest: last attempt, last success, failure count, next scheduled run |
| `POST` | `/api/v1/system/refresh` | Pipeline, System | Forces a full ingest now -- FIRMS pull, spatial join, reseed. Distinct from `/api/v1/sync`, which only re-imports the corpus already on disk |
| `GET` | `/api/v1/map/hexes` | Analytics, Surveillance | Detections aggregated server-side into H3 cells, so the map can draw the 2.04M-detection archive. Ordered by the highest priority a cell contains, then by volume |
| `GET` | `/api/v1/incident/{incident_id}/thermal-probe` | Validation | Sentinel-3 SLSTR F1/F2 brightness temperature with a **measured** background annulus, plus an attempted Dozier retrieval that returns `NO_SOLUTION` rather than a number when the mixture has no admissible solve |

---

## 5b. Model Performance and Its Limits

**Read this section before quoting any accuracy figure from this project.**

### 5b.1 Label Provenance

| Label source | Count | Nature |
| :--- | ---: | :--- |
| Verified ground truth | **0** | No public per-detection label set exists for Indian industrial thermal anomalies |
| Real FIRMS detections, heuristic-labelled | 2,044,295 | Rule-derived by `weak_label_real_detection()` (labelling rule **v8**) from recurrence and containment evidence |
| Synthetic augmented rows | **0** | Removed. See below. |

**Synthetic augmentation was removed.** Earlier builds added 1,850 rows drawn from uniform distributions -- including fabricated coordinates inside the India bounding box -- to give the rare `ACCIDENTAL_FIRE` class support on what was then a 2-day corpus. Two things made that indefensible once the corpus reached 12 months:

- `balance_corpus()` deliberately never subsamples `ACCIDENTAL_FIRE`, so while the majority classes shed most of their synthetic rows, **all 400 synthetic P0 rows survived intact** -- roughly 13% of the one class the system exists to detect.
- Those rows were drawn from clean, well-separated ranges and scored a provenance-slice macro F1 of **1.000**, against 0.9975 on real detections. They contributed no learnable signal while shaping the P0 decision boundary.

Removing them dropped `ACCIDENTAL_FIRE` from 3,084 to **2,684** -- exactly the 400 fabricated rows. `generate_training_corpus()` now raises rather than substituting generated data for a missing corpus, and drops coordinate-less rows instead of assigning them random positions. Four tests in `test_evaluation_integrity.py` hold that line.

Because the labels are rule-derived, **every score below measures how faithfully the model reproduces its own labelling rule -- not field accuracy against confirmed incidents.** That distinction is preserved in the metrics file itself (`label_provenance` and `caveats` keys) so the numbers cannot travel without their warnings.

### 5b.2 Corpus and Recurrence Granularity

Two defects made every temporal mechanism in the system inert. Both are now fixed, and the effect of each is measured separately below.

**Defect 1 -- corpus shorter than the recurrence window.** `n_30d` is a 30-day count; the original corpus was a 2-day snapshot.

**Defect 2 -- recurrence keyed too finely.** Recurrence accumulated per H3 resolution-9 cell (~174m edge). A refinery complex spans tens of square kilometres with multiple flare stacks, so successive detections of one continuously operating facility scattered across dozens of cells through pixel-centre variation, off-nadir growth and parallax. No cell ever reached the PERSISTENT_BASELINE threshold. The design documents call for tolerating *"spatial jitter up to 1,000 metres caused by satellite parallax effects and viewing geometry"* -- roughly six times looser than a resolution-9 cell. Recurrence is now keyed on **facility identity** inside mapped industrial polygons, and on the H3 cell only outside them, where a genuinely moving wildfire front *should* fragment.

| | 2-day corpus | 60-day, hex keying | 60-day, facility keying | **12-month, facility keying** |
| :--- | ---: | ---: | ---: | ---: |
| Detections | 1,514 | 32,696 | 32,696 | **2,044,295** |
| `PERSISTENT_BASELINE` | **0** | 2,146 | 7,968 | **246,917** |
| `ESCALATED_FLAREUP` | **0** | 109 | 294 | **11,388** |
| `ACCIDENTAL_FIRE` (P0) | 30 | 204 | 32 | **530** |
| Detections SUPPRESSED | **0** | 2,146 | 7,968 | **246,917** |

The 60-day P0 column was the headline at the time: **204 false emergencies fell to 32**, an 84% reduction, while suppression rose 3.7x.

The current corpus is the full 12 months. Of **185,691** detections inside industrial polygons, **167,370 (90.1%)** are recognised as routine operations and suppressed. **530 P0 alerts across 373 distinct locations** survive from 2.04 million detections -- a survival rate of 0.026%. That is the alert-fatigue mandate actually being discharged at national scale.

### 5b.2b Verified-Label Accuracy: The Independent Measurement

This is the **only** measurement in the project not taken against its own labelling rule. **21 events are defined**, up from 9, and **4 of them are outside India** -- up from 1. 20 fall inside a corpus; they are catalogued in full in section 5c, which is generated from the events themselves so it cannot drift from them.

The set was grown by changing what was searched for. Public reporting cannot tell you whether a fire was *detectable*, and searching recent industrial fires kept failing for one reason -- a modern plant fire is extinguished in hours and falls between overpasses. A fire with a **documented multi-day duration** necessarily straddled them, and the MODIS archive reaches back to 2000. Every accidental-fire event below was found that way.

| Verified event | n | Accuracy | Confidence | What it anchors |
| :--- | ---: | ---: | :--- | :--- |
| Khavda Renewable Energy Park (Gujarat) | 100 | **1.000** | medium | sun-glint negative control |
| Pavagada Solar Park (Karnataka) | 53 | **1.000** | medium | sun-glint negative control |
| Benban Solar Park (Aswan, Egypt) | 16 | 0.000 model / **1.000 served** | medium | glint control outside India |
| JSW Vijayanagar Works (steel, Karnataka) | 4,505 | **0.975** | high | continuous metallurgical heat |
| Visakhapatnam Steel Plant (RINL) | 2,591 | **0.965** | high | continuous metallurgical heat |
| Raniganj coalfield seam fires (West Bengal) | 6,236 | **0.962** | high | persistence with no operator |
| Reliance Jamnagar refinery, routine flaring | 539 | **0.941** | high | routine flaring |
| Jharia coalfield seam fires (Jharkhand) | 23,633 | **0.922** | high | persistence with no operator |
| Punjab post-Kharif paddy residue burning | 510 | 0.900 | medium | seasonal biomass |
| Punjab post-Rabi wheat residue burning | 1,608 | 0.883 | medium | seasonal biomass |
| Buncefield oil depot fire (UK, 2005) | 5 | 0.800 | high | an accident outside India |
| Baghjan gas well blowout (Oil India, Assam) | 705 | 0.793 | high | unmapped industrial accident |
| Jaipur IOC oil depot fire (2009) | 33 | 0.697 | high | mapped industrial accident |
| Bandipur National Park fire (2019) | 726 | 0.769 model / **0.741 served** | high | forest fire, served from land cover |
| Deonar landfill fire (Mumbai, 2016) | 29 | 0.552 | high | accident at a chronically burning site |
| Similipal Biosphere Reserve fires (2021) | 3,035 | 0.876 model / **0.860 served** | high | diffuse forest fire near the artifact floor |
| Uttarakhand Himalayan fires (2016) | 1,512 | 0.732 model / **0.562 served** | medium | forest fire where OSM cover thins |
| Brahmapuram waste plant fire (Kochi, 2023) | 30 | 0.533 | high | accident at a chronically burning site |
| ITC Deer Park terminal fire (Texas, 2019) | 11 | **0.182** | high | an accident outside India |
| Rumaila oil field flaring (Basra, Iraq) | 200 | **0.065** | medium | persistence outside India |
| **12-month national archive** | **39,775** | **0.923 accuracy / 0.520 macro F1** | | |

Rule v8 numbers, regenerated from the model artifact on disk. The full register -- every event's coordinates, window, radius, citation and scoring corpus -- is **section 5c**, which is generated rather than written so it cannot drift from `verified_labels.py`.

**The three forest events carry two accuracy figures because they have two truth classes.** A forest fire's correct *model* answer is `AGRICULTURAL_BURN`, the finest distinction a coordinate-free feature vector supports; its correct *served* answer is `FOREST_FIRE`, which land cover decides. See 5b.4f, and note what they exposed: the harvest artifact floor of 1.59 MW discards **45%** of Similipal and **44.5%** of Uttarakhand as artifacts.

**Read the two numbers on the last row together.** Accuracy of 0.923 is carried by the large persistent sites. Macro F1 of **0.520** weights every class equally and is the honest figure.

#### What the four new events bought

Each was added because it could fail in a way nothing already in the set could.

**The coal-seam fires** are persistence with no operator, no process and no burner -- Jharia has been alight since 1916, documented by a peer-reviewed thermal-anomaly literature independent of this project, and **58% of its detections fall outside any mapped industrial polygon**. Every other persistent anchor is a furnace or a flare inside a fence, which is also how the labelling rule thinks about persistence.

**Deer Park** is deliberately outside India: `inside_industrial` is structurally `False` for all 1,291 detections in its corpus, because no Indian OSM polygon covers Texas. It immediately found a defect that aggregate transferability testing had only measured in the abstract -- see 5b.4c.

**Deonar and Brahmapuram** are catastrophic fires at sites that were *already* burning. Deonar reached `n_30d` of 108 with an onset lag of 15 days before its disaster began, so neither of the rule's two accident tests could fire.

#### Outside India: the classes that had never been checked abroad

Fourteen of the first fifteen events were Indian, and the one that was not -- the
ITC Deer Park tank fire -- is an accidental fire. So the *persistent* and
*artifact* classes had never once been scored against a cited label outside the
region the whole system was calibrated in. Three events were added to close that.

| event | class | model | rule | served |
| :--- | :--- | ---: | ---: | ---: |
| Buncefield oil depot, UK (2005) | accidental fire | 0.800 | 1.000 | 0.800 |
| Benban Solar Park, Egypt | transient artifact | **0.000** | 1.000 | **1.000** |
| Rumaila oil field, Iraq | persistent baseline | **0.065** | **0.065** | **0.065** |

**Buncefield was added expecting it to fail the detectability test.** December in
the UK, MODIS only -- VIIRS did not exist in 2005 -- and a dense black plume that
could plausibly have masked the thermal signal from above. Terra caught four
detections at 21:25 on the 11th and Aqua one at 01:28 on the 12th, against zero
in the preceding three months. A four-to-five day fire produced two days of
detections, which is its own quiet measurement of what the plume cost.

**Benban is the rule/model gap, appearing in a third class.** The model calls all
16 detections `AGRICULTURAL_BURN`; the rule and the serving guard both correctly
withhold that outside the calibrated domain. The harness scores raw model output,
so the number reads 0.000 for a case the deployed system gets right -- a
reporting gap in the evaluator rather than a defect in the system, and worth
stating because the two figures differ.

**Rumaila is a new structural defect, and the most important finding here.**
Both the model *and* the rule score 0.065 on a field that flared 3.39 billion
cubic metres of gas. The mechanism:

```
inside_industrial      0 for all 200 detections
dist_to_industrial_km  999
n_30d                  median 4   (persistence threshold is 8)
```

**The persistence test depends on OpenStreetMap polygon coverage.** Inside India
recurrence is keyed on facility identity, so a whole refinery's detections pool
into one counter and cross the threshold easily. Outside it there are no
polygons, recurrence falls back to a ~900 m hex, and Rumaila's flaring is spread
across dozens of separate stacks each firing intermittently. No single cell ever
reaches `n_30d >= 8`, so a continuously flaring field reads as scattered
transient hotspots.

This is not a threshold to tune. It is the same dependency that made Deer Park
fail, in a different class: **the system's notion of "persistent" is borrowed
from a map, and the map stops at the border.** The direction is to key recurrence
on a coarser cell where no polygon exists -- roughly what `persistence_resolution`
already does, needing re-tuning against this event. Logged, not fixed.

#### Deonar: closed by rate, after onset and surge both failed

Deonar was the worst result in the verified set at **0.069**, and the mechanism
was specific: a site that smoulders chronically has *no usable onset* -- it
reached `n_30d` of 108 with a 15-day onset lag before its four-day disaster
began -- and *no usable surge*, since that disaster cleared `z > 3` on 13.8% of
detections against a routine refinery's 4.3%. Both accident tests were
structurally blind to it.

What neither uses is **rate**. `n_30d` smears four days of catastrophe across a
month; detections per day does not. `H3CellHistory.burst()` reports the trailing
24-hour count and that rate against the cell's own prior daily rate, with the
recent window excluded from the baseline so a large enough burst cannot inflate
the thing it is measured against.

**Burst alone is useless, and the first measurement of it misled me.**
Aggregated per site over a 2 km radius it looked decisive -- x18.2 for Deonar
against x0.9 for Jamnagar. At the detection granularity the rule actually
operates on, **18.6% of the 12-month archive exceeds a 3x burst**, because a
cell with one prior detection and two today is a large ratio and nothing else.
The signal is in the conjunction: an established baseline, several detections
within the day, *and* a multiple of the cell's own rate.

Swept before adoption, exactly as `INSIDE_SURGE_Z` was swept before rejection:

| config | Deonar | Jamnagar | Visakhapatnam | Jharia | unweighted mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| off | 0.069 | 0.974 | 0.991 | 0.910 | 0.7988 |
| `n24>=4 br>3` | 0.862 | 0.801 | 0.934 | 0.858 | 0.8268 |
| `n24>=4 br>5` | 0.552 | 0.915 | 0.960 | 0.894 | 0.8233 |
| **`n24>=6 br>5`** | **0.552** | **0.941** | **0.965** | **0.902** | **0.8265** |

`br>3` scores the same mean and costs Jamnagar 0.17. The adopted configuration
takes nearly all the gain for a third of the damage; no anchor loses more than
0.033, and corpus `ACCIDENTAL_FIRE` share moves 4.97% -> 5.13%.

**Why adopted where the surge test was rejected.** The surge test bought
**+0.0040** in the unweighted mean -- one event moving in a set of fourteen, and
negative when weighted by detections. This buys **+0.028**, seven times larger,
concentrated in an event that was catastrophically wrong rather than
marginally. Same harness, same procedure, opposite verdict. The rejected branch
is still off and `INSIDE_SURGE_Z` in `train_classifier.py` carries its table, so
the negative result stays readable next to the positive one.

**Shipping it once without the features was strictly negative, and worth
recording.** Rule v6 first went out with `n_24h` and `burst_ratio` declared in
`LABEL_RULE_FEATURES` but absent from `NUMERICAL_FEATURE_COLS`. The training log
said so plainly -- the audit ablated ten columns, not twelve -- and the result
was the rule scoring Deonar 0.552 while the **model still scored 0.069**, with
the anchors drifting down as it fitted noise where a decision it could not
express should have been. That is the same defect documented one section above,
reintroduced: a rule input the model lacks is not a conservative choice, it is a
label the model cannot learn. With both features in the vector (31 columns, up
from 29) the model reproduces the rule exactly on Deonar.

Measured, v5 -> v6, every event in the set:

| event | v5 | v6 | delta |
| :--- | ---: | ---: | ---: |
| **Deonar landfill fire** | 0.069 | **0.552** | **+0.483** |
| Punjab post-Kharif | 0.894 | 0.900 | +0.006 |
| Punjab post-Rabi | 0.880 | 0.883 | +0.003 |
| Baghjan blowout | 0.791 | 0.793 | +0.002 |
| Khavda / Pavagada / Jaipur / Brahmapuram | -- | -- | 0.000 |
| Raniganj coalfield | 0.968 | 0.962 | -0.006 |
| Jharia coalfield | 0.910 | 0.901 | -0.009 |
| JSW Vijayanagar | 1.000 | 0.975 | -0.025 |
| Visakhapatnam Steel | 0.991 | 0.965 | -0.026 |
| Reliance Jamnagar | 0.974 | 0.941 | -0.033 |
| **unweighted mean** | **0.8236** | **0.8540** | **+0.0304** |

No anchor lost more than 0.033, matching what the sweep predicted. Spatial-block
CV recovered to **0.9971 +/- 0.0008** once the model could express its own
labels, and the circularity delta **rose** from 0.2458 to **0.2502** -- because
the audit now ablates twelve columns instead of ten. A higher number measured
correctly is worth more than a lower one that was not.

**Honesty note.** The decision to adopt was taken on the same verified set the
result is reported against, so these events are partly a tuning set. The
threshold was chosen from four candidates rather than fitted, and `n_24h` and
`burst_ratio` are declared in `LABEL_RULE_FEATURES` so the circularity audit
ablates them -- but this is not independent confirmation.

#### Rule v7: the artifact floor was calibrated for scrub

The three forest events were added on a land-cover argument and immediately found a defect in the rule that had nothing to do with forests as such.

**2,225 detections whose cited truth was AGRICULTURAL_BURN were classified `TRANSIENT_HOTSPOT`, and 2,117 of them -- 95% -- were the three forest fires.** Their FRP median was 1.07 MW against a harvest artifact floor of 1.59 MW, and **2,174 of the 2,225 carried nominal rather than low detection confidence**. They were not instrument noise. One threshold was damaging two classes at once: it destroyed `AGRICULTURAL_BURN` recall *and* `TRANSIENT_HOTSPOT` precision, because 2,225 of the 3,391 things called transient were these.

The floor encodes a prior -- *low-energy open-ground detections are mostly specular glint* -- and that prior is a statement about **bare ground, water and metal**. Tree canopy is dark and diffuse, and a smouldering understorey fire is genuinely low-FRP and genuinely a fire.

**Why forest and not land cover generally.** OpenStreetMap maps **568,585 km2** of India's forest against the Forest Survey of India's ~713,000 -- about 80%. It maps **27,148 km2** of cropland against roughly 1.8 million, about **1.5%**, and **zero of 2,000** sampled verified crop-burn detections fall inside a mapped farmland polygon. The map knows where the forests are and does not know where the fields are, so only one of the two can carry a threshold. That asymmetry was measured before the feature was built, and it is why `is_farmland` does not exist.

Swept before adoption, as `INSIDE_SURGE_Z` was swept before rejection:

| Percentile | Forest floor | Verified detections recovered | Verified glint controls at risk |
| ---: | ---: | ---: | ---: |
| 2nd | 0.57 MW | 1,779 (79.3%) | 0 |
| **5th** | **0.82 MW** | **1,451 (64.7%)** | **0** |
| 10th | 1.21 MW | 640 (28.5%) | 0 |

All 153 verified `TRANSIENT_HOTSPOT` controls sit outside mapped forest, so the floor cannot damage them at any candidate. The 5th percentile was taken rather than the more aggressive 2nd because 0.57 MW sits near VIIRS's own detection floor, where confidence genuinely degrades: the extra 328 detections are not worth readmitting instrument noise.

Measured, v6 -> v7:

| | v6 | v7 |
| :--- | ---: | ---: |
| **Verified macro F1** | 0.4967 | **0.5425** |
| `AGRICULTURAL_BURN` F1 | 0.740 | **0.866** (recall 0.650 -> 0.839) |
| `TRANSIENT_HOTSPOT` F1 | 0.092 | **0.144** (precision 0.050 -> 0.079) |
| `PERSISTENT_BASELINE` F1 | 0.950 | 0.950 |
| `ACCIDENTAL_FIRE` F1 | 0.209 | 0.209 |
| Spatial-block CV | 0.9971 | **0.9981** |
| Circularity delta | 0.2502 over 12 | **0.2629 over 13** |

The two classes the diagnosis predicted would move, moved. The two anchors it could have damaged did not.

**It shipped once without the feature, and the log said so.** The first v7 run declared `in_forest` in `LABEL_RULE_FEATURES` but never added it to the feature vector. The rule changed the labels and the model could not see the input that explained them: verified macro F1 reached only **0.5028**, spatial CV *fell* to 0.9790, and the audit printed **12** ablated columns where it should have printed 13. This is the identical defect recorded one version above, and it was caught by the same single line. Same rule, same data, feature present: **0.5028 -> 0.5425** and a full point of spatial CV.

**What it cost.** Deonar fell **0.552 -> 0.517**. A landfill sitting inside a mapped forest polygon now has a lower artifact floor, which admits more of its chronic smoulder as real burning and slightly blurs the catastrophe against its own baseline. It is a small regression in the worst-understood event in the set and it is recorded rather than absorbed.

**Honesty note.** The floor was swept against the verified set, so those events are partly a tuning set -- the same caveat rule v6's burst test carries. The threshold was chosen from four candidates rather than fitted, and `in_forest` is declared in `LABEL_RULE_FEATURES` so the circularity audit ablates it, but this is not independent confirmation.

#### Rule v8: an onset only counts if the neighbourhood was quiet

`ACCIDENTAL_FIRE` was the worst class in the set at F1 **0.209**, and its precision of **0.142** had a single dominant cause: of 1,709 routine detections called accidents, **1,404 were Jharia**.

The mechanism is the onset test meeting a fire that moves. Outside mapped infrastructure a "source" is a ~900 m cell, and a coal-seam front migrating through unmapped terrain lights a succession of cells that are each individually new. The rule reads every one of them as a fresh ignition: `n_30d >= 8 AND onset_lag > 45 days -> ACCIDENTAL_FIRE`.

**The first attempt at this failed, and the failure is what produced the fix.** Reading onset from a coarser H3 res-6 neighbourhood took `onset_lag > 45` from **2.17% of the corpus to 56.04%** -- a res-6 cell is 48x a res-8 cell, so sparse agricultural areas that never reach the persistence threshold at res 8 cross it easily, and they cross it *during the burning season*. It did not teach the rule that Jharia was already alight; it taught the rule that Punjab agriculture began in October. Reverted, and recorded in `state_machine.py`.

What was wrong was the instrument, not the hypothesis. Redefining onset was the error; **adding evidence beside it** was the fix. `neighbourhood_active_keys` counts the distinct *other* recurrence keys seen within the surrounding area in the trailing 30 days, excluding the detection's own key -- so a source that has merely been burning a long time cannot count itself as a crowd.

The measurement that decided the threshold, taken over the detections where the onset branch actually fires:

| Gate | Jharia false positives kept | Baghjan true positives kept |
| :--- | ---: | ---: |
| `nk <= 8` | **0.0%** | 61.8% |
| **`nk <= 12`** | **0.0%** | **74.5%** |

Every onset-firing Jharia detection sits in a crowded neighbourhood; Baghjan's is sparse even five months into a blowout. The gate was set at the *loosest* threshold that still rejects all of them, because a missed industrial fire is the most expensive error this system makes and the recall cost looked real.

**It was not.** Measured v7 -> v8:

| | v7 | v8 |
| :--- | ---: | ---: |
| **Verified macro F1** | 0.5425 | **0.5544** |
| `ACCIDENTAL_FIRE` precision | 0.142 | **0.183** |
| `ACCIDENTAL_FIRE` recall | 0.395 | **0.396** |
| `ACCIDENTAL_FIRE` F1 | 0.209 | **0.250** |
| `PERSISTENT_BASELINE` recall | 0.916 | **0.929** |
| Jharia (verified) | 0.901 | **0.922** |
| Deonar (verified) | 0.517 | **0.552** |
| Spatial-block CV | 0.9981 | 0.9961 |
| Circularity delta | 0.2629 over 13 | **0.2631 over 14** |

Precision rose 29% and **recall did not move**. The prediction was that the gate would cost roughly a quarter of Baghjan's onset-caught detections; with `neighbourhood_active_keys` in the feature vector the model recovered them, which is the argument for giving the model every input the rule reads rather than only the rule. `PERSISTENT_BASELINE` recall rose as a side effect -- fewer of its detections stolen by false accidents -- and Deonar recovered the 0.035 it lost to v7's forest floor.

The cost is a small easing of spatial-block CV, 0.9981 -> 0.9961, and it is recorded rather than absorbed.

#### Deer Park: the rule is transferable, the model is not

The rule scores Deer Park 0.455 under v5. The model scores **0.182**, and the gap is structural rather than a training artifact.

Rule v5 declines to apply India's harvest calendar outside `SEASONALITY_DOMAIN`, which it decides from latitude and longitude. The model has neither: coordinates are withheld from the feature space on purpose, because a model handed them memorises where refineries are (5b.4). So the model cannot see that it is out of domain, and the training corpus is entirely Indian, so it has never seen an out-of-domain example to learn from either.

That is a deliberate trade: **the labelling rule generalises further than the trained model does.** It is not closed by tuning, and it is not closed by training either -- an "in domain" flag would be constant across every training row and carry no signal.

It is closed by **applying the rule over the model's output at serving time**. `apply_serving_guards()` withholds `AGRICULTURAL_BURN` outside `SEASONALITY_DOMAIN` and reports the detection as an unclassified transient instead, which is true where asserting Indian land use on another continent was not. It applies the non-combustion override for the same reason -- belt-and-braces over a rule the model does already learn, because a classification of "fire" at a site that cannot burn is the one error worth being redundant about. `POST /api/v1/classify` returns both `model_predicted_class` and the served class plus a `serving_guard_note`, so an override is visible rather than silent. Four tests cover it, including that the guard does **not** fire inside the domain -- one that fired everywhere would quietly delete the crop-burn class.

### 5b.3 Current Scores

Measured on the 12-month corpus, 2,044,295 detections, labelling rule **v8**, run `20260913T191719Z-15d55e3d`.

| Metric | Value | What it actually means |
| :--- | ---: | :--- |
| Spatial-block CV macro F1 (real detections) | **0.9961 +/- 0.0013** | 5-fold `GroupKFold` over 40 five-degree geographic blocks; every test fold is regions the model has never seen. Worst fold **0.9949** |
| Random-split macro F1 | 0.9970 | Optimistic; detections cluster spatially |
| Majority-class baseline | 0.2050 | Floor |
| Logistic-regression baseline | 0.8342 | Scaled features; the non-trivial floor |
| **Verified-label macro F1** | **0.5544** | **The only independent number**, pooled over all 20 scorable events. On the 12-month archive alone it is 0.5198 on 39,775 detections. See 5b.2b. |

The gap between 0.996 and 0.554 is the entire point of this section. The first says the model reproduces its own labelling rule almost perfectly across unseen geography. The second says what that rule is and is not right about.

### 5b.3b The Circularity Audit

The weak-labelling rule reads `inside_industrial`, `is_exact_match`, `n_30d`, `frp`, `z_frp`, `onset_lag_days` and -- since v2, though it went undeclared until v5 -- the calendar `month`. `run_circularity_audit()` retrains with those columns ablated and scores real detections only.

| Configuration | 2-day corpus | 60-day, hex | 60-day, facility | 12-month, 10-feature (v5) | 12-month, 12-feature (v6) | 12-month, 13-feature (v7) | **12-month, 14-feature (v8)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full feature set | 0.993 | 0.998 | 1.000 | 0.9945 | 0.9981 | 0.9963 | **0.9976** |
| Label-rule features ablated | 0.657 | 0.819 | 0.905 | 0.7487 | 0.7478 | 0.7335 | **0.7345** |
| **Delta (lower is better)** | **0.336** | **0.179** | **0.095** | **0.2458** | **0.2502** | **0.2629** | **0.2631** |

**Read the ablation count alongside the delta; neither means much alone.** The set has grown from 7 columns to 14. At v5, `month` and `is_harvest_season` were added because the rule had consulted the calendar since v2 and never declared it -- and the delta *fell* anyway, from 0.2938 to 0.2458, because removing two arbitrary branches from the rule left the model less arbitrary structure to memorise. At v6, `n_24h` and `burst_ratio` were added and the delta *rose*, 0.2458 to 0.2502. Both movements are the audit working: what changed was what it was measuring, and the count is what tells you so.

The reason is the mechanism this project has been claiming and could not previously demonstrate: **the delta falls when the rule gets less arbitrary, not when the model gets better at guessing it.** Rule v4 removed the onset asymmetry after the Jaipur fire and the delta fell 0.3024 -> 0.2938. Rule v5 removed two more asymmetries after Deonar and Deer Park, and it fell again. Each verified event removes an assumption the rule was making for free, and the model has correspondingly less arbitrary structure to memorise.

The reading is unchanged and still honest: **partial independence**. The model retains real signal without the label-rule features and still leans on them substantially. Closing this needs verified labels, not more data.

### 5b.4 Spatial Leakage: Explicitly Avoided

Raw `latitude`/`longitude` are **never** features. Industrial facilities are static, so a model given coordinates memorizes where refineries are instead of learning what industrial combustion looks like -- scoring near-perfectly under a random split and then collapsing on an unseen region. `tests/test_evaluation_integrity.py::test_no_coordinates_in_feature_space` enforces this permanently. Coordinates are carried through the corpus for one purpose only: constructing the spatial CV blocks.

### 5b.4b Optical Validation of the P0 Alert Set

The pipeline raises **530 `ACCIDENTAL_FIRE` alerts across 373 distinct locations** from 2,044,295 detections. Sentinel-2 dNBR is computed once per location and asks a question the thermal model cannot: *did anything actually burn?*

An industrial fire consumes stored fuel inside a bunded compound. It should leave the surrounding canopy untouched -- `|dNBR| < 0.10`. A P0 alert that instead shows `dNBR >= 0.27` consumed biomass, which means it was a vegetation or crop fire misfiled as an industrial emergency.

**Result over all 373 locations**, 302 of them measured (425 of 530 detections):

| Verdict | dNBR band | n | Share |
| :--- | :--- | ---: | ---: |
| Contained -- consistent with an industrial fire | \|dNBR\| < 0.10 | 245 | 81.1% |
| Ambiguous -- some surface change | 0.10 to 0.27 | 55 | 18.2% |
| **Refuted -- biomass burned** | dNBR >= 0.27 | **2** | **0.7%** |

`dNBR` min / median / max: `-0.3016 / +0.0207 / +0.3366`.

**Optical false-positive rate on P0 alerts: 0.7%.**

#### Coverage, and why it is not 100%

| Status | Locations | Share |
| :--- | ---: | ---: |
| `OK` | 302 | 81.0% |
| `NO_CLEAR_SCENE` | 66 | 17.7% |
| `NO_SCENES_IN_WINDOW` | 3 | 0.8% |
| `AWAITING_POST_SCENE` | 2 | 0.5% |
| `API_ERROR` | 0 | -- |

`compute_dnbr()` writes to `dnbr_cache.json` only on the `OK` path, so every cloud rejection is absent from it by construction. **Reading the cache to measure coverage reports a rejection rate of zero no matter what happened** -- and it was reported as zero twice during this work before the run log showed otherwise. The table above comes from the output parquet, which records every location's outcome.

The 71 unmeasured locations are Sentinel-2 unable to see the ground, not failures of this system. A scene is trusted only when at least 20% of its pixels survive the Scene Classification Layer mask; below that the location returns `dnbr = None` rather than a burn ratio computed from a handful of cloud-edge pixels. Rejections track the monsoon -- 0% in December, 3% in January, rising through 17-20% in March to May, and above 25% in June and September.

**A caveat that is not yet tested:** these are fire locations, and thick smoke can be classified as cloud by the SCL. If that is happening, dNBR is systematically weakest on the largest fires -- precisely the ones most worth checking -- which would cap how much the 0.7% is worth.

#### The two refutations

| Location | Date | dNBR | FRP | OSM facility tag |
| :--- | :--- | ---: | ---: | :--- |
| 28.344, 76.908 | 2026-03-07 | +0.337 | 35.3 | `general_industrial` |
| 29.174, 78.982 | 2025-10-19 | +0.314 | 10.1 | `petrochemical_refinery` |

Haryana during the Rabi harvest and western Uttar Pradesh in the post-Kharif window. Both are field fires that burned real vegetation, close enough to a tagged industrial polygon to inherit its facility association. The second inherited a **refinery** tag, which is how a crop fire became a P0 industrial emergency.

Neither location is covered by a verified label -- the optical layer found them independently. This is the same failure mode the verified Punjab labels measure at 0.694 and 0.712, reached from unrelated evidence.

#### What this does and does not establish

The distribution did not move as coverage grew. Between 111 and 302 measured locations -- a 2.7x increase in sample -- "contained" stayed within 80.3% to 82.0% and "refuted" within 0.7% to 1.8%. Nothing emerged at scale that was not visible in the first third.

So **99.3% of P0 alerts show no biomass consumption**, which is what an industrial fire should look like. That is the first evidence in this project, from a sensor and a physical quantity entirely independent of the thermal labelling rule, that the alert set is largely what it claims to be.

But **dNBR refutes without confirming.** A contained thermal anomaly could equally be routine flaring the state machine mislabelled; neither burns vegetation, so dNBR cannot separate them. This bounds the false-positive rate from one direction only: at least 0.7% of P0 alerts are wrong, and the remaining 99.3% are *not disproven*, which is weaker than verified. The 18.2% ambiguous band is genuinely inconclusive, not quiet good news.

### 5b.4c Transferability: the Permian Basin Test

The project's own research predicts the judge challenge verbatim -- *"change the bounding box to the Permian Basin and run it right now"*. It was run: 6,689 detections over 60 days across West Texas and south-eastern New Mexico, the largest gas-flaring region in North America.

**What transferred.** FIRMS ingestion ran unchanged. The recurrence state machine, which reasons purely about whether heat returns to the same ~900 m cell, correctly identified **38.8%** of detections as persistent thermal sources with no map data whatsoever. That mechanism is scale-free and geography-free, and it earned its place.

**What did not, and why.** The classifier initially reported **84.1%** of that basin as transient artifacts. Two causes, both measured:

| Cause | Evidence |
| :--- | :--- |
| No polygons outside India | The reference layer spans lon 68.50-96.15, lat 8.08-34.82. Every Permian detection falls outside it, so `inside_industrial` is false everywhere. |
| Absolute FRP thresholds | The rule required `n_30d >= 8` **and** `frp >= 5.0` MW. In the Permian 41.5% of detections recur but only 8.9% clear 5 MW, so 2,587 wellhead flares were labelled sun-glint artifacts. |

The second is a genuine defect and is fixed. Thresholds are now **percentiles of the local FRP distribution** (`derive_frp_thresholds`), and **recurrence outranks energy**: a source emitting from the same cell for a month is stationary, and stationary is the entire distinction between infrastructure and a fire front. Energy only ever said how big it is.

Dropping the energy gate was measured against the verified labels before it was adopted: **zero of 1,995 verified Punjab crop-burn detections reach `n_30d >= 8`**, because agricultural burning migrates field to field and never accumulates in one cell. The override cannot touch agriculture because agriculture does not recur.

| | Before | After | Reference |
| :--- | ---: | ---: | ---: |
| Permian persistent | 1.5% | **37.9%** | state machine 38.8% |
| Permian artifacts | 84.1% | 23.8% | -- |
| Baghjan verified accuracy | 0.545 | **0.773** | -- |
| India verified anchors | 1.000 / 0.996 / 0.974 | **unchanged** | -- |

The classifier now agrees with its own recurrence engine to within a percentage point on a continent it has never seen.

**The residual, and how it closed.** 34.8% of the Chihuahuan Desert was labelled agricultural burning, because `AGRICULTURAL_BURN` was the catch-all for "outside a polygon, not persistent, above the floor" and the taxonomy assumed an Indian landscape. That was recorded here as a data and ontology problem rather than a threshold one, and unfixed.

It was closed by a documented incident rather than by a threshold. The 2019 ITC Deer Park tank-farm fire in Texas -- three days, $150M damage, a US Chemical Safety Board investigation -- came back **`AGRICULTURAL_BURN` on 10 of its 11 detections**, because March is a Rabi harvest month in Punjab. The aggregate error over the Permian and the single misclassified regulator-documented fire were the same defect, and the second one made it impossible to describe as an ontology problem for later.

`HARVEST_MONTHS` is a calibrated prior about Indian agriculture, not a fact about combustion, so rule v5 scoped it to `SEASONALITY_DOMAIN` -- the region where it was established. Outside those bounds the month is not evidence, the agricultural default does not apply, and a detection the system cannot account for is reported as `TRANSIENT_HOTSPOT`: *open ground, no baseline, nothing further established*, which is true, where "agricultural burn" on another continent was an unearned claim about land use.

| Permian Basin, rule labels | v4 | v5 |
| :--- | ---: | ---: |
| `AGRICULTURAL_BURN` | 34.8% (Chihuahuan block) | **0.0%** |
| `PERSISTENT_BASELINE` | 37.9% | 37.9% (state machine 38.8%) |
| `TRANSIENT_HOTSPOT` | 23.8% | 56.8% |
| `ACCIDENTAL_FIRE` | -- | 5.3% |

The transferability gain from dropping the energy gate is preserved exactly; what changed is that the classes the system cannot justify abroad are now named as unestablished rather than as farming.

### 5b.4d Sub-Pixel Fire Temperature: Implemented, Validated, Rejected

The problem-statement research puts absolute combustion temperature first in the prescribed feature vector, retrieved by Planck-curve fitting. `src/pipeline/thermal_physics.py` implements it -- Planck inversion, the Dozier (1981) bi-spectral method, vectorised bisection over two million detections, background estimated per month and day/night stratum.

**It is deliberately not wired into the pipeline.** Validated against the verified events on 2026-09-13:

| Verified site | Retrieved | Physically expected |
| :--- | ---: | ---: |
| Reliance Jamnagar, refinery flare | **443 K** | 1700-2000 K |
| JSW Vijayanagar, blast furnace | **457 K** | 1500-1900 K |
| Visakhapatnam Steel | **487 K** | 1500-1900 K |
| Punjab crop burning | **522 K** | 800-1000 K |
| **Khavda Solar Park** | **466 K** | **no combustion** |

Two failures, either of which is disqualifying. Every regime collapses into a 443-522 K band, so the retrieval does not separate a refinery flare from a wheat field -- which is the only reason to compute temperature at all. And it returns a confident 466 K for a photovoltaic array, where the correct answer is that there is no fire.

**Why.** A VIIRS pixel is 375 m across; a flare stack is metres. The fire occupies a thousandth of the pixel, so the signal is dominated by the surrounding ground, and recovering the fire's temperature requires knowing that background. The FIRMS active-fire product does not carry it, so it is estimated -- and the estimate then governs the answer. The module's own header warned this would affect weak fires; the validation shows it affects strong ones too.

**The named fix was tried, on the right instrument, and it failed too.** The diagnosis above put the blame on the estimated background, so the obvious test was an instrument that supplies a measured one. Sentinel-3 SLSTR does: it is a full-swath imager, so the ground around a hot pixel is available, and its F1 channel does not saturate where S7 clips. `src/ingestion/slstr_client.py` fetches F1/F2 over the source and over an annulus of undisturbed ground on the same acquisition, and hands both to the same tested solver.

| Verified site | F1 max | S7 max | Measured background | Retrieved | Physically expected |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Reliance Jamnagar, refinery flare | 320.8 K | 312.3 K | 291.6 K | **no solution** | 1700-2000 K |
| JSW Vijayanagar, blast furnace | 316.5 K | 312.2 K | 300.1 K | **473 K** | 1500-1900 K |
| Visakhapatnam Steel | 316.1 K | 312.2 K | 295.1 K | **569 K** | 1500-1900 K |
| Punjab crop burning | 292.4 K | 291.5 K | 291.2 K | **no solution** | 800-1000 K |
| **Khavda Solar Park** | 317.8 K | 312.3 K | 300.6 K | **411 K** | **no combustion** |

411-569 K, and the photovoltaic array still solves. **The estimated background was not the binding constraint.** SLSTR resolves 1 km against VIIRS's 375 m, so a 10 m flare occupies about 1e-5 of a pixel instead of 7e-5 -- the measured background is a real improvement and the sub-pixel dilution is a roughly sevenfold regression, and the dilution wins.

This is worth more than the original rejection. The first result said a two-channel retrieval fails on FIRMS inputs; this one says it fails on a better instrument, for the reason the first result named as the fix. The hypothesis is eliminated rather than untested.

**What would fix it** remains the multi-band VIIRS Nightfire retrieval, which fits M7/M8/M10/M12/M13 simultaneously rather than inverting two channels. Those bands are in the VNF/VNP46 products, not in the active-fire product this project ingests.

One process note, because it cost an afternoon and would cost it again. The first SLSTR probe returned HTTP 401 `invalid_client` on every attempt and was written off as CDSE rate limiting. The credentials were fine: the probe script lived outside the project tree, and `load_dotenv()` with no argument searches upward from the *calling file*, not the working directory, so it read no `.env` at all. A missing credential and a throttled one are the same status code. The module now pins the path.

The physics and the solver are correct and tested. The inputs are insufficient, and a 466 K "fire temperature" for a solar farm is the same failure as synthetic weather: a number that looks like a measurement and is not. The validation result is recorded in the module header so nobody wires it in later believing it only needs plumbing.

### 5b.4e Additional Sensors: What Is Reachable

| Sensor | Status | Blocker |
| :--- | :--- | :--- |
| **Sentinel-3 SLSTR** | **Integrated** (`src/ingestion/slstr_client.py`, 14 tests, `GET /api/v1/incident/{id}/thermal-probe`) | None. It runs on the existing CDSE credentials -- no new registration. Its F1 channel does not saturate where S7 clips near 311 K, measured directly: Jamnagar on 2026-03-25 read S7 **312.3 K** against F1 **320.8 K**. VIIRS I-4 clips at 366.9 K, costing 3% of all detections and **12% of `ACCIDENTAL_FIRE` detections**. |
| **INSAT-3D / 3DS** | Not reachable | ISRO MOSDAC registration. |
| **VNP14IMG** | Not reachable | NASA Earthdata token, distinct from the FIRMS map key. |

**The case for INSAT is no longer theoretical, and it now rests on three fires rather than one.** Haldia (2026-06-30, ignited 04:00-04:30 local), ONGC Uran (2018-09-03, ignited ~07:00 and out by ~09:00) and the Anaj Mandi factory fire that killed 43 people in Delhi (2019-12-08, reported ~05:00) each opened and closed inside the gap between the ~01:30 and ~13:30 VIIRS passes, and each produced **zero detections** -- measured, with a working control in every case (5c.3). One such fire is bad luck; three is a duty cycle. A geostationary platform at a 15-to-30-minute cadence would have seen all three, which is a measured argument for the sensor rather than a specification box.

**What INSAT would not fix** is equally measured, and 5c.3 carries it: a confined boiler explosion at Neyveli, a steel-plant blast at Bhilai that the sensor did see and could not distinguish from routine operation, and the LG Polymers styrene release, which was not a fire at all. Cadence is one of several bounds, not the only one.

**What SLSTR did and did not buy.** It is a second independent instrument, it supplies a measured background where FIRMS supplies none, and it reads past the standard channel's ceiling. It did **not** rescue the sub-pixel temperature retrieval -- see 5b.4d, where the measured background was tried and the 1 km footprint defeated it anyway. The client is integrated for what it can actually establish, and the endpoint returns `NO_SOLUTION` rather than a number when the mixture has no admissible solve.

One operational note recorded here because it cost time twice, in opposite directions. The CDSE token endpoint answers heavy use with **HTTP 401 `invalid_client`**, not 429, which is indistinguishable from a bad secret unless the caller retries -- and on the second occasion a probe that returned exactly that turned out to have no credentials at all, because `load_dotenv()` searches upward from the calling file rather than the working directory and the script lived outside the project tree. Neither diagnosis is safe without checking the other.

### 5b.4f Forest Fires: Segregation by Land Cover, Not by Class

The mandate asks for industrial fires to be **"explicitly segregated from forest fires and natural thermal events"**. Until this was built they were not. A forest fire reaches the classifier as sustained open-ground combustion with no facility history, which is exactly what a crop fire looks like, and it was served `AGRICULTURAL_BURN`.

**The obvious fix is a fifth class, and it is the wrong one.** The model is coordinate-free on purpose (5b.4). It cannot see that one sustained open-ground fire is in the Western Ghats and another is in a Punjab wheat field, and the radiometry is genuinely the same in both. A fifth class the feature vector cannot separate would ask the model to guess, and it would guess from whatever spurious correlation the corpus happened to contain -- the same failure mode as spatial leakage, arrived at from the other direction.

What separates them is **land cover**, which is a fact about the map. So the model keeps classifying combustion *behaviour*, and `src/pipeline/forest_cover.py` answers what was burning, from the same national OSM extract the industrial layer comes from. `FOREST_FIRE` is therefore a **served class, never a trained one**: the model never predicts it, no training label carries it, and the verified harness never scores it.

| | |
| :--- | ---: |
| Forest polygons extracted (`landuse=forest`, `natural=wood`) | **76,186** |
| Mapped area, equal-area projection | **568,585 km2** |
| For scale: India's forest cover per Forest Survey of India | ~713,000 km2 |

`natural=scrub` is deliberately excluded. Scrubland abuts agricultural land across most of India and including it would pull crop burning into the forest class -- the exact confusion this exists to resolve.

#### The two ways this could fail silently, both measured

A land-cover guard fails **vacuously** if OSM forest coverage is so thin that nothing lands inside a polygon, or **greedily** if the polygons are so broad that they swallow the crop-burn class, which is larger and far better evidenced. Both were counted against the 12-month corpus before anything was claimed.

| Measurement | Result |
| :--- | ---: |
| Open-ground candidates in the corpus | 1,541,361 |
| Falling inside mapped forest | **261,864 (16.99%)** |
| Verified Punjab crop-burn detections affected | **0 of 2,015 (0.00%)** |
| Change to any verified event's served score | **none** |

Neither failure mode occurred. The determination fires on a sixth of open-ground detections and does not touch a single detection in the verified agricultural windows.

#### The seasonality is independent corroboration

The strongest evidence is one nobody designed for. Detections falling inside forest cover follow the **Indian forest fire calendar**, not the crop calendar:

| Month | Feb | Mar | Apr | May | Jul | Aug |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Detections in forest | 22,727 | **102,081** | **100,214** | 20,369 | 172 | 147 |

**93.7% fall in February to May**, peaking in March and April, and the monsoon months are effectively empty -- 319 detections across July and August combined, 0.12% of the annual total. That is the documented Indian forest fire season reproduced exactly, from a layer selected purely on OSM tags with no temporal input whatsoever. Punjab's crop windows peak in *October-November* and *April-May*; March carries 39% of the forest total and is not a major residue month. The land-cover layer and the burn calendar agree without having been told to.

#### Ordering, and why it is not arbitrary

`apply_serving_guards()` now runs three guards, and forest runs **last**:

1. **Non-combustion land use** -- a site that cannot burn, decided first.
2. **Out-of-domain harvest calendar** -- withholds an unearned claim.
3. **Forest cover** -- adds specificity to whatever survives.

Forest runs last because it only ever refines an `AGRICULTURAL_BURN` the first two guards have allowed to stand. A Texan open-ground burn is withheld by guard 2 before the Indian forest layer is ever consulted, which is correct: asserting Indian land cover in Texas is precisely the error guard 2 exists to prevent. A test asserts that ordering directly.

The determination is also **additive only**. Absence of a forest polygon is not absence of forest -- the same argument this project already makes about industry and Baghjan -- so a detection outside forest cover is left exactly as the model found it, and only `AGRICULTURAL_BURN` is eligible for refinement. A brick kiln in a forest clearing stays a brick kiln, and an `ACCIDENTAL_FIRE` is never relabelled, because that would downgrade it out of dispatch: the most expensive error this system can make.

#### The class now has evidence, and it immediately found something

`FOREST_FIRE` shipped on a land-cover argument and a seasonality signature. Three cited events now score it, chosen by the same rule every other verified event had to pass -- documented multi-day duration, then pull the window and count before writing anything down.

| Event | n | Model | Served | Inside mapped forest |
| :--- | ---: | ---: | ---: | ---: |
| Bandipur National Park, Feb 2019 | 726 | 0.669 | **0.640** | 95% |
| Similipal Biosphere Reserve, Feb-Mar 2021 | 3,035 | 0.534 | **0.518** | 97% |
| Uttarakhand Garhwal & Kumaon, Apr-May 2016 | 1,512 | 0.519 | **0.349** | 71% |

Each carries **two truth classes**, and the distinction is the point. The *model* should say `AGRICULTURAL_BURN`: sustained open-ground combustion with no facility history is the finest distinction a coordinate-free feature vector supports, and a forest fire is radiometrically identical to a crop fire. The *served* answer should be `FOREST_FIRE`. Scoring the model against `FOREST_FIRE` would penalise it for not knowing something it is deliberately never told, so `VerifiedEvent` gained a `served_label` field, defaulting to `label` so all eighteen existing events keep exactly the meaning they had.

**The defect these events exposed: the artifact floor eats half of a real forest fire.**

| Event | Median FRP | Classified `TRANSIENT_HOTSPOT` |
| :--- | ---: | ---: |
| Bandipur | 10.98 MW | 11.3% |
| Similipal | 1.72 MW | **45.0%** |
| Uttarakhand | 1.86 MW | **44.5%** |

`ARTIFACT_FRP_FLOOR_HARVEST` is 1.59 MW, and a smouldering understorey fire sits right on it. Bandipur, burning at a median of 10.98 MW and peaking at 4,252 MW, loses 11%. Similipal and Uttarakhand are ordinary forest fires -- cooler, more diffuse, longer -- and the floor discards nearly half of each as artifacts.

This is not a threshold to nudge. The floor is a percentile of the local FRP distribution and it exists because low-energy open-ground detections genuinely are dominated by noise and glint; lowering it globally would readmit exactly what it was calibrated to exclude. What the evidence says is narrower and more useful: **the floor is calibrated for crop residue and is wrong for forest**. A land-cover-aware floor is the obvious next move, and it is now a measurement rather than a hunch. Logged, not fixed.

**Uttarakhand is the weakest of the three and says so.** Its served score of 0.349 is the lowest, its confidence is `medium`, and both are earned: detections precede its window inside the same radius so there is no clean onset, and only ~71% of its detections fall inside mapped forest against 95-97% for the others. That is not a radius artifact -- the forest share is flat at 70/73/71/74/67/74 percent across 20 to 50 km, so the ceiling is **OSM forest coverage in Himalayan terrain**. The radius was set to 30 km on that evidence, because the wider label bought nothing but unrelated detections.

#### What is still open

A forest fire is **routed rather than dispatched** -- `P2_ADVISORY`, state `FOREST_FIRE`, which matches the design intent that a confirmed natural fire reaches forestry and environmental departments rather than an industrial response team. Segregated, not discarded.

The class is now scored against three cited events (above), and the remaining gap is narrower than it was: all three are Indian, so `FOREST_FIRE` has never been checked outside the region whose forest layer this project extracted. The determination is also bounded by OpenStreetMap coverage, which Uttarakhand measures at ~71% against Bandipur's 95%.

### 5b.5 Outstanding Work Before Any Accuracy Claim

1. **Keep growing the verified-label set.** 21 events (20 scorable, catalogued in 5c) is enough to expose defects and reject fixes; it is not enough to certify accuracy. The method is now proven and repeatable: search for fires with a **documented multi-day duration**, because those are the ones that necessarily straddled satellite overpasses. Every event added this way has found something -- Deonar and Brahmapuram found the chronic-baseline blind spot, Deer Park found the exported harvest calendar, Rumaila found that persistence does not transfer, the coal-seam fires are the only persistent anchor `inside_industrial` cannot explain, and the three forest fires found that the artifact floor is calibrated for crop residue. Each event removes an assumption the rule was making for free, which is why the circularity delta falls as the set grows.

2. **Persistence does not transfer, and Rumaila proves it.** Both model and rule
   score 0.065 on a documented non-Indian flare field, because `n_30d` is keyed
   on facility identity where a polygon exists and on a ~900 m hex where none
   does. A dispersed flare field never reaches the threshold. This was invisible
   until a cited persistent source outside India existed to expose it.

3. **`ACCIDENTAL_FIRE` is still the worst class at F1 0.250, and the remaining half is recall.** Rule v8 repaired precision (0.142 -> 0.183) by gating onset on neighbourhood activity. What it did not touch is recall, still **0.396**: **443 of the 490 missed accidents are Baghjan**, a five-month blowout that became its own baseline, so recurrence reports it as routine. The mechanism is named and the fix is architectural rather than a threshold -- the system classifies *detections* independently, but an incident is a temporal object, and a source that entered `ACCIDENTAL_FIRE` should stay there until it goes quiet. Hysteresis is the obvious next change and it has not been attempted.

4. **Deer Park at 0.182 is the worst result inside the domain**, second only to Rumaila's 0.065 -- and both are abroad, which is itself the finding. Deonar is no longer on this list: rule v6 moved it from 0.069 to **0.552** on a rate signal after the obvious threshold fix was tried and rejected on measurement (5b.2b), and v8 restored the 0.035 it briefly lost to v7's forest floor. The chronic-baseline blind spot is narrowed, not closed: Brahmapuram still reads 0.533.

5. **The labelling rule generalises further than the trained model.** Rule v5 declines to apply India's harvest calendar outside the region where it was calibrated; the model cannot see that gate, because coordinates are deliberately withheld to prevent spatial leakage and the training corpus is entirely Indian. Serving another region means retraining on its detections, not tuning.

6. **Two sensors still need registrations this project does not hold.** INSAT-3D/3DS needs ISRO MOSDAC; VNP14IMG needs a NASA Earthdata token distinct from the FIRMS map key. The INSAT case is measured rather than aspirational -- the Haldia fire of 2026-06-30 burned entirely between VIIRS overpasses and produced zero detections (5b.4e).

7. **Sub-pixel fire temperature stays unwired.** Implemented, validated against known regimes on two instruments, and rejected both times (5b.4d). The remaining path is the multi-band VIIRS Nightfire retrieval, which needs the VNF/VNP46 products.

8. **Re-run the circularity audit after any rule change, and read the ablation count as well as the delta.** The delta moved 0.3024 -> 0.2938 -> 0.2458 -> 0.2502 -> 0.2629 -> **0.2631** across rule versions 3 to 8, while the ablation set grew from 7 columns to **14**. The rise at v6 is the audit getting stricter, not the model getting worse -- and the count is what tells you which. Rule v6 was first shipped with two rule inputs missing from the feature vector; the log printed **ten** ablated columns where it should have printed twelve, and that single line was the difference between a rule the model could learn and one it could not.

---

## 5c. The Verified-Event Register

<!-- BEGIN GENERATED: verified-event register -->

**21 events are defined**, four of them outside the region this system was calibrated in. 20 currently match detections in a corpus on disk, together covering **46,077 detections**.

Section 5b.2b argues from this set; this section *is* the set. It is generated from `src/models/verified_labels.py` by `scripts/generate_verified_register.py`, so it cannot drift away from the objects the evaluation harness actually loads. Every figure below is computed at generation time from the model artifact and the corpora on disk.

### 5c.1 The register

Two accuracy columns, because they differ and a single number would mislead. **Model** is the raw classifier output and is what the evaluation harness scores. **Served** is what the API actually returns, after `apply_serving_guards()` applies the constraints the model structurally cannot learn -- chiefly that India's harvest calendar is a calibrated prior about Indian agriculture, not a fact about combustion, and is unearned outside the domain where it was measured.

**Which corpus an event is scored in is part of the result.** The recurrence features are computed from whatever history that corpus contains. Baghjan reads **0.793** here, against the pre-event pull -- and **0.075** against a corpus that starts at the blowout, where `n_30d` and the onset lag have no baseline to contrast the accident with. Each event is therefore scored in the first corpus that covers it, national archive first, and 5c.2 names that corpus beside every event so the figure can be reproduced.

Three events carry **two truth classes**, and that is not a hedge. A forest fire's correct *model* answer is `AGRICULTURAL_BURN`: sustained open-ground combustion with no facility history is the finest distinction a coordinate-free feature vector supports, and a forest fire is radiometrically identical to a crop fire. Its correct *served* answer is `FOREST_FIRE`, because `apply_forest_cover` consults land cover, which the model cannot. Scoring the model against `FOREST_FIRE` would penalise it for not knowing something it is deliberately never told.

| # | Event | Truth class | Region | Window | Radius | n | Model | Served | Conf. |
| ---: | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 1 | Khavda Renewable Energy Park (Gujarat) - reflective artifact | TRANSIENT_HOTSPOT | India | `2024-01-01` → `2026-12-31` | 4 km | 100 | 1.000 | **1.000** | medium |
| 2 | Pavagada Solar Park (Karnataka) - reflective artifact | TRANSIENT_HOTSPOT | India | `2024-01-01` → `2026-12-31` | 3 km | 53 | 1.000 | **1.000** | medium |
| 3 | Benban Solar Park (Aswan, Egypt) - reflective artifact | TRANSIENT_HOTSPOT | **abroad** | `2026-03-05` → `2026-05-31` | 6 km | 16 | 0.000 | **1.000** | medium |
| 4 | JSW Vijayanagar Works (Jindal Steel, Karnataka) | PERSISTENT_BASELINE | India | `2024-01-01` → `2026-12-31` | 4 km | 4,505 | 0.975 | **0.975** | high |
| 5 | Visakhapatnam Steel Plant (RINL, Andhra Pradesh) | PERSISTENT_BASELINE | India | `2024-01-01` → `2026-12-31` | 5 km | 2,591 | 0.965 | **0.965** | high |
| 6 | Raniganj coalfield seam fires (Paschim Bardhaman, West Bengal) | PERSISTENT_BASELINE | India | `2025-09-13` → `2026-09-12` | 10 km | 6,236 | 0.963 | **0.963** | high |
| 7 | Reliance Jamnagar refinery complex, routine flaring (Gujarat) | PERSISTENT_BASELINE | India | `2024-01-01` → `2026-12-31` | 6 km | 539 | 0.941 | **0.941** | high |
| 8 | Jharia coalfield seam fires (Dhanbad, Jharkhand) | PERSISTENT_BASELINE | India | `2025-09-13` → `2026-09-12` | 10 km | 23,633 | 0.922 | **0.922** | high |
| 9 | Punjab post-Kharif paddy residue burning (Oct-Nov window) | AGRICULTURAL_BURN | India | `2025-10-05` → `2025-11-30` | 25 km | 510 | 0.900 | **0.900** | medium |
| 10 | Punjab post-Rabi wheat residue burning (Apr-May window) | AGRICULTURAL_BURN | India | `2026-04-10` → `2026-05-31` | 25 km | 1,608 | 0.884 | 0.884 | medium |
| 11 | Similipal Biosphere Reserve fires (Mayurbhanj, Odisha) | AGRICULTURAL_BURN<br>&rarr; **FOREST_FIRE** | India | `2021-02-20` → `2021-03-10` | 25 km | 3,035 | 0.875 | 0.860 | high |
| 12 | Buncefield oil depot fire (Hemel Hempstead, United Kingdom) | ACCIDENTAL_FIRE | **abroad** | `2005-12-11` → `2005-12-15` | 3 km | 5 | 0.800 | 0.800 | high |
| 13 | Baghjan gas well blowout (Oil India, Tinsukia, Assam) | ACCIDENTAL_FIRE | India | `2020-06-09` → `2020-11-15` | 3 km | 705 | 0.793 | 0.793 | high |
| 14 | Bandipur National Park forest fire (Chamarajanagar, Karnataka) | AGRICULTURAL_BURN<br>&rarr; **FOREST_FIRE** | India | `2019-02-21` → `2019-02-25` | 15 km | 726 | 0.767 | 0.741 | high |
| 15 | Jaipur IOC oil depot fire (Sitapura, Rajasthan) | ACCIDENTAL_FIRE | India | `2009-10-29` → `2009-11-09` | 5 km | 33 | 0.697 | 0.697 | high |
| 16 | Uttarakhand Himalayan forest fires (Garhwal and Kumaon) | AGRICULTURAL_BURN<br>&rarr; **FOREST_FIRE** | India | `2016-04-25` → `2016-05-10` | 30 km | 1,512 | 0.731 | 0.562 | medium |
| 17 | Deonar landfill fire (Mumbai, Maharashtra) | ACCIDENTAL_FIRE | India | `2016-01-27` → `2016-01-31` | 2 km | 29 | 0.552 | 0.552 | high |
| 18 | Brahmapuram waste plant fire (Kochi, Kerala) | ACCIDENTAL_FIRE | India | `2023-03-02` → `2023-03-14` | 2 km | 30 | 0.533 | 0.533 | high |
| 19 | ITC Deer Park terminal fire (Harris County, Texas, USA) | ACCIDENTAL_FIRE | **abroad** | `2019-03-17` → `2019-03-23` | 1.5 km | 11 | 0.182 | **0.182** | high |
| 20 | Rumaila oil field flaring (Basra, Iraq) | PERSISTENT_BASELINE | **abroad** | `2026-03-01` → `2026-05-31` | 30 km | 200 | 0.065 | **0.065** | medium |

Defined but not currently scorable -- the event is verified, but no corpus on disk covers its window, so it contributes no rows to any accuracy figure:

- **Punjab post-Kharif paddy residue burning** &mdash; `2023-10-15` → `2023-11-30`, 30.5000, 75.8000

### 5c.2 Provenance: how each label was established

`VerifiedEvent.__post_init__` raises without a `source`. The constraint is deliberate and it is the point of the whole module: a label inferred from the model, from FIRMS recurrence, or from the weak-labelling rule is not evidence about any of them. If it cannot be cited, it does not go in the set.

**Khavda Renewable Energy Park (Gujarat) - reflective artifact** — `TRANSIENT_HOTSPOT` (negative control), medium confidence, 24.1191, 69.3708, 4 km, n=100, served **1.000** (`firms_industrial_joined_12m.parquet`)

> Verified by land use: a photovoltaic and wind generation park with no combustion process of any kind. Geometry from the 96.6 km2 'Khavda Renewable Energy Park' polygon in the reference layer. Thermal detections here cannot be industrial combustion and are consistent with specular reflection off the array.

Negative control for sun glint. The polygon is tagged industrial in OSM, so detections here are currently credited with facility recurrence and reported as a persistent thermal source -- which is precisely the false positive this label is meant to expose.

**Pavagada Solar Park (Karnataka) - reflective artifact** — `TRANSIENT_HOTSPOT` (negative control), medium confidence, 14.2622, 77.4315, 3 km, n=53, served **1.000** (`firms_industrial_joined_12m.parquet`)

> Verified by land use: one of the largest photovoltaic installations in India, with no combustion process. Geometry from the 44.3 km2 'Pavagada Solar Park' polygon in the reference layer.

Second glint control, geographically distant from Khavda.

**Benban Solar Park (Aswan, Egypt) - reflective artifact** — `TRANSIENT_HOTSPOT` (negative control), medium confidence, 24.4500, 32.7500, 6 km, n=16, served **1.000** (`firms_benban_2026_joined.parquet`)

> Verified by land use: one of the largest photovoltaic complexes in the world, built on open desert, with no combustion process of any kind. Thermal detections inside it cannot be industrial combustion and are consistent with specular reflection off the array -- the same argument that applies to Khavda and Pavagada, tested for the first time outside India.

Thin: 16 detections on 4 distinct days over three months, against 100 at Khavda and 53 at Pavagada. A desert array in Egypt produces far fewer spurious detections than the Indian parks do, which is worth knowing in itself. Scored as a negative control -- any detection here classified ACCIDENTAL_FIRE is a false positive of exactly the kind that causes alert fatigue.

**JSW Vijayanagar Works (Jindal Steel, Karnataka)** — `PERSISTENT_BASELINE` (continuous heat), high confidence, 15.1805, 76.6684, 4 km, n=4,505, served **0.975** (`firms_industrial_joined_12m.parquet`)

> Verified by facility type and geometry: an integrated steel works operating continuous blast furnaces. Geometry is the centroid of the 21.2 km2 'Jindal Steel Works' polygon in the reference layer.

Second steel anchor, geographically distant from Visakhapatnam.

**Visakhapatnam Steel Plant (RINL, Andhra Pradesh)** — `PERSISTENT_BASELINE` (continuous heat), high confidence, 17.6125, 83.1863, 5 km, n=2,591, served **0.965** (`firms_industrial_joined_12m.parquet`)

> Verified by facility type and geometry rather than by incident: an integrated steel plant whose blast furnaces and coke ovens run continuously by design. Geometry is the centroid of the 31.8 km2 'Visakapatnam Steel Plant' polygon in this project's own OSM/Bhuvan reference layer.

Continuous metallurgical heat, not episodic combustion.

**Raniganj coalfield seam fires (Paschim Bardhaman, West Bengal)** — `PERSISTENT_BASELINE` (continuous heat), high confidence, 23.6200, 87.1300, 10 km, n=6,236, served **0.963** (`firms_industrial_joined_12m.parquet`)

> India's oldest coalfield, with mine fires documented since 1906 and mapped repeatedly in the same thermal remote-sensing literature that covers Jharia. Included as a second, geographically separate coal-fire anchor so the class is not established by a single site.

6,236 detections on 303 distinct days in the 12-month corpus.

**Reliance Jamnagar refinery complex, routine flaring (Gujarat)** — `PERSISTENT_BASELINE` (continuous heat), high confidence, 22.3380, 69.8686, 6 km, n=539, served **0.941** (`firms_industrial_joined_12m.parquet`)

> Verified by facility type rather than by incident: one of the world's largest petroleum refining complexes, where continuous flare-stack operation is a designed, permanent feature. Geometry confirmed by the OSM industrial polygon in this project's own reference layer. Thermal behaviour confirmed independently by Sentinel-2 dNBR of +0.0345 (UNBURNED) at this complex, i.e. intense thermal activity with no biomass consumption.

A standing negative control. Any detection here classified ACCIDENTAL_FIRE is a false positive of exactly the kind that causes alert fatigue.

**Jharia coalfield seam fires (Dhanbad, Jharkhand)** — `PERSISTENT_BASELINE` (continuous heat), high confidence, 23.7500, 86.4200, 10 km, n=23,633, served **0.922** (`firms_industrial_joined_12m.parquet`)

> Established by a peer-reviewed remote-sensing literature independent of this project: subsurface and surface coal fires at Jharia have burned since first recorded in 1916, and thermal-anomaly mapping from Landsat TIR has tracked their extent at five-year intervals from 1988 to 2013, with around 70% of the coalfield's mines affected by surface or subsurface fire. The class follows from that documented fact -- a source alight every day for a century must never page an incident commander -- and not from this project's recurrence counter.

23,633 detections on 326 of 365 days in the 12-month corpus. 58% of them fall OUTSIDE any mapped industrial polygon, so this is also the only persistent event in the set that the `inside_industrial` feature cannot explain.

**Punjab post-Kharif paddy residue burning (Oct-Nov window)** — `AGRICULTURAL_BURN` (seasonal biomass), medium confidence, 30.5000, 75.8000, 25 km, n=510, served **0.900** (`firms_industrial_joined_12m.parquet`)

> Documented annual seasonal phenomenon: large-scale paddy stubble burning across the Punjab plain following the Kharif harvest. Independently corroborated for this project by a Sentinel-2 dNBR of +0.2775 (MODERATE_LOW burn severity) measured at this location from live Copernicus imagery.

Region-and-season label, hence the wide radius.

**Punjab post-Rabi wheat residue burning (Apr-May window)** — `AGRICULTURAL_BURN` (seasonal biomass), medium confidence, 30.5000, 75.8000, 25 km, n=1,608, served **0.884** (`firms_industrial_joined_12m.parquet`)

> Documented annual seasonal phenomenon: wheat residue burning following the Rabi harvest. This is the larger of the two Punjab windows in the corpus (1,568 detections in May against 361 in November), and the month originally omitted from the harvest-season feature.

Adding this window is what exposed is_harvest_season excluding May despite May carrying 10.8% of national open-ground burn detections.

**Similipal Biosphere Reserve fires (Mayurbhanj, Odisha)** — `AGRICULTURAL_BURN` (seasonal biomass), high confidence, 21.8500, 86.3500, 25 km, n=3,035, served **0.860** (`firms_similipal_2021_joined.parquet`)

> Extensively reported fires across the Similipal Biosphere Reserve and tiger reserve, Mayurbhanj district, Odisha, burning through late February and early March 2021. The scale drew national coverage and a state response involving forest squads, fire watchers and community volunteers across hundreds of active points.

A diffuse, weeks-long event rather than a single front: 3,027 detections on 16 days, 97% inside mapped forest. Median FRP is only 1.72 MW, which is what makes it valuable -- it sits near the artifact floor and exposes what that floor costs (5b.4f).

**Buncefield oil depot fire (Hemel Hempstead, United Kingdom)** — `ACCIDENTAL_FIRE` (accident), high confidence, 51.7666, -0.4255, 3 km, n=5, served **0.800** (`firms_buncefield_2005_joined.parquet`)

> UK Health and Safety Executive investigation into the explosion and fire at the Buncefield oil storage depot, which ignited at about 06:00 on 2005-12-11, injured over 40 people, forced 2,000 evacuations and produced the largest peacetime fire in Europe to that date. Flames were largely extinguished by the afternoon of 13 December, with one tank reigniting and left to burn.

Added expecting it to FAIL the detectability test, and it passed. December in the UK, MODIS only -- VIIRS did not exist in 2005 -- and a dense black plume that could plausibly have masked the thermal signal from above. Terra caught four detections at 21:25 on the 11th (FRP 20.8-57.0, confidence 73-100) and Aqua one at 01:28 on the 12th, with zero detections anywhere in the 10 km window during the preceding three months. The fire burned four to five days and produced two days of detections, which is its own quiet measurement of what the plume cost.

**Baghjan gas well blowout (Oil India, Tinsukia, Assam)** — `ACCIDENTAL_FIRE` (accident), high confidence, 27.5870, 95.3850, 3 km, n=705, served **0.793** (`firms_baghjan_preevent_joined.parquet`)

> Widely documented industrial disaster: Oil India Limited well Baghjan-5 blew out on 2020-05-27 and ignited on 2020-06-09, burning until it was capped in November 2020.

Sustained months-long accidental industrial fire. Optically hard to validate via dNBR because the burn period sits inside the Assam monsoon -- see sentinel2_client validation notes.

**Bandipur National Park forest fire (Chamarajanagar, Karnataka)** — `AGRICULTURAL_BURN` (seasonal biomass), high confidence, 11.6700, 76.6300, 15 km, n=726, served **0.741** (`firms_bandipur_2019_joined.parquet`)

> Widely documented forest fire in Bandipur National Park, Chamarajanagar district, Karnataka, which burned from 2019-02-21 across several thousand hectares of the tiger reserve. The Indian Air Force deployed Mi-17 helicopters for aerial water drops and hundreds of forest staff and volunteers were mobilised before it was brought under control on 2019-02-25.

The cleanest of the three: 709 detections over 5 days against 14 in the preceding three weeks, peaking at 4,252 MW. 95% of sampled detections fall inside mapped forest cover, so the FOREST_FIRE determination has the land-cover evidence it needs.

**Jaipur IOC oil depot fire (Sitapura, Rajasthan)** — `ACCIDENTAL_FIRE` (accident), high confidence, 26.7810, 75.8373, 5 km, n=33, served **0.697** (`firms_jaipur_2009_joined.parquet`)

> Widely documented industrial disaster: fuel released during a pipeline transfer at the Indian Oil Corporation POL terminal, Sitapura Industrial Area, Jaipur, ignited on 2009-10-29 and burned out of control for eleven days. Twelve people were killed, more than 300 injured, and roughly half a million evacuated. Recorded in the FABIG industrial-accident database (fabig.com/industrial-accidents/jaipur-oil-depot-india) and imaged from orbit by NASA Earth Observatory.

The second verified ACCIDENTAL_FIRE, and the reason it is usable where three documented 2026 fires were not: it burned for eleven days, so it straddles many overpasses. The onset is unambiguous -- 33 detections inside 5km between 2009-10-29 and 2009-11-04, peaking at 230.8 MW, against ZERO in the four weeks before ignition. MODIS only; VIIRS did not launch until 2012.

**Uttarakhand Himalayan forest fires (Garhwal and Kumaon)** — `AGRICULTURAL_BURN` (seasonal biomass), medium confidence, 30.1000, 79.3000, 30 km, n=1,512, served **0.562** (`firms_uttarakhand_2016_joined.parquet`)

> The 2016 Uttarakhand forest fires, which burned across the Garhwal and Kumaon divisions through late April and early May 2016, affecting thousands of hectares of pine forest. The National Disaster Response Force was deployed alongside state forest personnel, and the scale and duration were the subject of proceedings before the Supreme Court of India.

Confidence is medium and the reason is measured rather than cautious. Detections precede the window inside the same radius, so there is no clean onset: fires were already burning through April. And only ~71% of sampled detections fall inside mapped forest, against 95-97% for Bandipur and Similipal. That is not a radius problem -- the forest share is flat at 70/73/71/74/67/74 percent across 20 to 50 km, so the ceiling is OSM forest coverage in Himalayan terrain. The radius was set to 30 km on that evidence: the wider label bought nothing but unrelated detections.

**Deonar landfill fire (Mumbai, Maharashtra)** — `ACCIDENTAL_FIRE` (accident), high confidence, 19.0720, 72.9290, 2 km, n=29, served **0.552** (`firms_deonar_2016_joined.parquet`)

> NASA Earth Observatory records that sensors on Terra, Aqua and Suomi NPP began detecting smoke and fire at the 132-hectare Deonar dumping ground on 2016-01-27 and that it burned for four days, closing more than 70 schools and pushing Mumbai air quality to the worst level recorded since monitoring began in June 2015. Detectability is therefore stated by the observing agency rather than inferred: earthobservatory.nasa.gov/images/87429.

29 detections over 5 days inside 2 km, against 7 in the preceding 3.5 months -- a clean onset with no prior baseline. The centre is the detection centroid, about 2 km north of the site's nominal coordinates; using the nominal point would have placed the fire at the edge of its own window.

**Brahmapuram waste plant fire (Kochi, Kerala)** — `ACCIDENTAL_FIRE` (accident), high confidence, 9.9925, 76.3650, 2 km, n=30, served **0.533** (`firms_brahmapuram_2023_joined.parquet`)

> Widely documented disaster with an official response record: garbage piles across roughly 40 acres ignited on 2023-03-02 and were declared doused eleven days later after 23 fire engines, 32 excavators, four helicopters and more than 200 firefighters were committed; smoke was still rising at twelve days. Kochi AQI exceeded 320 and the Kerala state health service opened a dedicated incident page.

Deliberately the hardest accidental-fire case in the set. Unlike Deonar this site was already smouldering: 37 detections inside 2 km in the four months before ignition, against 30 over 7 days during it. A site with a chronic baseline that then suffers a catastrophic accident is the exact failure mode the recurrence counter creates, and nothing else in the verified set tests it.

**ITC Deer Park terminal fire (Harris County, Texas, USA)** — `ACCIDENTAL_FIRE` (accident), high confidence, 29.7320, -95.0920, 1.5 km, n=11, served **0.182** (`firms_deerpark_2019_joined.parquet`)

> US Chemical Safety Board final investigation report (2023) into the Intercontinental Terminals Company tank-farm fire that ignited 2019-03-17, burned for three days, reignited on 2019-03-22 and breached a second containment wall. Over $150M in facility damage and a $6.6M natural-resource-damage settlement with the Texas Attorney General and US DOJ. A regulator's own findings, which is the class of source this project could not previously obtain.

Deliberately outside India. Every other verified event sits inside this project's Indian OSM extract, so `inside_industrial` has never once been structurally False for a real, operating facility -- the model has never been scored on a documented fire where the map layer is simply absent. The E3 transferability work measured that against rule labels; this measures it against a regulator's.

**Rumaila oil field flaring (Basra, Iraq)** — `PERSISTENT_BASELINE` (continuous heat), medium confidence, 30.1500, 47.3500, 30 km, n=200, served **0.065** (`firms_rumaila_2026_joined.parquet`)

> World Bank satellite flaring data records Rumaila flaring 3.39 billion cubic metres of gas in a year, around 9.5 Mt CO2e. Iraq publishes no official flaring record, which is precisely why the World Bank tracks it from orbit -- making this a case where the independent evidence for the class is itself remote sensing, by a different programme, at a different cadence.

Confidence is medium, and the reason is a measurement rather than caution. The field is roughly 80 km long, so this is a field label rather than a point one -- and even at 30 km it appears on only 30% of days, against Jamnagar's 51% at a 3 km radius and Jharia's 89%. Median FRP is 3.5 MW, close to the detection floor. The documented ground truth is continuous flaring; FIRMS sees it intermittently. That gap is a property of the sensor, not of the label, and it is the first non-Indian persistent source the classifier has ever been scored against.

**Punjab post-Kharif paddy residue burning** — `AGRICULTURAL_BURN` (seasonal biomass), high confidence, 30.5000, 75.8000, 25 km, not currently scorable

> Verified as a seasonal regional phenomenon rather than a single incident: large-scale paddy stubble burning across Punjab in the post-Kharif window is extensively documented and regulated. Confirmed independently by Sentinel-2 dNBR of +0.2775 (MODERATE_LOW burn severity) on 2023-11-05.

Region-and-season label, not a point incident, hence the wide radius. Both dNBR measurements cited above were produced by this project's own optical validation against live Copernicus data.

### 5c.3 The counter-register: incidents that produced no usable signal

A record of failures is evidence too, and it is the half that systems built to impress usually omit. These six documented industrial accidents produced **no usable FIRMS signature** -- in most cases no detection whatsoever, and in one case detections that cannot be told apart from the site's ordinary Tuesday. They are deliberately *not* verified events: there is nothing to label, so adding them would inflate the event count without contributing one scorable row. They are here because they answer a question the verified set structurally cannot. Every accuracy figure in this project is conditional on the fire being visible to a polar-orbiting radiometer at the moment it passes overhead, and **these bound what the system may claim.**

Each was selected for a *different* mechanism of failure, because a counter-register whose entries all failed the same way bounds nothing that the first entry had not already bounded. Every zero below was measured by pulling the window and counting, and every entry carries a control showing the retrieval worked -- a negative from an instrument that was not looking proves nothing.

| Incident | Date | Mechanism of non-detection |
| :--- | :--- | :--- |
| ONGC Uran gas processing plant fire | `2018-09-03` | Burned entirely between two overpasses |
| Bhilai Steel Plant gas pipeline blast | `2018-10-09` | **Detected, but indistinguishable** from the site's routine operation |
| Anaj Mandi factory fire | `2019-12-08` | Enclosed building, and pre-dawn between overpasses |
| LG Polymers styrene vapour release | `2020-05-07` | **No combustion at all** -- an unignited toxic vapour release |
| NLC India Neyveli Thermal Power Station II boiler explosion | `2020-07-01` | Combustion confined inside boilers and stacks; no radiating flame |
| Haldia Petrochemicals naphtha pipeline fire | `2026-06-30` | Burned entirely between two overpasses |

**ONGC Uran gas processing plant fire (Raigad, Maharashtra)** — `2018-09-03`, 18.8700, 72.9400

> Fire at the ONGC Uran gas processing complex, Raigad district, Maharashtra, on the morning of 2018-09-03. Five people were killed, among them three CISF personnel who responded to it, and the fire was reported brought under control within roughly two hours.

- **Why it was missed:** The same mechanism as Haldia, which is why it is worth recording: a second instance makes the overpass gap a recurring property of the observing system rather than one unlucky fire. Ignition was around 07:00 local and the fire was out by roughly 09:00. VIIRS passes the region near 02:30 and 13:30 local, so the entire event opened and closed inside a single gap between passes.
- **Evidence:** Zero detections within 10 km on 2018-09-03. The two nearest in the 43-day window sit 1.9 km from the plant, on 2018-08-18 and 2018-09-05, at 0.9 and 3.7 MW -- so the site is visible to the sensor on ordinary days and was simply not being looked at on this one.

**Bhilai Steel Plant gas pipeline blast (Durg, Chhattisgarh)** — `2018-10-09`, 21.1886, 81.3911

> A blast on a gas pipeline at the SAIL Bhilai Steel Plant, Durg district, Chhattisgarh, on 2018-10-09 killed at least 14 people during maintenance work.

- **Why it was missed:** This one WAS detected, and that is exactly why it belongs here. The site is a working blast-furnace complex that registers almost every day, so the blast did not have to be invisible to be unfindable -- it only had to be unremarkable, and it was. The plant produced FEWER detections on the day of the blast than on any surrounding day, and its peak fell below what the same plant reaches in routine operation. No threshold on radiative power, count or recurrence separates this event from the Tuesday before it. This is the chronic-baseline blind spot of 5b.2b appearing at an operating steel plant rather than a landfill, and no rate signal rescues it there either.
- **Evidence:** 484 detections within 4 km on 40 of 43 days. On 2018-10-09: 4 detections, peak FRP 10.8 MW. On the four preceding days: 14, 12, 14 and 15 detections, and on 2018-10-07 a ROUTINE peak of 28.5 MW -- 2.6x the blast day. Site median FRP is 1.9 MW, 90th percentile 5.6 MW.

**Anaj Mandi factory fire (Rani Jhansi Road, Delhi)** — `2019-12-08`, 28.6600, 77.2100

> Fire before dawn on 2019-12-08 in a multi-storey building housing bag and packaging manufacturing units at Anaj Mandi, off Rani Jhansi Road, Delhi. Forty-three people died, most of them workers asleep inside -- the deadliest fire in Delhi since the Uphaar cinema fire of 1997.

- **Why it was missed:** Two mechanisms at once. The fire was reported around 05:00 local and fought down within a few hours, so it fell in the same gap between the 02:30 and 13:30 passes that hid Haldia and Uran. It was also entirely enclosed: a fire burning through the floors of a sealed building presents almost no radiating surface to a sensor looking straight down. The deadliest fire in this register by a wide margin is also among the least visible.
- **Evidence:** Zero detections within 10 km on 2019-12-08. A 2.8 MW detection sits 0.6 km away on 2019-12-07, the afternoon before -- almost certainly an unrelated waste fire in dense urban Delhi, and quoted here precisely because it shows the sensor resolving small sources at this exact location on an ordinary day.

**LG Polymers styrene vapour release (Visakhapatnam, Andhra Pradesh)** — `2020-05-07`, 17.7564, 83.2101

> Styrene vapour escaped from a storage tank at the LG Polymers India plant at RR Venkatapuram, Gopalapatnam, Visakhapatnam in the early hours of 2020-05-07, killing at least 11 people, hospitalising hundreds and forcing the evacuation of surrounding villages. The National Green Tribunal took suo motu cognisance the following day and directed an interim deposit of Rs 50 crore; the Government of Andhra Pradesh High Power Committee published its investigation report in July 2020.

- **Why it was missed:** There was nothing thermal to detect. The release was an unignited vapour cloud, not a fire: it killed by inhalation. This is the hardest bound in the register and it is categorical rather than circumstantial -- a sensor that measures radiative power cannot see a toxic release at ambient temperature, however many people it kills or however often the satellite passes. No cadence, no resolution and no additional band closes this gap.
- **Evidence:** Zero detections within 4 km on 2020-05-07, and one in the whole 43-day window around it (3.4 MW, 2.5 km away, three weeks before). The surrounding 100 km box carried 216 detections over 36 days, so the sensors were observing the region normally throughout.

**NLC India Neyveli Thermal Power Station II boiler explosion (Tamil Nadu)** — `2020-07-01`, 11.5548, 79.4429

> A boiler exploded at NLC India's Neyveli Thermal Power Station II, Cuddalore district, Tamil Nadu, on 2020-07-01, killing six workers. It followed an explosion at the same station on 2020-05-07 in which eight people were injured.

- **Why it was missed:** A confined explosion inside a boiler house presents no sustained open flame for a radiometer to integrate. The wider finding here matters more than the incident: this lignite-fired station produced ZERO detections across the entire 43-day window at any radius out to 10 km. A station of this size is among the largest continuous combustion sources in the state, and it is thermally invisible to FIRMS, because its heat leaves through boilers and stacks rather than as radiating flame.
- **Evidence:** Zero detections within 10 km across 2020-06-10 to 2020-07-22; the nearest detection anywhere in the pull is 13.3 km away. The surrounding 100 km box carried 67 detections over 27 days, confirming the retrieval worked. This bounds the corpus itself: absence from the detection record is not absence of industrial combustion.

**Haldia Petrochemicals naphtha pipeline fire (West Bengal)** — `2026-06-30`, 22.0669, 88.1130

> Reported 2026-06-30: fire in a naphtha pipeline at the Haldia Petrochemicals facility, Purba Medinipur district, West Bengal, which spread to housing at Chiranjibpur and injured at least 20 people, five critically. Twelve fire tenders were deployed. Business Standard, business-standard.com/india-news/haldia-refinery-fire-naphtha-pipeline-west-bengal-purba-medinipur-126063000197_1.html

- **Why it was missed:** Ignition was reported between 04:00 and 04:30 local time and the fire was fought down with 12 tenders. VIIRS overpasses the region at roughly 01:30 and 13:30 local, so the event began after the night pass and was suppressed before the afternoon one.
- **Evidence:** Zero detections within 4 km on 2026-06-30. The nearest are 2026-06-28 and 2026-07-02, all sub-3 MW routine flaring at neighbouring plants (Hooghly Met Coke, IOCL Refinery), correctly classified PERSISTENT_BASELINE.

### 5c.4 Extending the register

Two routes, both loaded by `load_verified_events()`: append a `VerifiedEvent` to `VERIFIED_EVENTS` in `src/models/verified_labels.py`, or add a row to `data/reference/verified_events.csv`, which takes the same columns and needs no Python. Malformed CSV rows are logged and skipped rather than dropped silently, because a lost verified label is expensive.

**What to search for.** Public reporting cannot tell you whether a fire was *detectable*, and searching recent industrial fires kept failing for one reason: a modern plant fire is extinguished in hours and falls between overpasses. Search instead for fires with a **documented multi-day duration** -- those necessarily straddled an overpass -- and remember the MODIS archive reaches back to 2000 where VIIRS begins in 2012. Then pull the window and count what landed *before* writing the label -- 5c.3 is what happens when you do not.

Regenerate this section with `python scripts/generate_verified_register.py` after any addition or retrain.

<!-- END GENERATED: verified-event register -->

---

## 6. Verification, Testing & Robustness Suite

The test suite consists of **376 automated pytest unit and integration tests** located in `tests/`:

```bash
# Run complete test suite
.venv\Scripts\python.exe -m pytest -v
```

### Test Breakdown:
- **`tests/test_api.py` (33 tests)**:
  - Verifies dashboard UI delivery, health endpoints, incident querying, status transitions, live classification, analytics, weather/dispersion, audit logging, and graceful degradation of optical validation.
- **`tests/test_firms_client.py` (4 tests)**:
  - Verifies NASA FIRMS NRT CSV parsing, empty GeoDataFrame generation, invalid API key handling, and coordinate parsing.
- **`tests/test_models.py` (4 tests)**:
  - Verifies XGBoost classifier loading, multi-class probability outputs, pipeline transforms, and TreeSHAP calculations.
- **`tests/test_simulation.py` (4 tests)**:
  - Tests the What-If simulation engine with variable FRP, brightness temperatures, facility overrides, and plume coordinate generation.
- **`tests/test_sitrep.py` (15 tests)**:
  - Validates WGS84 to MGRS coordinate translation, pure-Python UTM projections, Gaussian plume dispersion geometry, indicative chemical emission estimates, and -- critically -- that synthetic weather and unsourced emission factors declare themselves rather than presenting as measurements.
- **`tests/test_spatial_join.py` (4 tests)**:
  - Tests exact polygon containment, scan-angle-adaptive parallax buffer recovery, facility type classification, and nearest-boundary distance computations.
- **`tests/test_state_machine.py` (15 tests)**:
  - Validates Uber H3 resolution 9 binning, recurrence counters, persistent baseline suppression, and escalation triggers.
- **`tests/test_sentinel2.py` (36 tests)**:
  - Covers the Sentinel-2 dNBR client entirely offline: USGS severity breakpoints, latitude-corrected bounding boxes, dNBR arithmetic from stubbed scene statistics, cache round-trips, request rationing, and — critically — that a missing credential, a cloud-obscured scene, or a future event date all report `dnbr=None` rather than a fabricated `0.0`.
- **`tests/test_hex_aggregator.py` (18 tests)**:
  - Guards the corpus-scale map layer: that the H3 roll-up lands a fine cell inside its own parent, that the payload stays bounded and declares truncation, that a single P0 outranks a crowd of suppressed agricultural detections, and that a cell reports both its highest priority (for sorting) and its most common one (for colour) -- because colouring by the highest turns every 22 km cell in India red.
- **`tests/test_slstr.py` (14 tests)**:
  - Covers the Sentinel-3 SLSTR client offline: that an unconfigured client makes no network call, that the background annulus really is an annulus (opposite winding, hole larger than the hot area) so the fire is not measured against itself, that the background is pinned to the *same* acquisition as the source, and that a failed retrieval reports `NO_SOLUTION` rather than returning a number.
- **`tests/test_evaluation_integrity.py` (50 tests)**:
  - Guards the honesty of the evaluation itself: asserts no coordinate features can reach the model (spatial leakage), that the weak-labelling rule does not classify an established refinery flare as an accident, that `LABEL_RULE_FEATURES` stays in sync with the rule that uses it, and that the persisted metrics file always carries its label provenance, caveats, real-vs-synthetic split, spatial-block CV, and circularity audit.

- **`tests/test_replay_mode.py` (13 tests)**:
  - Guards the feature least likely to be exercised before it matters: that replay stops the refresh loop rather than retrying a dead network while an operator is talking, that leaving replay lets ingestion resume rather than stranding the badge, that the preflight reports `INCOMPLETE` when an asset is genuinely missing -- a preflight that always says READY is worse than none -- and that the basemap caveat, the one gap with no local fallback, is never quietly dropped.
- **`tests/test_forest_cover.py` (17 tests)**:
  - Guards the forest-fire segregation the mandate asks for: that a missing land-cover layer disables the determination rather than breaking classification, that latitude and longitude are not transposed, that only open-ground burns are refined -- a brick kiln in a forest clearing is still a brick kiln, and an accidental fire relabelled `FOREST_FIRE` would be downgraded out of dispatch -- and that the determination runs *behind* the guards which withhold unearned claims, so Indian land cover is never asserted in Texas.
- **`tests/test_verified_labels.py` (28 tests)**:
  - Guards the only measurement in the project that is independent of its own labelling rule: that a `VerifiedEvent` cannot be constructed without a citation, that overlapping windows resolve to the more specific claim, that the harness refuses to report accuracy when nothing matched rather than returning a flattering zero-support figure, and that the generated register in section 5c still contains every event, every source citation and every undetected incident -- so a stale evidence table fails the build instead of reading as authoritative.
- **`tests/test_compliance_register.py` (9 tests)**:
  - Guards the flaring register against the three ways it could libel an operator: misattributing a detection to the wrong facility, listing a solar farm as an emitter, and printing a carbon figure derived from an emission factor nobody sourced.
- **`tests/test_firms_archive.py` (23 tests)**:
  - Covers archive retrieval offline: that a span longer than the API's 5-day ceiling is paged rather than silently truncated, that NRT and SP sources are routed by date and deduplicated where they overlap, and that every configured sensor is actually queried.
- **`tests/test_dispatcher.py` (27 tests)**:
  - Covers alert dispatch safety: that dry-run is the default, that a failed send reports failure rather than success, and that suppression rules cannot be bypassed by a malformed payload.
- **`tests/test_postgis_layer.py` (22 tests)**:
  - Guards the datastore migration (6b). Chiefly: that a configured-but-unreachable PostGIS **raises instead of falling back**, because the geometry column exists only in PostGIS mode and degrading silently would leave every proximity query returning nothing from a schema that reports itself healthy. Also that `docker-compose.yml` and the application agree on credentials *and defaults* -- the test parses the compose file and compares -- and that the audit log's duplicate check survives a SQLite string meeting a PostgreSQL datetime, which is the bug that turned 31 evidence rows into 61.

### Operating System Resilience:
- **Responsive collapse, and the specificity bug that disabled it**: the shell's breakpoints at 1280px and 900px existed and had **never once applied**. `.shell[data-rail="collapsed"][data-dossier="closed"]` carries specificity 0,3,0; a media query adds none, so a bare `.shell` inside one always lost to attributes that are set on every load. Below roughly 800px the fixed 360px rail and 400px dossier left the map cell no room, its container reached zero width, and Leaflet threw `Invalid LatLng object: (NaN, NaN)` once per animation frame -- **36 uncaught errors per page load, with a blank map behind them**. Measured directly: a 1440px viewport produced 0, an 820px viewport produced 36. The breakpoint variants are now repeated inside each query so they carry equal specificity, and `.stage` has a 320px floor so the map can never be handed a zero-width container regardless. Verified at 820px and 1440px: **zero console errors at both**.
- **Windows Smart App Control Compliance**: All native binary dependencies (`h3_cy`, `scipy_qhull`, `pyproj`) include robust pure-Python mathematical fallbacks and native C++ XGBoost `pred_contribs=True` execution. The application runs seamlessly under strict Code Integrity policies.

---

## 6b. The Datastore: PostGIS, and What Stayed Out of It

**PostGIS is the primary datastore.** Incident geometry is a real
`geometry(Point, 4326)` column with a GiST index, and proximity queries are
served by `ST_DWithin` over `geography` rather than by scanning the table and
filtering in Python.

### 6b.1 Why SQLite was replaced

Three reasons, in order of how much they matter:

1. **Two concurrent writers.** `src/pipeline/auto_refresh.py` runs a daemon
   thread that opens its own session and writes incidents while the API is
   serving analyst writes -- dispatch, status changes, audit rows. SQLite
   serialises every write behind a single file lock and surfaces the
   contention as `database is locked`: intermittently, under exactly the load
   a demonstration does not reproduce. During the migration this path was
   exercised live -- the refresh thread added 653 incidents while the API was
   answering requests, with no contention.
2. **The audit log is regulatory evidence** for the CPCB register (5c).
   Evidence that can be lost to a half-completed file copy is not evidence.
   PostgreSQL has WAL archiving and point-in-time recovery; SQLite has a file.
3. **One API worker only.** A file-backed database cannot safely serve
   `uvicorn --workers N`.

### 6b.2 Why a latitude/longitude index is not a spatial index

A degree of longitude is 111 km at Kanyakumari and 96 km at Srinagar. A
bounding box computed in degrees is therefore a different real distance in
Kerala than in Kashmir, and "incidents within 5 km" means two different things
depending on where it is asked. `ST_DWithin` over `geography` measures metres
on the spheroid.

**Two spatial indexes are required, and discovering that took an EXPLAIN.**
GeoAlchemy2 creates `idx_incidents_geom`, a GiST index on the geometry column,
which serves geometry predicates -- map-extent queries, `ST_Intersects`
against a facility polygon. It does **not** serve `ST_DWithin(geom::geography,
...)`. Measured on 200,004 synthetic rows:

| Indexes present | Plan | Time |
| :--- | :--- | ---: |
| `idx_incidents_geom` only | Parallel Seq Scan | 268 ms |
| **+ `idx_incidents_geog`** on `(geom::geography)` | **Bitmap Index Scan** | **4.6 ms** |

The claim "GiST-indexed proximity search" was false without the second index,
and the assumption that one spatial index covers both cases is what the query
plan corrected. Both are declared in the model.

### 6b.3 What deliberately did not move into PostGIS

- **Bulk point-in-polygon enrichment stays in GeoPandas.** Measured on this
  corpus: 2,044,295 detections against 28,587 polygons joins in **1.22 s** with
  an in-memory R-tree. A per-row SQL join would be slower. PostGIS is used for
  *query-time* geometry, not batch enrichment.
- **The analytical corpus stays in Parquet.** 2.04M rows read columnar in
  **0.05 s**. Postgres is the wrong store for a scan-everything workload.

Three engines, each doing what it is good at. This is the answer to "your
reference documents specify PostGIS": the parts of the workload PostGIS is
right for now use it, and the parts it would slow down are named, measured,
and left alone.

### 6b.4 The mode is decided by configuration, never by connectivity

SQLite remains reachable for two cases: `pytest`, which must not require a
database server, and Historical Replay Mode on a disconnected machine. Both
are opt-in by setting `DATABASE_URL` to an explicit `sqlite://` URL.

An earlier draft fell back to SQLite automatically when PostGIS was
unreachable. That was wrong, and the reason is worth recording: the geometry
column is declared only in PostGIS mode, so a transient connection failure at
import time would have silently built a **different schema** -- an application
with no `incidents.geom`, answering every proximity query with nothing, while
`/api/v1/health` reported it healthy. A configured-but-unreachable PostGIS now
raises at startup and names the three ways to fix it. Running offline is a
decision an operator makes, not an accident a dropped connection makes for
them. `tests/test_postgis_layer.py` enforces this.

### 6b.5 Four defects the migration produced, and how each was found

None of these were found by reading the code. Each came from running it.

| # | Defect | Found by | Consequence had it shipped |
| :--- | :--- | :--- | :--- |
| 1 | Audit dedup key compared a SQLite **string** against a PostgreSQL **datetime**, so it could never match | Re-running the migration to test idempotency | 31 evidence rows became **61**. Duplicated evidence is corrupt evidence |
| 2 | `audit_logs.incident_id` copied verbatim while PostgreSQL assigned fresh `SERIAL` ids | Asking *why* the ids happened to line up | Audit rows silently pointing at the **wrong incidents** on any re-run |
| 3 | `index=True` on three primary keys built a duplicate B-tree per table | `pg_indexes_size` exceeding `pg_relation_size` | Every insert maintaining a redundant index; 920 kB of indexes on 896 kB of data |
| 4 | No foreign key between the audit trail and the incidents it describes | Constraint audit | Orphan evidence rows with nothing preventing them |

All four are fixed. The migration now **refuses to report success** if any
duplicate audit rows exist or if any audit row references an incident with a
different `detection_id`, because for an evidence trail a silently wrong link
is worse than a crash. The foreign key uses `ON DELETE SET NULL` rather than
`CASCADE`, deliberately: an audit entry whose incident was removed is still a
record that the action happened, and deleting it to satisfy a constraint would
be destroying evidence.

**Known and not yet fixed.** All four timestamp columns are `timestamp without
time zone`. Every writer stamps UTC so the stored data is correct, but the
database does not enforce it -- `timestamp_utc` is a column *name* carrying a
promise the *type* does not keep. Converting to `timestamptz` is a four-column
schema migration, deferred rather than done, and recorded here so it is not
mistaken for an oversight.

### 6b.5b Every Detection Is Classified

The incident seeding path was gated on `inside_industrial`: only detections
inside a mapped industrial polygon reached the model. It read as a sensible
cost optimisation and it had a consequence nobody had looked at.

**A crop burn is by definition not inside an industrial polygon.** So
`AGRICULTURAL_BURN` was filtered out before the classifier ever saw it, and
`FOREST_FIRE` was unreachable for the same reason, being an upgrade applied to
`AGRICULTURAL_BURN`. The model holds 207,521 agricultural training examples and
had no live path to ever predict one. **Two of five classes were structurally
absent from the running system**, which is visible the moment the class
distribution is read rather than assumed:

| Class | Gated | Ungated |
| :--- | ---: | ---: |
| `NOT_ASSESSED` | 2,189 (66.8%) | **0** |
| `TRANSIENT_HOTSPOT` | 425 | 472 (45.6%) |
| `PERSISTENT_BASELINE` | 651 | 321 (31.0%) |
| **`AGRICULTURAL_BURN`** | **0** | **227 (22.0%)** |
| **`FOREST_FIRE`** | **0** | **8 (0.8%)** |
| `ACCIDENTAL_FIRE` | 10 | 6 (0.6%) |

What replaces the gate is a cheaper split rather than a cheaper filter:
**predict everything, and spend TreeSHAP only on what becomes an alert.**
Attribution costs ~30 ms per detection -- measured -- and is worth it for
something a human will open, not for background thermal activity nobody will
click. `ExplainabilityEngine.predict_detection()` is the cheap path and applies
the same prior correction as the full one.

The serving guards matter far more without the gate, and are now applied at
seed time rather than only at `/classify`. Classifying everything means solar
farms, forest and cropland all reach a model that is coordinate-free by design
and trained on an entirely Indian corpus -- exactly the cases it cannot judge
for itself. The eight `FOREST_FIRE` rows above are guard upgrades; before this,
the guards never ran in the seeding path at all.

**A second fabrication was found in the same block.** The exception fallback
set `PERSISTENT_BASELINE` or `ACCIDENTAL_FIRE` by state-machine guess with a
hard-coded `conf = 0.95` -- the same invention `CONTROLLED_PROCESS` carried,
in a rarer branch where it would have been harder to notice. A failed
classification now records `NOT_ASSESSED` with a null confidence and names the
exception.

#### Confidence must not round up into certainty

88.6% of classified rows reported **exactly 100%**. Two separate things, and
only one was a defect.

The saturation is real: a typical raw probability here is **0.9996**, because
the model reproduces a *deterministic* labelling rule, so the decision boundary
is sharp and softmax saturates. Spatial-block CV macro F1 of 0.9983 against
those labels is the same fact stated differently.

The rounding was the defect. `round(99.96222, 1)` is `100.0`, so a model
expressing 0.9996 was displayed as certain. `_confidence_percent()` now floors
anything short of an exact 1.0 at **99.9%**. The change is cosmetically tiny
and the distinction is the entire point: 99.9% is a very confident model, 100%
is a model that cannot be wrong. After the fix, **zero of 1,034 rows claim
certainty**, and the range is 0.967 to 0.999.

### 6b.6 Running it

```bash
docker compose up -d db                              # PostGIS 16-3.4, healthchecked
python scripts/migrate_sqlite_to_postgis.py --dry-run
python scripts/migrate_sqlite_to_postgis.py
```

`docker-compose.yml` and `app/database.py` read the same `POSTGRES_*` variable
names **and the same defaults**, so a clean checkout needs no `.env` at all. An
earlier version defaulted to `postgres` with an empty password while compose
defaulted to `sih`/`sih_local_dev`; the mismatch is now pinned by a test that
parses `docker-compose.yml` and compares.

The migration is idempotent on `detection_id`, leaves the SQLite file intact as
the offline-replay database, and verifies after every run that the incident
count equals the count of rows carrying geometry -- a NULL geom is invisible to
every `ST_DWithin` query, so a partial population would be silent data loss
rather than an error.

**Offline escape hatch**, for a venue with no Docker:

```bash
DATABASE_URL=sqlite:///data/fire_db.sqlite python -m uvicorn app.main:app --port 8000
```

Reports `"database_mode": "sqlite_offline"` at `/api/v1/health`, and
`/api/v1/incidents/near` answers with `engine: sqlite_scan_haversine` instead
of `postgis_st_dwithin` -- correct results by table scan, and the difference is
stated rather than hidden.

---

## 7. Operational Quickstart Guide

### Starting the Datastore
```bash
docker compose up -d db
```
PostGIS must be running before the server starts. It does not fall back to
SQLite on its own -- see 6b.4 for why that is deliberate.

### Starting the C2 Tactical Server
```powershell
# Activate virtual environment and launch Uvicorn
d:\industrial-fire-classifier\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Accessing the System
- **Tactical C2 Interface**: Open `http://127.0.0.1:8000/` in any modern web browser.
- **Interactive REST API Docs (Swagger UI)**: `http://127.0.0.1:8000/docs`
- **ReDoc API Specifications**: `http://127.0.0.1:8000/redoc`

### Key Workflow Actions:
1. **Monitor Real-Time Detections**: Review active incidents in the left-hand telemetry feed. Filter by `P0 EMERGENCY` or `P1 ALERT`.
2. **Inspect Factor Attributions**: Click on any incident card to view its live risk score, thermal intensity, and SHAP driving factor breakdown.
3. **Generate a Tactical SitRep**: Click **"GENERATE SITREP"** on any incident to launch the Military Grid Reference System (MGRS) briefing with chemical release rates and atmospheric dispersion zones. Print or save to PDF via `Ctrl + P`.
4. **Run a "What-If" Disaster Simulation**: Open the **"WHAT-IF SANDBOX"** drawer. Drag the target reticle across the map, adjust FRP up to 500 MW, rotate the wind dial, and watch the dynamic downwind plume rotate and scale in real time.
