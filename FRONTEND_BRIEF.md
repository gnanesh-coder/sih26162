# Frontend build brief — SIH26162 Sentinel Thermal C2

Paste everything below the line into Claude. It is written to be self-contained:
it carries the API contract, the vocabulary, the non-negotiable constraints, and
the specific defects this project has already found and fixed, so a new build
does not reintroduce them.

Run it from the repository root so Claude can read `app/main.py` for anything
the brief does not cover.

---

Build the operator console for **SIH26162**, an NTRO problem statement on
AI-based detection and classification of industrial fires from NASA FIRMS,
OpenStreetMap and Sentinel-2 data.

You are replacing `app/templates/index.html`, a single Jinja2 template served by
FastAPI at `GET /`. Keep it a single self-contained file. Do not introduce a
build step, a bundler, or a JS framework that needs compiling — this is
submitted as a repository someone else runs with one command.

## Scope — read this before anything else

**Rebuild the look. Do not rebuild the behaviour.**

The file is roughly 881 lines of markup, 308 of CSS and 1,580 of controller
JavaScript. You are replacing the first two. The controller — 60-odd functions
covering map rendering, incident selection, the evidence dossier, analytics,
triage, audit, the what-if sandbox, dispatch and the SitRep modals — is working,
verified and carries fixes that took a day to find. Port it across essentially
intact.

Adapt a controller function only where the DOM it touches has genuinely changed
(panel show/hide, list rendering, class toggles). When you do, say which ones
and why.

Keep every element `id` the controller reads. Before you finish, check that no
`getElementById` in the script refers to an id the new markup does not define:

```
grep -o "getElementById(['\"\`][a-zA-Z0-9_-]*" app/templates/index.html | sort -u
```

Preserve these behaviours exactly — each replaced a defect that shipped:

- freshness computed from the newest `timestamp_utc`, never a literal
- per-incident `shap_factors` rendered from the API, empty state as a sentence
- recurrence read from `/api/v1/incident/{id}` -> `recurrence`, not from FRP
- `sourceTitle()` naming, including OSM placeholder handling
- status bar separating on-screen data from the training corpus
- the 5-minute poll that preserves selection, filter and search

If you find yourself deleting one of those, stop and ask.

## What the system does, in one paragraph

It ingests satellite thermal detections across India, decides which ones are
real industrial accidents rather than routine flaring, crop burning or sensor
artifacts, and dispatches only the ones worth a responder's attention. The
entire thesis is **alert-fatigue suppression**: of ~2 million detections a year
it forwards roughly 530. The console exists so an operator can see what was
suppressed and why, and overrule it.

## The API you are building against

All endpoints are live and documented at `/docs`. Verified response shapes:

```
GET  /api/v1/alerts/active?limit=2000&priority=&industrial_only=&status=
     -> { status, total, incidents: [ {
            id, detection_id, latitude, longitude, frp, bright_ti4, bright_ti5,
            timestamp_utc, satellite, inside_industrial, facility_name,
            facility_type, dist_to_industrial_km, h3_index, predicted_class,
            confidence, alert_priority, alert_state, shap_rationale,
            status, operator_notes, created_at } ] }

GET  /api/v1/incident/{id}
     -> the above, plus:
        shap_factors: [ { feature, label, shap_value, abs_shap,
                          feature_value, direction } ]   # MAY BE EMPTY
        shap_basis:   "TREESHAP_STORED" | "NO_PER_FEATURE_ATTRIBUTION"
        recurrence:   { status, h3_index, detections_30d, mean_frp_mw,
                        sigma_frp_mw, z_frp, persistence_threshold,
                        is_established, basis }

GET  /api/v1/map/hexes?resolution=3..8&bbox=w,s,e,n&state=&min_detections=&limit=
     -> { status, resolution, approx_cell_edge_km, detections_aggregated,
          detections_in_corpus, cells_returned, truncated, compression_ratio,
          ordering, cells: [ { h3, detections, days_observed, frp_max_mw,
            frp_median_mw, frp_total_mw, p0, p1, suppressed,
            top_priority, dominant_priority, dominant_state } ] }

GET  /api/v1/stats
     -> { total_detections, p0_emergencies, p1_alerts, p2_advisories,
          suppressed, fatigue_reduction_pct, industrial_incidents,
          status_breakdown }

GET  /api/v1/analytics/charts
     -> { frp_distribution{labels,counts}, facility_types{...},
          alert_priorities{...}, alert_states{...}, top_hotspots[] }

GET  /api/v1/compliance/register?min_detections=&limit=
     -> { status, purpose, observation_span_days, routine_detections,
          sites_listed, sites_continuous, assumptions{}, exclusions,
          caveat, entries[] }

GET  /api/v1/incident/{id}/optical-validation   # Sentinel-2 dNBR burn scar
GET  /api/v1/incident/{id}/thermal-probe        # Sentinel-3 SLSTR channels
GET  /api/v1/incident/{id}/sitrep               # printable briefing (HTML)
GET  /api/v1/weather/context?lat=&lon=&at=      # carries weather_is_measured
GET  /api/v1/audit/logs                         # analyst override trail
GET  /api/v1/system/refresh                     # ingest state
GET  /api/v1/health

POST /api/v1/incident/{id}/action    { action_type, operator_notes }
POST /api/v1/incident/{id}/status    { status, operator_notes }
POST /api/v1/incident/{id}/dispatch  # routes to SMS/email/webhook by priority
POST /api/v1/classify                # on-demand inference for arbitrary input
POST /api/v1/simulation/sitrep       # what-if scenario briefing
POST /api/v1/system/refresh          # forces a FIRMS ingest
POST /api/v1/sync                    # re-imports the on-disk corpus ONLY
```

Do not invent endpoints. If you need data that is not above, read
`app/main.py` and use what is there, or add the endpoint deliberately and say
that you did.

## Vocabulary — use these exact strings

```
predicted_class   PERSISTENT_BASELINE  ACCIDENTAL_FIRE  AGRICULTURAL_BURN
                  TRANSIENT_HOTSPOT  CONTROLLED_PROCESS
alert_priority    P0_EMERGENCY  P1_ALERT  P2_ADVISORY  NON_ALERT  SUPPRESSED
alert_state       PERSISTENT_BASELINE  ESCALATED_FLAREUP  MONITORING
                  TRANSIENT_SUSPICION  CONFIRMED_DISPATCH  ACCIDENTAL_FIRE
facility_type     petrochemical_refinery  steel_metallurgy  power_thermal
                  mining  brick_kiln  general_industrial  non_industrial
                  renewable_non_thermal
status            OPEN  ACKNOWLEDGED  DISPATCHED  RESOLVED  MUTED_ROUTINE
```

## Non-negotiable constraints

These are not style preferences. Each one exists because the opposite shipped
and had to be removed.

1. **Never render a number that is not computed from the response.** A previous
   build drew three TreeSHAP attribution bars — `+0.72`, `+1.14`, `-0.34` —
   hardcoded in the markup, identical for every incident, under a heading that
   said "TreeSHAP attribution". Another printed `frp / 8.0` labelled
   "x baseline", which knows nothing about the source's own history, so a 40 MW
   refinery flare and a 40 MW depot fire both scored "5.0x".

2. **Render absence as absence.** `shap_factors` is empty for detections
   classified by the recurrence state machine rather than the model — that is
   most of them. Show a sentence saying so. Never fall back to a placeholder
   bar, a zero, or a dash that looks like a measurement.

3. **Freshness is computed, never asserted.** A previous build printed a
   hardcoded `LIVE` while the newest detection on screen was five days old.
   Derive the badge from the newest `timestamp_utc` you hold. FIRMS NRT
   publishes ~3 h behind the overpass, so: under 6 h `LIVE`, 6–48 h
   `RECENT · N h old`, beyond that `STALE · N d old`, each with a distinct
   colour.

4. **Distinguish what is on screen from what the model was trained on.** The
   status bar once read "corpus 2,044,295 detections" under a map drawing 1,514.
   Both numbers were true and the juxtaposition was not.

5. **Name a source by what distinguishes it.** Unmapped detections all rendered
   as the same string — "Rural / Non-Industrial Node" — so the first thing a
   viewer saw was four identical rows from a system whose entire claim is
   telling thermal sources apart. An unmapped source has no name but it has a
   position; use coordinates as the headline. Treat OSM placeholders
   (`Unnamed Industrial Site` and similar) as unnamed: thousands of distinct
   polygons share that one string.

6. **Weather and emissions carry their provenance.** `/api/v1/weather/context`
   returns `weather_is_measured` and `weather_source`. When it is synthetic, say
   so on the surface that displays it, not in a tooltip.

7. **Colour is data.** See below.

## Design direction

**One rule runs through the palette: warm, saturated colour means combustion,
and nothing in the chrome is allowed to compete with it.**

An earlier build used amber for borders, brand, icons and active tabs — and
amber is also the colour of a P2 advisory, so the interface was shouting at
exactly the frequency of its own alerts. In a system whose entire pitch is
suppressing alert fatigue that is an information-design defect, not a taste
question. Replacing amber with a different accent only moves the problem;
removing accent colour from the chrome is what fixes it.

So: an achromatic instrument surface, with saturation reserved for five values
— `P0_EMERGENCY`, `P1_ALERT`, `P2_ADVISORY`, routine/suppressed, and the H3
archive layer — plus the map itself. Sensor imagery is fundamentally grayscale;
false colour is applied only where it means something. Build the interface the
same way.

Beyond that rule, make your own choices. Do not copy the current look. Pick a
palette and a typeface pairing that suit a thermal remote-sensing operations
console, avoid the generic dark-dashboard-with-one-neon-accent default, and
avoid Inter and Space Grotesk.

This is a **UI, not a document**: it is scanned and operated. Surface the
summary before the detail, encode state in form as well as number (a severity
stripe, a chip, a rail), and make what is interactive look interactive.

## Layout — a grid shell, not floating panels

The build before last positioned every panel `absolute` over a full-bleed map.
At real viewports they collided: the header ran into the map toolbar and clipped
a tab, the detail panel overflowed the right edge part-way through a
40-character sensor identifier, and the timeline sat on top of the attribution
bars. Use CSS Grid with named areas so overlap is impossible by construction,
and so collapsing a panel **reflows** the map rather than uncovering it:

```
masthead  masthead  masthead      ~52px
rail      stage     detail        1fr
status    status    status        ~26px
~340px    1fr       ~384px
```

Collapse the detail column below ~1280px and the rail below ~900px. Long
identifiers wrap or scroll inside their own cell; the body never scrolls
sideways.

## Technical traps, already paid for

- **Load `h3-js` BEFORE `deck.gl`.** deck treats it as an external peer and
  resolves it from the global at load time. In the wrong order `H3HexagonLayer`
  constructs, accepts data, then throws `getResolution is not a function` on
  every update while the map silently shows nothing.
- **Give the Leaflet container an explicit `z-index`.** Leaflet's panes run at
  400–700 and deck.gl's canvas joins them. At `z-index: auto` they compete with
  your own controls, and enabling the archive layer paints the deck canvas over
  the map toolbar — controls still present, still clickable, completely
  invisible.
- **Put a `ResizeObserver` on the map's grid cell.** Leaflet caches container
  size, so a cell that changes width when a panel opens leaves the map rendering
  a stale viewport. A hand-maintained list of `setTimeout(invalidateSize)` calls
  at each toggle site drifts out of sync with the CSS.
- **Poll for new detections, but preserve operator state.** The server ingests
  FIRMS every 3 hours. Re-running the first-load path on a poll yanks an open
  investigation away and flies the map elsewhere. Preserve selection, filter and
  search.
- Pinned CDN versions that work: `leaflet@1.9.4`, `h3-js@4.1.0`,
  `deck.gl@9.0.38`, `deck.gl-leaflet@1.3.1`, `chart.js`, Tailwind Play CDN.

## Screens to cover

1. **Surveillance** — map, incident rail, evidence panel. The primary screen.
2. **Analytics** — the four `/analytics/charts` distributions and top hotspots.
3. **Triage** — the P0/P1 queue with confirm and dispatch actions.
4. **Audit** — the analyst override trail from `/audit/logs`.
5. **What-if sandbox** — drag a point on the map, set FRP / brightness / wind,
   get live inference from `POST /api/v1/classify`.
6. **Compliance register** — `/compliance/register`, currently unsurfaced in the
   UI. Routine flaring logged for CPCB auditing; it reports the detections the
   pipeline *suppressed*, and deliberately does not compute gas volume or CO2.

The **archive layer** (`/api/v1/map/hexes`) deserves a first-class control, not
a checkbox inside a popover: it is the only view that draws all 2,044,295
detections at once, aggregated server-side into a few thousand H3 cells.

## How to work

Read `PROJECT_DOCUMENTATION.md` §4.7 for what the current build does and why.
Write the file, look at it once in a browser, fix what that shows, and stop.
Then tell me what you changed and what you deliberately left out.
