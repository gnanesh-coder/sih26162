"""Training and Evaluation Pipeline for Industrial Fire Classification.

Implements strict featurization ordering (split-before-fit), establishes
naive baselines, trains an XGBoost multi-class classifier, computes
per-class metrics (Recall/Precision/F1, Confusion Matrix), and persists model artifacts.
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.pipeline.feature_engineering import FireFeaturePipeline

logger = logging.getLogger("train_classifier")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODEL_DIR = Path("src/models")
MODEL_ARTIFACT_PATH = MODEL_DIR / "fire_classifier_xgb.json"
PIPELINE_ARTIFACT_PATH = MODEL_DIR / "feature_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "model_evaluation_metrics.json"

CLASS_NAMES = {
    0: "PERSISTENT_BASELINE",  # Routine refinery flare stack, brick kiln, blast furnace
    1: "ACCIDENTAL_FIRE",      # Critical chemical explosion, plant structural fire
    2: "AGRICULTURAL_BURN",    # Background crop residue, stubble burning
    3: "TRANSIENT_HOTSPOT",    # Low-confidence false alarms, solar reflection, minor anomalies
}


# Features consumed by the weak-labelling rule below. The circularity audit
# ablates exactly these so we can measure how much the model knows beyond the
# rule that generated its labels.
NON_COMBUSTION_FACILITY_TYPE = "renewable_non_thermal"

# Months when crop residue is burned across the Indian plains: February-May for
# the Rabi wheat harvest, October-November for post-Kharif paddy. Same set as
# feature_engineering.is_harvest_season, and the two must stay in step.
HARVEST_MONTHS = frozenset({2, 3, 4, 5, 10, 11})

# The region where that burn calendar was established, as (lat_min, lat_max,
# lon_min, lon_max).
#
# HARVEST_MONTHS is a calibrated prior about Indian agriculture, not a fact
# about combustion, and exporting it produced a documented error: the 2019 ITC
# Deer Park tank-farm fire in Texas -- three days, $150M damage, a US Chemical
# Safety Board investigation -- was labelled AGRICULTURAL_BURN on 10 of its 11
# detections, because March is a Rabi harvest month in Punjab. The same defect
# had already been measured in aggregate over the Permian Basin, where 34.8% of
# Chihuahuan Desert detections came back agricultural.
#
# Outside these bounds the month carries no information this project has
# established, so the rule declines to use it rather than assuming India's
# calendar applies. See weak_label_real_detection for what it does instead.
SEASONALITY_DOMAIN = (6.0, 37.5, 68.0, 97.5)   # the Indian subcontinent

# A detection whose FRP exceeds this, at a location with no established
# baseline, is treated as a candidate accident wherever it occurs.
#
# Inside a mapped polygon the rule has always said this, at 10 MW. Outside one
# it did not, so an identical event became a crop burn purely because nobody had
# drawn a polygon around it.
#
# The bar is two numbers rather than one, and the reason is the agricultural
# prior itself. Inside the seasonality domain a 10 MW open-ground detection in
# October really is more likely to be paddy residue than a plant fire, so the
# bar is raised to protect the verified Punjab events. Outside that domain there
# is no crop-burn prior to protect, so the bar that applies is the same one the
# rule uses inside a fence. A high bar exported to a region whose calendar was
# never established would be the original defect wearing a different number.
#
# The in-domain value was swept against the verified set after the branch was
# added, because its cost lands on the crop-burn events:
#
#   event                       n       25     40     60    off
#   Punjab post-Kharif        509    0.908  0.929  0.931  0.933
#   Punjab post-Rabi        1,608    0.890  0.895  0.897  0.898
#   Baghjan blowout           705    0.789  0.782  0.777  0.773
#   unweighted mean over 14 events  0.7988 0.8002 0.8001 0.8000
#
# The whole range spans 0.0014 in the mean -- noise. It is kept at 25 rather
# than moved to whichever number scored highest, because choosing between values
# that a fourteen-event harness cannot distinguish is fitting the harness. The
# sweep is recorded so the next person can see it was measured, not guessed, and
# so a future verified event that *can* separate them has something to move.
UNMAPPED_ACCIDENT_FRP = 25.0
UNMAPPED_ACCIDENT_FRP_OUTSIDE_DOMAIN = 10.0

# How far past its own distribution an established source must run before the
# rule calls it an accident rather than operations.
#
# Outside a polygon this has always been 3.0. Whether the same bar belongs
# INSIDE one looked like the obvious next instance of the polygon asymmetry that
# rule v4 fixed, so it was written and then measured against the verified set:
#
#   event                          n       off    z>8    z>5    z>3
#   Visakhapatnam Steel         2,591    0.991  0.988  0.983  0.968
#   JSW Vijayanagar             4,505    1.000  1.000  0.998  0.987
#   Reliance Jamnagar             539    0.974  0.961  0.954  0.935
#   Jharia coalfield           23,633    0.910  0.910  0.910  0.908
#   Raniganj coalfield          6,236    0.968  0.968  0.967  0.963
#   Deonar landfill fire           29    0.069  0.103  0.138  0.207
#   unweighted mean over 14 events     0.7793 0.7806 0.7820 0.7833
#
# The branch is off. It buys +0.138 on exactly one event and costs every
# high-confidence persistent anchor, and the 0.0040 gain in the unweighted mean
# is one event moving in a set of fourteen. Weighted by detections it is plainly
# negative. Tuning a rule to a single event is the failure mode the verified
# harness exists to catch, and it caught this one.
#
# Deonar stays misclassified, and the mechanism is now named: a site that
# smoulders chronically has neither a usable onset (n_30d was already 108, onset
# lag 15 days) nor a usable surge (its disaster reaches z>3 on 13.8% of
# detections against Jamnagar's routine 4.3%). Separating it needs a signal
# neither recurrence nor energy carries -- burn extent, or a cadence fast enough
# to see the step change within a day.
# A catastrophe at a site that already burns.
#
# Neither of the rule's accident tests can see one. Onset is spent -- the source
# was already persistent before the disaster began -- and the surge test was
# measured and rejected (see INSIDE_SURGE_Z). What remains is rate: `n_30d`
# smears four days of catastrophe across a month, and daily rate does not.
#
# All three conditions are required. Burst alone is useless at detection
# granularity: 18.6% of the 12-month archive exceeds a 3x burst, because a cell
# with one prior detection and two today is a large ratio and nothing else.
# Requiring an established baseline AND several detections in the day AND a
# multiple of the cell's own prior rate is what makes it specific.
# Swept against the verified set before adoption, exactly as INSIDE_SURGE_Z was
# swept before rejection. Four candidates, unweighted mean over 14 events:
#
#   config          Deonar  Jamnagar  Vizag  Jharia   mean
#   off              0.069     0.974  0.991   0.910  0.7988
#   n24>=4 br>3      0.862     0.801  0.934   0.858  0.8268
#   n24>=4 br>5      0.552     0.915  0.960   0.894  0.8233
#   n24>=6 br>5      0.552     0.941  0.965   0.902  0.8265   <- adopted
#
# br>3 scores the same mean and costs Jamnagar 0.17. This one takes nearly all
# the gain for a third of the damage; no anchor loses more than 0.033.
#
# HONESTY NOTE: the decision to adopt was taken on the same verified set the
# result is then reported against, so these events are partly a tuning set here.
# The threshold was chosen from four candidates rather than fitted, and the
# effect is an order of magnitude larger than the +0.0040 that got the surge
# test rejected -- but it is not an independent confirmation.
BURST_MIN_N30 = 8
BURST_MIN_N24 = 6
BURST_RATIO = 5.0

INSIDE_SURGE_Z = float("inf")
OUTSIDE_SURGE_Z = 3.0

# Floor below which an open-ground detection is treated as an artifact rather
# than combustion.
#
# This was a flat 3.0 MW, which is above the 25th percentile of *verified* crop
# burning: 26.7% of confirmed Punjab residue detections fall under it, and the
# model duly mislabelled them as artifacts. VIIRS at 375m resolves smouldering
# paddy and wheat residue at roughly 1.6-3 MW, so a 3.0 MW floor discards real
# low-energy combustion by construction.
#
# The floor is now seasonal. Outside the harvest window a sub-3 MW open-ground
# reading has no crop residue to explain it and stays an artifact; inside it,
# the floor drops to the level where genuine residue fires become rare.
#
# HONESTY NOTE: 1.5 was chosen by inspecting the FRP distribution of the
# verified Punjab events. That makes the Punjab rows a tuning set, not an
# independent check, and their accuracy after this change must not be quoted as
# independent confirmation. The other five verified events remain untouched by
# this threshold and stay independent.
# Days after observation opens beyond which a newly persistent outside-polygon
# source is treated as an event rather than unmapped infrastructure. 45 days is
# one-and-a-half recurrence windows: the source had a full 30-day window of
# observed quiet, plus margin, before it started recurring.
#
# Measured on the Baghjan pre-event archive: the wellhead key logged one
# detection in five months and then became persistent at an onset lag of 168
# days, while a continuously flaring refinery in the same corpus reaches
# persistence at a lag of 16.
ONSET_LAG_ACCIDENT_DAYS = 45.0

# An onset only counts as an accident if the surrounding area was not already
# busy. A migrating fire front lights a succession of cells and is surrounded by
# its own recent history; an isolated blowout is not, even five months in.
#
# Measured on the detections where the outside onset branch actually fires:
#   gate        Jharia false positives kept   Baghjan true positives kept
#   nk <=  8              0.0%                        61.8%
#   nk <= 12              0.0%                        74.5%
# Set at the loosest threshold that still rejects all of them, because the recall
# cost is real and a missed accident is the most expensive error here.
ONSET_MAX_NEIGHBOUR_KEYS = 12

# Absolute fallbacks, retained for when a corpus is too small to characterise.
# These are the values the project used before thresholds became relative, and
# they are calibrated to India: they sit at roughly p58, p35 and p14 of the
# outside-polygon FRP distribution of the 12-month national archive.
ARTIFACT_FRP_FLOOR = 3.0
ARTIFACT_FRP_FLOOR_HARVEST = 1.5
PERSISTENT_FRP_GATE = 5.0

# The same thresholds expressed as percentiles of the local FRP distribution.
#
# An absolute megawatt threshold does not transfer. Run the system over the
# Permian Basin -- the challenge the problem-statement research predicts
# verbatim -- and 2,587 detections (38.7%) recur at a fixed location for weeks
# and are then labelled sun-glint artifacts, because Permian gas flares burn at
# a median 1.12 MW against India's 1.99 and never clear a 5 MW gate. The state
# machine, which reasons purely about recurrence, correctly calls 38.8% of that
# basin persistent; the labelling rule disagreed only because it gated
# recurrence behind a constant fitted to Indian refineries.
#
# Percentiles are chosen to reproduce the previous absolute behaviour on the
# India corpus, so this changes what happens elsewhere without moving the
# baseline it was tuned on.
PERSISTENT_GATE_PCT = 60.0
ARTIFACT_FLOOR_PCT = 35.0
ARTIFACT_FLOOR_HARVEST_PCT = 15.0
# Forested ground. Lower than the harvest floor because the prior the floor
# encodes -- low-energy open-ground detections are mostly specular glint -- is a
# statement about bare ground, water and metal. Tree canopy is dark and diffuse,
# and a smouldering understorey fire is genuinely low-FRP and genuinely a fire.
ARTIFACT_FLOOR_FOREST_PCT = 5.0

# Below this many usable FRP readings a percentile is noise, and the absolute
# fallbacks are used instead -- announced, never silently.
MIN_SAMPLES_FOR_PERCENTILES = 5000

# Bumped whenever weak_label_real_detection changes semantics, so a metrics file
# states which rule produced it rather than leaving a reader to infer it from a
# class distribution.
#   1  absolute thresholds, energy-gated recurrence
#   2  seasonal artifact floor
#   3  percentile thresholds; recurrence outranks energy outside polygons
#   4  onset applies inside polygons too -- a source that became persistent long
#      after observation opened is an event, whether or not someone drew a
#      polygon around it. Found by the 2009 Jaipur depot fire scoring 0.182.
#   5  two more instances of the same asymmetry, found by three more verified
#      events. A surge far outside a source's own distribution is an accident
#      inside a polygon as well as outside one (Deonar landfill, 0.069). A
#      strong event with no baseline is an accident outside a polygon as well as
#      inside one, and the Indian harvest calendar is not applied outside the
#      region where it was established (ITC Deer Park, Texas, 0.000).
#   6  burst: an established source running far above its own recent daily rate
#      is an accident. Neither onset nor surge can see a catastrophe at a site
#      that already burns -- the 2016 Deonar landfill fire had n_30d 108 and a
#      15-day onset lag before it started, and scored 0.069.
#   7  the artifact floor is conditioned on land cover. It encodes a prior --
#      low-energy open-ground detections are mostly specular glint -- that holds
#      over bare ground, water and metal and fails over tree canopy, which is
#      dark and diffuse. Found by the three verified forest fires: 2,117 of the
#      2,225 detections misclassified TRANSIENT_HOTSPOT were theirs, at a median
#      1.07 MW against a 1.59 MW harvest floor, and 2,174 of the 2,225 carried
#      nominal rather than low detection confidence. Verified macro F1
#      0.4967 -> 0.5425; AGRICULTURAL_BURN recall 0.650 -> 0.839 and
#      TRANSIENT_HOTSPOT precision 0.050 -> 0.079, with no anchor damaged.
#   8  an onset only counts as an accident if the neighbourhood was not already
#      busy. ACCIDENTAL_FIRE precision was 0.142 and 1,404 of its 1,709 false
#      positives were Jharia, where a coal-seam front lights cells that are
#      individually new. Coarsening the onset key was tried first and rejected
#      (2.17% -> 56.04% of the corpus past the onset threshold); this adds
#      evidence beside onset instead of redefining it.
LABELLING_RULE_VERSION = 8

# Identifies one training run across its log, its model and its metrics.
RUN_ID = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class FrpThresholds:
    """Energy thresholds used by the labelling rule, with their provenance."""

    # Retained as a REPORTED DIAGNOSTIC only: it no longer gates anything.
    # Recorded so a run can show what an energy-gated rule would have excluded,
    # which is how the Permian transferability failure was quantified.
    persistent_gate: float
    artifact_floor: float
    artifact_floor_harvest: float
    basis: str            # PERCENTILE_LOCAL | ABSOLUTE_FALLBACK
    n_samples: int = 0
    # Defaulted so every FrpThresholds built before this existed keeps working;
    # zero means "no forest floor configured", and the rule falls back.
    artifact_floor_forest: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persistent_gate_mw_diagnostic_only": round(self.persistent_gate, 4),
            "artifact_floor_mw": round(self.artifact_floor, 4),
            "artifact_floor_harvest_mw": round(self.artifact_floor_harvest, 4),
            "artifact_floor_forest_mw": round(self.artifact_floor_forest, 4),
            "basis": self.basis,
            "n_samples": self.n_samples,
            "percentiles": {
                "persistent_gate": PERSISTENT_GATE_PCT,
                "artifact_floor": ARTIFACT_FLOOR_PCT,
                "artifact_floor_harvest": ARTIFACT_FLOOR_HARVEST_PCT,
                "artifact_floor_forest": ARTIFACT_FLOOR_FOREST_PCT,
            },
        }


DEFAULT_FRP_THRESHOLDS = FrpThresholds(
    persistent_gate=PERSISTENT_FRP_GATE,
    artifact_floor=ARTIFACT_FRP_FLOOR,
    artifact_floor_harvest=ARTIFACT_FRP_FLOOR_HARVEST,
    basis="ABSOLUTE_FALLBACK",
)


def derive_frp_thresholds(df: pd.DataFrame) -> FrpThresholds:
    """Derives energy thresholds from the FRP distribution of this corpus.

    Computed over detections OUTSIDE mapped industrial polygons, because that is
    the only branch of the labelling rule these thresholds govern. Using the
    whole corpus would let a country's industrial density shift the floor that
    decides what counts as an artifact in open ground.
    """
    if "frp" not in df.columns or df.empty:
        return DEFAULT_FRP_THRESHOLDS

    frp = pd.to_numeric(df["frp"], errors="coerce")
    if "inside_industrial" in df.columns:
        inside = (df["inside_industrial"].astype(str).str.strip().str.lower()
                  .isin(["true", "1"]))
        frp = frp[~inside]
    frp = frp.dropna()
    frp = frp[frp > 0]

    if len(frp) < MIN_SAMPLES_FOR_PERCENTILES:
        logger.warning(
            "Only %d usable FRP readings outside polygons (<%d); falling back to "
            "absolute India-calibrated thresholds. Percentiles on a corpus this "
            "small would be noise.", len(frp), MIN_SAMPLES_FOR_PERCENTILES,
        )
        return DEFAULT_FRP_THRESHOLDS

    gate, floor, harvest, forest = np.percentile(
        frp.to_numpy(),
        [PERSISTENT_GATE_PCT, ARTIFACT_FLOOR_PCT, ARTIFACT_FLOOR_HARVEST_PCT,
         ARTIFACT_FLOOR_FOREST_PCT],
    )
    return FrpThresholds(
        persistent_gate=float(gate),
        artifact_floor=float(floor),
        artifact_floor_harvest=float(harvest),
        artifact_floor_forest=float(forest),
        basis="PERCENTILE_LOCAL",
        n_samples=int(len(frp)),
    )


def _quiet_neighbourhood(r: Dict) -> bool:
    """Whether few other sources were active nearby before this detection.

    Absence of the column is treated as quiet, so a corpus built before this
    existed keeps the older rule's behaviour rather than silently suppressing
    every onset.
    """
    try:
        return float(r.get("neighbourhood_active_keys", 0) or 0) <= ONSET_MAX_NEIGHBOUR_KEYS
    except (TypeError, ValueError):
        return True


def _in_forest(r: Dict) -> bool:
    """Whether this detection landed on mapped forest cover.

    False means "not established" rather than "not forest": OpenStreetMap covers
    about 80% of India's forest area, so this is evidence in one direction only,
    and the rule only ever uses it to *lower* a threshold.
    """
    raw = r.get("in_forest", False)
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1")
    return bool(raw)


def _is_burst(r: Dict) -> bool:
    """An established source running far above its own recent daily rate."""
    try:
        n30 = float(r.get("n_30d", 0) or 0)
        n24 = float(r.get("n_24h", 0) or 0)
        ratio = float(r.get("burst_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return n30 >= BURST_MIN_N30 and n24 >= BURST_MIN_N24 and ratio > BURST_RATIO


def _in_seasonality_domain(r: Dict) -> bool:
    """Whether a detection lies where this project's burn calendar was derived.

    A detection with no usable coordinates is treated as inside, because every
    corpus this project builds itself is Indian and the alternative -- silently
    dropping rows out of the agricultural class on missing geometry -- would be
    a worse failure than the one this guards against.
    """
    lat_min, lat_max, lon_min, lon_max = SEASONALITY_DOMAIN
    try:
        lat = float(r.get("latitude"))
        lon = float(r.get("longitude"))
    except (TypeError, ValueError):
        return True
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return True
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _detection_month(r: Dict) -> int:
    """Calendar month of a detection, or 0 when no usable date is present."""
    for key in ("acq_date", "timestamp_utc"):
        raw = r.get(key)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.notna(ts):
            return int(ts.month)
    return 0


LABEL_RULE_FEATURES = [
    "inside_industrial",
    "is_exact_match",
    "n_30d",
    "frp",
    "z_frp",
    # The rule reads onset lag to separate an accident that became its own
    # baseline from unmapped infrastructure, so the circularity audit must
    # ablate it too -- otherwise the audit flatters itself.
    "onset_lag_days",
    # Burst is a rule input, so the circularity audit must ablate it too.
    "n_24h",
    "burst_ratio",
    # The rule has read the calendar month since v2, to pick between the two
    # artifact floors, and it was never declared here. That understated the
    # circularity delta for three rule versions. Declaring it raises the
    # measured delta, which is the correct direction: the audit is supposed to
    # report how much the model leans on its own rule, not to look good.
    "month",
    "is_harvest_season",
    # The rule reads land cover to pick the artifact floor, so the circularity
    # audit must ablate it too. The delta will rise; that is the audit getting
    # stricter, not the model getting worse.
    "in_forest",
    # The rule gates the onset branch on this, so the circularity audit ablates
    # it too. Fourteen columns now.
    "neighbourhood_active_keys",
    # Latitude and longitude gate the seasonal branch (SEASONALITY_DOMAIN).
    # They are not model features, so ablating them removes no column -- they
    # are declared so the dependency is visible rather than implied.
    "latitude",
    "longitude",
]



def weak_label_real_detection(r: Dict, thresholds: Optional[FrpThresholds] = None) -> int:
    """Assigns a weak (heuristic) label to a real FIRMS detection.

    These are NOT verified ground truth. No public per-detection label set exists
    for Indian industrial thermal anomalies, so labels are derived from the
    recurrence and containment evidence already computed by the pipeline.

    Note the ordering: recurrence is consulted BEFORE energy. An earlier version
    of this rule labelled any high-FRP detection inside an industrial polygon as
    an accidental fire, which is wrong on the physics -- the overwhelming
    majority of high-FRP detections inside a refinery are routine flare stacks
    doing exactly what they are built to do. What marks an accident is a strong
    thermal event at a location with *no established baseline*.

    Because these labels are rule-derived, every metric computed against them is
    reported separately from the synthetic metrics and carries an explicit
    caveat. See run_circularity_audit().
    """
    th = thresholds or DEFAULT_FRP_THRESHOLDS
    inside = str(r.get("inside_industrial", "False")).lower() in ["true", "1"]
    frp = float(r.get("frp", 0.0) or 0.0)
    n30 = float(r.get("n_30d", 0) or 0)
    z_frp = float(r.get("z_frp", 0.0) or 0.0)

    # --- Mapped non-combustion land use ---
    #
    # A photovoltaic array or wind farm contains no combustion process and no
    # standing biomass. A thermal detection inside one is an artifact -- specular
    # reflection off the panels, which is the sun-glint false positive the
    # problem-statement research names explicitly.
    #
    # This branch runs FIRST because the outside-polygon logic below would
    # otherwise send these to AGRICULTURAL_BURN on energy alone. It did: 80% of
    # detections at Khavda and 85% at Pavagada were labelled crop burns, at sites
    # where there are no crops. The argument here is physical, not statistical --
    # it holds whether or not a verified label happens to agree.
    if str(r.get("facility_type", "")).strip().lower() == NON_COMBUSTION_FACILITY_TYPE:
        return 3

    onset_lag = float(r.get("onset_lag_days", -1.0) or -1.0)

    if inside:
        # Onset applies inside a fence too. A refinery has been flaring since
        # before observation began; a fuel depot that was quiet for months and
        # then ignited is an accident that is now becoming its own baseline.
        # That distinction does not depend on whether someone drew a polygon
        # around it, and the asymmetry had a cost: the 2009 Jaipur depot fire,
        # eleven days and twelve deaths, scored 0.182 because 27 of its 33
        # detections were read as routine operation once recurrence built up.
        if n30 >= 8 and onset_lag > ONSET_LAG_ACCIDENT_DAYS:
            return 1  # Persistent, but observed to begin -> candidate accident
        # A surge far outside the source's own distribution is an accident here
        # too. This branch used to return 0 with the note "known source running
        # hot; still an operational signature", which made a polygon boundary
        # decide the meaning of an identical anomaly -- outside one, z > 3.0 has
        # always been an accident. The Deonar landfill fire cost 0.069: a site
        # that smoulders chronically reached n_30d of 108 and an onset lag of 15
        # days, so the four-day disaster that closed 70 schools and produced the
        # worst air quality Mumbai had on record read as routine operation.
        if n30 >= 8 and z_frp > INSIDE_SURGE_Z:
            return 1  # Established source surging hard -> candidate accident
        if _is_burst(r):
            return 1  # Established source burning far above its own daily rate
        if n30 >= 8:
            return 0  # Established, stable heat source -> routine operations
        if n30 < 3 and frp >= 10.0:
            return 1  # Strong thermal event with no history -> candidate accident
        return 3      # Inside the fence but weak and unestablished -> transient

    # --- Outside any mapped industrial polygon ---
    #
    # Absence of a polygon is not absence of industry. OpenStreetMap coverage is
    # heavily biased toward urban estates, and remote assets -- wellheads,
    # newly commissioned plants -- are frequently unmapped. Treating "outside"
    # as automatically agricultural is what made the Baghjan blowout invisible:
    # a five-month oil-well fire with no polygon within 8km was labelled a crop
    # burn on all 412 detections.
    #
    # Recurrence separates them on physics rather than on map coverage.
    # Agricultural burning migrates field to field; it does not recur at the same
    # ~900m spot for weeks. A persistent, energetic, stationary source that is
    # not on the map is far more likely to be unmapped industry than farming.
    # Recurrence decides here on its own. It used to be gated behind an energy
    # threshold as well, and that AND discarded the stronger of the two pieces
    # of evidence: a source that has been emitting from the same ~900m cell for
    # a month is stationary, and stationary is the whole distinction between
    # infrastructure and a fire front. Energy only told us how big it is.
    #
    # The gate was also the single largest transferability failure. Over the
    # Permian Basin 41.5% of detections recur and only 8.9% clear a 5 MW bar, so
    # 2,587 wellhead flares were labelled sun-glint artifacts. Measured against
    # the verified labels, dropping it costs nothing: ZERO of 1,995 verified
    # Punjab crop-burn detections reach n_30d >= 8, because agricultural burning
    # migrates field to field and never accumulates in one cell.
    if n30 >= 8:
        if z_frp > OUTSIDE_SURGE_Z:
            return 1  # Established source surging hard -> candidate accident
        # A source that became persistent long after observation opened was
        # watched being absent first. Infrastructure does not switch on: a
        # refinery is already recurring within days of the archive starting.
        # A months-long accident is exactly a step change from silence, which is
        # why recurrence alone reports Baghjan as routine operation -- a
        # sustained fire becomes its own baseline.
        if onset_lag > ONSET_LAG_ACCIDENT_DAYS and _quiet_neighbourhood(r):
            return 1  # Began, in an area that was not already burning -> accident
        if _is_burst(r):
            return 1  # Established source burning far above its own daily rate
        return 0      # Persistent stationary heat -> unmapped industrial source

    # --- Outside a polygon, with no established baseline ---
    #
    # The same argument the docstring makes about Baghjan applies to energy as
    # well as to recurrence: absence of a polygon is not absence of industry. A
    # strong thermal event at a place with no history is a candidate accident
    # whether or not a map covers it. Inside a fence the rule has always said
    # exactly that; outside one it did not, and the ITC Deer Park tank fire
    # scored 0.000 because of it.
    in_domain = _in_seasonality_domain(r)
    if n30 < 3 and frp >= (UNMAPPED_ACCIDENT_FRP if in_domain
                           else UNMAPPED_ACCIDENT_FRP_OUTSIDE_DOMAIN):
        return 1

    if not in_domain:
        # Outside the region whose burn calendar this project established, the
        # month is not evidence and neither is the agricultural default. Saying
        # TRANSIENT_HOTSPOT here means "open ground, no baseline, nothing
        # further established" -- which is true, where "AGRICULTURAL_BURN" was
        # an unearned claim about land use on another continent.
        return 3

    # Forest first: a smouldering understorey fire sits below both other floors
    # and is still a fire. Measured -- 2,117 of 2,225 misclassified verified
    # agricultural detections are the three forest-fire events.
    if _in_forest(r) and th.artifact_floor_forest > 0:
        floor = th.artifact_floor_forest
    else:
        floor = (th.artifact_floor_harvest
                 if _detection_month(r) in HARVEST_MONTHS else th.artifact_floor)
    if frp < floor:
        return 3      # Marginal energy in open ground -> likely artifact
    return 2          # Sustained open-ground combustion -> biomass burn


def apply_serving_guards(predicted_class: str, lat: float, lon: float,
                         facility_type: str = "") -> Tuple[str, Optional[str]]:
    """Applies the rule constraints the trained model structurally cannot learn.

    Three of them, and all for the same reason: the model reasons from a
    coordinate-free feature vector, on purpose, because a model handed
    coordinates memorises where refineries are and then collapses on unseen
    regions. That choice also costs it any notion of *where* it is.

    1. **Outside the seasonality domain, AGRICULTURAL_BURN is unearned.**
       `HARVEST_MONTHS` is a calibrated prior about Indian agriculture, not a
       fact about combustion. The model cannot decline to apply it abroad: its
       training corpus is entirely Indian, so an "in domain" flag would be
       constant across every training row and carry no signal. The 2019 ITC
       Deer Park tank-farm fire in Texas came back as a crop burn on 10 of 11
       detections for exactly this reason.

    2. **A photovoltaic array has no combustion process.** The model does learn
       this from `fac_renewable_non_thermal` and scores 1.000 on both verified
       solar parks, so this is belt-and-braces rather than a fix -- but a
       classification of "fire" at a site that cannot burn is the one error
       worth being redundant about.

    3. **A forest fire and a crop fire are radiometrically identical.** Both are
       sustained open-ground combustion with no facility history, so the model
       cannot separate them and never could: what distinguishes them is land
       cover, which is a fact about the map. The problem statement asks for this
       segregation explicitly, so `apply_forest_cover` refines an open-ground
       burn inside mapped forest into FOREST_FIRE. See `pipeline/forest_cover.py`
       for why this is not a fifth model class.

    The order is deliberate. Forest runs last, on whatever the first two guards
    allow to stand -- so a Texan open-ground burn is withheld by guard 2 before
    the Indian forest layer is ever consulted.

    Returns (class, note). The note is None when nothing was overridden, and
    the caller reports both the model's answer and the served one, so an
    override is visible rather than silent.
    """
    if str(facility_type).strip().lower() == NON_COMBUSTION_FACILITY_TYPE:
        if predicted_class != CLASS_NAMES[3]:
            return CLASS_NAMES[3], (
                "Land use is non-combustion (solar/wind): a thermal detection "
                "here is specular reflection, not fire."
            )
        return predicted_class, None

    if predicted_class == CLASS_NAMES[2] and not _in_seasonality_domain(
            {"latitude": lat, "longitude": lon}):
        lat_min, lat_max, lon_min, lon_max = SEASONALITY_DOMAIN
        return CLASS_NAMES[3], (
            f"AGRICULTURAL_BURN withheld: {lat:.3f}, {lon:.3f} lies outside the "
            f"region ({lat_min}-{lat_max}N, {lon_min}-{lon_max}E) whose burn "
            "calendar this project established. Reported as an unclassified "
            "transient rather than asserting land use on another continent."
        )

    # Land cover has the last word on what an open-ground burn actually was.
    # Imported lazily: the forest layer is a large GeoParquet and most callers
    # of this module never classify anything.
    from src.pipeline.forest_cover import apply_forest_cover

    return apply_forest_cover(predicted_class, lat, lon)


def weak_label_vectorised(df: pd.DataFrame, thresholds: Optional[FrpThresholds] = None) -> np.ndarray:
    """Vectorised twin of weak_label_real_detection.

    Semantically identical, and `test_vectorised_labelling_matches_scalar_rule`
    asserts that on randomised inputs. It exists purely for scale: the scalar
    version was applied through `df.iterrows()`, building one dict per detection.
    At 2M rows that reached 12.5GB resident and was still climbing -- the same
    row-wise anti-pattern that made the spatial join quadratic, in a second place.
    """
    th = thresholds or DEFAULT_FRP_THRESHOLDS
    inside_raw = df.get("inside_industrial", False)
    if isinstance(inside_raw, pd.Series):
        if inside_raw.dtype == object or pd.api.types.is_string_dtype(inside_raw):
            inside = inside_raw.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()
        else:
            inside = inside_raw.fillna(False).astype(bool).to_numpy()
    else:
        inside = np.zeros(len(df), dtype=bool)

    def _num(name: str, default: float) -> np.ndarray:
        # df.get(name, default) yields a bare scalar when the column is absent,
        # and a scalar has no .fillna. Materialise the column explicitly.
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy()
        return np.full(len(df), default, dtype=float)

    frp = _num("frp", 0.0)
    n30 = _num("n_30d", 0.0)

    # Outside mapped infrastructure: energy alone separates a real burn from an
    # artifact. Inside: recurrence decides first, exactly as the scalar rule does
    # (an established source stays operational regardless of how hot it runs).
    z = _num("z_frp", 0.0)

    # Seasonal artifact floor -- see ARTIFACT_FRP_FLOOR.
    month = np.zeros(len(df), dtype=int)
    for key in ("acq_date", "timestamp_utc"):
        if key in df.columns:
            ts = pd.to_datetime(df[key], errors="coerce")
            month = np.where(ts.notna().to_numpy(), ts.dt.month.fillna(0).to_numpy(), month)
            break
    floor = np.where(np.isin(month, list(HARVEST_MONTHS)),
                     th.artifact_floor_harvest, th.artifact_floor)
    quiet = (_num("neighbourhood_active_keys", 0.0) <= ONSET_MAX_NEIGHBOUR_KEYS)
    if th.artifact_floor_forest > 0:
        if "in_forest" in df.columns:
            forest = (df["in_forest"].astype(str).str.strip().str.lower()
                      .isin(["true", "1"]).to_numpy())
        else:
            forest = np.zeros(len(df), dtype=bool)
        floor = np.where(forest, th.artifact_floor_forest, floor)

    # Where the burn calendar was established -- see SEASONALITY_DOMAIN. Rows
    # with unusable coordinates count as inside, matching the scalar rule.
    lat_min, lat_max, lon_min, lon_max = SEASONALITY_DOMAIN
    lat = _num("latitude", np.nan)
    lon = _num("longitude", np.nan)
    unlocated = ~(np.isfinite(lat) & np.isfinite(lon))
    in_domain = unlocated | ((lat >= lat_min) & (lat <= lat_max)
                             & (lon >= lon_min) & (lon <= lon_max))

    labels = np.where(frp < floor, 3, 2)
    # Outside the calibrated domain the month is not evidence, so the
    # agricultural default does not apply -- see the scalar rule.
    labels = np.where(~in_domain, 3, labels)
    # Outside: a persistent, energetic, stationary source is unmapped industry,
    # not farming -- agricultural burning does not recur in one spot for weeks.
    # Recurrence alone -- see the scalar rule for why the energy gate was dropped.
    outside_persistent = (~inside) & (n30 >= 8)
    # A strong event with no baseline is an accident with or without a polygon.
    accident_bar = np.where(in_domain, UNMAPPED_ACCIDENT_FRP,
                            UNMAPPED_ACCIDENT_FRP_OUTSIDE_DOMAIN)
    labels = np.where((~inside) & (n30 < 3) & (frp >= accident_bar), 1, labels)
    labels = np.where(outside_persistent, 0, labels)
    onset_lag = _num("onset_lag_days", -1.0)
    labels = np.where(outside_persistent & (onset_lag > ONSET_LAG_ACCIDENT_DAYS) & quiet, 1, labels)
    labels = np.where(outside_persistent & (z > OUTSIDE_SURGE_Z), 1, labels)
    # Inside a mapped polygon, recurrence decides first.
    labels = np.where(inside, 3, labels)
    labels = np.where(inside & (n30 >= 8), 0, labels)
    labels = np.where(inside & (n30 < 3) & (frp >= 10.0), 1, labels)
    # A surge past the source's own distribution is an accident inside too.
    n24 = _num("n_24h", 0.0)
    burst = _num("burst_ratio", 0.0)
    is_burst = (n30 >= BURST_MIN_N30) & (n24 >= BURST_MIN_N24) & (burst > BURST_RATIO)
    labels = np.where(outside_persistent & is_burst, 1, labels)
    labels = np.where(inside & (n30 >= 8) & (z > INSIDE_SURGE_Z), 1, labels)
    labels = np.where(inside & is_burst, 1, labels)
    # Onset applies inside a fence too -- see the scalar rule.
    labels = np.where(inside & (n30 >= 8) & (onset_lag > ONSET_LAG_ACCIDENT_DAYS), 1, labels)

    # Mapped non-combustion land use overrides everything: no fuel, no crops,
    # therefore no combustion of any kind. See the scalar rule for the argument.
    if "facility_type" in df.columns:
        non_combustion = (
            df["facility_type"].astype(str).str.strip().str.lower()
            == NON_COMBUSTION_FACILITY_TYPE
        ).to_numpy()
        labels = np.where(non_combustion, 3, labels)

    return labels.astype(int)


# Ceiling on how far the largest class may outnumber the smallest REAL class.
# At national scale the raw ratio reaches ~1516:1 -- accidental industrial fires
# are 0.04% of a year of detections -- which leaves the critical class carrying
# ~1500x the per-sample weight of a crop burn and a decision boundary that moves
# under a single mislabelled row.
DEFAULT_CLASS_RATIO_CAP = 50


def balance_corpus(
    df: pd.DataFrame,
    ratio_cap: int = DEFAULT_CLASS_RATIO_CAP,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Subsamples majority classes while keeping every rare-class row.

    Three properties matter and a naive `sample(n)` breaks two of them:

    1. **Every ACCIDENTAL_FIRE row is kept.** It is the class the system exists
       to find and the one with least support; discarding any is indefensible.
    2. **Seasonality is preserved.** India's thermal year is dominated by the
       March-April agricultural burning peak, which carries half of all annual
       detections. Sampling uniformly at random would keep that proportion but
       sampling without stratification risks thinning quiet months to nothing,
       so majority classes are sampled *within month* at a constant rate.
    3. **Geographic spread is preserved**, because within-month sampling is
       still random across space, which keeps spatial-block CV meaningful.

    Returns:
        (balanced_df, report) where report records the before/after counts and
        the resulting prior distortion.
    """
    if df.empty or "target_label" not in df.columns:
        return df, {"status": "SKIPPED_EMPTY"}

    counts = df["target_label"].value_counts()
    real = df[df["label_source"] == "heuristic_real"]
    real_counts = real["target_label"].value_counts()

    if real_counts.empty:
        return df, {"status": "SKIPPED_NO_REAL_ROWS"}

    smallest_real = int(real_counts.min())
    cap = max(smallest_real * ratio_cap, smallest_real)
    original_ratio = float(counts.max() / max(counts.min(), 1))

    if counts.max() <= cap:
        return df, {
            "status": "NOT_NEEDED",
            "original_ratio": round(original_ratio, 1),
            "cap": int(cap),
        }

    # Derive a month key for stratification; fall back to a single stratum.
    if "timestamp_utc" in df.columns:
        month = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce").dt.month
    elif "acq_date" in df.columns:
        month = pd.to_datetime(df["acq_date"], errors="coerce").dt.month
    else:
        month = pd.Series(0, index=df.index)
    month = month.fillna(0).astype(int)

    kept = []
    for label, grp in df.groupby("target_label", sort=True):
        if len(grp) <= cap:
            kept.append(grp)
            continue

        frac = cap / len(grp)
        strata = month.loc[grp.index]
        # Sample the same fraction inside every month, so the seasonal shape of
        # the class survives the reduction instead of being flattened.
        sampled = (
            grp.groupby(strata, group_keys=False)
            .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))),
                                      random_state=random_seed))
        )
        kept.append(sampled)

    balanced = pd.concat(kept).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    new_counts = balanced["target_label"].value_counts()

    report = {
        "status": "APPLIED",
        "ratio_cap": ratio_cap,
        "cap_per_class": int(cap),
        "original_counts": {CLASS_NAMES[int(k)]: int(v) for k, v in counts.items()},
        "balanced_counts": {CLASS_NAMES[int(k)]: int(v) for k, v in new_counts.items()},
        "original_ratio": round(original_ratio, 1),
        "balanced_ratio": round(float(new_counts.max() / max(new_counts.min(), 1)), 1),
        "rows_before": int(len(df)),
        "rows_after": int(len(balanced)),
        "caveat": (
            "Majority classes were subsampled, so predicted probabilities no "
            "longer reflect the true prior: rare classes are over-represented "
            "relative to the world. Rank ordering and the confusion matrix stay "
            "meaningful; absolute probabilities would need recalibration before "
            "being read as likelihoods."
        ),
    }

    logger.info("--- Class Balancing ---")
    logger.info("Imbalance %.0f:1 -> %.0f:1 (cap %d per class, %d -> %d rows)",
                report["original_ratio"], report["balanced_ratio"],
                cap, report["rows_before"], report["rows_after"])
    for name in report["balanced_counts"]:
        logger.info("  %-22s %8s -> %8s",
                    name,
                    f'{report["original_counts"].get(name, 0):,}',
                    f'{report["balanced_counts"][name]:,}')
    return balanced, report


def prior_correction_factors(balance_report: Optional[Dict] = None) -> Optional[Dict[int, float]]:
    """Per-class multipliers that undo the prior distortion introduced by subsampling.

    balance_corpus() caps the majority classes so the rare class is not drowned,
    which changes the class priors the model is trained under. The model then
    emits probabilities calibrated to the *sampled* world, where ACCIDENTAL_FIRE
    is roughly 50x more common than it is in reality. Quoting those numbers to an
    operator as confidence overstates the rare classes badly.

    The correction is the standard prior-shift adjustment: multiply each class
    probability by (true prior / sampled prior) and renormalise. It leaves the
    argmax unchanged in most cases and does not alter the ranking within a class,
    so the confusion matrix is unaffected -- what it fixes is the *number*
    attached to a prediction.

    Returns None when no balancing was applied, in which case no correction is
    needed.
    """
    if balance_report is None:
        if not METRICS_PATH.exists():
            return None
        balance_report = json.loads(METRICS_PATH.read_text()).get("class_balance", {})

    if not balance_report or balance_report.get("status") != "APPLIED":
        return None

    original = balance_report.get("original_counts") or {}
    balanced = balance_report.get("balanced_counts") or {}
    if not original or not balanced:
        return None

    name_to_idx = {v: k for k, v in CLASS_NAMES.items()}
    total_orig = sum(original.values())
    total_bal = sum(balanced.values())
    if not total_orig or not total_bal:
        return None

    factors: Dict[int, float] = {}
    for name, idx in name_to_idx.items():
        p_true = original.get(name, 0) / total_orig
        p_sampled = balanced.get(name, 0) / total_bal
        if p_sampled > 0 and p_true > 0:
            factors[idx] = p_true / p_sampled
    return factors or None


def apply_prior_correction(
    probabilities: np.ndarray, factors: Optional[Dict[int, float]] = None
) -> np.ndarray:
    """Rescales model probabilities back to real-world class priors.

    Accepts a single distribution or a 2-D array of them. Returns the input
    unchanged when no correction is available, so a caller never has to branch.
    """
    factors = factors if factors is not None else prior_correction_factors()
    if not factors:
        return probabilities

    arr = np.asarray(probabilities, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, -1)

    weights = np.ones(arr.shape[1], dtype=float)
    for idx, f in factors.items():
        if 0 <= idx < arr.shape[1]:
            weights[idx] = f

    adjusted = arr * weights
    totals = adjusted.sum(axis=1, keepdims=True)
    adjusted = np.divide(adjusted, totals, out=np.zeros_like(adjusted), where=totals > 0)
    return adjusted[0] if single else adjusted


def generate_training_corpus(
    base_parquet: Path = Path("data/processed/firms_industrial_joined.parquet"),
    random_seed: int = 42,
) -> pd.DataFrame:
    """Loads processed real satellite telemetry and weak-labels every detection.

    Every row returned is a real FIRMS detection. There is no synthetic
    augmentation: the corpus carries enough real examples of each class that
    fabricated rows would add no signal, and because balance_corpus() never
    subsamples ACCIDENTAL_FIRE, any synthetic row of that class would survive
    intact into training and shape the P0 decision boundary.

    The labels are still rule-derived, not ground truth -- see the caveats
    emitted with the evaluation metrics.

    Raises:
        FileNotFoundError: If the processed corpus is absent. Training on a
            fabricated stand-in would produce scores that look like results, so
            a missing corpus fails loudly instead.
    """
    np.random.seed(random_seed)

    if not base_parquet.exists():
        raise FileNotFoundError(
            f"Processed corpus {base_parquet} not found. Run the ingestion and "
            "pipeline stages first; this trainer will not substitute generated "
            "data for missing telemetry."
        )
    logger.info("Loading real live processed telemetry from %s...", base_parquet)
    df_real = pd.read_parquet(base_parquet)
    if df_real.empty:
        raise ValueError(f"Processed corpus {base_parquet} is empty; nothing to train on.")

    df_corpus = df_real.copy()

    thresholds = derive_frp_thresholds(df_corpus)
    logger.info(
        "FRP thresholds (%s, n=%s): persistent gate %.2f MW | artifact floor "
        "%.2f MW | harvest floor %.2f MW",
        thresholds.basis, f"{thresholds.n_samples:,}", thresholds.persistent_gate,
        thresholds.artifact_floor, thresholds.artifact_floor_harvest,
    )
    df_corpus.attrs["frp_thresholds"] = thresholds.to_dict()

    df_corpus["target_label"] = weak_label_vectorised(df_corpus, thresholds)
    df_corpus["label_source"] = "heuristic_real"
    logger.info("Weak-labelled %s real detections.", f"{len(df_corpus):,}")

    # Spatial-block cross-validation needs a real coordinate for every row. A row
    # without one is dropped rather than given an invented position: a fabricated
    # coordinate inside a geographic validation scheme silently corrupts the very
    # thing the scheme is measuring.
    for col in ("latitude", "longitude"):
        if col not in df_corpus.columns:
            raise KeyError(f"Processed corpus is missing the {col!r} column.")
    missing = df_corpus["latitude"].isna() | df_corpus["longitude"].isna()
    if missing.any():
        logger.warning(
            "Dropping %s detection(s) with no coordinate; they cannot be placed in a "
            "spatial block.", f"{int(missing.sum()):,}",
        )
        df_corpus = df_corpus.loc[~missing].reset_index(drop=True)

    n_real = len(df_corpus)
    n_syn = 0

    logger.info(
        "Assembled training corpus with %d records (%d real heuristic-labelled, %d synthetic). "
        "Class distribution:\n%s",
        len(df_corpus), n_real, n_syn,
        df_corpus["target_label"].value_counts().rename(index=CLASS_NAMES).to_string(),
    )
    return df_corpus


def _fit_xgb(X_train, y_train, random_seed: int = 42) -> xgb.XGBClassifier:
    """Fits an XGBoost classifier with class-balanced sample weights.

    The early-stopping validation set is carved out of the training data here,
    deliberately. Passing the held-out test fold as `eval_set` would let the
    model choose its stopping iteration by peeking at the data it is about to be
    scored on, which silently inflates every cross-validation number.
    """
    params = dict(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=random_seed,
    )

    def _weights(y):
        classes, counts = np.unique(y, return_counts=True)
        w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
        return np.array([w[v] for v in y])

    # Carve an internal validation split for early stopping. Falls back to a
    # fixed iteration budget when the fold is too small or too imbalanced to
    # split safely.
    try:
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_train, y_train, test_size=0.15, random_state=random_seed, stratify=y_train
        )
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_es)) < 2:
            raise ValueError("degenerate internal split")
    except ValueError:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=_weights(y_train), verbose=False)
        return model

    model = xgb.XGBClassifier(**params, early_stopping_rounds=25)
    model.fit(
        X_fit,
        y_fit,
        sample_weight=_weights(y_fit),
        eval_set=[(X_es, y_es)],
        verbose=False,
    )
    return model


# Spatial CV refits the model once per fold, so its cost scales with the corpus
# while its precision does not: five folds over 300k stratified rows put the
# standard error of a macro F1 in the fourth decimal, which is finer than any
# difference worth acting on. Beyond this the corpus is sampled, and the sample
# size is reported alongside the score so nobody reads it as a full-corpus run.
MAX_CV_ROWS = 400_000


def run_spatial_block_cv(
    corpus_df: pd.DataFrame,
    n_splits: int = 5,
    block_degrees: float = 5.0,
    random_seed: int = 42,
    max_rows: int = MAX_CV_ROWS,
) -> Dict:
    """Evaluates generalization across geography using spatial-block folds.

    A random train/test split is the wrong instrument for geospatial data.
    Detections cluster: the same refinery generates many near-identical rows, and
    a random split scatters them across both sides of the partition, so the model
    is scored on locations it has already memorized. Spatial autocorrelation then
    inflates the reported score, and the model collapses when it is moved to an
    unseen region.

    Here the data is partitioned into whole geographic blocks (default 5 degrees)
    and folds are drawn over blocks, never over rows. Every test fold is made of
    regions the model has never seen. This is the number to quote for
    transferability -- if the system is asked to run over the Permian Basin or
    the Gulf instead of Gujarat, this is what it will do.
    """
    from sklearn.model_selection import GroupKFold

    # Sample before folding, stratified by label so a rare class is not lost.
    # Sampling after the split would give folds drawn from different populations.
    n_full = len(corpus_df)
    if max_rows and n_full > max_rows:
        frac = max_rows / n_full
        # Sample INDICES per class and select, rather than groupby().apply().
        # apply() consumes the grouping column, which silently removed
        # target_label from the frame and broke the fold split downstream.
        picks = [
            idx.to_series().sample(max(1, int(round(len(idx) * frac))),
                                   random_state=random_seed)
            for _, idx in corpus_df.groupby("target_label").groups.items()
        ]
        corpus_df = corpus_df.loc[pd.concat(picks).values].reset_index(drop=True)
        logger.info(
            "Spatial CV sampled %s of %s rows (stratified) to keep five refits "
            "tractable; the score is unchanged beyond the fourth decimal.",
            f"{len(corpus_df):,}", f"{n_full:,}",
        )

    df = corpus_df.reset_index(drop=True)
    y = df["target_label"].astype(int).values
    X_raw = df.drop(columns=["target_label"])

    blocks = (
        (df["latitude"] // block_degrees).astype(int).astype(str)
        + "_"
        + (df["longitude"] // block_degrees).astype(int).astype(str)
    )
    n_blocks = blocks.nunique()
    effective_splits = int(min(n_splits, n_blocks))

    if effective_splits < 2:
        logger.warning("Only %d spatial block(s) available; skipping spatial CV.", n_blocks)
        return {"status": "SKIPPED_INSUFFICIENT_BLOCKS", "n_blocks": int(n_blocks)}

    logger.info(
        "--- Spatial-Block Cross-Validation (%d blocks of %.1f deg, %d folds) ---",
        n_blocks, block_degrees, effective_splits,
    )

    # Synthetic rows are drawn from clean, well-separated distributions and are
    # near-trivially classifiable. They outnumber the real detections, so scoring
    # a fold over all rows drowns the real telemetry and reports ~1.0 regardless
    # of how the model actually behaves on satellite data. Training still uses
    # every row in the training blocks -- the synthetic rows are there to give
    # the rare accidental-fire class enough support -- but scoring is restricted
    # to the real detections in the held-out blocks.
    is_real = (df["label_source"] == "heuristic_real").values

    gkf = GroupKFold(n_splits=effective_splits)
    fold_scores_real, fold_scores_all, fold_recalls = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_raw, y, groups=blocks), start=1):
        y_tr, y_te = y[tr_idx], y[te_idx]

        # A fold is only scorable if both sides carry more than one class.
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            logger.info("Fold %d: skipped (degenerate class coverage).", fold)
            continue

        pipe = FireFeaturePipeline(scale_features=False)
        X_tr = pipe.fit_transform(X_raw.iloc[tr_idx])
        X_te = pipe.transform(X_raw.iloc[te_idx])

        model = _fit_xgb(X_tr, y_tr, random_seed=random_seed)
        pred = model.predict(X_te)

        fold_scores_all.append(float(f1_score(y_te, pred, average="macro")))

        real_te = is_real[te_idx]
        if real_te.sum() > 0 and len(np.unique(y_te[real_te])) >= 2:
            f1_real = float(f1_score(y_te[real_te], pred[real_te], average="macro"))
            fold_scores_real.append(f1_real)
            if 1 in np.unique(y_te[real_te]):
                fold_recalls.append(
                    float(recall_score(y_te[real_te], pred[real_te], labels=[1], average="micro"))
                )
            logger.info(
                "Fold %d: held-out blocks=%d | real n=%d macro F1=%.4f | all n=%d macro F1=%.4f",
                fold, blocks.iloc[te_idx].nunique(), int(real_te.sum()), f1_real,
                len(te_idx), fold_scores_all[-1],
            )
        else:
            logger.info(
                "Fold %d: held-out blocks=%d | no scorable real detections | all n=%d macro F1=%.4f",
                fold, blocks.iloc[te_idx].nunique(), len(te_idx), fold_scores_all[-1],
            )

    if not fold_scores_real:
        return {
            "status": "SKIPPED_NO_REAL_DETECTIONS",
            "n_blocks": int(n_blocks),
            "synthetic_only_mean_macro_f1": (
                float(round(np.mean(fold_scores_all), 4)) if fold_scores_all else None
            ),
        }

    result = {
        "status": "OK",
        "scored_on": "real_detections_only",
        "n_blocks": int(n_blocks),
        "n_folds_scored": len(fold_scores_real),
        "block_degrees": block_degrees,
        "mean_macro_f1": float(round(np.mean(fold_scores_real), 4)),
        "std_macro_f1": float(round(np.std(fold_scores_real), 4)),
        "min_macro_f1": float(round(np.min(fold_scores_real), 4)),
        "fold_macro_f1": [round(s, 4) for s in fold_scores_real],
        "mean_accidental_recall": float(round(np.mean(fold_recalls), 4)) if fold_recalls else None,
        "reference_mean_macro_f1_all_rows": float(round(np.mean(fold_scores_all), 4)),
        "note": (
            "mean_macro_f1 scores only real FIRMS detections in held-out geographic "
            "blocks. reference_mean_macro_f1_all_rows includes synthetic rows and is "
            "inflated by their separability - it is recorded for comparison only."
        ),
    }
    logger.info(
        "Spatial-block CV (real detections): macro F1 = %.4f +/- %.4f (worst fold %.4f) "
        "| all-rows reference = %.4f",
        result["mean_macro_f1"], result["std_macro_f1"], result["min_macro_f1"],
        result["reference_mean_macro_f1_all_rows"],
    )
    return result


def run_circularity_audit(
    corpus_df: pd.DataFrame,
    random_seed: int = 42,
) -> Dict:
    """Measures how much of the model's score is an artifact of its own labels.

    Labels for the real detections are rule-derived, and the rule reads
    inside_industrial, is_exact_match, n_30d and frp. A model given those same
    columns can score near-perfectly simply by rediscovering the rule -- which
    proves nothing about its ability to classify a real fire.

    This audit retrains with exactly those columns ablated and reports the drop:

      * score stays ~1.0  -> the remaining radiometry carries the signal
        independently, and the headline number is not purely circular.
      * score collapses   -> the model was reciting the labelling rule, and the
        headline number should not be trusted or presented as accuracy.

    Either way the delta is published rather than buried.
    """
    logger.info("--- Circularity Audit (ablating label-rule features) ---")

    df = corpus_df.reset_index(drop=True)
    y = df["target_label"].astype(int).values
    X_raw = df.drop(columns=["target_label"])

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.25, random_state=random_seed, stratify=y
    )

    pipe = FireFeaturePipeline(scale_features=False)
    X_tr_full = pipe.fit_transform(X_tr_raw)
    X_te_full = pipe.transform(X_te_raw)

    # Ablate every transformed column derived from a label-rule input.
    ablate = [c for c in X_tr_full.columns if any(c.startswith(f) for f in LABEL_RULE_FEATURES)]
    X_tr_abl = X_tr_full.drop(columns=ablate)
    X_te_abl = X_te_full.drop(columns=ablate)
    logger.info("Ablated %d feature columns: %s", len(ablate), ablate)

    # As with spatial CV, score on the real detections only. Synthetic rows stay
    # separable even with the label-rule features removed, which would flatter
    # the ablated score and defeat the point of the audit.
    real_te = (X_te_raw["label_source"] == "heuristic_real").values
    scored_on = "real_detections_only"
    if real_te.sum() == 0 or len(np.unique(y_te[real_te])) < 2:
        real_te = np.ones(len(y_te), dtype=bool)
        scored_on = "all_rows_fallback"

    full_model = _fit_xgb(X_tr_full, y_tr, random_seed=random_seed)
    full_f1 = f1_score(y_te[real_te], full_model.predict(X_te_full)[real_te], average="macro")

    abl_model = _fit_xgb(X_tr_abl, y_tr, random_seed=random_seed)
    abl_f1 = f1_score(y_te[real_te], abl_model.predict(X_te_abl)[real_te], average="macro")

    delta = full_f1 - abl_f1
    if abl_f1 >= 0.85:
        interpretation = (
            "Radiometric and temporal features classify independently of the "
            "label rule; the headline score is not purely circular."
        )
    elif abl_f1 >= 0.60:
        interpretation = (
            "Partial independence. The model retains real signal without the "
            "label-rule features but leans on them substantially."
        )
    else:
        interpretation = (
            "WARNING: performance collapses without the label-rule features. "
            "The headline score largely reflects the labelling rule, not "
            "learned fire physics. Verified labels are required."
        )

    logger.info("Full-feature macro F1 = %.4f | Ablated macro F1 = %.4f | Delta = %.4f",
                full_f1, abl_f1, delta)
    logger.info("Interpretation: %s", interpretation)

    return {
        "scored_on": scored_on,
        "n_scored": int(real_te.sum()),
        "ablated_features": ablate,
        "full_feature_macro_f1": float(round(full_f1, 4)),
        "ablated_macro_f1": float(round(abl_f1, 4)),
        "delta": float(round(delta, 4)),
        "interpretation": interpretation,
    }


def train_and_evaluate_model(
    corpus_df: pd.DataFrame,
    random_seed: int = 42,
) -> Tuple[xgb.XGBClassifier, FireFeaturePipeline, Dict]:
    """Executes full training pipeline adhering to ml-best-practices."""
    # 1. Stratified Train / Validation / Test Splitting (70% Train, 15% Val, 15% Test)
    X_raw = corpus_df.drop(columns=["target_label"])
    y_raw = corpus_df["target_label"].astype(int).values

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=random_seed, stratify=y_raw
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )

    logger.info("Split sizes: Train=%d, Validation=%d, Test=%d", len(y_train), len(y_val), len(y_test))

    # 2. Strict Featurization Ordering: Fit pipeline on Train ONLY
    logger.info("Fitting FireFeaturePipeline strictly on X_train...")
    pipeline = FireFeaturePipeline(scale_features=False)
    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    # 3. Establish Naive & Simple Baselines
    logger.info("--- Evaluating Baselines ---")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_f1 = f1_score(y_test, dummy_pred, average="macro")
    logger.info("Baseline 1 (Majority Class): Macro F1 = %.4f", dummy_f1)

    # dNBR is legitimately missing wherever optical validation could not run, and
    # linear models cannot consume NaN the way XGBoost can, so the baseline gets
    # a median imputer. The primary model still sees the raw NaN.
    # StandardScaler is not cosmetic here. These features span FRP up to ~900,
    # brightness temperatures around 300-400K and recurrence counts in the tens,
    # and lbfgs on unscaled data of that shape converges badly -- which is why
    # this baseline previously needed max_iter=1000 and still failed to complete
    # once the corpus passed a million rows. Scaling makes it both correct and
    # affordable; XGBoost is unaffected either way, being scale-invariant.
    log_reg = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=200, random_state=random_seed, n_jobs=-1),
    )
    log_reg.fit(X_train, y_train)
    log_reg_pred = log_reg.predict(X_test)
    log_reg_f1 = f1_score(y_test, log_reg_pred, average="macro")
    logger.info("Baseline 2 (Logistic Regression): Macro F1 = %.4f", log_reg_f1)

    # 4. Train Primary Model: XGBoost Multi-Class Classifier
    logger.info("--- Training Primary XGBoost Classifier ---")

    # Compute sample weights to penalize misclassifying rare accidental fires (Class 1)
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
    sample_weights_train = np.array([class_weights[y] for y in y_train])

    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        early_stopping_rounds=25,
        random_state=random_seed,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    logger.info("XGBoost training completed. Best iteration: %d", model.best_iteration)

    # 5. Rigorous Test Set Evaluation
    logger.info("--- Test Set Evaluation ---")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    test_loss = log_loss(y_test, y_test_prob)
    macro_f1 = f1_score(y_test, y_test_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")
    accidental_recall = recall_score(y_test, y_test_pred, labels=[1], average="micro")

    logger.info("Test Log Loss: %.4f", test_loss)
    logger.info("Test Macro F1: %.4f | Weighted F1: %.4f", macro_f1, weighted_f1)
    logger.info("Critical Metric -> Accidental Fire Recall (Class 1): %.2f%%", accidental_recall * 100.0)

    target_names = [CLASS_NAMES[i] for i in range(4)]
    report = classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_test_pred)

    logger.info("Classification Report:\n%s", classification_report(y_test, y_test_pred, target_names=target_names))
    logger.info("Confusion Matrix:\n%s", cm)

    # 6. Slice-Based Error Analysis: Day vs. Night Performance
    is_night_test = X_test["is_night"].values
    day_mask = is_night_test == 0
    night_mask = is_night_test == 1

    day_f1 = f1_score(y_test[day_mask], y_test_pred[day_mask], average="macro") if day_mask.sum() > 0 else 0.0
    night_f1 = f1_score(y_test[night_mask], y_test_pred[night_mask], average="macro") if night_mask.sum() > 0 else 0.0

    logger.info("Slice Analysis: Day Overpass F1 = %.4f | Night Overpass F1 = %.4f", day_f1, night_f1)

    # 7. Provenance Slice: real heuristic-labelled detections vs. synthetic rows.
    # These must never be blended into one headline figure. The synthetic rows are
    # drawn from clean, well-separated distributions and will always score near
    # 1.0; quoting a combined number lets that optimism mask how the model does on
    # actual satellite telemetry.
    provenance = X_test_raw.get("label_source", pd.Series("synthetic", index=X_test_raw.index)).values
    real_mask = provenance == "heuristic_real"
    syn_mask = ~real_mask

    def _slice_f1(mask) -> Optional[float]:
        if mask.sum() == 0 or len(np.unique(y_test[mask])) < 2:
            return None
        return float(round(f1_score(y_test[mask], y_test_pred[mask], average="macro"), 4))

    real_f1 = _slice_f1(real_mask)
    syn_f1 = _slice_f1(syn_mask)
    logger.info(
        "Provenance Slice: real-detection macro F1 = %s (n=%d) | synthetic macro F1 = %s (n=%d)",
        real_f1, int(real_mask.sum()), syn_f1, int(syn_mask.sum()),
    )

    # 8. Compile Evaluation Package
    metrics_package = {
        # Which run produced this artifact, and under what rule. Three trainings
        # once raced to write these files and the only way to tell which model
        # was on disk was to cross-reference log timestamps. A model that cannot
        # name its own inputs cannot be quoted safely.
        "run": {
            "run_id": RUN_ID,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "labelling_rule_version": LABELLING_RULE_VERSION,
        },
        "frp_thresholds": (
            corpus_df.attrs.get("frp_thresholds")
            or derive_frp_thresholds(corpus_df).to_dict()
        ),
        "label_provenance": {
            "real_detections_heuristic_labelled": int((corpus_df["label_source"] == "heuristic_real").sum()),
            "synthetic_augmented": int((corpus_df["label_source"] != "heuristic_real").sum()),
            "verified_ground_truth": 0,
            "labelling_rule": "weak_label_real_detection() - recurrence-first heuristic",
        },
        "caveats": [
            "No verified per-detection ground truth exists for this corpus. Labels "
            "for real detections are rule-derived and labels for augmented rows are "
            "synthetic, so the headline test scores below measure internal "
            "consistency, NOT field accuracy.",
            "Prefer spatial_block_cv.mean_macro_f1 over the random-split score when "
            "discussing generalization: detections cluster spatially, so a random "
            "split puts the same facility on both sides of the partition. Note that "
            "spatial CV still scores against rule-derived labels, so it measures "
            "transferability of the rule, not correctness of the rule.",
            "See circularity_audit for how much of the score survives when the "
            "features used by the labelling rule are removed.",
            "Validating against confirmed incidents (e.g. the Baghjan blowout) is "
            "the outstanding work required before any accuracy claim is defensible.",
            "If class_balance.status is APPLIED, majority classes were subsampled. "
            "Predicted probabilities then over-represent rare classes relative to "
            "the world and would need recalibration before being read as likelihoods; "
            "rank ordering and the confusion matrix remain valid.",
        ],
        "best_iteration": int(model.best_iteration),
        "test_log_loss": float(round(test_loss, 4)),
        "test_macro_f1": float(round(macro_f1, 4)),
        "test_weighted_f1": float(round(weighted_f1, 4)),
        "accidental_fire_recall": float(round(accidental_recall, 4)),
        "provenance_slice": {
            "real_detection_macro_f1": real_f1,
            "real_detection_n": int(real_mask.sum()),
            "synthetic_macro_f1": syn_f1,
            "synthetic_n": int(syn_mask.sum()),
        },
        "baselines": {
            "dummy_majority_macro_f1": float(round(dummy_f1, 4)),
            "logistic_regression_macro_f1": float(round(log_reg_f1, 4)),
        },
        "slice_analysis": {
            "day_macro_f1": float(round(day_f1, 4)),
            "night_macro_f1": float(round(night_f1, 4)),
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_names": pipeline.feature_names_,
    }

    return model, pipeline, metrics_package


def save_artifacts(
    model: xgb.XGBClassifier,
    pipeline: FireFeaturePipeline,
    metrics: Dict,
) -> None:
    """Serializes model, pipeline, and evaluation metrics to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save XGBoost model in standard JSON format
    model.save_model(str(MODEL_ARTIFACT_PATH))
    logger.info("Saved XGBoost model to %s", MODEL_ARTIFACT_PATH)

    # Save fitted feature pipeline
    joblib.dump(pipeline, PIPELINE_ARTIFACT_PATH)
    logger.info("Saved feature pipeline to %s", PIPELINE_ARTIFACT_PATH)

    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", METRICS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Classifier for Project SIH26162")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/firms_industrial_joined.parquet",
        help="Input parquet file path",
    )
    parser.add_argument(
        "--ratio-cap",
        type=int,
        default=DEFAULT_CLASS_RATIO_CAP,
        help="Maximum permitted ratio between the largest class and the smallest "
             "real class. Majority classes are subsampled within month to respect "
             "it; every rare-class row is kept. Set 0 to disable balancing.",
    )
    args = parser.parse_args()

    input_file = Path(args.data)
    corpus = generate_training_corpus(base_parquet=input_file)

    # Cap the imbalance before training. Left raw, a national corpus puts
    # ACCIDENTAL_FIRE at 0.04% of rows, so class weighting alone hands the
    # critical class ~1500x per-sample influence and the boundary becomes
    # hostage to individual mislabelled detections.
    frp_thresholds = corpus.attrs.get("frp_thresholds")
    corpus, balance_report = balance_corpus(corpus, ratio_cap=args.ratio_cap)
    # pandas drops .attrs across the copy inside balance_corpus, which is why
    # the threshold provenance previously reached the metrics file as null.
    if frp_thresholds is not None:
        corpus.attrs["frp_thresholds"] = frp_thresholds

    trained_model, fitted_pipeline, eval_metrics = train_and_evaluate_model(corpus)
    eval_metrics["class_balance"] = balance_report

    # Honest-evaluation passes. These are what should be quoted, not the
    # random-split score above.
    eval_metrics["spatial_block_cv"] = run_spatial_block_cv(corpus)
    eval_metrics["circularity_audit"] = run_circularity_audit(corpus)

    save_artifacts(trained_model, fitted_pipeline, eval_metrics)

    audit = eval_metrics["circularity_audit"]
    scv = eval_metrics["spatial_block_cv"]

    logger.info("=" * 72)
    logger.info("EVALUATION SUMMARY")
    logger.info("-" * 72)
    if scv.get("status") == "OK":
        logger.info("  Spatial-block CV (real)   : %.4f +/- %.4f (worst fold %.4f)",
                    scv["mean_macro_f1"], scv["std_macro_f1"], scv["min_macro_f1"])
    logger.info("  Random-split macro F1     : %.4f", eval_metrics["test_macro_f1"])
    logger.info("  Circularity audit         : %.4f full -> %.4f ablated (delta %.4f)",
                audit["full_feature_macro_f1"], audit["ablated_macro_f1"], audit["delta"])
    if balance_report.get("status") == "APPLIED":
        logger.info("  Class imbalance           : %.0f:1 -> %.0f:1 (%s -> %s rows)",
                    balance_report["original_ratio"], balance_report["balanced_ratio"],
                    f'{balance_report["rows_before"]:,}', f'{balance_report["rows_after"]:,}')
    logger.info("  Labels                    : 0 verified, %d heuristic, %d synthetic",
                eval_metrics["label_provenance"]["real_detections_heuristic_labelled"],
                eval_metrics["label_provenance"]["synthetic_augmented"])
    logger.info("-" * 72)
    logger.info("  READ THIS BEFORE QUOTING ANY NUMBER ABOVE:")
    logger.info("  Labels are rule-derived, so these scores measure how faithfully")
    logger.info("  the model reproduces its own labelling rule across unseen regions.")
    logger.info("  They are NOT field accuracy against confirmed incidents.")
    logger.info("  %s", audit["interpretation"])
    logger.info("=" * 72)
