"""Verified reference labels and an evaluation harness that uses only them.

WHY THIS EXISTS
---------------
Every accuracy figure this project has produced so far is computed against labels
the project generated itself. `weak_label_real_detection()` derives a class from
`inside_industrial`, `is_exact_match`, `n_30d` and `frp`, and the model is then
scored on how well it reproduces that rule. The circularity audit quantifies the
problem directly: with the label-rule features ablated, macro F1 falls from
0.998 to 0.819. That residual gap does not close with more data. It closes with
ground truth.

This module is the place ground truth goes.

WHAT COUNTS AS A VERIFIED LABEL
-------------------------------
An entry here asserts that a specific class of thermal event occurred at a
specific place and time, and it must carry a `source` recording how that was
established -- an incident report, a regulatory filing, a news record, an
operator disclosure, or direct analyst adjudication of imagery.

Do NOT add entries inferred from the model, from FIRMS recurrence, or from the
weak-labelling rule. A label derived from the thing being evaluated is not
evidence about it, and silently mixing such entries in would make this harness
report the same circularity it exists to detect.

If you cannot cite how you know, it does not belong here.

THE SEED SET IS DELIBERATELY SMALL
----------------------------------
Only events that can be stated with confidence are seeded below, and each is
marked with its confidence level. This is not a limitation to work around by
padding the list -- a handful of genuinely verified labels is worth more than
hundreds of plausible-looking guesses, because the entire purpose is to be
independent of guesswork.

Extend it either in VERIFIED_EVENTS or, more conveniently for non-developers,
via data/reference/verified_events.csv (same columns, loaded automatically).
"""

import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("verified_labels")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CSV_PATH = Path("data/reference/verified_events.csv")

# Mirrors CLASS_NAMES in train_classifier; duplicated as names rather than
# indices so a CSV written by hand stays readable and order changes cannot
# silently re-map labels.
CLASS_NAME_TO_INDEX = {
    "PERSISTENT_BASELINE": 0,
    "ACCIDENTAL_FIRE": 1,
    "AGRICULTURAL_BURN": 2,
    "TRANSIENT_HOTSPOT": 3,
}

# Classes the deployed system can report but the model can never predict,
# because they are decided outside the feature vector. FOREST_FIRE comes from
# land cover (see pipeline/forest_cover.py): the model is coordinate-free on
# purpose, so it cannot tell a forest fire from a crop fire, and it is not asked
# to. An event carrying one of these has a `label` for the model and a
# `served_label` for the system, and they are legitimately different.
SERVED_ONLY_CLASSES = frozenset({"FOREST_FIRE"})


@dataclass
class VerifiedEvent:
    """One adjudicated space-time window with a known thermal-event class."""

    name: str
    lat: float
    lon: float
    start_date: str          # inclusive, YYYY-MM-DD
    end_date: str            # inclusive, YYYY-MM-DD
    label: str               # a key of CLASS_NAME_TO_INDEX
    confidence: str          # "high" | "medium"
    source: str              # how this was established -- required
    radius_km: float = 3.0
    notes: str = ""
    # What the *deployed system* should report, when that differs from what the
    # model should predict. Defaults to `label`, so every event written before
    # this existed keeps exactly the meaning it had.
    served_label: str = ""

    def __post_init__(self):
        # `label` first. It is the field an author writes, and `served_label`
        # defaults to it -- so checking the derived field first reports a
        # mistyped label under the wrong name.
        if self.label not in CLASS_NAME_TO_INDEX:
            raise ValueError(
                f"{self.name}: unknown label {self.label!r}. "
                f"Expected one of {sorted(CLASS_NAME_TO_INDEX)}"
            )

        if not self.served_label:
            self.served_label = self.label
        if self.served_label not in CLASS_NAME_TO_INDEX and \
                self.served_label not in SERVED_ONLY_CLASSES:
            raise ValueError(
                f"{self.name}: unknown served_label {self.served_label!r}. "
                f"Expected a model class or one of {sorted(SERVED_ONLY_CLASSES)}"
            )
        if not str(self.source).strip():
            raise ValueError(
                f"{self.name}: a verified label must record how it was established. "
                "Set `source` to a citation, report reference, or adjudication note."
            )
        if self.confidence not in {"high", "medium"}:
            raise ValueError(f"{self.name}: confidence must be 'high' or 'medium'.")

    @property
    def label_index(self) -> int:
        return CLASS_NAME_TO_INDEX[self.label]


# ---------------------------------------------------------------------------
# Seed set
#
# Each entry states what is known and how. Anything that could not be stated
# with confidence was left out rather than guessed at.
# ---------------------------------------------------------------------------

VERIFIED_EVENTS: List[VerifiedEvent] = [
    # ---- PERSISTENT_BASELINE: integrated steel, continuous by design ----
    VerifiedEvent(
        name="Visakhapatnam Steel Plant (RINL, Andhra Pradesh)",
        lat=17.6125,
        lon=83.1863,
        start_date="2024-01-01",
        end_date="2026-12-31",
        label="PERSISTENT_BASELINE",
        confidence="high",
        source=(
            "Verified by facility type and geometry rather than by incident: an "
            "integrated steel plant whose blast furnaces and coke ovens run "
            "continuously by design. Geometry is the centroid of the 31.8 km2 "
            "'Visakapatnam Steel Plant' polygon in this project's own OSM/Bhuvan "
            "reference layer."
        ),
        radius_km=5.0,
        notes="Continuous metallurgical heat, not episodic combustion.",
    ),
    VerifiedEvent(
        name="JSW Vijayanagar Works (Jindal Steel, Karnataka)",
        lat=15.1805,
        lon=76.6684,
        start_date="2024-01-01",
        end_date="2026-12-31",
        label="PERSISTENT_BASELINE",
        confidence="high",
        source=(
            "Verified by facility type and geometry: an integrated steel works "
            "operating continuous blast furnaces. Geometry is the centroid of the "
            "21.2 km2 'Jindal Steel Works' polygon in the reference layer."
        ),
        radius_km=4.0,
        notes="Second steel anchor, geographically distant from Visakhapatnam.",
    ),

    # ---- TRANSIENT_HOTSPOT: reflective artifacts, no combustion possible ----
    #
    # Solar parks are the textbook sun-glint false positive the problem statement
    # research calls out: "highly reflective surfaces, such as the expansive
    # metallic rooftops of large factory complexes or solar arrays, can reflect
    # intense solar radiation directly into the sensor optics". A photovoltaic
    # farm contains no combustion process at all, so a thermal anomaly inside one
    # is an artifact unless something is genuinely alight.
    #
    # Confidence is "medium", not "high", precisely because that caveat is real:
    # scrub under the panels or an equipment fire would be a true detection. The
    # tight radii keep the window well inside the array rather than sweeping in
    # surrounding farmland.
    VerifiedEvent(
        name="Khavda Renewable Energy Park (Gujarat) - reflective artifact",
        lat=24.1191,
        lon=69.3708,
        start_date="2024-01-01",
        end_date="2026-12-31",
        label="TRANSIENT_HOTSPOT",
        confidence="medium",
        source=(
            "Verified by land use: a photovoltaic and wind generation park with no "
            "combustion process of any kind. Geometry from the 96.6 km2 'Khavda "
            "Renewable Energy Park' polygon in the reference layer. Thermal "
            "detections here cannot be industrial combustion and are consistent "
            "with specular reflection off the array."
        ),
        radius_km=4.0,
        notes=(
            "Negative control for sun glint. The polygon is tagged industrial in "
            "OSM, so detections here are currently credited with facility "
            "recurrence and reported as a persistent thermal source -- which is "
            "precisely the false positive this label is meant to expose."
        ),
    ),
    VerifiedEvent(
        name="Pavagada Solar Park (Karnataka) - reflective artifact",
        lat=14.2622,
        lon=77.4315,
        start_date="2024-01-01",
        end_date="2026-12-31",
        label="TRANSIENT_HOTSPOT",
        confidence="medium",
        source=(
            "Verified by land use: one of the largest photovoltaic installations "
            "in India, with no combustion process. Geometry from the 44.3 km2 "
            "'Pavagada Solar Park' polygon in the reference layer."
        ),
        radius_km=3.0,
        notes="Second glint control, geographically distant from Khavda.",
    ),

    # ---- AGRICULTURAL_BURN: documented seasonal residue burning ----
    #
    # These are region-and-season labels rather than point incidents. Punjab's
    # crop residue burning is an extensively documented, regulated and litigated
    # annual phenomenon with two distinct windows: post-Kharif paddy in
    # October-November, and post-Rabi wheat in April-May.
    #
    # Confidence is "medium" because asserting the phenomenon recurred in one
    # specific year is an inference from its documented annual character rather
    # than a verified observation of that year. The corroborating detail is that
    # the corpus shows exactly the expected bimodal concentration in this belt --
    # 1,568 detections in May and 361 in November against 1 to 7 per month from
    # December through March.
    VerifiedEvent(
        name="Punjab post-Kharif paddy residue burning (Oct-Nov window)",
        lat=30.500,
        lon=75.800,
        start_date="2025-10-05",
        end_date="2025-11-30",
        label="AGRICULTURAL_BURN",
        confidence="medium",
        source=(
            "Documented annual seasonal phenomenon: large-scale paddy stubble "
            "burning across the Punjab plain following the Kharif harvest. "
            "Independently corroborated for this project by a Sentinel-2 dNBR of "
            "+0.2775 (MODERATE_LOW burn severity) measured at this location from "
            "live Copernicus imagery."
        ),
        radius_km=25.0,
        notes="Region-and-season label, hence the wide radius.",
    ),
    VerifiedEvent(
        name="Punjab post-Rabi wheat residue burning (Apr-May window)",
        lat=30.500,
        lon=75.800,
        start_date="2026-04-10",
        end_date="2026-05-31",
        label="AGRICULTURAL_BURN",
        confidence="medium",
        source=(
            "Documented annual seasonal phenomenon: wheat residue burning "
            "following the Rabi harvest. This is the larger of the two Punjab "
            "windows in the corpus (1,568 detections in May against 361 in "
            "November), and the month originally omitted from the harvest-season "
            "feature."
        ),
        radius_km=25.0,
        notes=(
            "Adding this window is what exposed is_harvest_season excluding May "
            "despite May carrying 10.8% of national open-ground burn detections."
        ),
    ),


    VerifiedEvent(
        name="Jaipur IOC oil depot fire (Sitapura, Rajasthan)",
        # Median of the 33 MODIS detections the fire itself produced, not a
        # recalled address. The depot was rebuilt after 2009 and the current OSM
        # extract carries no polygon for the facility as it then stood, so the
        # detections are the only geometry that can be trusted here.
        lat=26.7810,
        lon=75.8373,
        start_date="2009-10-29",
        end_date="2009-11-09",
        label="ACCIDENTAL_FIRE",
        confidence="high",
        source=(
            "Widely documented industrial disaster: fuel released during a "
            "pipeline transfer at the Indian Oil Corporation POL terminal, "
            "Sitapura Industrial Area, Jaipur, ignited on 2009-10-29 and burned "
            "out of control for eleven days. Twelve people were killed, more "
            "than 300 injured, and roughly half a million evacuated. Recorded in "
            "the FABIG industrial-accident database "
            "(fabig.com/industrial-accidents/jaipur-oil-depot-india) and imaged "
            "from orbit by NASA Earth Observatory."
        ),
        radius_km=5.0,
        notes=(
            "The second verified ACCIDENTAL_FIRE, and the reason it is usable "
            "where three documented 2026 fires were not: it burned for eleven "
            "days, so it straddles many overpasses. The onset is unambiguous -- "
            "33 detections inside 5km between 2009-10-29 and 2009-11-04, peaking "
            "at 230.8 MW, against ZERO in the four weeks before ignition. "
            "MODIS only; VIIRS did not launch until 2012."
        ),
    ),
    VerifiedEvent(
        name="Baghjan gas well blowout (Oil India, Tinsukia, Assam)",
        lat=27.587,
        lon=95.385,
        start_date="2020-06-09",
        end_date="2020-11-15",
        label="ACCIDENTAL_FIRE",
        confidence="high",
        source=(
            "Widely documented industrial disaster: Oil India Limited well "
            "Baghjan-5 blew out on 2020-05-27 and ignited on 2020-06-09, burning "
            "until it was capped in November 2020."
        ),
        radius_km=3.0,
        notes=(
            "Sustained months-long accidental industrial fire. Optically hard to "
            "validate via dNBR because the burn period sits inside the Assam "
            "monsoon -- see sentinel2_client validation notes."
        ),
    ),
    VerifiedEvent(
        name="Reliance Jamnagar refinery complex, routine flaring (Gujarat)",
        # Coordinates are the centroid of the 32 km2 "Reliance Refinery" polygon
        # in data/reference/industrial_boundaries_merged.parquet, NOT a recalled
        # figure. An earlier hand-entered coordinate was ~20km off and matched
        # zero detections, which is exactly how a verified label quietly becomes
        # worthless -- so these are taken from the reference geometry.
        lat=22.3380,
        lon=69.8686,
        start_date="2024-01-01",
        end_date="2026-12-31",
        label="PERSISTENT_BASELINE",
        confidence="high",
        source=(
            "Verified by facility type rather than by incident: one of the world's "
            "largest petroleum refining complexes, where continuous flare-stack "
            "operation is a designed, permanent feature. Geometry confirmed by the "
            "OSM industrial polygon in this project's own reference layer. Thermal "
            "behaviour confirmed independently by Sentinel-2 dNBR of +0.0345 "
            "(UNBURNED) at this complex, i.e. intense thermal activity with no "
            "biomass consumption."
        ),
        radius_km=6.0,
        notes=(
            "A standing negative control. Any detection here classified "
            "ACCIDENTAL_FIRE is a false positive of exactly the kind that causes "
            "alert fatigue."
        ),
    ),
    VerifiedEvent(
        name="Punjab post-Kharif paddy residue burning",
        lat=30.500,
        lon=75.800,
        start_date="2023-10-15",
        end_date="2023-11-30",
        label="AGRICULTURAL_BURN",
        confidence="high",
        source=(
            "Verified as a seasonal regional phenomenon rather than a single "
            "incident: large-scale paddy stubble burning across Punjab in the "
            "post-Kharif window is extensively documented and regulated. "
            "Confirmed independently by Sentinel-2 dNBR of +0.2775 "
            "(MODERATE_LOW burn severity) on 2023-11-05."
        ),
        radius_km=25.0,
        notes=(
            "Region-and-season label, not a point incident, hence the wide radius. "
            "Both dNBR measurements cited above were produced by this project's own "
            "optical validation against live Copernicus data."
        ),
    ),

    # ---- ACCIDENTAL_FIRE: found by searching for *duration*, not for fame ----
    #
    # The 2009 Jaipur depot fire established the method. Public reporting cannot
    # tell you whether a fire was detectable, but a fire with a documented
    # multi-day duration necessarily straddled overpasses. Searching recent
    # fires kept failing for one reason: a modern plant fire is extinguished in
    # hours and falls between them. So the search key is the duration, and the
    # archives reach back to 2000.
    VerifiedEvent(
        name="Deonar landfill fire (Mumbai, Maharashtra)",
        lat=19.0720,
        lon=72.9290,
        start_date="2016-01-27",
        end_date="2016-01-31",
        label="ACCIDENTAL_FIRE",
        confidence="high",
        source=(
            "NASA Earth Observatory records that sensors on Terra, Aqua and Suomi "
            "NPP began detecting smoke and fire at the 132-hectare Deonar dumping "
            "ground on 2016-01-27 and that it burned for four days, closing more "
            "than 70 schools and pushing Mumbai air quality to the worst level "
            "recorded since monitoring began in June 2015. Detectability is "
            "therefore stated by the observing agency rather than inferred: "
            "earthobservatory.nasa.gov/images/87429."
        ),
        radius_km=2.0,
        notes=(
            "29 detections over 5 days inside 2 km, against 7 in the preceding "
            "3.5 months -- a clean onset with no prior baseline. The centre is the "
            "detection centroid, about 2 km north of the site's nominal "
            "coordinates; using the nominal point would have placed the fire at "
            "the edge of its own window."
        ),
    ),
    VerifiedEvent(
        name="Brahmapuram waste plant fire (Kochi, Kerala)",
        lat=9.9925,
        lon=76.3650,
        start_date="2023-03-02",
        end_date="2023-03-14",
        label="ACCIDENTAL_FIRE",
        confidence="high",
        source=(
            "Widely documented disaster with an official response record: garbage "
            "piles across roughly 40 acres ignited on 2023-03-02 and were declared "
            "doused eleven days later after 23 fire engines, 32 excavators, four "
            "helicopters and more than 200 firefighters were committed; smoke was "
            "still rising at twelve days. Kochi AQI exceeded 320 and the Kerala "
            "state health service opened a dedicated incident page."
        ),
        radius_km=2.0,
        notes=(
            "Deliberately the hardest accidental-fire case in the set. Unlike "
            "Deonar this site was already smouldering: 37 detections inside 2 km "
            "in the four months before ignition, against 30 over 7 days during it. "
            "A site with a chronic baseline that then suffers a catastrophic "
            "accident is the exact failure mode the recurrence counter creates, "
            "and nothing else in the verified set tests it."
        ),
    ),

    VerifiedEvent(
        name="ITC Deer Park terminal fire (Harris County, Texas, USA)",
        lat=29.7320,
        lon=-95.0920,
        start_date="2019-03-17",
        end_date="2019-03-23",
        label="ACCIDENTAL_FIRE",
        confidence="high",
        source=(
            "US Chemical Safety Board final investigation report (2023) into the "
            "Intercontinental Terminals Company tank-farm fire that ignited "
            "2019-03-17, burned for three days, reignited on 2019-03-22 and "
            "breached a second containment wall. Over $150M in facility damage "
            "and a $6.6M natural-resource-damage settlement with the Texas "
            "Attorney General and US DOJ. A regulator's own findings, which is "
            "the class of source this project could not previously obtain."
        ),
        radius_km=1.5,
        notes=(
            "Deliberately outside India. Every other verified event sits inside "
            "this project's Indian OSM extract, so `inside_industrial` has never "
            "once been structurally False for a real, operating facility -- the "
            "model has never been scored on a documented fire where the map layer "
            "is simply absent. The E3 transferability work measured that against "
            "rule labels; this measures it against a regulator's."
        ),
    ),

    # ---- PERSISTENT_BASELINE: coal-seam fires, a different physical source ----
    #
    # Every persistent event above is a furnace or a flare -- engineered
    # combustion inside a facility, which is also how the labelling rule thinks
    # about persistence. A coal-seam fire is none of those things: no operator,
    # no process, no polygon drawn around a burner, and it has been alight for
    # over a century. If the rule only recognises persistence that looks
    # industrial, these are where it shows.
    VerifiedEvent(
        name="Jharia coalfield seam fires (Dhanbad, Jharkhand)",
        lat=23.7500,
        lon=86.4200,
        start_date="2025-09-13",
        end_date="2026-09-12",
        label="PERSISTENT_BASELINE",
        confidence="high",
        source=(
            "Established by a peer-reviewed remote-sensing literature independent "
            "of this project: subsurface and surface coal fires at Jharia have "
            "burned since first recorded in 1916, and thermal-anomaly mapping from "
            "Landsat TIR has tracked their extent at five-year intervals from 1988 "
            "to 2013, with around 70% of the coalfield's mines affected by surface "
            "or subsurface fire. The class follows from that documented fact -- a "
            "source alight every day for a century must never page an incident "
            "commander -- and not from this project's recurrence counter."
        ),
        radius_km=10.0,
        notes=(
            "23,633 detections on 326 of 365 days in the 12-month corpus. 58% of "
            "them fall OUTSIDE any mapped industrial polygon, so this is also the "
            "only persistent event in the set that the `inside_industrial` feature "
            "cannot explain."
        ),
    ),
    # ---- Outside India: the classes that had never been checked abroad ----
    #
    # Fourteen of the first fifteen events were Indian, and the one that was not
    # -- the ITC Deer Park tank fire -- is an accidental fire. So the persistent
    # and artifact classes had never once been scored against a cited label
    # outside the region the entire system was calibrated in.
    VerifiedEvent(
        name="Buncefield oil depot fire (Hemel Hempstead, United Kingdom)",
        lat=51.7666,
        lon=-0.4255,
        start_date="2005-12-11",
        end_date="2005-12-15",
        label="ACCIDENTAL_FIRE",
        confidence="high",
        source=(
            "UK Health and Safety Executive investigation into the explosion and "
            "fire at the Buncefield oil storage depot, which ignited at about "
            "06:00 on 2005-12-11, injured over 40 people, forced 2,000 "
            "evacuations and produced the largest peacetime fire in Europe to "
            "that date. Flames were largely extinguished by the afternoon of "
            "13 December, with one tank reigniting and left to burn."
        ),
        radius_km=3.0,
        notes=(
            "Added expecting it to FAIL the detectability test, and it passed. "
            "December in the UK, MODIS only -- VIIRS did not exist in 2005 -- "
            "and a dense black plume that could plausibly have masked the "
            "thermal signal from above. Terra caught four detections at 21:25 "
            "on the 11th (FRP 20.8-57.0, confidence 73-100) and Aqua one at "
            "01:28 on the 12th, with zero detections anywhere in the 10 km "
            "window during the preceding three months. The fire burned four to "
            "five days and produced two days of detections, which is its own "
            "quiet measurement of what the plume cost."
        ),
    ),
    VerifiedEvent(
        name="Rumaila oil field flaring (Basra, Iraq)",
        lat=30.1500,
        lon=47.3500,
        start_date="2026-03-01",
        end_date="2026-05-31",
        label="PERSISTENT_BASELINE",
        confidence="medium",
        source=(
            "World Bank satellite flaring data records Rumaila flaring 3.39 "
            "billion cubic metres of gas in a year, around 9.5 Mt CO2e. Iraq "
            "publishes no official flaring record, which is precisely why the "
            "World Bank tracks it from orbit -- making this a case where the "
            "independent evidence for the class is itself remote sensing, by a "
            "different programme, at a different cadence."
        ),
        radius_km=30.0,
        notes=(
            "Confidence is medium, and the reason is a measurement rather than "
            "caution. The field is roughly 80 km long, so this is a field label "
            "rather than a point one -- and even at 30 km it appears on only "
            "30% of days, against Jamnagar's 51% at a 3 km radius and Jharia's "
            "89%. Median FRP is 3.5 MW, close to the detection floor. The "
            "documented ground truth is continuous flaring; FIRMS sees it "
            "intermittently. That gap is a property of the sensor, not of the "
            "label, and it is the first non-Indian persistent source the "
            "classifier has ever been scored against."
        ),
    ),
    VerifiedEvent(
        name="Benban Solar Park (Aswan, Egypt) - reflective artifact",
        lat=24.4500,
        lon=32.7500,
        start_date="2026-03-05",
        end_date="2026-05-31",
        label="TRANSIENT_HOTSPOT",
        confidence="medium",
        source=(
            "Verified by land use: one of the largest photovoltaic complexes in "
            "the world, built on open desert, with no combustion process of any "
            "kind. Thermal detections inside it cannot be industrial combustion "
            "and are consistent with specular reflection off the array -- the "
            "same argument that applies to Khavda and Pavagada, tested for the "
            "first time outside India."
        ),
        radius_km=6.0,
        notes=(
            "Thin: 16 detections on 4 distinct days over three months, against "
            "100 at Khavda and 53 at Pavagada. A desert array in Egypt produces "
            "far fewer spurious detections than the Indian parks do, which is "
            "worth knowing in itself. Scored as a negative control -- any "
            "detection here classified ACCIDENTAL_FIRE is a false positive of "
            "exactly the kind that causes alert fatigue."
        ),
    ),

    VerifiedEvent(
        name="Raniganj coalfield seam fires (Paschim Bardhaman, West Bengal)",
        lat=23.6200,
        lon=87.1300,
        start_date="2025-09-13",
        end_date="2026-09-12",
        label="PERSISTENT_BASELINE",
        confidence="high",
        source=(
            "India's oldest coalfield, with mine fires documented since 1906 and "
            "mapped repeatedly in the same thermal remote-sensing literature that "
            "covers Jharia. Included as a second, geographically separate "
            "coal-fire anchor so the class is not established by a single site."
        ),
        radius_km=10.0,
        notes="6,236 detections on 303 distinct days in the 12-month corpus.",
    ),
    # ---- FOREST_FIRE: the class the mandate names, served from land cover ----
    #
    # `label` is AGRICULTURAL_BURN on all three, and that is not a mistake. It is
    # what the *model* should say: sustained open-ground combustion with no
    # facility history is the finest distinction a coordinate-free feature vector
    # supports. `served_label` is FOREST_FIRE, which is what the deployed system
    # should report once land cover has been consulted.
    VerifiedEvent(
        name="Bandipur National Park forest fire (Chamarajanagar, Karnataka)",
        lat=11.6700,
        lon=76.6300,
        start_date="2019-02-21",
        end_date="2019-02-25",
        label="AGRICULTURAL_BURN",
        served_label="FOREST_FIRE",
        confidence="high",
        radius_km=15.0,
        source=(
            "Widely documented forest fire in Bandipur National Park, "
            "Chamarajanagar district, Karnataka, which burned from 2019-02-21 "
            "across several thousand hectares of the tiger reserve. The Indian "
            "Air Force deployed Mi-17 helicopters for aerial water drops and "
            "hundreds of forest staff and volunteers were mobilised before it "
            "was brought under control on 2019-02-25."
        ),
        notes=(
            "The cleanest of the three: 709 detections over 5 days against 14 in "
            "the preceding three weeks, peaking at 4,252 MW. 95% of sampled "
            "detections fall inside mapped forest cover, so the FOREST_FIRE "
            "determination has the land-cover evidence it needs."
        ),
    ),
    VerifiedEvent(
        name="Similipal Biosphere Reserve fires (Mayurbhanj, Odisha)",
        lat=21.8500,
        lon=86.3500,
        start_date="2021-02-20",
        end_date="2021-03-10",
        label="AGRICULTURAL_BURN",
        served_label="FOREST_FIRE",
        confidence="high",
        radius_km=25.0,
        source=(
            "Extensively reported fires across the Similipal Biosphere Reserve "
            "and tiger reserve, Mayurbhanj district, Odisha, burning through "
            "late February and early March 2021. The scale drew national "
            "coverage and a state response involving forest squads, fire "
            "watchers and community volunteers across hundreds of active points."
        ),
        notes=(
            "A diffuse, weeks-long event rather than a single front: 3,027 "
            "detections on 16 days, 97% inside mapped forest. Median FRP is only "
            "1.72 MW, which is what makes it valuable -- it sits near the "
            "artifact floor and exposes what that floor costs (5b.4f)."
        ),
    ),
    VerifiedEvent(
        name="Uttarakhand Himalayan forest fires (Garhwal and Kumaon)",
        lat=30.1000,
        lon=79.3000,
        start_date="2016-04-25",
        end_date="2016-05-10",
        label="AGRICULTURAL_BURN",
        served_label="FOREST_FIRE",
        confidence="medium",
        radius_km=30.0,
        source=(
            "The 2016 Uttarakhand forest fires, which burned across the Garhwal "
            "and Kumaon divisions through late April and early May 2016, "
            "affecting thousands of hectares of pine forest. The National "
            "Disaster Response Force was deployed alongside state forest "
            "personnel, and the scale and duration were the subject of "
            "proceedings before the Supreme Court of India."
        ),
        notes=(
            "Confidence is medium and the reason is measured rather than "
            "cautious. Detections precede the window inside the same radius, so "
            "there is no clean onset: fires were already burning through April. "
            "And only ~71% of sampled detections fall inside mapped forest, "
            "against 95-97% for Bandipur and Similipal. That is not a radius "
            "problem -- the forest share is flat at 70/73/71/74/67/74 percent "
            "across 20 to 50 km, so the ceiling is OSM forest coverage in "
            "Himalayan terrain. The radius was set to 30 km on that evidence: "
            "the wider label bought nothing but unrelated detections."
        ),
    ),
]


def load_verified_events(csv_path: Path = CSV_PATH) -> List[VerifiedEvent]:
    """Returns the seed events plus any defined in the CSV.

    The CSV exists so analysts can contribute labels without editing Python.
    Malformed rows are reported and skipped rather than silently dropped -- a
    lost verified label is expensive, since they are the scarce resource here.
    """
    events = list(VERIFIED_EVENTS)

    path = Path(csv_path)
    if not path.exists():
        logger.info("No %s found; using the %d seeded events only.", path, len(events))
        return events

    added, rejected = 0, 0
    with path.open(newline="", encoding="utf-8") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            if not row.get("name", "").strip():
                continue
            try:
                events.append(
                    VerifiedEvent(
                        name=row["name"].strip(),
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                        start_date=row["start_date"].strip(),
                        end_date=row["end_date"].strip(),
                        label=row["label"].strip().upper(),
                        confidence=row.get("confidence", "medium").strip().lower(),
                        source=row.get("source", "").strip(),
                        radius_km=float(row.get("radius_km") or 3.0),
                        notes=row.get("notes", "").strip(),
                        served_label=row.get("served_label", "").strip().upper(),
                    )
                )
                added += 1
            except (ValueError, KeyError) as e:
                rejected += 1
                logger.warning("%s line %d rejected: %s", path, line_no, e)

    logger.info("Loaded %d verified events (%d seeded, %d from CSV, %d rejected).",
                len(events), len(VERIFIED_EVENTS), added, rejected)
    return events


# ---------------------------------------------------------------------------
# Documented incidents the sensor could not see
# ---------------------------------------------------------------------------
#
# These are real, sourced industrial fires that produced NO usable FIRMS
# signature. They are deliberately NOT VerifiedEvents: there are no detections
# to label, so adding them to the evaluation set would inflate the event count
# without adding a single scorable row.
#
# They are recorded because they answer a question the verified set cannot:
# how often does a genuine industrial fire simply not reach the data? Every
# accuracy figure in this project is conditional on the event being visible to a
# polar-orbiting radiometer at the moment it passes overhead. These are the
# counter-examples, and they bound what the system can claim.


@dataclass(frozen=True)
class UndetectedIncident:
    """A documented industrial fire with no usable thermal signature."""

    name: str
    date: str
    lat: float
    lon: float
    source: str
    why_missed: str
    evidence: str


KNOWN_UNDETECTED_INCIDENTS: List[UndetectedIncident] = [
    UndetectedIncident(
        name="Haldia Petrochemicals naphtha pipeline fire (West Bengal)",
        date="2026-06-30",
        # Centroid of the 3.46 km2 "Haldia Petrochemicals Ltd." polygon in this
        # project's own OSM reference layer, not a recalled coordinate.
        lat=22.0669,
        lon=88.1130,
        source=(
            "Reported 2026-06-30: fire in a naphtha pipeline at the Haldia "
            "Petrochemicals facility, Purba Medinipur district, West Bengal, "
            "which spread to housing at Chiranjibpur and injured at least 20 "
            "people, five critically. Twelve fire tenders were deployed. "
            "Business Standard, "
            "business-standard.com/india-news/haldia-refinery-fire-naphtha-"
            "pipeline-west-bengal-purba-medinipur-126063000197_1.html"
        ),
        why_missed=(
            "Ignition was reported between 04:00 and 04:30 local time and the "
            "fire was fought down with 12 tenders. VIIRS overpasses the region "
            "at roughly 01:30 and 13:30 local, so the event began after the "
            "night pass and was suppressed before the afternoon one."
        ),
        evidence=(
            "Zero detections within 4 km on 2026-06-30. The nearest are "
            "2026-06-28 and 2026-07-02, all sub-3 MW routine flaring at "
            "neighbouring plants (Hooghly Met Coke, IOCL Refinery), correctly "
            "classified PERSISTENT_BASELINE."
        ),
    ),
    # --- no combustion at all -----------------------------------------------
    UndetectedIncident(
        name="LG Polymers styrene vapour release (Visakhapatnam, Andhra Pradesh)",
        date="2020-05-07",
        # Centroid of the 0.29 km2 "LG Polymers" polygon in this project's own
        # reference layer, not a recalled coordinate.
        lat=17.7564,
        lon=83.2101,
        source=(
            "Styrene vapour escaped from a storage tank at the LG Polymers "
            "India plant at RR Venkatapuram, Gopalapatnam, Visakhapatnam in the "
            "early hours of 2020-05-07, killing at least 11 people, "
            "hospitalising hundreds and forcing the evacuation of surrounding "
            "villages. The National Green Tribunal took suo motu cognisance the "
            "following day and directed an interim deposit of Rs 50 crore; the "
            "Government of Andhra Pradesh High Power Committee published its "
            "investigation report in July 2020."
        ),
        why_missed=(
            "There was nothing thermal to detect. The release was an unignited "
            "vapour cloud, not a fire: it killed by inhalation. This is the "
            "hardest bound in the register and it is categorical rather than "
            "circumstantial -- a sensor that measures radiative power cannot "
            "see a toxic release at ambient temperature, however many people it "
            "kills or however often the satellite passes. No cadence, no "
            "resolution and no additional band closes this gap."
        ),
        evidence=(
            "Zero detections within 4 km on 2020-05-07, and one in the whole "
            "43-day window around it (3.4 MW, 2.5 km away, three weeks before). "
            "The surrounding 100 km box carried 216 detections over 36 days, so "
            "the sensors were observing the region normally throughout."
        ),
    ),
    # --- burned entirely between two overpasses ------------------------------
    UndetectedIncident(
        name="ONGC Uran gas processing plant fire (Raigad, Maharashtra)",
        date="2018-09-03",
        # Not present in the OSM extract; this is the plant location used for
        # the retrieval. The zero below holds to 10 km, so a small error in the
        # coordinate does not explain the result.
        lat=18.8700,
        lon=72.9400,
        source=(
            "Fire at the ONGC Uran gas processing complex, Raigad district, "
            "Maharashtra, on the morning of 2018-09-03. Five people were "
            "killed, among them three CISF personnel who responded to it, and "
            "the fire was reported brought under control within roughly two "
            "hours."
        ),
        why_missed=(
            "The same mechanism as Haldia, which is why it is worth recording: "
            "a second instance makes the overpass gap a recurring property of "
            "the observing system rather than one unlucky fire. Ignition was "
            "around 07:00 local and the fire was out by roughly 09:00. VIIRS "
            "passes the region near 02:30 and 13:30 local, so the entire event "
            "opened and closed inside a single gap between passes."
        ),
        evidence=(
            "Zero detections within 10 km on 2018-09-03. The two nearest in the "
            "43-day window sit 1.9 km from the plant, on 2018-08-18 and "
            "2018-09-05, at 0.9 and 3.7 MW -- so the site is visible to the "
            "sensor on ordinary days and was simply not being looked at on this "
            "one."
        ),
    ),
    # --- industrial heat that never leaves the building ----------------------
    UndetectedIncident(
        name="NLC India Neyveli Thermal Power Station II boiler explosion (Tamil Nadu)",
        date="2020-07-01",
        # Centroid of the 3.16 km2 "Neyveli Thermal Power Station II" polygon in
        # this project's own reference layer.
        lat=11.5548,
        lon=79.4429,
        source=(
            "A boiler exploded at NLC India's Neyveli Thermal Power Station II, "
            "Cuddalore district, Tamil Nadu, on 2020-07-01, killing six "
            "workers. It followed an explosion at the same station on "
            "2020-05-07 in which eight people were injured."
        ),
        why_missed=(
            "A confined explosion inside a boiler house presents no sustained "
            "open flame for a radiometer to integrate. The wider finding here "
            "matters more than the incident: this lignite-fired station "
            "produced ZERO detections across the entire 43-day window at any "
            "radius out to 10 km. A station of this size is among the largest "
            "continuous combustion sources in the state, and it is thermally "
            "invisible to FIRMS, because its heat leaves through boilers and "
            "stacks rather than as radiating flame."
        ),
        evidence=(
            "Zero detections within 10 km across 2020-06-10 to 2020-07-22; the "
            "nearest detection anywhere in the pull is 13.3 km away. The "
            "surrounding 100 km box carried 67 detections over 27 days, "
            "confirming the retrieval worked. This bounds the corpus itself: "
            "absence from the detection record is not absence of industrial "
            "combustion."
        ),
    ),
    # --- enclosed, pre-dawn, and deadly --------------------------------------
    UndetectedIncident(
        name="Anaj Mandi factory fire (Rani Jhansi Road, Delhi)",
        date="2019-12-08",
        lat=28.6600,
        lon=77.2100,
        source=(
            "Fire before dawn on 2019-12-08 in a multi-storey building housing "
            "bag and packaging manufacturing units at Anaj Mandi, off Rani "
            "Jhansi Road, Delhi. Forty-three people died, most of them workers "
            "asleep inside -- the deadliest fire in Delhi since the Uphaar "
            "cinema fire of 1997."
        ),
        why_missed=(
            "Two mechanisms at once. The fire was reported around 05:00 local "
            "and fought down within a few hours, so it fell in the same gap "
            "between the 02:30 and 13:30 passes that hid Haldia and Uran. It "
            "was also entirely enclosed: a fire burning through the floors of a "
            "sealed building presents almost no radiating surface to a sensor "
            "looking straight down. The deadliest fire in this register by a "
            "wide margin is also among the least visible."
        ),
        evidence=(
            "Zero detections within 10 km on 2019-12-08. A 2.8 MW detection "
            "sits 0.6 km away on 2019-12-07, the afternoon before -- almost "
            "certainly an unrelated waste fire in dense urban Delhi, and quoted "
            "here precisely because it shows the sensor resolving small sources "
            "at this exact location on an ordinary day."
        ),
    ),
    # --- detected, and useless anyway ----------------------------------------
    UndetectedIncident(
        name="Bhilai Steel Plant gas pipeline blast (Durg, Chhattisgarh)",
        date="2018-10-09",
        # Centroid of the 15.4 km2 "Bhilai Steel Plant" polygon in this
        # project's own reference layer.
        lat=21.1886,
        lon=81.3911,
        source=(
            "A blast on a gas pipeline at the SAIL Bhilai Steel Plant, Durg "
            "district, Chhattisgarh, on 2018-10-09 killed at least 14 people "
            "during maintenance work."
        ),
        why_missed=(
            "This one WAS detected, and that is exactly why it belongs here. "
            "The site is a working blast-furnace complex that registers almost "
            "every day, so the blast did not have to be invisible to be "
            "unfindable -- it only had to be unremarkable, and it was. The "
            "plant produced FEWER detections on the day of the blast than on "
            "any surrounding day, and its peak fell below what the same plant "
            "reaches in routine operation. No threshold on radiative power, "
            "count or recurrence separates this event from the Tuesday before "
            "it. This is the chronic-baseline blind spot of 5b.2b appearing at "
            "an operating steel plant rather than a landfill, and no rate "
            "signal rescues it there either."
        ),
        evidence=(
            "484 detections within 4 km on 40 of 43 days. On 2018-10-09: 4 "
            "detections, peak FRP 10.8 MW. On the four preceding days: 14, 12, "
            "14 and 15 detections, and on 2018-10-07 a ROUTINE peak of 28.5 MW "
            "-- 2.6x the blast day. Site median FRP is 1.9 MW, 90th percentile "
            "5.6 MW."
        ),
    ),
]


def undetected_incident_report() -> str:
    """Human-readable summary of documented fires the sensor never saw."""
    lines = ["Documented industrial fires with no usable FIRMS signature:", ""]
    for inc in KNOWN_UNDETECTED_INCIDENTS:
        lines.append(f"  {inc.date}  {inc.name}")
        lines.append(f"      why missed : {inc.why_missed}")
        lines.append(f"      evidence   : {inc.evidence}")
        lines.append("")
    return chr(10).join(lines)


def attach_verified_labels(
    detections: pd.DataFrame,
    events: Optional[List[VerifiedEvent]] = None,
) -> pd.DataFrame:
    """Tags detections that fall inside a verified event's space-time window.

    Adds `verified_label` (class index, NaN when unmatched), `verified_event`,
    and `verified_confidence`.

    Where windows overlap, the smaller radius wins: a point incident inside a
    wide regional window is the more specific claim about that detection.
    """
    events = events if events is not None else load_verified_events()
    out = detections.copy()
    out["verified_label"] = np.nan
    out["verified_event"] = None
    out["verified_confidence"] = None

    if out.empty:
        return out

    ts = pd.to_datetime(out.get("acq_date", out.get("timestamp_utc")), errors="coerce", utc=True)
    lat = pd.to_numeric(out["latitude"], errors="coerce")
    lon = pd.to_numeric(out["longitude"], errors="coerce")

    for ev in sorted(events, key=lambda e: -e.radius_km):
        start = pd.Timestamp(ev.start_date, tz="UTC")
        end = pd.Timestamp(ev.end_date, tz="UTC") + pd.Timedelta(days=1)

        # Equirectangular approximation is ample at these radii and avoids a
        # projection dependency.
        dlat_km = (lat - ev.lat) * 111.32
        dlon_km = (lon - ev.lon) * 111.32 * np.cos(np.radians(ev.lat))
        within = np.sqrt(dlat_km ** 2 + dlon_km ** 2) <= ev.radius_km

        mask = within & ts.between(start, end)
        if mask.any():
            out.loc[mask, "verified_label"] = ev.label_index
            out.loc[mask, "verified_event"] = ev.name
            out.loc[mask, "verified_confidence"] = ev.confidence

    n = int(out["verified_label"].notna().sum())
    logger.info("Verified labels attached to %d / %d detections (%.2f%%).",
                n, len(out), 100.0 * n / max(len(out), 1))
    return out


def evaluate_against_verified(
    detections: pd.DataFrame,
    model,
    pipeline,
    events: Optional[List[VerifiedEvent]] = None,
    min_support: int = 20,
) -> Dict:
    """Scores the model using ONLY verified labels.

    This is the one metric in the project that is not measured against its own
    labelling rule. It is reported separately and never blended with the
    heuristic scores, because averaging an independent measurement into a
    circular one destroys exactly the information that makes it worth having.
    """
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    tagged = attach_verified_labels(detections, events)
    matched = tagged[tagged["verified_label"].notna()].copy()

    if matched.empty:
        return {
            "status": "NO_VERIFIED_MATCHES",
            "detail": (
                "No detection fell inside a verified event window. Either the "
                "corpus does not overlap the seeded events in time, or more "
                "verified events are needed. Extend data/reference/verified_events.csv."
            ),
            "n_events": len(events if events is not None else load_verified_events()),
        }

    y_true = matched["verified_label"].astype(int).values
    y_pred = model.predict(pipeline.transform(matched))

    per_event = {}
    for name, grp in matched.groupby("verified_event"):
        idx = matched.index.get_indexer(grp.index)
        truth, pred = y_true[idx], y_pred[idx]
        per_event[name] = {
            "n": int(len(grp)),
            "accuracy": float(round((truth == pred).mean(), 4)),
            "confidence": grp["verified_confidence"].iloc[0],
        }

    result = {
        "status": "OK" if len(matched) >= min_support else "LOW_SUPPORT",
        "n_verified_detections": int(len(matched)),
        "n_events_matched": int(matched["verified_event"].nunique()),
        "accuracy": float(round((y_true == y_pred).mean(), 4)),
        "macro_f1": float(round(f1_score(y_true, y_pred, average="macro"), 4)),
        "per_event": per_event,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=sorted(CLASS_NAME_TO_INDEX.values())
        ).tolist(),
        "caveat": (
            "Computed on verified labels only. This is the project's sole "
            "independent accuracy measurement; every other score is against "
            "rule-derived labels. Treat a small n accordingly."
        ),
    }

    if len(matched) < min_support:
        result["detail"] = (
            f"Only {len(matched)} verified detections matched (min_support="
            f"{min_support}). Directionally useful, not yet a defensible accuracy "
            "claim. Add more verified events."
        )

    logger.info("Verified-label evaluation: n=%d, accuracy=%.4f, macro F1=%.4f",
                result["n_verified_detections"], result["accuracy"], result["macro_f1"])
    for name, stats in per_event.items():
        logger.info("  %-55s n=%-5d acc=%.3f (%s confidence)",
                    name[:55], stats["n"], stats["accuracy"], stats["confidence"])
    return result


def write_csv_template(path: Path = CSV_PATH) -> Path:
    """Writes a CSV template for analysts to extend, without overwriting data."""
    path = Path(path)
    if path.exists():
        logger.info("%s already exists; leaving it untouched.", path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["name", "lat", "lon", "start_date", "end_date", "label",
             "confidence", "source", "radius_km", "notes"]
        )
        writer.writerow(
            ["# EXAMPLE - delete this row. label must be one of: "
             + " / ".join(sorted(CLASS_NAME_TO_INDEX)),
             "", "", "", "", "", "", "", "", ""]
        )
        writer.writerow(
            ["# `source` is required: record HOW this was verified. Never add a "
             "label inferred from the model or from FIRMS recurrence.",
             "", "", "", "", "", "", "", "", ""]
        )
    logger.info("Wrote verified-events template to %s", path)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verified-label harness (SIH26162)")
    parser.add_argument("--detections", type=str,
                        default="data/processed/firms_industrial_joined.parquet")
    parser.add_argument("--template", action="store_true",
                        help="Write the analyst CSV template and exit.")
    args = parser.parse_args()

    if args.template:
        write_csv_template()
        sys.exit(0)

    evs = load_verified_events()
    print(f"\n{len(evs)} verified event(s) defined:\n")
    for e in evs:
        print(f"  {e.name}")
        print(f"    {e.label:20s} {e.start_date} -> {e.end_date}  "
              f"({e.lat}, {e.lon}) r={e.radius_km}km  [{e.confidence}]")
        print(f"    source: {e.source[:110]}...\n")

    det_path = Path(args.detections)
    if det_path.exists():
        df = pd.read_parquet(det_path)
        tagged = attach_verified_labels(df, evs)
        n = int(tagged["verified_label"].notna().sum())
        print(f"Corpus {det_path}: {n} of {len(df)} detections carry a verified label.")
        if n:
            print(tagged[tagged.verified_label.notna()]
                  .groupby("verified_event").size().to_string())
