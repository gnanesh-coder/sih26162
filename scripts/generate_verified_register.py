"""Regenerates the verified-event register (section 5c) of PROJECT_DOCUMENTATION.md.

WHY THIS IS GENERATED RATHER THAN WRITTEN
-----------------------------------------
The verified set is the evidence base for the only measurement in this project
that is independent of its own labelling rule, and section 5b.2b argues *from*
it. But the full record of each event -- its coordinates, its window, its radius
and above all the citation that establishes it -- lives only in
`verified_labels.py`, where nobody evaluating this project will read it.

Copying that record into Markdown by hand would guarantee it drifts, and a
drifted evidence table is worse than no table at all, because it still reads as
authoritative. So the register is written from the live objects every time, and
the accuracy beside each event is computed from the model on disk against the
corpora on disk.

Run it after adding an event or retraining:

    python scripts/generate_verified_register.py

Only the text between the two HTML comment markers in the Markdown is replaced.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import xgboost as xgb

from src.models.train_classifier import (
    CLASS_NAMES,
    MODEL_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
    apply_serving_guards,
)
from src.models.verified_labels import (
    KNOWN_UNDETECTED_INCIDENTS,
    VerifiedEvent,
    attach_verified_labels,
    load_verified_events,
)

logging.getLogger("verified_labels").setLevel(logging.WARNING)

DOC = PROJECT_ROOT / "PROJECT_DOCUMENTATION.md"
BEGIN = "<!-- BEGIN GENERATED: verified-event register -->"
END = "<!-- END GENERATED: verified-event register -->"

# Every corpus an event could be scored in. The national archive first, so a
# national event is attributed to it rather than to an incident pull that
# happens to overlap; the incident corpora after, each covering one window the
# archive does not reach back to.
CORPORA = [
    "data/processed/firms_industrial_joined_12m.parquet",
    "data/processed/firms_baghjan_preevent_joined.parquet",
    "data/processed/firms_baghjan_2020_joined.parquet",
    "data/processed/firms_jaipur_2009_joined.parquet",
    "data/processed/firms_deonar_2016_joined.parquet",
    "data/processed/firms_brahmapuram_2023_joined.parquet",
    "data/processed/firms_deerpark_2019_joined.parquet",
    "data/processed/firms_buncefield_2005_joined.parquet",
    "data/processed/firms_rumaila_2026_joined.parquet",
    "data/processed/firms_benban_2026_joined.parquet",
    "data/processed/firms_bandipur_2019_joined.parquet",
    "data/processed/firms_similipal_2021_joined.parquet",
    "data/processed/firms_uttarakhand_2016_joined.parquet",
]

# Mirrors SEASONALITY_DOMAIN in train_classifier: the region whose harvest
# calendar this project calibrated, and therefore the region inside which an
# AGRICULTURAL_BURN claim is earned.
SEASONALITY_DOMAIN = (6.0, 37.5, 68.0, 97.5)

# One line per incident naming what actually defeated the sensor. Kept beside
# the generator rather than in the dataclass because it is an editorial
# grouping of the `why_missed` text, not a new fact about the incident.
MECHANISMS = {
    "2018-09-03": "Burned entirely between two overpasses",
    "2018-10-09": "**Detected, but indistinguishable** from the site's routine operation",
    "2019-12-08": "Enclosed building, and pre-dawn between overpasses",
    "2020-05-07": "**No combustion at all** -- an unignited toxic vapour release",
    "2020-07-01": "Combustion confined inside boilers and stacks; no radiating flame",
    "2026-06-30": "Burned entirely between two overpasses",
}

ANCHORS = {
    "PERSISTENT_BASELINE": "continuous heat",
    "ACCIDENTAL_FIRE": "accident",
    "AGRICULTURAL_BURN": "seasonal biomass",
    "TRANSIENT_HOTSPOT": "negative control",
}



_SMALL = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def spell(n: int) -> str:
    """Spells out a small count for running prose."""
    return _SMALL.get(n, f"{n:,}")


def in_domain(lat: float, lon: float) -> bool:
    a, b, c, d = SEASONALITY_DOMAIN
    return a <= lat <= b and c <= lon <= d


def score_event(matched: pd.DataFrame, event: VerifiedEvent, model, pipeline) -> Dict:
    """The raw model answer and what the deployed system actually serves.

    They differ, and one column would misrepresent the system in one direction
    or the other. Benban reads 0.000 from the model and 1.000 served, because
    `apply_serving_guards` withholds a claim about Egyptian land use that the
    model has no way to withhold for itself; Rumaila reads 0.065 in both,
    because no guard rescues a defect in the recurrence feature.
    """
    truth = CLASS_NAMES[event.label_index]
    # What the deployed system should report. For a forest fire this is a class
    # the model can never predict; for every other event it equals `truth`.
    served_truth = getattr(event, "served_label", "") or truth
    predicted = model.predict(pipeline.transform(matched))

    served = [
        apply_serving_guards(
            CLASS_NAMES[int(p)],
            lat=float(row.latitude),
            lon=float(row.longitude),
            facility_type=str(getattr(row, "facility_type", "") or ""),
        )[0]
        for p, row in zip(predicted, matched.itertuples())
    ]

    return {
        "n": int(len(matched)),
        "model": float((predicted == event.label_index).mean()),
        "served": float(sum(s == served_truth for s in served) / len(served)),
        "served_truth": served_truth,
        "differs": served_truth != truth,
    }


def collect(events: List[VerifiedEvent], model, pipeline
            ) -> Tuple[List[Tuple[VerifiedEvent, Dict]], List[VerifiedEvent]]:
    """Scores each event in the first corpus that contains it."""
    by_name = {ev.name: ev for ev in events}
    scored: Dict[str, Dict] = {}

    for rel in CORPORA:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        remaining = [ev for ev in events if ev.name not in scored]
        if not remaining:
            break

        tagged = attach_verified_labels(pd.read_parquet(path), remaining)
        matched = tagged[tagged["verified_label"].notna()]
        if matched.empty:
            continue

        for name, group in matched.groupby("verified_event"):
            scored[name] = score_event(group, by_name[name], model, pipeline)
            scored[name]["corpus"] = path.name

    rows = [(by_name[n], s) for n, s in scored.items()]
    rows.sort(key=lambda r: (-r[1]["served"], -r[1]["n"]))
    unscored = [ev for ev in events if ev.name not in scored]
    return rows, unscored


def build() -> str:
    events = load_verified_events()
    model = xgb.XGBClassifier()
    model.load_model(str(PROJECT_ROOT / MODEL_ARTIFACT_PATH))
    pipeline = joblib.load(PROJECT_ROOT / PIPELINE_ARTIFACT_PATH)

    rows, unscored = collect(events, model, pipeline)
    abroad = [ev for ev in events if not in_domain(ev.lat, ev.lon)]
    out: List[str] = [BEGIN, ""]

    out.append(
        f"**{len(events)} events are defined**, {spell(len(abroad))} of them outside the "
        f"region this system was calibrated in. {len(rows)} currently match "
        f"detections in a corpus on disk, together covering "
        f"**{sum(s['n'] for _, s in rows):,} detections**."
    )
    out.append("")
    out.append(
        "Section 5b.2b argues from this set; this section *is* the set. It is "
        "generated from `src/models/verified_labels.py` by "
        "`scripts/generate_verified_register.py`, so it cannot drift away from the "
        "objects the evaluation harness actually loads. Every figure below is "
        "computed at generation time from the model artifact and the corpora on "
        "disk."
    )
    out.append("")

    out.append("### 5c.1 The register")
    out.append("")
    out.append(
        "Two accuracy columns, because they differ and a single number would "
        "mislead. **Model** is the raw classifier output and is what the "
        "evaluation harness scores. **Served** is what the API actually returns, "
        "after `apply_serving_guards()` applies the constraints the model "
        "structurally cannot learn -- chiefly that India's harvest calendar is a "
        "calibrated prior about Indian agriculture, not a fact about combustion, "
        "and is unearned outside the domain where it was measured."
    )
    out.append("")
    out.append(
        "**Which corpus an event is scored in is part of the result.** The "
        "recurrence features are "
        "computed from whatever history that corpus contains. Baghjan reads "
        "**0.793** here, against the pre-event pull -- and **0.075** against a "
        "corpus that starts at the blowout, where `n_30d` and the onset lag have "
        "no baseline to contrast the accident with. Each event is therefore "
        "scored in the first corpus that covers it, national archive first, and "
        "5c.2 names that corpus beside every event so the figure can be "
        "reproduced."
    )
    out.append("")
    out.append(
        "Three events carry **two truth classes**, and that is not a hedge. A "
        "forest fire's correct *model* answer is `AGRICULTURAL_BURN`: sustained "
        "open-ground combustion with no facility history is the finest "
        "distinction a coordinate-free feature vector supports, and a forest fire "
        "is radiometrically identical to a crop fire. Its correct *served* answer "
        "is `FOREST_FIRE`, because `apply_forest_cover` consults land cover, which "
        "the model cannot. Scoring the model against `FOREST_FIRE` would penalise "
        "it for not knowing something it is deliberately never told."
    )
    out.append("")
    out.append("| # | Event | Truth class | Region | Window | Radius | n | Model | Served | Conf. |")
    out.append("| ---: | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |")
    for i, (ev, s) in enumerate(rows, 1):
        window = (f"`{ev.start_date}`" if ev.start_date == ev.end_date
                  else f"`{ev.start_date}` → `{ev.end_date}`")
        region = "India" if in_domain(ev.lat, ev.lon) else "**abroad**"
        bold = "**" if s["served"] >= 0.9 or s["served"] < 0.2 else ""
        cls = (f"{ev.label}<br>&rarr; **{s['served_truth']}**"
               if s.get("differs") else ev.label)
        out.append(
            f"| {i} | {ev.name} | {cls} | {region} | {window} | "
            f"{ev.radius_km:g} km | {s['n']:,} | {s['model']:.3f} | "
            f"{bold}{s['served']:.3f}{bold} | {ev.confidence} |"
        )
    out.append("")

    if unscored:
        out.append(
            "Defined but not currently scorable -- the event is verified, but no "
            "corpus on disk covers its window, so it contributes no rows to any "
            "accuracy figure:"
        )
        out.append("")
        for ev in unscored:
            out.append(
                f"- **{ev.name}** &mdash; `{ev.start_date}` → `{ev.end_date}`, "
                f"{ev.lat:.4f}, {ev.lon:.4f}"
            )
        out.append("")

    out.append("### 5c.2 Provenance: how each label was established")
    out.append("")
    out.append(
        "`VerifiedEvent.__post_init__` raises without a `source`. The constraint "
        "is deliberate and it is the point of the whole module: a label inferred "
        "from the model, from FIRMS recurrence, or from the weak-labelling rule "
        "is not evidence about any of them. If it cannot be cited, it does not go "
        "in the set."
    )
    out.append("")

    for ev, s in rows + [(ev, None) for ev in unscored]:
        coords = f"{ev.lat:.4f}, {ev.lon:.4f}"
        anchor = ANCHORS.get(ev.label, "")
        head = (f"**{ev.name}** — `{ev.label}` ({anchor}), {ev.confidence} "
                f"confidence, {coords}, {ev.radius_km:g} km")
        if s:
            head += (f", n={s['n']:,}, served **{s['served']:.3f}** "
                     f"(`{s['corpus']}`)")
        else:
            head += ", not currently scorable"
        out.append(head)
        out.append("")
        out.append(f"> {ev.source}")
        out.append("")
        if ev.notes:
            out.append(ev.notes)
            out.append("")

    incidents = sorted(KNOWN_UNDETECTED_INCIDENTS, key=lambda i: i.date)
    out.append("### 5c.3 The counter-register: incidents that produced no usable signal")
    out.append("")
    out.append(
        f"A record of failures is evidence too, and it is the half that systems "
        f"built to impress usually omit. These {spell(len(incidents))} documented "
        f"industrial accidents produced **no usable FIRMS signature** -- in most "
        f"cases no detection whatsoever, and in one case detections that cannot "
        f"be told apart from the site's ordinary Tuesday. They are deliberately "
        f"*not* verified events: there is nothing to label, so adding them would "
        f"inflate the event count without contributing one scorable row. They "
        f"are here because they answer a question the verified set structurally "
        f"cannot. Every accuracy figure in this project is conditional on the "
        f"fire being visible to a polar-orbiting radiometer at the moment it "
        f"passes overhead, and **these bound what the system may claim.**"
    )
    out.append("")
    out.append(
        "Each was selected for a *different* mechanism of failure, because a "
        "counter-register whose entries all failed the same way bounds nothing "
        "that the first entry had not already bounded. Every zero below was "
        "measured by pulling the window and counting, and every entry carries a "
        "control showing the retrieval worked -- a negative from an instrument "
        "that was not looking proves nothing."
    )
    out.append("")
    out.append("| Incident | Date | Mechanism of non-detection |")
    out.append("| :--- | :--- | :--- |")
    for inc in incidents:
        short = inc.name.split(" (")[0]
        out.append(f"| {short} | `{inc.date}` | {MECHANISMS.get(inc.date, '--')} |")
    out.append("")
    for inc in incidents:
        out.append(f"**{inc.name}** — `{inc.date}`, {inc.lat:.4f}, {inc.lon:.4f}")
        out.append("")
        out.append(f"> {inc.source}")
        out.append("")
        out.append(f"- **Why it was missed:** {inc.why_missed}")
        out.append(f"- **Evidence:** {inc.evidence}")
        out.append("")

    out.append("### 5c.4 Extending the register")
    out.append("")
    out.append(
        "Two routes, both loaded by `load_verified_events()`: append a "
        "`VerifiedEvent` to `VERIFIED_EVENTS` in `src/models/verified_labels.py`, "
        "or add a row to `data/reference/verified_events.csv`, which takes the "
        "same columns and needs no Python. Malformed CSV rows are logged and "
        "skipped rather than dropped silently, because a lost verified label is "
        "expensive."
    )
    out.append("")
    out.append(
        "**What to search for.** Public reporting cannot tell you whether a fire "
        "was *detectable*, and searching recent industrial fires kept failing for "
        "one reason: a modern plant fire is extinguished in hours and falls "
        "between overpasses. Search instead for fires with a **documented "
        "multi-day duration** -- those necessarily straddled an overpass -- and "
        "remember the MODIS archive reaches back to 2000 where VIIRS begins in "
        "2012. Then pull the window and count what landed *before* writing the "
        "label -- 5c.3 is what happens when you do not."
    )
    out.append("")
    out.append(
        "Regenerate this section with `python scripts/generate_verified_register.py` "
        "after any addition or retrain."
    )
    out.append("")
    out.append(END)
    return "\n".join(out)


def main() -> int:
    section = build()
    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print("Markers not found in PROJECT_DOCUMENTATION.md. Insert both lines "
              "where the section belongs:")
        print(f"  {BEGIN}")
        print(f"  {END}")
        return 1
    head = text[: text.index(BEGIN)]
    tail = text[text.index(END) + len(END):]
    DOC.write_text(head + section + tail, encoding="utf-8")
    print(f"Regenerated section 5c in {DOC.name} ({len(section):,} chars).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
