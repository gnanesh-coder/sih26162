"""Passive compliance register: routine flaring, logged rather than dispatched.

The alerting table in the problem statement gives routine industrial flaring its
own row -- not an alert, but *"passive logging for CPCB compliance auditing and
carbon liability calculations"*. Every other row of that table is built; this was
the empty one.

The argument for it is in the research itself: regulators have historically
relied on **operator self-reporting** for flaring activity. Satellite thermal
observation is independent of the operator, which is what makes it an audit
rather than a summary. The same approach found Permian supply-chain emissions
roughly 80% above bottom-up estimates.

So this module deliberately reports the detections the alerting pipeline
*suppressed*. A refinery flare that is correctly withheld from an incident
commander is exactly the record a compliance auditor wants: persistent, routine,
and unremarkable operationally, while still being a continuous emission.

**On what is and is not quantified.** Detection counts, observed days, and Fire
Radiative Power come straight from the sensor. Fire Radiative Energy is FRP
integrated over time and is derivable from them, with one stated assumption.
Flared gas volume and CO2 liability are NOT computed: converting radiative
energy to combusted volume needs a calibrated combustion efficiency and heating
value that this project does not have. An earlier version of this codebase
carried hand-tuned emission factors described as EPA AP-42; inventing a second
set here for carbon liability would repeat that mistake with legal consequences
attached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("compliance_register")

# States the alerting pipeline treats as routine industrial operation. These are
# suppressed from dispatch by design, and are precisely what an auditor wants.
ROUTINE_STATES = ("PERSISTENT_BASELINE", "MONITORING")

# Land uses that cannot emit, and must never appear in an emissions register.
#
# A photovoltaic array has no combustion process. Thermal detections inside one
# are specular reflection -- the sun-glint artifact the problem statement names
# explicitly. Four solar parks reached the first draft of this register on 7 to
# 14 observed days apiece, which is a glint signature rather than a flaring one.
# Reporting them under a CPCB heading would attribute emissions to an operator
# that produces none, and the burden of disproving that would fall on them.
NON_EMITTING_FACILITY_TYPES = ("renewable_non_thermal",)

# A source counted as continuously flaring when observed on at least this share
# of the days it could have been observed. Below it, the record still stands but
# is marked intermittent rather than continuous.
CONTINUOUS_THRESHOLD = 0.30

# VIIRS observes a point roughly twice daily. Treating each detection as
# representative of a half-day is the coarsest defensible assumption and is
# stated wherever the derived energy appears.
ASSUMED_HOURS_PER_DETECTION = 12.0

FRE_ASSUMPTION = (
    "Fire Radiative Energy is Fire Radiative Power integrated over time. FRP is "
    "sampled only at satellite overpass, so this assumes the source emitted at "
    "the observed power for 12 hours per detection -- the nominal VIIRS revisit "
    "half-interval. It is an order-of-magnitude figure for ranking sites against "
    "one another, not a metered quantity."
)

VOLUME_NOT_COMPUTED = (
    "Flared gas volume and CO2 liability are not computed. Converting radiative "
    "energy to combusted volume requires a calibrated combustion efficiency and "
    "gas heating value that this project has not established. Publishing an "
    "uncalibrated figure under a regulatory heading would be a fabricated "
    "measurement with legal consequences attached."
)


@dataclass(frozen=True)
class RegisterEntry:
    """One facility's observed flaring record over the corpus period."""

    facility_name: str
    facility_type: str
    osm_id: str
    latitude: float
    longitude: float
    detections: int
    days_observed: int
    span_days: int
    observation_ratio: float
    regime: str                 # CONTINUOUS | INTERMITTENT
    frp_median_mw: float
    frp_max_mw: float
    frp_total_mw: float
    estimated_fre_gj: float
    suppressed_from_dispatch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facility_name": self.facility_name,
            "facility_type": self.facility_type,
            "osm_id": self.osm_id,
            "latitude": round(self.latitude, 5),
            "longitude": round(self.longitude, 5),
            "detections": self.detections,
            "days_observed": self.days_observed,
            "span_days": self.span_days,
            "observation_ratio": round(self.observation_ratio, 4),
            "regime": self.regime,
            "frp_median_mw": round(self.frp_median_mw, 2),
            "frp_max_mw": round(self.frp_max_mw, 2),
            "frp_total_mw": round(self.frp_total_mw, 1),
            "estimated_fre_gj": round(self.estimated_fre_gj, 1),
            "suppressed_from_dispatch": self.suppressed_from_dispatch,
        }


def build_compliance_register(
    df: pd.DataFrame,
    min_detections: int = 10,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregates routine industrial thermal activity per facility.

    Args:
        df: Processed detections carrying state, facility and FRP columns.
        min_detections: Sites below this are omitted. A handful of detections
            cannot establish a flaring regime, and listing them under a
            regulatory heading would imply a confidence the data lacks.
        limit: Optional cap on entries returned, largest total FRP first.
    """
    required = {"state", "frp", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Compliance register needs columns {sorted(missing)}.")

    routine = df[df["state"].astype(str).isin(ROUTINE_STATES)].copy()

    if "facility_type" in routine.columns:
        non_emitting = (routine["facility_type"].astype(str).str.strip().str.lower()
                        .isin(NON_EMITTING_FACILITY_TYPES))
        if non_emitting.any():
            logger.info(
                "Excluded %s detection(s) at non-emitting land use (solar/wind) "
                "from the compliance register.", f"{int(non_emitting.sum()):,}",
            )
            routine = routine[~non_emitting]

    if routine.empty:
        return _empty_report(len(df))

    routine["frp"] = pd.to_numeric(routine["frp"], errors="coerce").fillna(0.0)
    dates = pd.to_datetime(
        routine.get("acq_date", routine.get("timestamp_utc")), errors="coerce", utc=True
    )
    routine["_day"] = dates.dt.date

    # Identity is the POLYGON, not the name. Names are not unique: thousands of
    # distinct OSM polygons share the placeholder "Unnamed Industrial Site", and
    # grouping on the name merged 94,088 detections from sites across the
    # country into a single register row -- which under a regulatory heading
    # would attribute one operator's emissions to another.
    #
    # osm_id identifies the polygon. Name is display only. A site with neither
    # falls back to its recurrence key so an unmapped but persistently flaring
    # source still appears rather than being dropped for lacking a name.
    def _clean(col: str) -> pd.Series:
        if col not in routine.columns:
            return pd.Series("", index=routine.index, dtype=str)
        return (routine[col].astype(str).str.strip()
                .replace({"nan": "", "None": "", "null": "", "<NA>": ""}))

    osm = _clean("osm_id")
    name = _clean("facility_name")
    fallback = _clean("recurrence_key")

    routine["_key"] = np.where(
        osm != "", "osm:" + osm,
        np.where(name != "", "name:" + name, "key:" + fallback),
    )
    routine["_display"] = np.where(
        name != "", name,
        np.where(osm != "", "OSM way " + osm, "unmapped source"),
    )

    span_days = max(int((dates.max() - dates.min()).days), 1) if dates.notna().any() else 1

    entries: List[RegisterEntry] = []
    for key, g in routine.groupby("_key", sort=False):
        if len(g) < min_detections:
            continue
        days = int(g["_day"].nunique())
        frp = g["frp"]
        ratio = days / span_days
        entries.append(RegisterEntry(
            facility_name=str(g["_display"].iloc[0]),
            facility_type=str(g.get("facility_type", pd.Series(["unknown"])).iloc[0]),
            osm_id=str(g.get("osm_id", pd.Series([""])).iloc[0]),
            latitude=float(g["latitude"].median()),
            longitude=float(g["longitude"].median()),
            detections=int(len(g)),
            days_observed=days,
            span_days=span_days,
            observation_ratio=float(ratio),
            regime="CONTINUOUS" if ratio >= CONTINUOUS_THRESHOLD else "INTERMITTENT",
            frp_median_mw=float(frp.median()),
            frp_max_mw=float(frp.max()),
            frp_total_mw=float(frp.sum()),
            # MW * hours = MWh; * 3.6 = GJ.
            estimated_fre_gj=float(frp.sum() * ASSUMED_HOURS_PER_DETECTION * 3.6),
            suppressed_from_dispatch=int(
                g.get("suppressed", pd.Series([False] * len(g))).astype(bool).sum()
            ),
        ))

    entries.sort(key=lambda e: e.frp_total_mw, reverse=True)
    if limit:
        entries = entries[:limit]

    continuous = [e for e in entries if e.regime == "CONTINUOUS"]
    logger.info(
        "Compliance register: %d site(s) above %d detections, %d continuous, "
        "from %s routine detections over %d days.",
        len(entries), min_detections, len(continuous), f"{len(routine):,}", span_days,
    )

    return {
        "status": "OK",
        "purpose": (
            "Independent observation of routine industrial flaring for CPCB "
            "compliance auditing. These are detections the alerting pipeline "
            "SUPPRESSED as routine operation: unremarkable to an incident "
            "commander, and exactly the continuous-emission record an auditor "
            "cannot get from operator self-reporting."
        ),
        "observation_span_days": span_days,
        "routine_detections": int(len(routine)),
        "sites_listed": len(entries),
        "sites_continuous": len(continuous),
        "min_detections": min_detections,
        "assumptions": {
            "fire_radiative_energy": FRE_ASSUMPTION,
            "gas_volume_and_co2": VOLUME_NOT_COMPUTED,
            "continuous_threshold": (
                f"A site observed on at least {CONTINUOUS_THRESHOLD:.0%} of the "
                "days in the observation span is reported as CONTINUOUS."
            ),
        },
        "exclusions": (
            "Non-combustion land uses (solar, wind) are excluded. They have no "
            "emissions to audit, and thermal detections there are specular "
            "reflection rather than flaring."
        ),
        "caveat": (
            "Classification is model-derived from satellite thermal data and has "
            "not been confirmed on the ground. This register is evidence for "
            "further inquiry, not a finding of non-compliance."
        ),
        "entries": [e.to_dict() for e in entries],
    }


def _empty_report(n_total: int) -> Dict[str, Any]:
    return {
        "status": "NO_ROUTINE_DETECTIONS",
        "detail": (
            f"None of {n_total:,} detections were classified as routine "
            f"industrial operation ({' or '.join(ROUTINE_STATES)}). With no "
            "suppressed flaring there is nothing to audit."
        ),
        "entries": [],
    }
