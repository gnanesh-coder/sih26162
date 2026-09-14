"""Recurrence & Suppression State Machine for Industrial Fire Classification.

Implements Uber H3 geospatial binning (res=9) and temporal recurrence tracking.
Distinguishes between routine operational flares/kilns and critical industrial fire emergencies.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

try:
    import h3
except (ImportError, OSError):
    h3 = None

import numpy as np
import pandas as pd

logger = logging.getLogger("state_machine")


def latlng_to_h3(lat: float, lng: float, resolution: int = 9) -> str:
    """Converts lat/lng coordinates to an H3 cell index (compatible with h3 v3 and v4),
    with pure-python deterministic hex bin fallback if native h3 is unavailable.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.
        resolution: H3 grid resolution (default: 9, ~174m edge length).

    Returns:
        H3 hexagon index string.
    """
    if h3 is not None:
        try:
            if hasattr(h3, "latlng_to_cell"):
                return str(h3.latlng_to_cell(lat, lng, resolution))
            elif hasattr(h3, "geo_to_h3"):
                return str(h3.geo_to_h3(lat, lng, resolution))
        except Exception:
            pass

    # Deterministic spatial hex cell fallback (~175m at res=9)
    step = 0.0015 * (10 - min(resolution, 10))
    lat_bin = int(round((lat + 90.0) / max(step, 0.0001)))
    lng_bin = int(round((lng + 180.0) / max(step, 0.0001)))
    return f"89{abs(lat_bin):06x}{abs(lng_bin):06x}ffff"


class AlertState(str, Enum):
    """Categorical classification states determined by recurrence & energy dynamics."""
    PERSISTENT_BASELINE = "PERSISTENT_BASELINE"
    ESCALATED_FLAREUP = "ESCALATED_FLAREUP"
    ACCIDENTAL_FIRE = "ACCIDENTAL_FIRE"
    TRANSIENT_SUSPICION = "TRANSIENT_SUSPICION"
    MONITORING = "MONITORING"


class AlertPriority(str, Enum):
    """Dispatch priority levels for incident response."""
    P0_EMERGENCY = "P0_EMERGENCY"  # High urgency accidental industrial fire
    P1_ALERT = "P1_ALERT"          # Significant operational anomaly / flare-up
    P2_ADVISORY = "P2_ADVISORY"    # Minor/developing thermal anomaly
    SUPPRESSED = "SUPPRESSED"      # Known baseline; do not disturb operators
    NON_ALERT = "NON_ALERT"        # Outside industrial boundary, background/agri


@dataclass
class AlertDecision:
    """Output decision package produced by the state machine for each detection."""
    state: AlertState
    priority: AlertPriority
    tag: str
    suppressed: bool
    rationale: str
    n_30d: int
    frp: float
    mu_frp: float
    var_frp: float
    z_frp: float
    frp_ratio: float
    inside_industrial: bool
    # Days since this source was first observed. Zero for a source appearing for
    # the first time. Causal: derived only from past observations.
    source_age_days: float = 0.0
    # Days between observation opening and this source becoming persistent.
    # -1.0 while it has never been persistent.
    onset_lag_days: float = -1.0
    # Distinct OTHER sources active nearby in the trailing window. High for a
    # migrating front surrounded by its own recent history, low for an isolated
    # source. See the module note on the onset gate.
    neighbourhood_active_keys: int = 0

    # Detections in the trailing 24 h, and that rate against the cell's own
    # prior daily rate. See H3CellHistory.burst for why n_30d cannot show this.
    n_24h: int = 0
    burst_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "priority": self.priority.value,
            "tag": self.tag,
            "suppressed": self.suppressed,
            "rationale": self.rationale,
            "n_30d": self.n_30d,
            "n_24h": self.n_24h,
            "burst_ratio": round(self.burst_ratio, 3),
            "frp": round(self.frp, 2),
            "mu_frp": round(self.mu_frp, 2),
            "var_frp": round(self.var_frp, 2),
            "z_frp": round(self.z_frp, 2),
            "frp_ratio": round(self.frp_ratio, 2),
            "inside_industrial": self.inside_industrial,
            "source_age_days": round(self.source_age_days, 2),
            "onset_lag_days": round(self.onset_lag_days, 2),
            "neighbourhood_active_keys": int(self.neighbourhood_active_keys),
        }


class RecurrenceStateMachine:
    """Evaluates hotspot recurrence statistics against industrial alerting rules."""

    @staticmethod
    def evaluate(
        n_30d: int,
        frp: float,
        mu_frp: float,
        var_frp: float,
        inside_industrial: bool,
    ) -> AlertDecision:
        """Determines alert state and priority for a thermal detection.

        State Machine Transitions:
          - PERSISTENT_BASELINE: N_30 >= 8 and Z_FRP <= 2.5
            -> Suppress alerts (tagged as flare/kiln).
          - ESCALATED_FLAREUP: N_30 >= 8 and (Z_FRP > 3.0 or FRP > 3 * mu_FRP)
            -> Trigger P1 Alert.
          - ACCIDENTAL_FIRE: N_30 < 3 and inside industrial polygon and FRP >= 10 MW
            -> Trigger P0 Emergency Alert.
          - TRANSIENT_SUSPICION: N_30 < 3 and outside industrial polygon
            -> Non-alert background/agri burn.

        Args:
            n_30d: Hotspot count in this H3 cell over past 30 days.
            frp: Current Fire Radiative Power (MW).
            mu_frp: Historical mean/EMA of FRP (MW).
            var_frp: Historical variance of FRP (MW^2).
            inside_industrial: True if hotspot intersects an industrial polygon.

        Returns:
            AlertDecision with state, priority, and diagnostic rationale.
        """
        std_frp = math.sqrt(max(var_frp, 0.0))
        z_frp = (frp - mu_frp) / std_frp if std_frp > 1e-4 else 0.0
        frp_ratio = (frp / mu_frp) if mu_frp > 1e-4 else 1.0

        # Rule 1: High Recurrence Persistent Thermal Sites (Refinery flares, smelters, brick kilns)
        if n_30d >= 8:
            if z_frp > 3.0 or frp > (3.0 * mu_frp):
                return AlertDecision(
                    state=AlertState.ESCALATED_FLAREUP,
                    priority=AlertPriority.P1_ALERT,
                    tag="escalated_flareup",
                    suppressed=False,
                    rationale=(
                        f"High recurrence site (N_30={n_30d}) experienced anomalous surge: "
                        f"FRP={frp:.1f}MW vs mu={mu_frp:.1f}MW (Z={z_frp:.2f}, ratio={frp_ratio:.2f}x)."
                    ),
                    n_30d=n_30d,
                    frp=frp,
                    mu_frp=mu_frp,
                    var_frp=var_frp,
                    z_frp=z_frp,
                    frp_ratio=frp_ratio,
                    inside_industrial=inside_industrial,
                )
            elif z_frp <= 2.5:
                return AlertDecision(
                    state=AlertState.PERSISTENT_BASELINE,
                    priority=AlertPriority.SUPPRESSED,
                    tag="flare_or_kiln",
                    suppressed=True,
                    rationale=(
                        f"Persistent operational emission (N_30={n_30d}, Z={z_frp:.2f} <= 2.5). "
                        "Routine flare stack or kiln baseline; alert suppressed."
                    ),
                    n_30d=n_30d,
                    frp=frp,
                    mu_frp=mu_frp,
                    var_frp=var_frp,
                    z_frp=z_frp,
                    frp_ratio=frp_ratio,
                    inside_industrial=inside_industrial,
                )
            else:
                # Borderline elevation (2.5 < Z <= 3.0)
                return AlertDecision(
                    state=AlertState.MONITORING,
                    priority=AlertPriority.P2_ADVISORY,
                    tag="moderate_flare_elevation",
                    suppressed=False,
                    rationale=(
                        f"Persistent site (N_30={n_30d}) with moderate FRP elevation (2.5 < Z={z_frp:.2f} <= 3.0). "
                        "Flagged for ongoing surveillance."
                    ),
                    n_30d=n_30d,
                    frp=frp,
                    mu_frp=mu_frp,
                    var_frp=var_frp,
                    z_frp=z_frp,
                    frp_ratio=frp_ratio,
                    inside_industrial=inside_industrial,
                )

        # Rule 2: Low Recurrence Hotspots (N_30 < 3)
        if n_30d < 3:
            if inside_industrial:
                if frp >= 10.0:
                    return AlertDecision(
                        state=AlertState.ACCIDENTAL_FIRE,
                        priority=AlertPriority.P0_EMERGENCY,
                        tag="accidental_industrial_fire",
                        suppressed=False,
                        rationale=(
                            f"CRITICAL: New sudden thermal source (N_30={n_30d}) inside industrial perimeter "
                            f"with high energy FRP={frp:.1f}MW >= 10MW. Accidental fire confirmed."
                        ),
                        n_30d=n_30d,
                        frp=frp,
                        mu_frp=mu_frp,
                        var_frp=var_frp,
                        z_frp=z_frp,
                        frp_ratio=frp_ratio,
                        inside_industrial=inside_industrial,
                    )
                else:
                    return AlertDecision(
                        state=AlertState.MONITORING,
                        priority=AlertPriority.P2_ADVISORY,
                        tag="low_frp_industrial_anomaly",
                        suppressed=False,
                        rationale=(
                            f"New hotspot inside industrial zone with low energy FRP={frp:.1f}MW (<10MW). "
                            "Advisory notification generated."
                        ),
                        n_30d=n_30d,
                        frp=frp,
                        mu_frp=mu_frp,
                        var_frp=var_frp,
                        z_frp=z_frp,
                        frp_ratio=frp_ratio,
                        inside_industrial=inside_industrial,
                    )
            else:
                return AlertDecision(
                    state=AlertState.TRANSIENT_SUSPICION,
                    priority=AlertPriority.NON_ALERT,
                    tag="background_agri_burn",
                    suppressed=True,
                    rationale=(
                        # Says what was done, not what the detection is. The
                        # previous wording -- "Classified as background
                        # agricultural/stubble" -- asserted a classification on
                        # a detection this branch deliberately never sends to
                        # the classifier, and named a class ("stubble") the
                        # system does not have.
                        f"Outside any mapped industrial polygon (N_30={n_30d}). "
                        "Not submitted to the classifier: background thermal "
                        "activity is filtered before inference, so no class and "
                        "no confidence were computed for this detection."
                    ),
                    n_30d=n_30d,
                    frp=frp,
                    mu_frp=mu_frp,
                    var_frp=var_frp,
                    z_frp=z_frp,
                    frp_ratio=frp_ratio,
                    inside_industrial=inside_industrial,
                )

        # Rule 3: Intermediate Recurrence (3 <= N_30 < 8)
        priority = AlertPriority.P1_ALERT if inside_industrial and frp >= 10.0 else AlertPriority.P2_ADVISORY
        return AlertDecision(
            state=AlertState.MONITORING,
            priority=priority,
            tag="developing_cluster",
            suppressed=False,
            rationale=(
                f"Developing hotspot cluster (N_30={n_30d}, FRP={frp:.1f}MW, inside_industrial={inside_industrial}). "
                "Classified into active monitoring pool."
            ),
            n_30d=n_30d,
            frp=frp,
            mu_frp=mu_frp,
            var_frp=var_frp,
            z_frp=z_frp,
            frp_ratio=frp_ratio,
            inside_industrial=inside_industrial,
        )


@dataclass
class H3CellHistory:
    """Historical thermal observations and rolling statistics for one source.

    Observations are held in a deque and pruned from the left. Detections are fed
    in chronological order, so the deque is always time-sorted and expiry is
    O(number actually evicted) rather than O(history length).

    The previous implementation rebuilt the entire list on every detection. That
    is fine for a short corpus but grows with history depth, and at national
    scale a busy facility accumulates thousands of observations: the state
    machine took 5m28s of a 6m34s run over 2M detections -- 83% of pipeline time
    spent re-copying lists. Running sums make the statistics O(1) as well.
    """
    observations: Deque[Tuple[datetime, float]] = field(default_factory=deque)
    ema_frp: float = 0.0
    ema_alpha: float = 0.2
    _sum_frp: float = 0.0
    _sum_sq: float = 0.0
    # First time this source was ever seen, kept outside the 30-day window so it
    # survives pruning. A source that has been emitting since the corpus opened
    # is infrastructure; one that appeared partway through and then became
    # persistent is an event that turned into its own baseline -- which is
    # exactly what a months-long blowout looks like, and why Baghjan reads as
    # routine operation on recurrence alone.
    first_seen: Optional[datetime] = None
    # When this source first became *persistent* -- the moment its 30-day count
    # crossed the recurrence threshold -- as distinct from the first photon ever
    # recorded at the location.
    #
    # The distinction is not academic. In the five months before the Baghjan
    # blowout the wellhead key logged exactly one detection, on 28 January. Keyed
    # on first_seen, the source looks 130 days old by the time it ignites and
    # reads as established infrastructure. Keyed on when it started *recurring*,
    # it is four months of silence followed by a step change.
    became_persistent_at: Optional[datetime] = None

    def prune(self, current_time: datetime, window_days: int = 30) -> None:
        """Evicts observations older than window_days relative to current_time."""
        cutoff = current_time - timedelta(days=window_days)
        obs = self.observations
        while obs and obs[0][0] < cutoff:
            _, frp = obs.popleft()
            self._sum_frp -= frp
            self._sum_sq -= frp * frp

    def add_observation(self, timestamp: datetime, frp: float) -> None:
        """Records a new hotspot observation and updates running aggregates."""
        if self.first_seen is None:
            self.first_seen = timestamp
        self.observations.append((timestamp, frp))
        self._sum_frp += frp
        self._sum_sq += frp * frp
        if self.ema_frp == 0.0:
            self.ema_frp = frp
        else:
            self.ema_frp = self.ema_alpha * frp + (1.0 - self.ema_alpha) * self.ema_frp

    def note_persistence(self, timestamp: datetime, n_30d: int, threshold: int = 8) -> None:
        """Records the first moment this source's rolling count crossed `threshold`."""
        if self.became_persistent_at is None and n_30d >= threshold:
            self.became_persistent_at = timestamp

    def onset_lag_days(self, corpus_start: Optional[datetime]) -> float:
        """Days between observation opening and this source becoming persistent.

        Near zero for infrastructure that was already running when the archive
        began. Large for a source that was observed to be quiet and then started.
        Returns -1.0 while the source has never been persistent, which keeps
        "not yet persistent" distinguishable from "persistent since day one".

        Bounded by corpus length: on an archive that opens after a plant is
        already running, that plant is indistinguishable from one commissioned on
        day one. A longer pre-event window is the only fix for that.
        """
        if self.became_persistent_at is None or corpus_start is None:
            return -1.0
        return max((self.became_persistent_at - corpus_start).total_seconds() / 86400.0, 0.0)

    def source_age_days(self, current_time: datetime) -> float:
        """Days between this source's first ever observation and `current_time`.

        Causal by construction: it looks only at what has already been observed,
        so it is available at inference time on a live feed. Note that it is
        bounded by the corpus length -- on a 12-month archive an installation
        running since before the archive began is indistinguishable from one that
        started on day one.
        """
        if self.first_seen is None:
            return 0.0
        return max((current_time - self.first_seen).total_seconds() / 86400.0, 0.0)

    def burst(self, current_time: datetime, window_hours: float = 24.0,
              window_days: int = 30) -> Tuple[int, float]:
        """Detections in the last `window_hours`, and that rate against the
        cell's own prior daily rate.

        WHY THIS EXISTS
        ---------------
        `n_30d` smears a four-day catastrophe across a month, so a site that
        already burns cannot show one. The 2016 Deonar landfill fire reached
        n_30d of 108 with a 15-day onset lag *before* the disaster started: no
        onset to detect, and its surge cleared z > 3 on 13.8% of detections
        against a routine refinery's 4.3%. Both of the rule's accident tests
        were blind to it, and it scored 0.069.

        Daily rate does not smear. Measured over the corpora on disk:

            Deonar landfill fire      0.318/d baseline -> 5.80/d   x18.2
            Brahmapuram waste fire    0.327/d          -> 4.29/d   x13.1
            Jaipur depot fire         0.000/d          -> 6.60/d      inf
            Jamnagar refinery         2.096/d          -> 1.79/d    x0.9
            Jharia coalfield         91.627/d          -> 77.8/d    x0.8
            Visakhapatnam Steel       9.710/d          -> 10.0/d    x1.0

        An order of magnitude of clear air between every catastrophe and every
        routine source.

        The prior rate deliberately EXCLUDES the recent window. Including it
        lets a large enough burst inflate its own baseline and suppress the
        very signal being measured.
        """
        cutoff = current_time - timedelta(hours=window_hours)
        n_recent = 0
        for ts, _ in reversed(self.observations):
            if ts < cutoff:
                break
            n_recent += 1

        n_total = len(self.observations)
        prior_days = max(window_days - (window_hours / 24.0), 1.0)
        prior_rate = (n_total - n_recent) / prior_days

        # A cell with no prior history has no baseline to be a multiple of.
        # That case belongs to the existing "strong event, no history" rule, so
        # it reports 0.0 rather than infinity.
        if prior_rate <= 0.0:
            return n_recent, 0.0
        return n_recent, float(n_recent / (window_hours / 24.0) / prior_rate)

    def compute_stats(self) -> Tuple[int, float, float]:
        """Count, mean and variance over the current window, from running sums."""
        n = len(self.observations)
        if n == 0:
            return 0, 0.0, 0.0

        mu = self._sum_frp / n
        if n > 1:
            # E[x^2] - (E[x])^2, floored at zero against float drift.
            var = max(self._sum_sq / n - mu * mu, 0.0)
        else:
            var = 0.0
        return n, float(mu), float(var)


class H3RecurrenceTracker:
    """Spatial registry maintaining 30-day thermal baselines.

    Recurrence is accumulated per *source*, and what counts as a source depends on
    whether the detection landed on mapped infrastructure.

    Keying purely on an H3 resolution-9 cell (~174m edge) was measurably wrong for
    industrial sites. A large refinery complex spans tens of square kilometres and
    contains multiple flare stacks; successive detections of the same continuously
    operating facility scatter across dozens of cells through pixel-centre
    variation, off-nadir growth and plume parallax. Each cell therefore
    accumulates only a handful of observations and never reaches the
    PERSISTENT_BASELINE threshold, so a permanently flaring refinery is reported
    as a series of unrelated transient hotspots.

    That is not hypothetical: at the Reliance Jamnagar complex, 43 detections
    spread across 18 distinct cells (2.4 per cell, n_30d median 1 against a
    threshold of 8) and 42 of them were misclassified. The project's own design
    documents call for tolerating "spatial jitter up to 1,000 metres caused by
    satellite parallax effects and viewing geometry" -- roughly six times looser
    than a resolution-9 cell.

    So when a detection falls inside a known industrial polygon, its history is
    keyed on the facility. Outside mapped infrastructure the H3 cell remains the
    key, which is correct for wildfires: a spreading fire front *should* fragment
    across cells, because it genuinely is moving.
    """

    def __init__(
        self,
        resolution: int = 9,
        window_days: int = 30,
        ema_alpha: float = 0.2,
        persistence_resolution: int = 8,
        neighbourhood_resolution: int = 6,
    ) -> None:
        # The area over which "is anything else burning here?" is asked. Res 6
        # spans roughly 36 km2 -- the scale a fire front moves through in a
        # month. Note this is only ever used to *count neighbours*; onset itself
        # stays keyed on the source, because coarsening that was measured and
        # rejected (see process_hotspot).
        self.neighbourhood_resolution = neighbourhood_resolution
        self.neighbourhood: Dict[str, Deque[Tuple[datetime, str]]] = {}
        self.resolution = resolution
        # Resolution used for the recurrence key OUTSIDE mapped infrastructure.
        # Resolution 8 spans roughly 900m, matching the "spatial jitter up to
        # 1,000 metres caused by satellite parallax effects and viewing geometry"
        # the design documents call for. Resolution 9 (~174m) is ~6x tighter, and
        # at that granularity an unmapped static source -- a remote wellhead, a
        # facility missing from OpenStreetMap -- scatters across cells and never
        # establishes a baseline, so it is reported as a series of unrelated
        # events. The Baghjan blowout burned continuously for five months and
        # still reached a median n_30d of only 4.
        self.persistence_resolution = persistence_resolution
        self.window_days = window_days
        self.ema_alpha = ema_alpha
        self.registry: Dict[str, H3CellHistory] = {}
        # Earliest timestamp this tracker has processed: the moment observation
        # opened. Onset is meaningless without it -- "appeared late" is only
        # meaningful relative to when watching began.
        self.corpus_start: Optional[datetime] = None

    @staticmethod
    def recurrence_key(h3_index: str, facility_id: Optional[str] = None) -> str:
        """Returns the registry key a detection accumulates against.

        Facility identity wins over cell geometry wherever it is known. The
        prefixes keep the two namespaces from colliding and make the choice
        visible in diagnostics.
        """
        if facility_id is not None:
            fid = str(facility_id).strip()
            if fid and fid.lower() not in ("nan", "none", "null", ""):
                return f"fac:{fid}"
        return f"hex:{h3_index}"

    def neighbourhood_keys(self, lat: float, lng: float, key: str,
                           timestamp: datetime) -> int:
        """Distinct other recurrence keys active nearby in the trailing window.

        Causal by construction: it reads only what has already been observed, so
        it is available at inference time on a live feed. The detection's own key
        is excluded, which is the whole point -- a source that has simply been
        burning a long time must not count itself as a crowd.
        """
        area = latlng_to_h3(lat, lng, self.neighbourhood_resolution)
        dq = self.neighbourhood.get(area)
        if dq is None:
            dq = self.neighbourhood[area] = deque()

        cutoff = timestamp - timedelta(days=self.window_days)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        n = len({k for _, k in dq if k != key})
        dq.append((timestamp, key))
        return n

    def get_or_create_cell(self, key: str) -> H3CellHistory:
        if key not in self.registry:
            self.registry[key] = H3CellHistory(ema_alpha=self.ema_alpha)
        return self.registry[key]

    def process_hotspot(
        self,
        lat: float,
        lng: float,
        frp: float,
        timestamp: datetime,
        inside_industrial: bool = False,
        facility_id: Optional[str] = None,
    ) -> Tuple[str, AlertDecision]:
        """Records detection, computes recurrence, and evaluates alert decision.

        Args:
            lat: Latitude of hotspot.
            lng: Longitude of hotspot.
            frp: Fire Radiative Power (MW).
            timestamp: UTC acquisition datetime.
            inside_industrial: Flag indicating intersection with industrial zone.
            facility_id: Stable identifier of the containing industrial facility
                (e.g. an OSM way id). When present, recurrence accumulates against
                the facility rather than the H3 cell, so a large complex is not
                fragmented into unrelated transient hotspots. See the class
                docstring for why this matters.

        Returns:
            Tuple of (h3_index, AlertDecision). The H3 index is still returned for
            display and geospatial binning; only the recurrence key differs.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # The fine cell is kept for display and geospatial binning; recurrence
        # uses facility identity where known, else a coarser cell.
        h3_index = latlng_to_h3(lat, lng, self.resolution)
        fid = facility_id if inside_industrial else None
        coarse = h3_index if fid else latlng_to_h3(lat, lng, self.persistence_resolution)
        key = self.recurrence_key(coarse, fid)
        cell = self.get_or_create_cell(key)

        # Prune and compute baseline PRIOR to adding current observation (to evaluate surprise)
        cell.prune(timestamp, self.window_days)
        n_prev, mu_prev, var_prev = cell.compute_stats()

        # If cell has no prior observations, use current FRP as initial baseline
        mu_eval = mu_prev if n_prev > 0 else frp
        var_eval = var_prev if n_prev > 1 else 1.0

        decision = RecurrenceStateMachine.evaluate(
            n_30d=n_prev,
            frp=frp,
            mu_frp=mu_eval,
            var_frp=var_eval,
            inside_industrial=inside_industrial,
        )
        # Age measured before this detection joins the history, so a source seen
        # for the first time reports 0 rather than its own timestamp difference.
        decision.source_age_days = cell.source_age_days(timestamp)

        if self.corpus_start is None or timestamp < self.corpus_start:
            self.corpus_start = timestamp
        cell.note_persistence(timestamp, n_prev, threshold=8)

        # Onset stays keyed on the source, not on a neighbourhood.
        #
        # Reading it from an H3 res-6 area was tried and measured: it took
        # `onset_lag > 45 days` from 2.17% of the corpus to 56.04%. A res-6 cell
        # is 48x a res-8 cell, so sparse agricultural areas that never reach the
        # persistence threshold at res 8 cross it easily at res 6 -- and they
        # cross it during the burning season, months into the corpus. The change
        # did not teach the rule that Jharia was already alight; it taught the
        # rule that Punjab agriculture began in October.
        #
        # The defect it aimed at is real (1,404 of 1,709 false accidents are
        # Jharia). The fix is a conjunction, not a coarser key: keep onset where
        # it is and additionally require a quiet neighbourhood before an onset
        # counts as an accident -- the same shape the burst test needed.
        decision.onset_lag_days = cell.onset_lag_days(self.corpus_start)
        decision.neighbourhood_active_keys = self.neighbourhood_keys(
            lat, lng, key, timestamp)

        # Add current detection to history, THEN read the burst: this detection
        # is part of the burst being measured, and excluding it would make a
        # single-detection spike invisible.
        cell.add_observation(timestamp, frp)
        decision.n_24h, decision.burst_ratio = cell.burst(timestamp)

        return h3_index, decision


def evaluate_dataframe(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    frp_col: str = "frp",
    time_col: str = "timestamp_utc",
    industrial_col: str = "inside_industrial",
    resolution: int = 9,
    tracker: Optional[H3RecurrenceTracker] = None,
    facility_id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Evaluates a DataFrame of hotspot detections through the recurrence state machine.

    Args:
        df: Input DataFrame with latitude, longitude, frp, and timestamp columns.
        lat_col: Column name for latitude.
        lon_col: Column name for longitude.
        frp_col: Column name for FRP.
        time_col: Column name for timestamp.
        industrial_col: Column name for industrial polygon intersection boolean.
        resolution: H3 resolution level.
        tracker: Optional existing H3RecurrenceTracker instance.
        facility_id_cols: Candidate columns identifying the containing facility,
            tried in order. Defaults to osm_id -> facility_name -> index_right,
            which covers OSM polygons, Bhuvan polygons (no osm_id), and anything
            else the spatial join matched. Detections inside a facility
            accumulate recurrence against it instead of against their H3 cell.

    Returns:
        DataFrame augmented with H3 indices, the recurrence key actually used,
        and AlertDecision metrics.
    """
    if df.empty:
        return df.copy()

    tracker = tracker or H3RecurrenceTracker(resolution=resolution)
    id_cols = facility_id_cols or ["osm_id", "facility_name", "index_right"]

    # Ensure chronological processing
    sorted_df = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(sorted_df)

    # Columns are extracted to flat sequences up front, and only the NEW fields
    # are accumulated. The earlier approach built one dict per detection holding
    # every original column and then re-assembled a DataFrame from the list; at
    # national scale (~2M detections x ~44 columns) that duplicates the entire
    # corpus in Python objects before pandas ever sees it.
    lats = pd.to_numeric(sorted_df[lat_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lngs = pd.to_numeric(sorted_df[lon_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    frps = pd.to_numeric(sorted_df.get(frp_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    times = pd.to_datetime(sorted_df[time_col], utc=True, errors="coerce").dt.to_pydatetime()

    if industrial_col in sorted_df.columns:
        col = sorted_df[industrial_col]
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            insides = col.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()
        else:
            insides = col.fillna(False).astype(bool).to_numpy()
    else:
        insides = np.zeros(n, dtype=bool)

    # Resolve facility identity once, vectorised, instead of per row.
    fac_ids = pd.Series([None] * n, dtype=object)
    for col_name in id_cols:
        if col_name not in sorted_df.columns:
            continue
        candidate = sorted_df[col_name].astype(str).str.strip()
        usable = (
            fac_ids.isna()
            & sorted_df[col_name].notna()
            & ~candidate.str.lower().isin(["nan", "none", "null", ""])
        )
        fac_ids = fac_ids.mask(usable, candidate)
    fac_ids = fac_ids.to_numpy()

    h3_out: List[str] = [""] * n
    key_out: List[str] = [""] * n
    dec_out: List[Dict[str, Any]] = [None] * n

    for i in range(n):
        inside_ind = bool(insides[i])
        facility_id = fac_ids[i] if inside_ind else None

        ts = times[i]
        if ts is None or (isinstance(ts, float) and np.isnan(ts)):
            ts = datetime.now(timezone.utc)

        h3_idx, decision = tracker.process_hotspot(
            lat=float(lats[i]),
            lng=float(lngs[i]),
            frp=float(frps[i]),
            timestamp=ts,
            inside_industrial=inside_ind,
            facility_id=facility_id,
        )

        h3_out[i] = h3_idx
        key_out[i] = tracker.recurrence_key(
            h3_idx if facility_id
            else latlng_to_h3(float(lats[i]), float(lngs[i]), tracker.persistence_resolution),
            facility_id,
        )
        dec_out[i] = decision.to_dict()

    out = sorted_df.copy()
    out["h3_index"] = h3_out
    out["recurrence_key"] = key_out
    for field_name in (dec_out[0].keys() if n else []):
        out[field_name] = [d[field_name] for d in dec_out]

    return out
