"""Historical Replay Mode: the demonstration's insurance policy.

WHAT THIS IS FOR
----------------
Every live dependency in this system is someone else's uptime. FIRMS, Copernicus
and Open-Meteo are all third-party APIs, and the basemap is tiles fetched from
Esri. A venue with bad wifi, a rate limit, or a maintenance window can take any
of them away in the middle of a presentation.

The project's own research documents name the remedy directly: *"if the live
ingestion pipeline fails during the presentation, the frontend must possess a
Historical Replay Mode that seamlessly switches to querying the local
instance."* The 12-month archive is already on disk. This is the switch.

WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------
It **stops the refresh loop from attempting the network**, and it **relabels the
freshness badge REPLAY rather than STALE**. That relabelling is the whole point
and is not cosmetic: STALE means "nobody has ingested anything and you should
worry", which is a defect. REPLAY means "ingestion is deliberately frozen and
you are looking at the archive", which is a choice. Presenting the second as the
first would be a lie in the operator's favour; presenting the first as the second
would be a lie in ours.

It does **not** reseed the incident database from the archive. Seeding walks
2,044,295 rows one at a time; it is not something to trigger from a button four
minutes before a demonstration. Every archive-backed surface -- the compliance
register, the H3 map layer, the optical overlay -- already reads the parquet
directly and works with the network unplugged.

It does **not** fabricate data to fill gaps. Panels that genuinely need the
network (the live Sentinel-2 probe, the Sentinel-3 probe, observed weather)
continue to report their own honest degraded states, which they were already
built to do.

THE PREFLIGHT IS THE USEFUL HALF
--------------------------------
Knowing the mode exists is worth less than knowing what survives without a
network. `preflight()` answers that per capability, so a rehearsal can be run
against a checklist instead of a hope.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "enabled": False,
    "since_utc": None,
    "reason": None,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enabled() -> bool:
    """Whether replay mode is currently on.

    The environment variable lets a deployment boot straight into replay -- the
    right setting for a machine that will be demonstrated on a venue network
    nobody has tested.
    """
    if _state["enabled"]:
        return True
    raw = os.getenv("REPLAY_MODE", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def set_enabled(on: bool, reason: str = "") -> Dict[str, Any]:
    with _lock:
        _state["enabled"] = bool(on)
        _state["since_utc"] = _now_iso() if on else None
        _state["reason"] = (reason or
                            ("Operator switched to the local archive."
                             if on else None))
    return status()


def status() -> Dict[str, Any]:
    out = dict(_state)
    out["enabled"] = enabled()
    out["env_override"] = bool(os.getenv("REPLAY_MODE", "").strip())
    return out


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
#
# Each entry is one capability, whether it needs a network, and what it does
# when there isn't one. "Degrades honestly" is a claim this project can make
# because those paths are individually tested; where it cannot, the entry says
# the capability is simply unavailable.

def _exists(rel: str) -> bool:
    p = Path(rel)
    return (p if p.is_absolute() else PROJECT_ROOT / p).exists()


def preflight() -> Dict[str, Any]:
    """What works with the network unplugged, checked against what is on disk.

    Written to be run *before* a demonstration, not after it fails.
    """
    archive = "data/processed/firms_industrial_joined_12m.parquet"
    live = "data/processed/firms_industrial_joined.parquet"

    checks: List[Dict[str, Any]] = [
        {
            "capability": "Incident stream, map markers, triage queue",
            "needs_network": False,
            "ready": _exists("data/fire_db.sqlite"),
            "source": "data/fire_db.sqlite",
            "offline_behaviour": "Fully available. Served from the incident database.",
        },
        {
            "capability": "Compliance register",
            "needs_network": False,
            "ready": _exists(archive) or _exists(live),
            "source": archive,
            "offline_behaviour": "Fully available. Read directly from the processed corpus.",
        },
        {
            "capability": "H3 archive density layer",
            "needs_network": False,
            "ready": _exists(archive) or _exists(live),
            "source": archive,
            "offline_behaviour": "Fully available. Aggregated server-side from the parquet.",
        },
        {
            "capability": "dNBR burn-scar overlay",
            "needs_network": False,
            "ready": _exists("data/processed/p0_alerts_dnbr.parquet"),
            "source": "data/processed/p0_alerts_dnbr.parquet",
            "offline_behaviour": "Fully available. The validation run is precomputed and permanent.",
        },
        {
            "capability": "Analytics charts, audit trail, SitRep",
            "needs_network": False,
            "ready": _exists("data/fire_db.sqlite"),
            "source": "data/fire_db.sqlite",
            "offline_behaviour": "Fully available.",
        },
        {
            "capability": "Model provenance",
            "needs_network": False,
            "ready": _exists("src/models/model_evaluation_metrics.json"),
            "source": "src/models/model_evaluation_metrics.json",
            "offline_behaviour": "Fully available. Read from the metrics artifact.",
        },
        {
            "capability": "Classification and TreeSHAP",
            "needs_network": False,
            "ready": _exists("src/models/fire_classifier_xgb.json"),
            "source": "src/models/fire_classifier_xgb.json",
            "offline_behaviour": "Fully available. The model runs locally.",
        },
        {
            "capability": "Forest-fire land cover",
            "needs_network": False,
            "ready": _exists("data/reference/osm_india_forest_from_pbf.parquet"),
            "source": "data/reference/osm_india_forest_from_pbf.parquet",
            "offline_behaviour": "Fully available. Absent, FOREST_FIRE is withheld and nothing else changes.",
        },
        {
            "capability": "FIRMS ingestion",
            "needs_network": True,
            "ready": True,
            "source": "NASA FIRMS API",
            "offline_behaviour": "Paused in replay. The freshness badge reads REPLAY, not STALE.",
        },
        {
            "capability": "Sentinel-2 per-incident probe",
            "needs_network": True,
            "ready": True,
            "source": "Copernicus",
            "offline_behaviour": "Reports its own failure status. The precomputed overlay is unaffected.",
        },
        {
            "capability": "Sentinel-3 thermal probe",
            "needs_network": True,
            "ready": True,
            "source": "Copernicus",
            "offline_behaviour": "Reports NO_SCENE or API_ERROR rather than a number.",
        },
        {
            "capability": "Observed weather for dispersion",
            "needs_network": True,
            "ready": True,
            "source": "Open-Meteo",
            "offline_behaviour": "Falls back to the flagged synthetic values, declared on four surfaces.",
        },
        {
            "capability": "Basemap tiles",
            "needs_network": True,
            "ready": True,
            "source": "Esri World Imagery",
            "offline_behaviour": (
                "UNRESOLVED. Tiles already fetched stay in the browser cache, but "
                "panning to an unvisited area shows blank tiles. Rehearse the exact "
                "map extent you intend to show while the network is up."
            ),
        },
    ]

    offline_ready = [c for c in checks if not c["needs_network"]]
    missing = [c for c in offline_ready if not c["ready"]]

    return {
        "replay": status(),
        "offline_capable": len(offline_ready) - len(missing),
        "offline_total": len(offline_ready),
        "network_dependent": sum(1 for c in checks if c["needs_network"]),
        "missing_assets": [c["source"] for c in missing],
        "verdict": (
            "READY" if not missing else "INCOMPLETE"
        ),
        "advice": (
            "Every offline capability has its asset on disk. Rehearse with the "
            "network disabled, and note the basemap caveat below."
            if not missing else
            "Some offline capabilities have no asset on disk and will be empty. "
            "Build them before demonstrating."
        ),
        "checks": checks,
    }
