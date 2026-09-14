"""Tests for the CPCB passive compliance register.

This output carries a regulatory heading, so its failure modes are not cosmetic:
naming the wrong operator, or attributing emissions to a site that produces
none, is worse than producing no register at all.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.compliance_register import (
    NON_EMITTING_FACILITY_TYPES,
    build_compliance_register,
)


def _frame(rows):
    return pd.DataFrame(rows)


def _routine(n, osm="1", name="Refinery A", ftype="petrochemical_refinery",
             lat=22.0, lon=70.0, start="2026-01-01"):
    days = pd.date_range(start, periods=n, freq="D")
    return [{"state": "PERSISTENT_BASELINE", "frp": 10.0, "latitude": lat,
             "longitude": lon, "acq_date": d, "facility_name": name,
             "facility_type": ftype, "osm_id": osm, "recurrence_key": f"fac:{osm}",
             "suppressed": True} for d in days]


def test_identity_is_the_polygon_not_the_name():
    """Thousands of OSM polygons share "Unnamed Industrial Site".

    Grouping on the name merged 94,088 detections from sites across the country
    into one row, which under a regulatory heading attributes one operator's
    emissions to another.
    """
    rows = (_routine(30, osm="111", name="Unnamed Industrial Site", lat=22.0, lon=70.0)
            + _routine(30, osm="222", name="Unnamed Industrial Site", lat=28.0, lon=77.0))
    r = build_compliance_register(_frame(rows), min_detections=10)

    assert r["sites_listed"] == 2, "same-named distinct polygons were merged"
    assert {e["osm_id"] for e in r["entries"]} == {"111", "222"}


def test_solar_farms_never_appear_in_an_emissions_register():
    """A photovoltaic array has no combustion process and nothing to audit."""
    rows = (_routine(40, osm="1", name="Refinery A")
            + _routine(40, osm="9", name="Pavagada Solar Park",
                       ftype=NON_EMITTING_FACILITY_TYPES[0], lat=14.2, lon=77.4))
    r = build_compliance_register(_frame(rows), min_detections=10)

    names = {e["facility_name"] for e in r["entries"]}
    assert "Pavagada Solar Park" not in names
    assert "Refinery A" in names
    assert "solar" in r["exclusions"].lower()


def test_thin_records_are_omitted():
    """A handful of detections cannot establish a flaring regime."""
    rows = _routine(40, osm="1") + _routine(3, osm="2", name="Barely Seen", lat=25.0, lon=80.0)
    r = build_compliance_register(_frame(rows), min_detections=10)

    assert {e["osm_id"] for e in r["entries"]} == {"1"}


def test_continuous_and_intermittent_are_distinguished():
    rows = _routine(300, osm="1", name="Always On", start="2026-01-01")
    rows += [dict(x, osm_id="2", facility_name="Now And Then", recurrence_key="fac:2")
             for x in _routine(20, osm="2", start="2026-01-01")]
    r = build_compliance_register(_frame(rows), min_detections=10)

    regimes = {e["facility_name"]: e["regime"] for e in r["entries"]}
    assert regimes["Always On"] == "CONTINUOUS"
    assert regimes["Now And Then"] == "INTERMITTENT"


def test_gas_volume_and_co2_are_not_invented():
    """Uncalibrated carbon figures under a regulatory heading are fabrications."""
    r = build_compliance_register(_frame(_routine(40)), min_detections=10)

    # Check the DATA fields, not the whole payload: the disclaimer legitimately
    # contains the words "gas volume" while stating that none is computed.
    fields = set().union(*(e.keys() for e in r["entries"]))
    forbidden = {f for f in fields
                 if any(t in f.lower() for t in ("co2", "carbon", "volume", "tonne", "scm"))}
    assert not forbidden, f"register publishes uncalibrated figures: {forbidden}"
    assert "not computed" in r["assumptions"]["gas_volume_and_co2"].lower()


def test_energy_estimate_states_its_assumption():
    r = build_compliance_register(_frame(_routine(40)), min_detections=10)

    assumption = r["assumptions"]["fire_radiative_energy"].lower()
    assert "overpass" in assumption
    assert "not a metered quantity" in assumption


def test_register_carries_the_unconfirmed_caveat():
    r = build_compliance_register(_frame(_routine(40)), min_detections=10)
    assert "not been confirmed on the ground" in r["caveat"]
    assert "not a finding of non-compliance" in r["caveat"]


def test_no_routine_activity_is_reported_honestly():
    rows = [dict(x, state="ACCIDENTAL_FIRE") for x in _routine(20)]
    r = build_compliance_register(_frame(rows), min_detections=10)

    assert r["status"] == "NO_ROUTINE_DETECTIONS"
    assert r["entries"] == []


def test_missing_columns_fail_loudly():
    with pytest.raises(KeyError):
        build_compliance_register(pd.DataFrame({"state": ["PERSISTENT_BASELINE"]}))
