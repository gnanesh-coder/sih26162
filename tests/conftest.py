"""Shared test fixtures.

The weather provider reaches a third-party archive over the network. Left live,
every test that renders a SitRep or computes a dispersion cone makes a blocking
outbound request: the suite went from 5 to 50 seconds, and its result started
depending on someone else's uptime.

Tests that care about the observed path mock the provider explicitly. Everything
else gets the labelled synthetic fallback, which is also what a machine with no
network sees -- so the default test environment matches the harshest real one.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# The suite must not write to the database the dashboard is serving.
#
# It did. `test_analyst_action_and_audit_logs` posts a real analyst action, so
# every `pytest` run appended an audit row to data/fire_db.sqlite and flipped a
# live incident to DISPATCHED with a canned operator note. Running the tests
# before a demonstration therefore polluted the demonstration -- and the note
# it left ("Tactical operator verified thermal spike via optical satellite
# pass") reads exactly like fabricated field data.
#
# `DATABASE_URL` is resolved at import time in app.database, so this has to be
# set before anything imports it. conftest is loaded first, which is why it
# lives here rather than in a fixture. The live database is copied rather than
# started empty, so the endpoint tests still run against realistic rows.
# ---------------------------------------------------------------------------
_LIVE_DB = PROJECT_ROOT / "data" / "fire_db.sqlite"
_TEST_DB = Path(tempfile.gettempdir()) / "sih_pytest_fire_db.sqlite"
if _LIVE_DB.exists():
    shutil.copy2(_LIVE_DB, _TEST_DB)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"


@pytest.fixture(autouse=True)
def no_live_weather_calls(monkeypatch, request):
    """Disable the live weather lookup unless a test opts in.

    Opt in with @pytest.mark.live_weather for a test that must exercise the
    real archive.
    """
    if request.node.get_closest_marker("live_weather"):
        return

    from src.alerting import sitrep_generator as sg

    def refuse(*args, **kwargs):
        raise AssertionError(
            "A test made a live weather request. Mock fetch_observed_weather, or "
            "mark the test with @pytest.mark.live_weather if the call is the point."
        )

    monkeypatch.setattr(sg.requests, "get", refuse)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live_weather: test may call the real weather archive"
    )
