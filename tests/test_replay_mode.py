"""Tests for Historical Replay Mode.

Replay exists for one moment: the live feed dies during a presentation. That
makes it the feature least likely to be exercised before it matters, and most
likely to be discovered broken at the worst possible time.

The properties worth guarding are therefore about honesty and blast radius, not
about the switch itself:

  * turning it on must stop the refresh loop, or a dead network gets retried in
    a loop while the operator is talking
  * turning it off must let ingestion resume, or the badge sits at REPLAY
    forever with nothing behind it
  * the preflight must report a capability as unavailable when its asset is
    genuinely missing -- a preflight that always says READY is worse than none
  * nothing here may reseed or fabricate
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import auto_refresh, replay_mode


@pytest.fixture(autouse=True)
def reset_replay():
    """Replay is process-global state; never leak it into another test."""
    replay_mode.set_enabled(False)
    yield
    replay_mode.set_enabled(False)


class TestSwitch:
    def test_off_by_default(self):
        assert replay_mode.enabled() is False

    def test_enabling_records_when_and_why(self):
        state = replay_mode.set_enabled(True, reason="venue wifi failed")
        assert state["enabled"] is True
        assert state["since_utc"]
        assert "venue wifi" in state["reason"]

    def test_disabling_clears_the_timestamp(self):
        replay_mode.set_enabled(True)
        state = replay_mode.set_enabled(False)
        assert state["enabled"] is False
        assert state["since_utc"] is None

    def test_environment_can_boot_straight_into_replay(self, monkeypatch):
        """The right setting for a machine going to an untested network."""
        monkeypatch.setenv("REPLAY_MODE", "true")
        assert replay_mode.enabled() is True
        assert replay_mode.status()["env_override"] is True


class TestRefreshInteraction:
    def test_replay_stops_the_refresh_loop(self, monkeypatch):
        """A loop retrying a dead network during a demo is pure noise."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("AUTO_REFRESH_ENABLED", "true")
        assert auto_refresh.refresh_enabled() is True

        replay_mode.set_enabled(True)
        assert auto_refresh.refresh_enabled() is False

    def test_leaving_replay_lets_ingestion_resume(self, monkeypatch):
        """Otherwise the badge sits at REPLAY with nothing behind it."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("AUTO_REFRESH_ENABLED", "true")
        replay_mode.set_enabled(True)
        replay_mode.set_enabled(False)
        assert auto_refresh.refresh_enabled() is True

    def test_pytest_still_wins_over_everything(self):
        """The suite must never reach FIRMS, replay or no replay."""
        replay_mode.set_enabled(False)
        assert auto_refresh.refresh_enabled() is False


class TestPreflight:
    def test_reports_both_halves(self):
        p = replay_mode.preflight()
        assert p["offline_total"] >= 8
        assert p["network_dependent"] >= 4
        assert p["verdict"] in {"READY", "INCOMPLETE"}

    def test_every_check_states_its_offline_behaviour(self):
        """A capability with no stated degraded behaviour cannot be rehearsed."""
        for check in replay_mode.preflight()["checks"]:
            assert check["offline_behaviour"].strip()
            assert isinstance(check["needs_network"], bool)
            assert check["source"].strip()

    def test_a_missing_asset_makes_the_verdict_incomplete(self, monkeypatch):
        """A preflight that always says READY is worse than no preflight."""
        monkeypatch.setattr(replay_mode, "_exists", lambda rel: False)
        p = replay_mode.preflight()
        assert p["verdict"] == "INCOMPLETE"
        assert p["missing_assets"]
        assert p["offline_capable"] == 0

    def test_the_basemap_caveat_is_not_quietly_dropped(self):
        """The one gap with no local fallback must stay visible."""
        checks = replay_mode.preflight()["checks"]
        basemap = next(c for c in checks if "asemap" in c["capability"])
        assert basemap["needs_network"] is True
        assert "UNRESOLVED" in basemap["offline_behaviour"]

    def test_replay_state_travels_with_the_preflight(self):
        replay_mode.set_enabled(True, reason="rehearsal")
        p = replay_mode.preflight()
        assert p["replay"]["enabled"] is True
        assert "rehearsal" in p["replay"]["reason"]


class TestNoSideEffects:
    def test_switching_touches_no_data(self, tmp_path):
        """Replay reseeds nothing and writes nothing. It is a flag."""
        before = {p: p.stat().st_mtime
                  for p in (PROJECT_ROOT / "data" / "processed").glob("*.parquet")}
        replay_mode.set_enabled(True)
        replay_mode.preflight()
        replay_mode.set_enabled(False)
        after = {p: p.stat().st_mtime
                 for p in (PROJECT_ROOT / "data" / "processed").glob("*.parquet")}
        assert before == after
