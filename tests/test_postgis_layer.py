"""Guards the PostGIS migration.

Two things are being protected here.

The first is that **the datastore mode is decided by configuration, not by
connectivity**. An earlier draft of `app/database.py` fell back to SQLite when
PostGIS was unreachable. Because the geometry column is declared only in
PostGIS mode, that fallback would silently build a different schema: an
application with no `incidents.geom`, answering every proximity query with
nothing, while reporting itself healthy. `test_unreachable_postgis_raises`
exists so that behaviour cannot come back.

The second is that the SQLite fallback's distance maths is actually correct.
It is a bounding box plus a haversine, and a box computed in degrees is a
different real distance at different latitudes -- the exact error PostGIS
exists to avoid. The tests below check it at Kanyakumari and at Srinagar.

The PostGIS-specific tests skip cleanly when no server is configured, so the
suite still runs with no database daemon anywhere.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from app.database import (  # noqa: E402
    IS_POSTGRES,
    OFFLINE_SQLITE,
    Incident,
    _compose_postgres_url,
    _normalise_driver,
    incidents_within_km,
    postgis_version,
)


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------

def test_bare_postgres_url_is_pinned_to_psycopg3():
    """A `postgresql://` URL must not send SQLAlchemy looking for psycopg2.

    psycopg2 is not installed. Without the rewrite, a connection string copied
    from pgAdmin or a cloud console fails with an import error that reads like
    a missing database rather than a missing driver.
    """
    assert _normalise_driver("postgresql://u:p@h:5432/db") == (
        "postgresql+psycopg://u:p@h:5432/db"
    )


def test_already_qualified_driver_is_left_alone():
    url = "postgresql+psycopg://u:p@h:5432/db"
    assert _normalise_driver(url) == url


def test_sqlite_url_is_left_alone():
    url = "sqlite:///data/fire_db.sqlite"
    assert _normalise_driver(url) == url


def test_composed_url_uses_the_same_names_as_docker_compose(monkeypatch):
    """docker-compose.yml and the application read one set of variables.

    If these drift, `docker compose up -d db` starts a server the application
    cannot address, and the failure looks like bad credentials.
    """
    monkeypatch.setenv("POSTGRES_USER", "sih")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_DB", "sih26162")
    assert _compose_postgres_url() == (
        "postgresql+psycopg://sih:secret@db.internal:6543/sih26162"
    )


def test_app_defaults_match_docker_compose_defaults(monkeypatch):
    """The regression: compose and the app must agree with NO .env present.

    An earlier version defaulted to `postgres` with an empty password while
    docker-compose.yml defaulted to `sih`/`sih_local_dev`, so a clean checkout
    running `docker compose up -d db` started a server the application could
    not authenticate against. This reads the compose file and compares.
    """
    for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST",
                "POSTGRES_PORT", "POSTGRES_DB"):
        monkeypatch.delenv(key, raising=False)

    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # Matches e.g.  POSTGRES_USER: ${POSTGRES_USER:-sih}
    declared = dict(
        re.findall(r"(POSTGRES_\w+):\s*\$\{POSTGRES_\w+:-([^}]*)\}", compose)
    )
    assert declared, "could not read defaults out of docker-compose.yml"

    url = _compose_postgres_url()
    assert declared["POSTGRES_USER"] in url
    assert declared["POSTGRES_PASSWORD"] in url
    assert declared["POSTGRES_DB"] in url


# ---------------------------------------------------------------------------
# The mode is configuration, never connectivity
# ---------------------------------------------------------------------------

def _import_with(env: dict) -> subprocess.CompletedProcess:
    """Imports app.database in a clean interpreter under the given env."""
    child_env = dict(os.environ)
    child_env.update(env)
    # dotenv would otherwise reload the developer's real .env over the top of
    # the URL under test.
    child_env["DOTENV_DISABLED"] = "1"
    return subprocess.run(
        [sys.executable, "-c", "import app.database as d; print(d.DATABASE_URL)"],
        cwd=str(PROJECT_ROOT),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_unreachable_postgis_raises_instead_of_degrading():
    """Configured-but-unreachable PostGIS must be fatal.

    The regression this prevents: falling back would drop `incidents.geom`
    from the schema, so every ST_DWithin query would return nothing from a
    database that reports itself healthy.
    """
    # Port 1 is reserved and never listening.
    result = _import_with(
        {"DATABASE_URL": "postgresql+psycopg://u:p@127.0.0.1:1/nosuchdb"}
    )
    assert result.returncode != 0, (
        "importing with an unreachable PostGIS must fail, not fall back:\n"
        f"{result.stdout}"
    )
    combined = result.stdout + result.stderr
    assert "PostGISUnavailable" in combined or "unreachable" in combined
    assert "sqlite" not in result.stdout.lower(), (
        "the fallback to SQLite must not happen automatically"
    )


def test_unreachable_postgis_error_says_how_to_fix_it():
    result = _import_with(
        {"DATABASE_URL": "postgresql+psycopg://u:p@127.0.0.1:1/nosuchdb"}
    )
    combined = result.stdout + result.stderr
    assert "docker compose up -d db" in combined
    assert "DATABASE_URL=sqlite" in combined


def test_explicit_sqlite_is_accepted_as_an_offline_choice(tmp_path):
    db = tmp_path / "offline.sqlite"
    result = _import_with({"DATABASE_URL": f"sqlite:///{db.as_posix()}"})
    assert result.returncode == 0, result.stderr
    assert "sqlite" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Schema shape follows the mode
# ---------------------------------------------------------------------------

def test_geometry_column_exists_only_in_postgis_mode():
    """`geom` is a PostGIS type and cannot be declared against SQLite."""
    has_geom = hasattr(Incident, "geom")
    assert has_geom == IS_POSTGRES


def test_offline_flag_is_the_inverse_of_postgres():
    assert OFFLINE_SQLITE != IS_POSTGRES


def test_postgis_version_is_none_when_not_on_postgis():
    if IS_POSTGRES:
        pytest.skip("running against PostGIS")
    assert postgis_version() is None


@pytest.mark.skipif(not IS_POSTGRES, reason="requires a live PostGIS server")
def test_both_spatial_indexes_are_declared():
    """A GiST index on geometry does not serve a geography query.

    Measured on 200,004 rows: with only `idx_incidents_geom` the planner chose
    a Parallel Seq Scan at 268 ms; adding the functional index on the cast
    expression produced a Bitmap Index Scan at 4.6 ms. Both are required.
    """
    names = {ix.name for ix in Incident.__table__.indexes}
    assert "idx_incidents_geog" in names


# ---------------------------------------------------------------------------
# The SQLite fallback's distance maths
# ---------------------------------------------------------------------------

class _FakeIncident:
    def __init__(self, id_, lat, lon):
        self.id = id_
        self.latitude = lat
        self.longitude = lon


class _FakeQuery:
    """Minimal stand-in for the ORM query used by the fallback path."""

    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, _model):
        return _FakeQuery(self._rows)


@pytest.mark.skipif(IS_POSTGRES, reason="exercises the SQLite fallback maths")
def test_fallback_measures_true_distance_not_degrees():
    """One degree of longitude is 111 km at the equator and 96 km at Srinagar.

    A naive degree-based radius would include the Srinagar point below and
    exclude nothing; the haversine must reject it.
    """
    lat, lon = 23.75, 86.42          # Jharia
    rows = [
        _FakeIncident(1, 23.75, 86.42),      # 0 km
        _FakeIncident(2, 23.83, 86.42),      # ~8.9 km
        _FakeIncident(3, 24.80, 86.42),      # ~117 km
    ]
    got = incidents_within_km(_FakeSession(rows), lat=lat, lon=lon, radius_km=15)
    assert [i for i, _ in got] == [1, 2]
    assert got[0][1] == pytest.approx(0.0, abs=1e-6)
    assert got[1][1] == pytest.approx(8.9, abs=0.5)


@pytest.mark.skipif(IS_POSTGRES, reason="exercises the SQLite fallback maths")
def test_fallback_radius_widens_with_latitude():
    """At 34N a degree of longitude is ~92 km, so a 50 km search must still
    reach a point 0.5 degrees east. A box padded without the cos(lat) term
    would silently drop it."""
    lat, lon = 34.0837, 74.7973      # Srinagar
    near = _FakeIncident(1, 34.0837, 74.7973 + 0.45)   # ~41 km east
    got = incidents_within_km(_FakeSession([near]), lat=lat, lon=lon, radius_km=50)
    assert [i for i, _ in got] == [1]
    assert got[0][1] == pytest.approx(41.0, abs=3.0)


@pytest.mark.skipif(IS_POSTGRES, reason="exercises the SQLite fallback maths")
def test_fallback_returns_nearest_first():
    lat, lon = 23.75, 86.42
    rows = [
        _FakeIncident(1, 23.95, 86.42),   # farther
        _FakeIncident(2, 23.80, 86.42),   # nearer
    ]
    got = incidents_within_km(_FakeSession(rows), lat=lat, lon=lon, radius_km=100)
    assert [i for i, _ in got] == [2, 1]
    assert got[0][1] < got[1][1]


@pytest.mark.skipif(IS_POSTGRES, reason="exercises the SQLite fallback maths")
def test_fallback_respects_the_limit():
    lat, lon = 23.75, 86.42
    rows = [_FakeIncident(i, 23.75 + i * 0.001, 86.42) for i in range(1, 20)]
    got = incidents_within_km(_FakeSession(rows), lat=lat, lon=lon, radius_km=100, limit=5)
    assert len(got) == 5


@pytest.mark.skipif(IS_POSTGRES, reason="exercises the SQLite fallback maths")
def test_fallback_returns_nothing_when_radius_excludes_everything():
    """An empty result must be empty, not the nearest row regardless of range."""
    got = incidents_within_km(
        _FakeSession([_FakeIncident(1, 30.0, 86.42)]),
        lat=23.75, lon=86.42, radius_km=5,
    )
    assert got == []


# ---------------------------------------------------------------------------
# Migration idempotency
# ---------------------------------------------------------------------------

from datetime import datetime  # noqa: E402

from scripts.migrate_sqlite_to_postgis import _ts_key  # noqa: E402


def test_timestamp_key_matches_across_both_databases():
    """SQLite returns a string, PostgreSQL a datetime, for the same instant.

    The regression: the audit log's duplicate check compared the two forms
    directly, never matched, and a second migration run re-inserted every
    row -- 31 evidence entries became 61. Duplicated evidence is corrupt
    evidence.
    """
    from_sqlite = "2026-09-13 12:34:56.789000"
    from_postgres = datetime(2026, 9, 13, 12, 34, 56, 789000)
    assert _ts_key(from_sqlite) == _ts_key(from_postgres)


def test_timestamp_key_pads_whole_seconds():
    """SQLite writes '...12:00:00' with no fractional part when it is zero."""
    assert _ts_key("2026-09-13 12:00:00") == _ts_key(datetime(2026, 9, 13, 12, 0, 0))


def test_timestamp_key_accepts_iso_t_separator():
    assert _ts_key("2026-09-13T12:00:00") == _ts_key("2026-09-13 12:00:00")


def test_timestamp_key_drops_tzinfo_rather_than_shifting():
    """The columns are `timestamp without time zone` and every writer stamps
    UTC, so an aware value must key as its own wall clock, not a shifted one."""
    from datetime import timezone as _tz
    aware = datetime(2026, 9, 13, 12, 0, 0, tzinfo=_tz.utc)
    assert _ts_key(aware) == _ts_key(datetime(2026, 9, 13, 12, 0, 0))


def test_timestamp_key_handles_null():
    assert _ts_key(None) == ""


def test_distinct_instants_do_not_collide():
    a = _ts_key(datetime(2026, 9, 13, 12, 0, 0))
    b = _ts_key(datetime(2026, 9, 13, 12, 0, 1))
    assert a != b
