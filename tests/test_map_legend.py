"""Guards the map legend against the marker colours it claims to describe.

The defect these exist to prevent: the legend popover on the map showed four
swatches -- P0, P1, P2, Routine -- under a heading that presented them as what
the map draws. The map draws seven colours from a first-match cascade mixing
alert priority with predicted class, and only one of them (crimson for P0) was
anywhere near a legend entry.

Two consequences, both worse than a cosmetic mismatch:

  * Green meant opposite things. In the legend it was `--routine`, the
    suppressed tier. On the map it is FOREST_FIRE, an actively burning forest.
  * The single largest visual group on screen -- slate, for detections that were
    never sent to the classifier -- had no legend entry at all, so the one
    colour whose entire purpose is to say "this was not assessed" read as an
    unexplained shade of grey.

The fix was structural, not editorial: `MARKER_LEGEND` is now the only place a
marker colour is written down, `renderMapMarkers()` colours from it, and the
legend renders from it. These tests hold that arrangement in place.
"""

import re
from pathlib import Path

import pytest

from src.models.verified_labels import CLASS_NAME_TO_INDEX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")


def _marker_legend_block() -> str:
    start = UI.index("const MARKER_LEGEND = [")
    end = UI.index("\n    ];", start)
    return UI[start:end]


def _legend_colours() -> list:
    return re.findall(r"colour:\s*'(#[0-9a-fA-F]{6})'", _marker_legend_block())


def test_marker_legend_table_exists():
    assert "const MARKER_LEGEND = [" in UI


def test_map_markers_colour_from_the_legend_table():
    """`renderMapMarkers` must not carry colours of its own.

    An if-chain here is how the original drift happened: the chain grew a
    branch, the hand-written legend did not, and nothing failed.
    """
    start = UI.index("function renderMapMarkers()")
    end = UI.index("function initHeatmap()", start)
    body = UI[start:end]

    assert "MARKER_LEGEND.find(" in body, (
        "renderMapMarkers no longer resolves its colour through MARKER_LEGEND"
    )
    assert "fillColor: color," in body

    # Only the *fill* is a classification. The white stroke is chrome -- the
    # same ring on every dot, there so a marker stays visible against both
    # satellite imagery and the light basemap -- so it is deliberately not
    # driven by the table and is excluded here rather than silently matched.
    stray = re.findall(r"(?:color\s*=|fillColor:)\s*'(#[0-9a-fA-F]{6})'", body)
    assert not stray, f"hard-coded fill colours are back in renderMapMarkers: {stray}"


def test_legend_is_rendered_from_the_table_not_hand_written():
    assert 'id="legend-markers"' in UI
    assert "function renderMarkerLegend()" in UI
    assert "MARKER_LEGEND.map(" in UI


def test_legend_renderer_runs_at_startup():
    """A legend built at runtime is empty until something builds it."""
    start = UI.index("window.addEventListener('DOMContentLoaded'")
    end = UI.index("function initMap()", start)
    assert "renderMarkerLegend();" in UI[start:end], (
        "renderMarkerLegend() is defined but never called, so the legend "
        "renders as an empty box"
    )


def test_every_marker_rule_has_a_label_and_a_colour():
    block = _marker_legend_block()
    rules = re.findall(r"\{\s*colour:", block)
    labels = re.findall(r"label:\s*'", block)
    tests = re.findall(r"test:\s*", block)
    assert len(rules) == len(labels) == len(tests), (
        "a rule is missing a label or a test, so it would draw an "
        "undocumented colour"
    )


def test_colours_are_distinct():
    """Two rules sharing a colour make the legend a lie by construction."""
    colours = _legend_colours()
    assert len(colours) == len(set(colours)), f"duplicate marker colours: {colours}"


def test_the_cascade_ends_in_a_rule_that_always_matches():
    """`Array.find` returns undefined when nothing matches.

    Without a terminal rule an unusual incident would throw on `rule.colour`
    and take the whole marker layer down with it.
    """
    assert "test: () => true" in _marker_legend_block()


def test_agricultural_burn_is_tested_before_not_assessed():
    """Order is the logic, and this pair is the fragile part of it.

    A crop burn is by definition outside any industrial polygon, so the
    not-assessed rule matches it too. Only the ordering keeps the 323 live
    agricultural detections yellow instead of slate.
    """
    block = _marker_legend_block()
    assert block.index("AGRICULTURAL_BURN") < block.index("NOT_ASSESSED")


def test_priority_is_tested_before_class():
    """An emergency must stay crimson whatever the classifier decided."""
    block = _marker_legend_block()
    assert block.index("P0_EMERGENCY") < block.index("PERSISTENT_BASELINE")


def test_forest_green_is_not_presented_as_the_routine_tier():
    """The mismatch that started this: one green, two meanings.

    The tier legend's green is `--routine`. The map's green is FOREST_FIRE.
    They must not be described as the same thing in the same panel.
    """
    start = UI.index('<span class="label block mb-1.5">Map markers</span>')
    end = UI.index("Alert tier &middot; stream rail", start)
    marker_section = UI[start:end]
    assert "var(--routine)" not in marker_section, (
        "the marker legend is using the routine tier token for a class colour"
    )


def test_tier_legend_is_labelled_as_the_stream_rail_not_the_map():
    assert "Alert tier &middot; stream rail" in UI
    assert '<span class="label block mb-1.5">Alert tier</span>' not in UI, (
        "the bare 'Alert tier' heading is back, which is what implied these "
        "swatches were the map's marker colours"
    )


def test_the_p1_caveat_is_stated_rather_than_hidden():
    """P1 has no marker colour of its own; the legend must say so.

    P1 and P2 are not neighbouring shades of the same idea -- P1 sends an
    email and P2 does not. Drawing them identically is a real limitation, and
    a legend that quietly omits it is the same defect in a smaller font.
    """
    assert "P1 has no colour of its own" in UI


@pytest.mark.parametrize(
    "stale",
    [
        "NOT_IMPLEMENTED</span>. A stub returning",
        "The SMS adapter has no provider wired",
    ],
)
def test_sms_is_no_longer_described_as_unimplemented(stale):
    """SmsChannel is implemented; the popover said otherwise.

    Understating a capability is a smaller sin than overstating one, but it is
    still the interface describing code that no longer exists.
    """
    assert stale not in UI


def test_sms_appears_in_the_p0_routing_line():
    from src.alerting.dispatcher import PRIORITY_ROUTING

    assert "sms" in PRIORITY_ROUTING["P0_EMERGENCY"]
    start = UI.index("Alert tier &middot; stream rail")
    end = UI.index("Archive layer", start)
    assert "sms + email + hook" in UI[start:end]


# ---------------------------------------------------------------------------
# "Never classified" must mean never classified
# ---------------------------------------------------------------------------

def test_not_assessed_rule_tests_only_the_class():
    """Containment is not evidence that inference was skipped.

    The rule read `predicted_class === 'NOT_ASSESSED' || !inc.inside_industrial`.
    That was true while the seeding path gated inference on `inside_ind`. After
    the gate came out, every detection reached the classifier and the second
    clause became a lie: 462 of 1,816 live rows were drawn slate and labelled
    "never classified" while holding TRANSIENT_HOTSPOT at 0.999 confidence.

    This is the CONTROLLED_PROCESS defect inverted. That one showed a
    confidence the model never produced; this one denied a classification the
    model did produce.
    """
    block = _marker_legend_block()
    rule = [ln for ln in block.splitlines() if "'NOT_ASSESSED'" in ln]
    assert rule, "the not-assessed rule is gone"
    assert len(rule) == 1, f"NOT_ASSESSED is tested in {len(rule)} places"
    assert "inside_industrial" not in rule[0], (
        "the not-assessed rule is keying on containment again, so classified "
        "detections outside industry will be labelled 'never classified'"
    )


def test_every_trained_class_has_a_marker_rule():
    """A class the model can emit must be nameable on the map.

    TRANSIENT_HOTSPOT had no rule at all, which is why it was being absorbed by
    a catch-all that described it wrongly. An unnamed class does not go
    undrawn -- it goes drawn as something else.
    """
    block = _marker_legend_block()
    for name in CLASS_NAME_TO_INDEX:
        assert f"'{name}'" in block, f"{name} has no marker rule and will fall through"


def test_no_rule_claims_a_detection_was_unclassified_by_position():
    """No marker rule may infer 'unassessed' from geography.

    Whether a detection sits inside an industrial polygon is a feature, not a
    statement about whether the classifier ran.
    """
    block = _marker_legend_block()
    for line in block.splitlines():
        if "inside_industrial" in line:
            assert "NOT_ASSESSED" not in line, f"position used as proof of non-assessment: {line.strip()}"


# ---------------------------------------------------------------------------
# A detection is a pixel, not a point
#
# FIRMS reports the CENTROID of a pixel that is 375 m (VIIRS) to 1 km (MODIS)
# across at nadir, and larger toward the edge of the swath. Drawn as a
# fixed-size dot at high zoom that reads as "the fire is at this rooftop",
# which is the reason markers appear not to sit on the plant they belong to.
# ---------------------------------------------------------------------------

def test_marker_footprint_is_drawn_in_metres_not_pixels():
    """L.circleMarker takes screen pixels and never scales with zoom.

    The footprint has to be L.circle, whose radius is metres, or it cannot
    represent a real distance on the ground.
    """
    start = UI.index("function renderMapMarkers()")
    end = UI.index("function initHeatmap()", start)
    body = UI[start:end]

    assert "L.circle(" in body, "no ground-truth footprint is drawn"
    assert "scan_km" in body and "track_km" in body


def test_footprint_is_withheld_when_the_pixel_size_is_unknown():
    """A guessed footprint is worse than none.

    A corpus built before scan/track were stored has no values, and defaulting
    to a nominal size would assert a precision that was never measured.
    """
    start = UI.index("function renderMapMarkers()")
    end = UI.index("function initHeatmap()", start)
    body = UI[start:end]

    assert "scanKm > 0 && trackKm > 0" in body, (
        "the footprint must be conditional on the corpus actually carrying it"
    )


def test_the_dossier_states_the_pixel_size_beside_the_coordinate():
    """Four decimals is ~11 m against a 375-1000 m pixel. Say both."""
    assert "km pixel" in UI


def test_basemaps_declare_a_native_zoom_ceiling():
    """Past a provider's real ceiling Esri returns a "Map data not yet
    available" placeholder, not nothing -- which is what filled the map when an
    operator zoomed into an incident."""
    assert UI.count("maxNativeZoom") >= 4, (
        "tile layers must cap requests at the provider's real ceiling and "
        "upscale, rather than requesting tiles that do not exist"
    )
