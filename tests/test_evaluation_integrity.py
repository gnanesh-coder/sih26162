"""Tests guarding the integrity of the model evaluation.

These exist because the project previously reported a single blended macro F1 of
0.9986 against labels it had generated itself. That number was not wrong
arithmetically -- it was wrong epistemically. These tests keep the machinery that
exposes that problem from silently regressing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_classifier import (
    LABEL_RULE_FEATURES,
    METRICS_PATH,
    weak_label_real_detection,
)
from src.pipeline.feature_engineering import NUMERICAL_FEATURE_COLS, FireFeaturePipeline


# --------------------------------------------------------------------------
# Spatial leakage: the trap the problem-statement research explicitly warns of
# --------------------------------------------------------------------------

def test_no_coordinates_in_feature_space():
    """Latitude and longitude must never reach the model.

    Industrial facilities are static. A model handed raw coordinates memorizes
    where refineries are instead of learning what industrial combustion looks
    like, scores near-perfectly under a random split, and then fails completely
    when pointed at an unseen region.
    """
    pipeline = FireFeaturePipeline()
    forbidden = {"latitude", "longitude", "lat", "lon", "lng", "x", "y"}

    for col in NUMERICAL_FEATURE_COLS:
        assert col.lower() not in forbidden, f"Coordinate feature '{col}' would cause spatial leakage"


def test_pipeline_ignores_coordinate_columns_when_present(sample_detections):
    """Even when coordinates are in the input frame, they must not be emitted."""
    pipeline = FireFeaturePipeline()
    X = pipeline.fit_transform(sample_detections)

    emitted = {c.lower() for c in X.columns}
    assert "latitude" not in emitted
    assert "longitude" not in emitted


# --------------------------------------------------------------------------
# Weak labelling rule
# --------------------------------------------------------------------------

def test_established_flare_is_not_labelled_an_accident():
    """A long-running refinery flare is routine operation, not a disaster.

    This is the specific defect the original rule had: it labelled any high-FRP
    detection inside an industrial polygon as an accidental fire, which would
    classify every refinery's normal flare stack as an emergency.
    """
    established_flare = {
        "inside_industrial": True,
        "frp": 48.0,          # high energy
        "n_30d": 25,          # but seen constantly for a month
        "z_frp": 0.4,         # and entirely in line with its own baseline
    }
    assert weak_label_real_detection(established_flare) == 0  # PERSISTENT_BASELINE


def test_sudden_event_at_unestablished_location_is_an_accident():
    """High energy with no history inside a facility is the accident signature."""
    sudden = {
        "inside_industrial": True,
        "frp": 60.0,
        "n_30d": 0,
        "z_frp": 8.0,
    }
    assert weak_label_real_detection(sudden) == 1  # ACCIDENTAL_FIRE


def test_open_ground_biomass_and_artifacts_are_separated():
    burn = {"inside_industrial": False, "frp": 14.0, "n_30d": 0, "z_frp": 0.0}
    artifact = {"inside_industrial": False, "frp": 1.2, "n_30d": 0, "z_frp": 0.0}

    assert weak_label_real_detection(burn) == 2       # AGRICULTURAL_BURN
    assert weak_label_real_detection(artifact) == 3   # TRANSIENT_HOTSPOT


def test_label_rule_features_are_actually_used_by_the_rule():
    """The circularity audit is only meaningful if it ablates the right columns.

    If someone changes the labelling rule to read a new column without adding it
    to LABEL_RULE_FEATURES, the audit silently understates circularity. This
    test flips each declared feature and asserts it can change the outcome.
    """
    # FRP sits between the inside-polygon accident bar (10 MW) and the
    # outside-polygon one (25 MW), which is the only window where flipping
    # `inside_industrial` still changes the answer. Under rule v5 a detection
    # above 25 MW with no baseline is an accident on either side of a polygon
    # boundary -- deliberately, because absence of a map is not absence of
    # industry, and the ITC Deer Park tank fire scored 0.000 when it was not.
    base = {"inside_industrial": True, "frp": 12.0, "n_30d": 0, "z_frp": 0.0}
    assert weak_label_real_detection(base) == 1

    # Each of these perturbations touches a declared label-rule input.
    assert weak_label_real_detection({**base, "inside_industrial": False}) != 1
    assert weak_label_real_detection({**base, "frp": 1.0}) != 1
    assert weak_label_real_detection({**base, "n_30d": 30}) != 1
    # z_frp decides outside a polygon: an established unmapped source surging
    # past its own distribution is an accident rather than infrastructure. It
    # deliberately does NOT decide inside one -- that branch was written,
    # measured against the verified set, and removed. See INSIDE_SURGE_Z.
    persistent_outside = {"inside_industrial": False, "frp": 12.0,
                          "n_30d": 30, "z_frp": 0.5}
    assert weak_label_real_detection(persistent_outside) == 0
    assert weak_label_real_detection({**persistent_outside, "z_frp": 8.0}) == 1

    for feat in ("inside_industrial", "frp", "n_30d", "z_frp", "month"):
        assert feat in LABEL_RULE_FEATURES


# --------------------------------------------------------------------------
# Persisted metrics must stay honest
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def metrics():
    assert METRICS_PATH.exists(), "Model metrics missing - run src.models.train_classifier"
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_metrics_declare_label_provenance(metrics):
    """Anyone reading the metrics must see where the labels came from."""
    prov = metrics["label_provenance"]
    assert prov["verified_ground_truth"] == 0
    assert prov["real_detections_heuristic_labelled"] > 0
    assert "labelling_rule" in prov


def test_metrics_carry_explicit_caveats(metrics):
    """The headline numbers must not travel without their warnings attached."""
    caveats = metrics["caveats"]
    assert isinstance(caveats, list) and len(caveats) >= 3
    joined = " ".join(caveats).lower()
    assert "not field accuracy" in joined or "not field accuracy" in joined.replace(",", "")
    assert "verified" in joined


def test_metrics_report_real_and_synthetic_separately(metrics):
    """A blended score hides poor real-data performance behind easy synthetic rows."""
    slice_ = metrics["provenance_slice"]
    assert "real_detection_macro_f1" in slice_
    assert "synthetic_macro_f1" in slice_
    assert slice_["real_detection_n"] > 0


def test_spatial_block_cv_scores_real_detections_only(metrics):
    """Spatial CV must not be diluted by the synthetic majority."""
    scv = metrics["spatial_block_cv"]
    if scv.get("status") != "OK":
        pytest.skip(f"Spatial CV not scored: {scv.get('status')}")

    assert scv["scored_on"] == "real_detections_only"
    assert scv["n_blocks"] >= 2
    assert 0.0 <= scv["mean_macro_f1"] <= 1.0
    # The all-rows figure is kept only for comparison and must be labelled as such.
    assert "reference_mean_macro_f1_all_rows" in scv


def test_circularity_audit_present_and_interpreted(metrics):
    """The audit must record the ablated score and say what it means."""
    audit = metrics["circularity_audit"]
    assert audit["ablated_features"], "audit ablated nothing"
    assert 0.0 <= audit["ablated_macro_f1"] <= 1.0
    assert audit["full_feature_macro_f1"] >= audit["ablated_macro_f1"] - 0.05
    assert len(audit["interpretation"]) > 20


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def sample_detections():
    return pd.DataFrame(
        {
            "latitude": [22.47, 21.05, 19.10],
            "longitude": [70.05, 72.68, 72.90],
            "frp": [45.0, 12.0, 1.4],
            "bright_ti4": [360.0, 340.0, 310.0],
            "bright_ti5": [300.0, 295.0, 299.0],
            "scan": [0.4, 0.5, 0.6],
            "track": [0.4, 0.4, 0.5],
            "confidence": ["h", "n", "l"],
            "daynight": ["N", "D", "D"],
            "timestamp_utc": pd.to_datetime(
                ["2026-05-01 20:00", "2026-05-02 08:00", "2026-05-03 09:00"], utc=True
            ),
            "inside_industrial": [True, True, False],
            "is_exact_match": [True, False, False],
            "dist_to_industrial_km": [0.0, 0.0, 30.0],
            "facility_type": ["petrochemical_refinery", "steel_metallurgy", "non_industrial"],
            "n_30d": [1, 20, 0],
            "mu_frp": [8.0, 11.0, 0.0],
            "z_frp": [7.5, 0.3, 0.0],
            "frp_ratio": [5.6, 1.1, 1.0],
        }
    )


# --------------------------------------------------------------------------
# Confidence normalization across sensor conventions
# --------------------------------------------------------------------------

def test_viirs_and_modis_confidence_share_one_scale(sample_detections):
    """VIIRS 'l'/'n'/'h' and MODIS 0-100 must land on the same numeric axis."""
    pipeline = FireFeaturePipeline()
    X = pipeline.fit_transform(sample_detections)

    conf = X["detection_confidence"].tolist()
    assert conf[0] > conf[1] > conf[2], "high > nominal > low must be preserved"
    assert all(0.0 <= c <= 100.0 for c in conf)


def test_modis_numeric_confidence_passes_through():
    pipeline = FireFeaturePipeline()
    df = pd.DataFrame(
        {
            "frp": [10.0, 10.0],
            "confidence": ["85", "12"],
            "facility_type": ["non_industrial", "non_industrial"],
            "daynight": ["D", "D"],
            "timestamp_utc": pd.to_datetime(["2026-05-01", "2026-05-01"], utc=True),
        }
    )
    X = pipeline.fit_transform(df)
    assert X["detection_confidence"].tolist() == [85.0, 12.0]


def test_missing_confidence_falls_back_without_crashing():
    pipeline = FireFeaturePipeline()
    df = pd.DataFrame(
        {
            "frp": [10.0],
            "facility_type": ["non_industrial"],
            "daynight": ["D"],
            "timestamp_utc": pd.to_datetime(["2026-05-01"], utc=True),
        }
    )
    X = pipeline.fit_transform(df)
    assert X["detection_confidence"].notna().all()


# --------------------------------------------------------------------------
# Vectorised labelling must be semantically identical to the scalar rule
#
# The scalar rule was applied via df.iterrows(), building one dict per
# detection; at 2M rows that reached 12.5 GB resident and was still climbing.
# The vectorised replacement is only safe if it agrees exactly -- a silent
# divergence would relabel the corpus and invalidate every metric downstream.
# --------------------------------------------------------------------------

def test_vectorised_labelling_matches_scalar_rule():
    """Randomised inputs spanning every branch of the rule must agree."""
    from src.models.train_classifier import (
        weak_label_real_detection,
        weak_label_vectorised,
    )

    rng = np.random.default_rng(7)
    n = 4000
    df = pd.DataFrame({
        "inside_industrial": rng.choice([True, False], size=n),
        # Ranges deliberately straddle every threshold in the rule: frp 3.0 and
        # 10.0, n_30d 3 and 8.
        "frp": rng.uniform(0.0, 30.0, size=n).round(2),
        "n_30d": rng.integers(0, 15, size=n),
        "z_frp": rng.uniform(-2.0, 12.0, size=n).round(2),
    })

    vector = weak_label_vectorised(df)
    scalar = np.array([weak_label_real_detection(r) for r in df.to_dict("records")])

    mismatches = int((vector != scalar).sum())
    assert mismatches == 0, (
        f"{mismatches}/{n} labels diverge between the scalar and vectorised rules"
    )


def test_vectorised_labelling_handles_stringified_booleans():
    """Parquet round-trips turn inside_industrial into 'True'/'False' strings."""
    from src.models.train_classifier import weak_label_vectorised

    df = pd.DataFrame({
        "inside_industrial": ["True", "False", "true", "FALSE", "False"],
        "frp": [40.0, 12.0, 40.0, 1.0, 40.0],
        "n_30d": [0, 0, 20, 0, 0],
        "z_frp": [8.0, 0.0, 0.5, 0.0, 0.0],
    })
    labels = weak_label_vectorised(df)

    assert labels[0] == 1   # inside, no history, strong -> ACCIDENTAL_FIRE
    assert labels[1] == 2   # outside, moderate energy -> AGRICULTURAL_BURN
    assert labels[2] == 0   # inside, established -> PERSISTENT_BASELINE
    assert labels[3] == 3   # outside, marginal -> TRANSIENT_HOTSPOT
    # Rule v5: past UNMAPPED_ACCIDENT_FRP, a source with no baseline is an
    # accident whether or not a polygon covers it. Row 4 differs from row 1 only
    # in that nobody mapped it.
    assert labels[4] == 1


def test_vectorised_labelling_on_empty_and_missing_columns():
    from src.models.train_classifier import weak_label_vectorised

    assert len(weak_label_vectorised(pd.DataFrame({"frp": []}))) == 0
    # A frame missing inside_industrial must not raise; treat as outside.
    out = weak_label_vectorised(pd.DataFrame({"frp": [12.0], "n_30d": [0]}))
    assert out.tolist() == [2]
    # And 50 MW with no baseline is an accident, mapped or not.
    out = weak_label_vectorised(pd.DataFrame({"frp": [50.0], "n_30d": [0]}))
    assert out.tolist() == [1]


# --------------------------------------------------------------------------
# Class balancing
# --------------------------------------------------------------------------

def _imbalanced_corpus():
    rng = np.random.default_rng(3)
    rows = []
    # 0.04% ACCIDENTAL_FIRE, mirroring the national corpus.
    for label, count in [(2, 12000), (3, 6000), (0, 2000), (1, 20)]:
        for _ in range(count):
            rows.append({
                "target_label": label,
                "label_source": "heuristic_real",
                "acq_date": f"2026-{rng.integers(1, 13):02d}-15",
                "frp": float(rng.uniform(1, 50)),
            })
    return pd.DataFrame(rows)


def test_balancing_keeps_every_rare_class_row():
    """The class the system exists to find must never be sampled away."""
    from src.models.train_classifier import balance_corpus

    df = _imbalanced_corpus()
    balanced, report = balance_corpus(df, ratio_cap=50)

    assert report["status"] == "APPLIED"
    assert (balanced.target_label == 1).sum() == (df.target_label == 1).sum() == 20


def test_balancing_reduces_the_ratio_below_the_cap():
    from src.models.train_classifier import balance_corpus

    balanced, report = balance_corpus(_imbalanced_corpus(), ratio_cap=50)

    assert report["original_ratio"] > 100
    assert report["balanced_ratio"] <= 55, "ratio should respect the cap"


def test_balancing_preserves_monthly_spread():
    """Seasonality must survive: India's burning peak is half the annual signal."""
    from src.models.train_classifier import balance_corpus

    df = _imbalanced_corpus()
    balanced, _ = balance_corpus(df, ratio_cap=50)

    before = pd.to_datetime(df[df.target_label == 2].acq_date).dt.month.nunique()
    after = pd.to_datetime(balanced[balanced.target_label == 2].acq_date).dt.month.nunique()
    assert after == before, "every month present before balancing must survive it"


def test_balancing_is_skipped_when_already_balanced():
    from src.models.train_classifier import balance_corpus

    df = pd.DataFrame({
        "target_label": [0, 1, 2, 3] * 50,
        "label_source": ["heuristic_real"] * 200,
        "acq_date": ["2026-03-15"] * 200,
    })
    _, report = balance_corpus(df, ratio_cap=50)
    assert report["status"] == "NOT_NEEDED"


def test_balancing_report_carries_the_prior_distortion_caveat():
    """Subsampling breaks probability calibration; that must not go unsaid."""
    from src.models.train_classifier import balance_corpus

    _, report = balance_corpus(_imbalanced_corpus(), ratio_cap=50)
    assert "recalibration" in report["caveat"].lower()


# --------------------------------------------------------------------------
# Corpus provenance: every training row must be a real detection
# --------------------------------------------------------------------------
#
# The trainer used to augment the corpus with ~1,850 rows drawn from uniform
# distributions, including fabricated coordinates inside the India bounding box.
# Because balance_corpus() deliberately never subsamples ACCIDENTAL_FIRE, all
# 400 synthetic rows of that class survived into training -- roughly 13% of the
# one class the system exists to detect, drawn from distributions so clean they
# scored a macro F1 of 1.000. These tests keep that from coming back.

def test_corpus_contains_no_synthetic_rows(tmp_path):
    """Every row generate_training_corpus() returns must be a real detection."""
    from src.models.train_classifier import generate_training_corpus

    frame = pd.DataFrame({
        "latitude": [22.40, 30.50, 15.18],
        "longitude": [70.05, 75.80, 76.67],
        "frp": [12.0, 40.0, 9.0],
        "bright_ti4": [340.0, 360.0, 335.0],
        "bright_ti5": [300.0, 305.0, 298.0],
        "n_30d": [12, 0, 20],
        "z_frp": [0.4, 5.0, 0.2],
        "inside_industrial": [True, False, True],
        "is_exact_match": [True, False, True],
        "acq_date": pd.to_datetime(["2026-03-15", "2026-03-16", "2026-03-17"]),
    })
    path = tmp_path / "corpus.parquet"
    frame.to_parquet(path)

    corpus = generate_training_corpus(base_parquet=path)

    assert len(corpus) == len(frame)
    assert set(corpus["label_source"].unique()) == {"heuristic_real"}


def test_corpus_generation_refuses_to_invent_a_missing_dataset(tmp_path):
    """A missing corpus must fail loudly, not fall back to generated data.

    Substituting a fabricated stand-in produces scores that look like results.
    """
    from src.models.train_classifier import generate_training_corpus

    with pytest.raises(FileNotFoundError):
        generate_training_corpus(base_parquet=tmp_path / "does_not_exist.parquet")


def test_corpus_drops_rows_without_coordinates_rather_than_inventing_them(tmp_path):
    """A row with no coordinate is dropped, never given a random position.

    Spatial-block CV partitions on geography; an invented coordinate silently
    corrupts the thing that validation is measuring.
    """
    from src.models.train_classifier import generate_training_corpus

    frame = pd.DataFrame({
        "latitude": [22.40, np.nan],
        "longitude": [70.05, 75.80],
        "frp": [12.0, 40.0],
        "n_30d": [12, 0],
        "z_frp": [0.4, 5.0],
        "inside_industrial": [True, False],
        "is_exact_match": [True, False],
        "acq_date": pd.to_datetime(["2026-03-15", "2026-03-16"]),
    })
    path = tmp_path / "corpus.parquet"
    frame.to_parquet(path)

    corpus = generate_training_corpus(base_parquet=path)

    assert len(corpus) == 1
    assert corpus["latitude"].notna().all()
    assert corpus["longitude"].notna().all()


def test_reported_metrics_declare_zero_synthetic_rows():
    """The published metrics file must not claim synthetic augmentation."""
    if not METRICS_PATH.exists():
        pytest.skip("no metrics file on disk yet")
    provenance = json.loads(METRICS_PATH.read_text())["label_provenance"]
    assert provenance["synthetic_augmented"] == 0


# --------------------------------------------------------------------------
# Physics branches in the labelling rule
# --------------------------------------------------------------------------

def test_non_combustion_sites_are_never_labelled_crop_burns():
    """A photovoltaic array has no biomass, so it cannot produce a crop burn.

    80% of detections at Khavda and 85% at Pavagada were labelled
    AGRICULTURAL_BURN by the outside-polygon energy branch before this.
    """
    from src.models.train_classifier import weak_label_real_detection, weak_label_vectorised

    row = {
        "inside_industrial": False, "facility_type": "renewable_non_thermal",
        "frp": 25.0, "n_30d": 40, "z_frp": 0.5, "acq_date": "2026-03-15",
    }
    assert weak_label_real_detection(row) == 3
    assert weak_label_vectorised(pd.DataFrame([row]))[0] == 3

    # The override must beat every other branch, including persistent recurrence.
    hot = dict(row, frp=120.0, z_frp=9.0)
    assert weak_label_real_detection(hot) == 3
    assert weak_label_vectorised(pd.DataFrame([hot]))[0] == 3


def test_artifact_floor_is_seasonal():
    """Smouldering residue at 2 MW is a crop burn in harvest season, not an artifact."""
    from src.models.train_classifier import weak_label_real_detection, weak_label_vectorised

    base = {"inside_industrial": False, "facility_type": "non_industrial",
            "frp": 2.0, "n_30d": 1, "z_frp": 0.1}

    harvest = dict(base, acq_date="2025-11-05")   # post-Kharif
    off = dict(base, acq_date="2025-07-05")       # monsoon, no residue burning

    assert weak_label_real_detection(harvest) == 2
    assert weak_label_real_detection(off) == 3
    assert weak_label_vectorised(pd.DataFrame([harvest]))[0] == 2
    assert weak_label_vectorised(pd.DataFrame([off]))[0] == 3

    # The floor still exists in harvest season.
    assert weak_label_real_detection(dict(harvest, frp=1.0)) == 3


def test_vectorised_matches_scalar_with_dates_present():
    """The equivalence test must exercise the seasonal branch, not bypass it."""
    from src.models.train_classifier import weak_label_real_detection, weak_label_vectorised

    rng = np.random.default_rng(11)
    rows = []
    for _ in range(400):
        rows.append({
            "inside_industrial": bool(rng.integers(0, 2)),
            "facility_type": str(rng.choice(
                ["non_industrial", "renewable_non_thermal", "petrochemical_refinery"])),
            "frp": float(rng.uniform(0.2, 60.0)),
            "n_30d": int(rng.integers(0, 40)),
            "z_frp": float(rng.uniform(-2.0, 10.0)),
            "acq_date": f"2026-{int(rng.integers(1, 13)):02d}-15",
        })
    frame = pd.DataFrame(rows)

    scalar = np.array([weak_label_real_detection(r) for r in rows])
    vector = weak_label_vectorised(frame)
    assert (scalar == vector).all()


# --------------------------------------------------------------------------
# Probability calibration after class subsampling
# --------------------------------------------------------------------------
#
# balance_corpus() caps the majority classes, so the model is trained under
# priors that do not match the world. Its raw probabilities over-state the rare
# classes, and those numbers were being served to operators as confidence.

def test_prior_correction_undoes_the_subsampling_distortion():
    from src.models.train_classifier import prior_correction_factors

    report = {
        "status": "APPLIED",
        "original_counts": {"PERSISTENT_BASELINE": 100, "ACCIDENTAL_FIRE": 10,
                            "AGRICULTURAL_BURN": 1000, "TRANSIENT_HOTSPOT": 100},
        "balanced_counts": {"PERSISTENT_BASELINE": 100, "ACCIDENTAL_FIRE": 10,
                            "AGRICULTURAL_BURN": 100, "TRANSIENT_HOTSPOT": 100},
    }
    factors = prior_correction_factors(report)

    # AGRICULTURAL_BURN was cut 10x, so it must be scaled back up; the untouched
    # classes are correspondingly over-represented and scale down.
    assert factors[2] > 1.0
    assert factors[1] < 1.0
    assert factors[0] < 1.0


def test_prior_correction_returns_a_distribution():
    from src.models.train_classifier import apply_prior_correction

    raw = np.array([0.05, 0.80, 0.10, 0.05])
    out = apply_prior_correction(raw)

    assert out.shape == raw.shape
    assert abs(out.sum() - 1.0) < 1e-9
    assert (out >= 0).all()


def test_prior_correction_lowers_rare_class_confidence():
    """The rare class is over-represented in training, so its confidence must fall."""
    from src.models.train_classifier import apply_prior_correction, prior_correction_factors

    factors = prior_correction_factors()
    if not factors:
        pytest.skip("no balancing applied in the current metrics file")

    raw = np.array([0.05, 0.80, 0.10, 0.05])
    out = apply_prior_correction(raw)
    assert out[1] < raw[1]


def test_prior_correction_is_a_no_op_without_balancing():
    from src.models.train_classifier import apply_prior_correction

    raw = np.array([0.1, 0.2, 0.3, 0.4])
    unchanged = apply_prior_correction(raw, factors=None if False else {})
    assert np.allclose(unchanged, raw)


def test_served_probabilities_declare_their_basis():
    """An operator must be able to see that the number was adjusted."""
    from src.models.train_classifier import METRICS_PATH
    if not METRICS_PATH.exists():
        pytest.skip("no trained model on disk")

    import inspect
    from src.models import explainability
    src = inspect.getsource(explainability)
    assert "PRIOR_CORRECTED" in src
    assert "confidence_percent_uncorrected" in src


# --------------------------------------------------------------------------
# FRP thresholds must be relative, not absolute
# --------------------------------------------------------------------------
#
# A 5 MW "this is industrial" gate is calibrated to Indian refineries. Permian
# gas flares burn at a median 1.12 MW, so 2,587 detections that recurred at a
# fixed wellhead for weeks were labelled sun-glint artifacts.

def test_thresholds_are_derived_from_the_local_distribution():
    from src.models.train_classifier import derive_frp_thresholds

    hot = pd.DataFrame({"frp": np.linspace(1.0, 100.0, 20000),
                        "inside_industrial": [False] * 20000})
    cool = pd.DataFrame({"frp": np.linspace(0.1, 5.0, 20000),
                         "inside_industrial": [False] * 20000})

    th_hot = derive_frp_thresholds(hot)
    th_cool = derive_frp_thresholds(cool)

    assert th_hot.basis == "PERCENTILE_LOCAL"
    assert th_cool.basis == "PERCENTILE_LOCAL"
    # A cooler region must get a lower bar for "this is real combustion".
    assert th_cool.persistent_gate < th_hot.persistent_gate
    assert th_cool.artifact_floor < th_hot.artifact_floor


def test_small_corpus_falls_back_to_absolutes_and_says_so():
    """A percentile over a handful of readings is noise, not calibration."""
    from src.models.train_classifier import derive_frp_thresholds, DEFAULT_FRP_THRESHOLDS

    th = derive_frp_thresholds(pd.DataFrame({"frp": [1.0, 2.0, 3.0],
                                             "inside_industrial": [False] * 3}))
    assert th.basis == "ABSOLUTE_FALLBACK"
    assert th.persistent_gate == DEFAULT_FRP_THRESHOLDS.persistent_gate


def test_thresholds_are_computed_outside_polygons_only():
    """Industrial density must not move the floor for open ground."""
    from src.models.train_classifier import derive_frp_thresholds

    df = pd.DataFrame({
        "frp": np.concatenate([np.full(10000, 500.0), np.linspace(0.1, 4.0, 10000)]),
        "inside_industrial": [True] * 10000 + [False] * 10000,
    })
    th = derive_frp_thresholds(df)
    assert th.persistent_gate < 10.0  # the 500 MW industrial block is excluded


def test_a_low_energy_recurrent_source_survives_a_cool_region():
    """The Permian case: recurrence plus modest energy must not read as artifact."""
    from src.models.train_classifier import (
        FrpThresholds, weak_label_real_detection, weak_label_vectorised,
    )

    cool = FrpThresholds(persistent_gate=1.33, artifact_floor=0.87,
                         artifact_floor_harvest=0.58, basis="PERCENTILE_LOCAL",
                         n_samples=6689)
    flare = {"inside_industrial": False, "facility_type": "non_industrial",
             "frp": 2.0, "n_30d": 25, "z_frp": 0.2, "acq_date": "2026-08-15"}

    assert weak_label_real_detection(flare, cool) == 0            # unmapped industry
    assert weak_label_vectorised(pd.DataFrame([flare]), cool)[0] == 0
    # Recurrence now decides on its own, so the same detection reads as
    # industrial under India-calibrated thresholds too. That is the contract:
    # a source stationary for a month is infrastructure wherever it is, and how
    # brightly it burns is a separate question from what it is.
    assert weak_label_real_detection(flare) == 0
    assert weak_label_vectorised(pd.DataFrame([flare]))[0] == 0


def test_percentiles_reproduce_the_previous_india_behaviour():
    """The change must adapt elsewhere without moving the corpus it was tuned on."""
    from src.models.train_classifier import (
        ARTIFACT_FRP_FLOOR, PERSISTENT_FRP_GATE, derive_frp_thresholds,
    )

    corpus = Path("data/processed/firms_industrial_joined_12m.parquet")
    if not corpus.exists():
        pytest.skip("national corpus not present")

    th = derive_frp_thresholds(pd.read_parquet(corpus, columns=["frp", "inside_industrial"]))
    assert abs(th.persistent_gate - PERSISTENT_FRP_GATE) < 0.5
    assert abs(th.artifact_floor - ARTIFACT_FRP_FLOOR) < 0.5



def test_recurrence_outranks_energy_outside_polygons():
    """A month of recurrence in one cell is industrial however weakly it burns.

    The energy gate was the largest single transferability failure: over the
    Permian Basin it discarded 2,587 recurring wellhead flares as artifacts.
    """
    from src.models.train_classifier import weak_label_real_detection, weak_label_vectorised

    faint = {"inside_industrial": False, "facility_type": "non_industrial",
             "frp": 0.9, "n_30d": 30, "z_frp": 0.1, "acq_date": "2026-08-15"}
    assert weak_label_real_detection(faint) == 0
    assert weak_label_vectorised(pd.DataFrame([faint]))[0] == 0


def test_agricultural_burning_is_untouched_by_the_override():
    """Crop fires migrate, so they never accumulate recurrence in one cell.

    Zero of 1,995 verified Punjab crop-burn detections reach n_30d >= 8, which
    is what makes the override safe rather than merely convenient.
    """
    from src.models.train_classifier import weak_label_real_detection

    crop = {"inside_industrial": False, "facility_type": "non_industrial",
            "frp": 6.0, "n_30d": 1, "z_frp": 0.4, "acq_date": "2025-11-05"}
    assert weak_label_real_detection(crop) == 2


# --------------------------------------------------------------------------
# A model must be able to name its own inputs
# --------------------------------------------------------------------------
#
# Three trainings once raced to write the same two files, and the only way to
# tell which model was on disk was to cross-reference log timestamps against
# class distributions. These make that a one-line check.

def test_metrics_identify_the_run_that_produced_them():
    if not METRICS_PATH.exists():
        pytest.skip("no trained model on disk")
    run = json.loads(METRICS_PATH.read_text()).get("run", {})

    assert run.get("run_id"), "metrics do not identify their run"
    assert run.get("trained_at_utc"), "metrics do not record when they were trained"
    assert isinstance(run.get("labelling_rule_version"), int)


def test_metrics_record_the_thresholds_the_rule_actually_used():
    """Logged-but-not-persisted provenance is provenance nobody can check."""
    if not METRICS_PATH.exists():
        pytest.skip("no trained model on disk")
    th = json.loads(METRICS_PATH.read_text()).get("frp_thresholds")

    assert th is not None, "frp_thresholds reached the metrics file as null"
    assert th["basis"] in {"PERCENTILE_LOCAL", "ABSOLUTE_FALLBACK"}
    assert th["artifact_floor_mw"] > 0
    assert "persistent_gate_mw_diagnostic_only" in th


def test_rule_version_is_documented():
    """Every version must have a history line, so a bump cannot be silent.

    Asserting a literal number here only caught a stale constant; it did not
    catch the rule changing underneath it. Requiring the history to cover the
    current version means a bump forces an explanation of what changed.
    """
    import inspect

    from src.models import train_classifier as tc

    src = inspect.getsource(tc)
    header = src[: src.index("LABELLING_RULE_VERSION =")]
    for v in range(1, tc.LABELLING_RULE_VERSION + 1):
        assert f"#   {v}  " in header, f"rule version {v} has no history line"


def test_trained_metrics_match_the_current_rule_version():
    """A model trained under an older rule must not be quoted as current."""
    from src.models.train_classifier import LABELLING_RULE_VERSION

    if not METRICS_PATH.exists():
        pytest.skip("no trained model on disk")
    recorded = json.loads(METRICS_PATH.read_text())["run"]["labelling_rule_version"]

    assert recorded == LABELLING_RULE_VERSION, (
        f"artifact was trained under rule v{recorded}, source is now "
        f"v{LABELLING_RULE_VERSION}; retrain before quoting its metrics"
    )


# --------------------------------------------------------------------------
# Serving guards: the rule generalises further than the model can
# --------------------------------------------------------------------------

def test_agricultural_burn_is_withheld_outside_the_calibrated_region():
    """The model cannot learn this and never will.

    Its feature vector is coordinate-free on purpose, and its training corpus is
    entirely Indian -- so an "in domain" flag would be constant across every
    training row and carry no signal. The 2019 ITC Deer Park tank-farm fire in
    Texas came back AGRICULTURAL_BURN on 10 of 11 detections.
    """
    from src.models.train_classifier import apply_serving_guards

    served, note = apply_serving_guards("AGRICULTURAL_BURN", lat=29.73, lon=-95.09)
    assert served == "TRANSIENT_HOTSPOT"
    assert note and "outside the region" in note


def test_agricultural_burn_survives_inside_the_calibrated_region():
    """A guard that fires everywhere would delete the crop-burn class."""
    from src.models.train_classifier import apply_serving_guards

    served, note = apply_serving_guards("AGRICULTURAL_BURN", lat=30.75, lon=75.50)
    assert served == "AGRICULTURAL_BURN"
    assert note is None


def test_non_combustion_land_use_can_never_be_reported_as_fire():
    from src.models.train_classifier import apply_serving_guards

    served, note = apply_serving_guards(
        "ACCIDENTAL_FIRE", lat=24.11, lon=69.37, facility_type="renewable_non_thermal")
    assert served == "TRANSIENT_HOTSPOT"
    assert note and "non-combustion" in note


def test_guards_leave_everything_else_alone():
    from src.models.train_classifier import apply_serving_guards

    for cls in ("ACCIDENTAL_FIRE", "PERSISTENT_BASELINE", "TRANSIENT_HOTSPOT"):
        assert apply_serving_guards(cls, lat=22.34, lon=69.87) == (cls, None)


def test_burst_requires_all_three_conditions():
    """Burst alone is useless at detection granularity: 18.6% of the 12-month
    archive exceeds a 3x burst, because one prior detection and two today is a
    large ratio and nothing else."""
    from src.models.train_classifier import (
        BURST_MIN_N24, BURST_MIN_N30, BURST_RATIO, weak_label_real_detection,
    )

    burst = {"inside_industrial": True, "frp": 12.0, "z_frp": 0.5,
             "n_30d": BURST_MIN_N30, "n_24h": BURST_MIN_N24,
             "burst_ratio": BURST_RATIO + 1}
    assert weak_label_real_detection(burst) == 1, "a real burst must read as an accident"

    # Drop each condition in turn; none alone may trigger it.
    assert weak_label_real_detection({**burst, "burst_ratio": BURST_RATIO}) == 0
    assert weak_label_real_detection({**burst, "n_24h": BURST_MIN_N24 - 1}) == 0
    assert weak_label_real_detection({**burst, "n_30d": BURST_MIN_N30 - 1}) != 1
