"""Unit tests for ML Feature Pipeline, XGBoost Classifier, and SHAP Explainability."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.models.explainability import FireExplainer
from src.models.train_classifier import (
    CLASS_NAMES,
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
)
from src.pipeline.feature_engineering import FireFeaturePipeline


@pytest.fixture
def sample_raw_dataframe():
    """Generates synthetic dataframe mimicking joined FIRMS and OSM data."""
    return pd.DataFrame(
        {
            "frp": [25.0, 65.0, 5.0],
            "bright_ti4": [345.0, 375.0, 315.0],
            "bright_ti5": [295.0, 300.0, 298.0],
            "scan": [0.4, 0.4, 0.5],
            "track": [0.4, 0.4, 0.5],
            "daynight": ["D", "N", "D"],
            "timestamp_utc": [
                pd.Timestamp("2026-09-08 05:00:00", tz="UTC"),
                pd.Timestamp("2026-09-08 20:30:00", tz="UTC"),
                pd.Timestamp("2026-10-15 09:00:00", tz="UTC"),
            ],
            "inside_industrial": [True, True, False],
            "is_exact_match": [True, True, False],
            "dist_to_industrial_km": [0.0, 0.0, 25.4],
            "facility_type": ["steel_metallurgy", "petrochemical_refinery", "non_industrial"],
            "n_30d": [15, 1, 0],
            "mu_frp": [24.0, 12.0, 0.0],
            "z_frp": [0.5, 6.8, 0.0],
            "frp_ratio": [1.04, 5.4, 1.0],
        }
    )


def test_feature_pipeline_fit_transform(sample_raw_dataframe):
    """Verify feature pipeline outputs correct schema, columns, and types."""
    pipeline = FireFeaturePipeline()
    X = pipeline.fit_transform(sample_raw_dataframe)

    assert isinstance(X, pd.DataFrame)
    assert len(X) == 3
    assert "temp_diff" in X.columns
    assert "frp_density" in X.columns
    assert "is_night" in X.columns
    assert "fac_steel_metallurgy" in X.columns
    assert "fac_petrochemical_refinery" in X.columns
    assert "detection_confidence" in X.columns
    assert "dnbr" in X.columns

    # `dnbr` is deliberately NaN when Sentinel-2 validation has not run, so that
    # "unknown" stays distinguishable from a verified dNBR of 0.0 (no burn scar).
    # XGBoost consumes the NaN natively. Every other column must be populated.
    assert X["dnbr"].isnull().all(), "dnbr should be NaN when no optical validation is present"
    assert X["dnbr_available"].eq(0).all()
    assert not X.drop(columns=["dnbr"]).isnull().values.any()


def test_model_artifact_exists_and_predicts(sample_raw_dataframe):
    """Verify saved XGBoost model artifact loads and outputs valid probabilities."""
    assert MODEL_ARTIFACT_PATH.exists()
    assert PIPELINE_ARTIFACT_PATH.exists()

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_ARTIFACT_PATH))

    pipeline = FireFeaturePipeline()
    pipeline.fit(sample_raw_dataframe)
    X = pipeline.transform(sample_raw_dataframe)

    probs = model.predict_proba(X)
    assert probs.shape == (3, 4)
    # Probabilities must sum to 1.0 per sample
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), rtol=1e-5)


def test_shap_explainer_local_explanation():
    """Verify SHAP explainability engine produces required keys and narrative text."""
    explainer = FireExplainer()

    incident = {
        "frp": 70.0,
        "bright_ti4": 375.0,
        "bright_ti5": 298.0,
        "scan": 0.4,
        "track": 0.4,
        "daynight": "N",
        "timestamp_utc": "2026-09-08 20:30:00+00:00",
        "inside_industrial": True,
        "is_exact_match": True,
        "dist_to_industrial_km": 0.0,
        "facility_type": "petrochemical_refinery",
        "n_30d": 1,
        "mu_frp": 12.0,
        "z_frp": 6.8,
        "frp_ratio": 5.8,
    }

    result = explainer.explain_detection(incident, top_k=3)

    assert "predicted_class" in result
    assert "confidence_percent" in result
    assert "top_driving_factors" in result
    assert len(result["top_driving_factors"]) == 3
    assert "narrative_rationale" in result
    assert "Classified as" in result["narrative_rationale"]


def test_saved_metrics_evaluation():
    """Verify saved metrics file contains required evaluation metrics and thresholds."""
    assert METRICS_PATH.exists()
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert "test_macro_f1" in metrics
    assert "accidental_fire_recall" in metrics
    assert "confusion_matrix" in metrics
    assert "baselines" in metrics

    # Accidental fire recall must meet NTRO requirements (>= 95%)
    assert metrics["accidental_fire_recall"] >= 0.95
    # Must outperform majority class baseline
    assert metrics["test_macro_f1"] > metrics["baselines"]["dummy_majority_macro_f1"]
