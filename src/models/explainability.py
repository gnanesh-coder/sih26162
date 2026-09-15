"""SHAP (SHapley Additive exPlanations) Explainability Engine for Project SIH26162.

Computes exact game-theoretic feature attribution using TreeSHAP on the trained XGBoost model.
Provides both global feature importance rankings and local per-incident factor breakdowns.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
try:
    import shap
except (ImportError, OSError):
    shap = None

import xgboost as xgb

from src.models.train_classifier import (
    CLASS_NAMES,
    MODEL_ARTIFACT_PATH,
    PIPELINE_ARTIFACT_PATH,
    apply_prior_correction,
)

logger = logging.getLogger("explainability")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _confidence_percent(probability: float) -> float:
    """Percent confidence that never rounds up into a claim of certainty.

    The model reproduces a deterministic labelling rule, so its decision
    boundary is sharp and softmax saturates: a typical maximum here is
    0.9996. `round(99.96222, 1)` is 100.0, and displaying 100% asserts a
    certainty the model did not express -- on 88.6% of rows, as measured.

    Anything short of an exact 1.0 is therefore floored at 99.9%. The change
    is cosmetically tiny and the distinction is the whole point: 99.9% is a
    very confident model, 100% is a model that cannot be wrong.
    """
    if probability >= 1.0:
        return 100.0
    return float(min(round(probability * 100.0, 1), 99.9))


class FireExplainer:
    """Provides game-theoretic SHAP explanations for XGBoost fire classification decisions."""

    def __init__(
        self,
        model_path: Path = MODEL_ARTIFACT_PATH,
        pipeline_path: Path = PIPELINE_ARTIFACT_PATH,
    ):
        if not model_path.exists() or not pipeline_path.exists():
            raise FileNotFoundError("Model or feature pipeline artifact not found. Please train model first.")

        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))

        # Load fitted pipeline
        self.pipeline = joblib.load(pipeline_path)

        # Initialize TreeExplainer (or use native XGBoost TreeSHAP engine)
        self.explainer = None
        if shap is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized shap.TreeExplainer successfully.")
            except Exception as e:
                logger.warning("shap.TreeExplainer init failed (%s); using native XGBoost TreeSHAP.", e)
        else:
            logger.info("Using native XGBoost TreeSHAP engine.")

        self.feature_names = self.pipeline.feature_names_
        logger.info("Initialized FireExplainer with %d features.", len(self.feature_names))

    def _compute_shap_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Internal helper computing TreeSHAP contributions via SHAP package or native XGBoost."""
        if self.explainer is not None:
            try:
                shap_raw = self.explainer.shap_values(X)
                return shap_raw
            except Exception:
                pass

        # Native XGBoost TreeSHAP fallback (pred_contribs=True)
        dmat = xgb.DMatrix(X)
        contribs = self.model.get_booster().predict(dmat, pred_contribs=True)
        return contribs

    def compute_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculates global mean absolute SHAP values across all classes and samples.

        Args:
            X: Feature matrix DataFrame.

        Returns:
            DataFrame with 'feature' and 'importance' sorted descending.
        """
        shap_values = self._compute_shap_matrix(X)
        # Check if 3D array from native XGBoost (n_samples, n_classes, n_features + 1)
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Drop the bias column at index -1
            mean_abs = np.abs(shap_values[:, :, :len(self.feature_names)]).mean(axis=(0, 1))
        elif isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return importance_df

    def predict_detection(self, hotspot_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Class and calibrated confidence, without the SHAP attribution.

        `explain_detection` costs about 30 ms per detection and almost all of
        it is TreeSHAP. That is the right price for something a human is going
        to read, and the wrong price for the roughly two thousand background
        detections per refresh that nobody will ever open.

        The split is: predict everything, explain what becomes an alert. The
        prior correction is applied here exactly as it is there -- raw
        probabilities are calibrated to the subsampled corpus, where the rare
        classes are roughly 50x over-represented, so reporting them unshifted
        would overstate a P0 call substantially.
        """
        df_single = pd.DataFrame([hotspot_dict])
        X_single = self.pipeline.transform(df_single)

        raw_probs = self.model.predict_proba(X_single)[0]
        probs = apply_prior_correction(raw_probs)
        pred_class_idx = int(np.argmax(probs))

        return {
            "predicted_class": CLASS_NAMES.get(pred_class_idx, "UNKNOWN"),
            "predicted_class_id": pred_class_idx,
            "confidence_percent": _confidence_percent(probs[pred_class_idx]),
        }

    def explain_detection(self, hotspot_dict: Dict[str, Any], top_k: int = 4) -> Dict[str, Any]:
        """Produces local decision explanation for an individual thermal hotspot detection.

        Args:
            hotspot_dict: Dictionary containing raw hotspot attributes (lat, lon, frp, etc.).
            top_k: Number of key driving factors to return.

        Returns:
            Dictionary with predicted class, probabilities, top driving factors, and textual narrative.
        """
        df_single = pd.DataFrame([hotspot_dict])
        X_single = self.pipeline.transform(df_single)

        # Predict class & probability distribution.
        #
        # Raw probabilities are calibrated to the SUBSAMPLED corpus, where the
        # rare classes are roughly 50x over-represented relative to the world.
        # Reporting those as confidence overstates a P0 call substantially, so
        # the priors are shifted back before the number is shown to anyone.
        raw_probs = self.model.predict_proba(X_single)[0]
        probs = apply_prior_correction(raw_probs)
        pred_class_idx = int(np.argmax(probs))
        pred_class_name = CLASS_NAMES.get(pred_class_idx, "UNKNOWN")
        confidence_pct = _confidence_percent(probs[pred_class_idx])
        uncorrected_pct = float(round(raw_probs[int(np.argmax(raw_probs))] * 100.0, 1))

        # Compute SHAP values for this single sample
        shap_raw = self._compute_shap_matrix(X_single)
        if isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 3:
            # Native XGBoost shape: (1, n_classes, n_features + 1)
            sample_shap = shap_raw[0, pred_class_idx, :len(self.feature_names)]
        elif isinstance(shap_raw, list):
            sample_shap = shap_raw[pred_class_idx][0]
        elif isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 2:
            sample_shap = shap_raw[0]
        else:
            sample_shap = np.zeros(len(self.feature_names))

        # Rank features by absolute impact on predicted class
        factor_pairs = []
        for feat_name, shap_val, feat_val in zip(self.feature_names, sample_shap, X_single.iloc[0]):
            factor_pairs.append({
                "feature": feat_name,
                "shap_value": float(round(shap_val, 3)),
                "feature_value": float(round(feat_val, 2)) if isinstance(feat_val, (int, float, np.number)) else str(feat_val),
            })

        # Sort by impact direction (positive factors drive prediction toward this class)
        factor_pairs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_factors = factor_pairs[:top_k]

        # Construct human-readable narrative for NTRO intelligence operators
        narratives = []
        for f in top_factors:
            sign = "+" if f["shap_value"] > 0 else "-"
            narratives.append(f"{sign}{abs(f['shap_value']):.2f} via {f['feature']} (val={f['feature_value']})")

        rationale_text = (
            f"Classified as {pred_class_name} (Confidence: {confidence_pct}%) driven by: "
            + "; ".join(narratives)
        )

        return {
            "predicted_class_id": pred_class_idx,
            "predicted_class": pred_class_name,
            "confidence_percent": confidence_pct,
            "class_probabilities": {CLASS_NAMES[i]: float(round(p, 4)) for i, p in enumerate(probs)},
            # Provenance of the number above: corrected back to real-world class
            # priors, with the uncorrected figure kept so the size of the
            # distortion is visible rather than hidden.
            "probability_basis": "PRIOR_CORRECTED",
            "confidence_percent_uncorrected": uncorrected_pct,
            "top_driving_factors": top_factors,
            "narrative_rationale": rationale_text,
        }


# Matches the narrative this module writes at line ~169: "+1.14 via frp (val=85.4)".
# Parsing our own formatted output is a coupling worth naming: the rationale is
# stored on the incident as a single string, and the dashboard needs the numbers
# back. The alternative -- re-running TreeSHAP on every dossier open -- costs an
# inference per click for values that were already computed once.
_FACTOR_RE = re.compile(
    r"(?P<sign>[+-])(?P<mag>\d+(?:\.\d+)?)\s+via\s+(?P<feature>[A-Za-z0-9_]+)"
    r"(?:\s*\(val=(?P<value>[^)]*)\))?"
)

# Human labels for the feature names the model actually uses. Anything not
# listed falls back to the raw column name -- a missing label should look like a
# missing label, not like a different feature.
FEATURE_LABELS = {
    "frp": "Radiative power",
    "frp_density": "Radiative power density",
    "frp_ratio": "FRP vs local baseline",
    "z_frp": "FRP z-score",
    "n_30d": "30-day recurrence",
    "mu_frp": "Mean baseline FRP",
    "onset_lag_days": "Onset lag",
    "dist_to_industrial_km": "Distance to facility",
    "inside_industrial": "Inside mapped polygon",
    "is_exact_match": "Exact polygon match",
    "bright_ti4": "Brightness temp (I-4)",
    "bright_ti5": "Brightness temp (I-5)",
    "temp_diff": "I-4 minus I-5",
    "ti4_ratio": "I-4 ratio",
    "hour_utc": "Hour of day",
    "month": "Month",
    "is_night": "Night overpass",
    "is_harvest_season": "Harvest season",
    "dnbr": "Burn scar (dNBR)",
    "dnbr_available": "dNBR measured",
    "detection_confidence": "Sensor confidence",
}


def parse_rationale_factors(rationale: Optional[str], top_k: int = 4) -> List[Dict[str, Any]]:
    """Recovers the per-feature SHAP contributions from a stored rationale.

    Returns [] when the rationale carries no attribution -- which is the honest
    answer for detections classified by the state machine rather than the model.
    A caller that renders bars must render none of them in that case: the
    dashboard previously showed three fixed values for every incident, which is
    a fabricated measurement sitting under the heading "TreeSHAP attribution".
    """
    if not rationale:
        return []

    factors: List[Dict[str, Any]] = []
    for m in _FACTOR_RE.finditer(str(rationale)):
        magnitude = float(m.group("mag"))
        signed = magnitude if m.group("sign") == "+" else -magnitude
        name = m.group("feature")
        if name in FEATURE_LABELS:
            label = FEATURE_LABELS[name]
        elif name.startswith("fac_"):
            # One-hot facility columns are generated, so they cannot be listed
            # individually without the table going stale the next time the OSM
            # taxonomy grows.
            label = "Facility type: " + name[4:].replace("_", " ")
        else:
            label = name.replace("_", " ")
        factors.append({
            "feature": name,
            "label": label,
            "shap_value": round(signed, 4),
            "abs_shap": round(abs(signed), 4),
            "feature_value": (m.group("value") or "").strip() or None,
            "direction": "toward" if signed > 0 else "against",
        })

    factors.sort(key=lambda f: f["abs_shap"], reverse=True)
    return factors[:top_k]


if __name__ == "__main__":
    explainer = FireExplainer()

    # Test sample hotspot: sudden high-energy anomaly inside refinery polygon
    test_incident = {
        "frp": 65.0,
        "bright_ti4": 372.0,
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
        "frp_ratio": 5.4,
    }

    explanation = explainer.explain_detection(test_incident)
    print("\n=== SAMPLE SHAP LOCAL EXPLANATION ===")
    print(f"Prediction: {explanation['predicted_class']} ({explanation['confidence_percent']}%)")
    print(f"Rationale: {explanation['narrative_rationale']}")
    print("\nTop Driving Factors:")
    for f in explanation["top_driving_factors"]:
        print(f"  * {f['feature']}: SHAP = {f['shap_value']:+.3f} (Value: {f['feature_value']})")
