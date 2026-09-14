"""Feature Engineering Pipeline for Project SIH26162.

Extracts multi-spectral thermal physics, geospatial facility proximity,
temporal diurnal cycles, and recurrence baseline metrics from joined FIRMS detections.
Enforces strict featurization ordering (split-before-fit) to eliminate data leakage.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("feature_engineering")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

FACILITY_TYPES = [
    "steel_metallurgy",
    "petrochemical_refinery",
    "brick_kiln",
    "power_thermal",
    "general_industrial",
    "mining",
    # Photovoltaic and wind sites. Tagged industrial in OSM but incapable of
    # combustion, so a thermal detection there is a reflective artifact rather
    # than a heat source. Kept as its own category so the model can learn that
    # distinction instead of lumping them with refineries.
    "renewable_non_thermal",
    "non_industrial",
]

NUMERICAL_FEATURE_COLS = [
    "frp",
    "bright_ti4",
    "bright_ti5",
    "temp_diff",
    "frp_density",
    "ti4_ratio",
    "detection_confidence",
    "dist_to_industrial_km",
    "n_30d",
    "mu_frp",
    "z_frp",
    "frp_ratio",
    "hour_utc",
    "month",
    "dnbr",
    # Days between observation opening and this source becoming persistent.
    # Separates infrastructure (already recurring when the archive began) from
    # an event that was watched starting. -1 while never persistent, which is
    # also the default when the column is absent.
    "onset_lag_days",
    # Trailing-24h count, and that rate against the cell's own prior daily rate.
    #
    # These reach the model because the labelling rule reads them. Rule v6 was
    # first shipped without them and the result was strictly negative: the rule
    # scored the Deonar landfill fire 0.552 while the MODEL still scored 0.069,
    # because it was being asked to reproduce a decision made from features it
    # could not see -- and the anchors drifted down as it fitted noise in their
    # place. A rule input the model lacks is not a conservative choice; it is a
    # label the model cannot learn.
    #
    # Neither is a coordinate, so neither reintroduces spatial leakage. Both are
    # declared in LABEL_RULE_FEATURES, so the circularity audit ablates them.
    "n_24h",
    "burst_ratio",
    # Distinct other sources active nearby in the trailing window. The rule gates
    # the onset branch on it, so the model must be able to see it.
    "neighbourhood_active_keys",
]

CATEGORICAL_FEATURE_COLS = [
    "inside_industrial",
    "is_exact_match",
    "is_night",
    "is_harvest_season",
    "dnbr_available",
    # Land cover. The labelling rule reads this to pick the artifact floor -- a
    # smouldering understorey fire is genuinely low-FRP -- so the model must be
    # able to see it. Rule v6 shipped once with two rule inputs missing from this
    # list and the model could not learn its own labels; the training log said so
    # by printing the wrong ablation count, and this is the same trap.
    "in_forest",
]

# VIIRS reports confidence as a categorical class; MODIS reports a 0-100 integer.
# Both are mapped onto a common 0-100 scale so the model sees one feature.
VIIRS_CONFIDENCE_MAP = {
    "l": 20.0,   # low
    "n": 60.0,   # nominal
    "h": 90.0,   # high
}
DEFAULT_CONFIDENCE = 50.0


class FireFeaturePipeline:
    """Transforms joined satellite and GIS records into model-ready numerical feature matrices."""

    def __init__(self, scale_features: bool = False):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.encoder = OneHotEncoder(categories=[FACILITY_TYPES], handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        self.feature_names_: List[str] = []

    @staticmethod
    def _numeric_col(df: pd.DataFrame, name: str, default: float) -> pd.Series:
        """Fetches a column as a numeric Series, tolerating its absence.

        `df.get(name, default)` returns the bare scalar when the column is
        missing, which then fails on any Series method. Live FIRMS payloads
        always carry these fields, but simulation requests and partial frames do
        not, so the column is materialized explicitly here.
        """
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    @staticmethod
    def _bool_col(df: pd.DataFrame, name: str) -> pd.Series:
        """Fetches a boolean column as 0/1 ints, tolerating absence and string forms.

        Parquet round-trips turn these into the strings "True"/"False", so both
        the native bool and the stringified form must be accepted.
        """
        if name not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)
        col = df[name]
        if col.dtype == "object" or pd.api.types.is_string_dtype(col):
            return col.astype(str).str.strip().str.lower().isin(["true", "1"]).astype(int)
        return col.fillna(False).astype(bool).astype(int)

    @staticmethod
    def _normalize_confidence(raw, index: pd.Index) -> pd.Series:
        """Maps FIRMS confidence onto a common 0-100 numeric scale.

        VIIRS (VNP14IMGTDL_NRT / VJ114IMGTDL_NRT) emits the categorical classes
        'l' / 'n' / 'h'; MODIS (MCD14DL) emits a 0-100 integer. Both appear in the
        combined NRT feed, so a single normalized column is derived here rather
        than forcing the model to reconcile two encodings.
        """
        if raw is None:
            return pd.Series(DEFAULT_CONFIDENCE, index=index, dtype=float)

        if not isinstance(raw, pd.Series):
            raw = pd.Series(raw, index=index)

        # MODIS numeric path first; non-numeric entries fall through as NaN.
        numeric = pd.to_numeric(raw, errors="coerce")

        # VIIRS categorical path for whatever did not parse as a number.
        categorical = (
            raw.astype(str).str.strip().str.lower().str[:1].map(VIIRS_CONFIDENCE_MAP)
        )

        return numeric.fillna(categorical).fillna(DEFAULT_CONFIDENCE).clip(lower=0.0, upper=100.0)

    def _extract_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes derived physics, temporal, and spatial metrics from input DataFrame."""
        feat = pd.DataFrame(index=df.index)

        # 1. Thermal Physics Metrics
        frp = self._numeric_col(df, "frp", 0.0).clip(lower=0.0)
        ti4 = self._numeric_col(df, "bright_ti4", 300.0)
        ti5 = self._numeric_col(df, "bright_ti5", 290.0)
        scan = self._numeric_col(df, "scan", 0.4).clip(lower=0.1)
        track = self._numeric_col(df, "track", 0.4).clip(lower=0.1)

        feat["frp"] = frp
        feat["bright_ti4"] = ti4
        feat["bright_ti5"] = ti5
        feat["temp_diff"] = ti4 - ti5
        feat["frp_density"] = frp / (scan * track)
        feat["ti4_ratio"] = ti4 / (ti5 + 1e-4)

        # 1b. Sensor Detection Confidence (normalized across VIIRS and MODIS conventions)
        # Low-confidence detections are the dominant signature of solar glint off
        # metallic factory roofing and other transient radiometric artifacts, so
        # this is a direct discriminator for the TRANSIENT_HOTSPOT class.
        feat["detection_confidence"] = self._normalize_confidence(df.get("confidence"), index=df.index)

        # 1c. Sentinel-2 Optical Burn-Scar Validation (dNBR)
        # NaN where optical validation was never run or the scene was cloud-obscured.
        # XGBoost routes NaN natively; `dnbr_available` lets the model separate
        # "verified no burn scar" (contained industrial event) from "unknown".
        if "dnbr" in df.columns:
            dnbr_raw = pd.to_numeric(df["dnbr"], errors="coerce")
        else:
            dnbr_raw = pd.Series(np.nan, index=df.index, dtype=float)
        feat["dnbr"] = dnbr_raw.clip(lower=-1.0, upper=2.0)
        feat["dnbr_available"] = feat["dnbr"].notna().astype(int)

        # 1d. Onset lag: days between observation opening and this source
        # becoming persistent. -1 means "never persistent", and is also the
        # default when the state machine has not run, so a caller building a
        # feature vector by hand is not obliged to supply it.
        feat["onset_lag_days"] = self._numeric_col(df, "onset_lag_days", -1.0)
        # Burst. Defaults of 0 are the correct "no burst observed" reading, and
        # also what a corpus built before the recurrence tracker gained these
        # columns should produce -- a missing column must not become a signal.
        feat["n_24h"] = self._numeric_col(df, "n_24h", 0.0)
        feat["burst_ratio"] = self._numeric_col(df, "burst_ratio", 0.0)
        feat["neighbourhood_active_keys"] = self._numeric_col(
            df, "neighbourhood_active_keys", 0.0)

        # 2. Spatial Proximity Metrics
        feat["inside_industrial"] = self._bool_col(df, "inside_industrial")
        feat["is_exact_match"] = self._bool_col(df, "is_exact_match")
        # False means "not established", not "not forest": OSM maps about 80% of
        # India's forest area, so absence is weak evidence and the rule only ever
        # uses it to lower a threshold.
        feat["in_forest"] = self._bool_col(df, "in_forest")

        dist = self._numeric_col(df, "dist_to_industrial_km", 999.0)
        feat["dist_to_industrial_km"] = dist.clip(lower=0.0, upper=100.0)

        # 3. Recurrence Metrics (from State Machine)
        feat["n_30d"] = self._numeric_col(df, "n_30d", 0.0).astype(int)
        mu_frp = self._numeric_col(df, "mu_frp", 0.0)
        feat["mu_frp"] = mu_frp
        feat["z_frp"] = self._numeric_col(df, "z_frp", 0.0).clip(lower=-5.0, upper=20.0)
        feat["frp_ratio"] = self._numeric_col(df, "frp_ratio", 1.0).clip(lower=0.0, upper=50.0)

        # 4. Diurnal & Seasonal Temporal Signals
        ts = pd.to_datetime(df.get("timestamp_utc", pd.NaT), utc=True, errors="coerce")
        if "daynight" in df.columns:
            daynight = df["daynight"].astype(str).str.upper()
        else:
            daynight = pd.Series("D", index=df.index, dtype=object)
        feat["is_night"] = (daynight == "N").astype(int)

        hour = ts.dt.hour.fillna(12).astype(int)
        month = ts.dt.month.fillna(9).astype(int)
        feat["hour_utc"] = hour
        feat["month"] = month
        # Peak Indian crop residue burning windows. February-May covers the Rabi
        # harvest (wheat), October-November the post-Kharif paddy residue window.
        #
        # May was originally omitted, which was wrong on the data: across a full
        # national year, May carries 10.8% of open-ground burn detections -- more
        # than November (5.9%) and nearly double October (1.9%). In the Punjab
        # belt specifically it is the single largest month, with 1,568 detections
        # against November's 361. February likewise carries 12.1%. The original
        # [3,4,10,11] window captured 62.3% of burn detections; this one captures
        # 85.2%.
        feat["is_harvest_season"] = month.isin([2, 3, 4, 5, 10, 11]).astype(int)

        # Facility classification string
        feat["facility_type"] = df.get("facility_type", "non_industrial").fillna("non_industrial").astype(str)

        return feat

    def fit(self, df: pd.DataFrame) -> "FireFeaturePipeline":
        """Fits encoder (and scaler if enabled) on the training set ONLY."""
        raw = self._extract_raw_features(df)
        facility_types = raw[["facility_type"]].values
        self.encoder.fit(facility_types)

        # Construct feature names
        enc_names = [f"fac_{cat}" for cat in self.encoder.categories_[0]]
        self.feature_names_ = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS + enc_names

        if self.scale_features:
            num_data = raw[NUMERICAL_FEATURE_COLS].values
            self.scaler.fit(num_data)

        self.is_fitted = True
        logger.info("Fitted feature pipeline with %d total features.", len(self.feature_names_))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame into the model-ready feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform().")

        raw = self._extract_raw_features(df)

        # Encode facility types
        encoded_fac = self.encoder.transform(raw[["facility_type"]].values)
        enc_names = [f"fac_{cat}" for cat in self.encoder.categories_[0]]
        enc_df = pd.DataFrame(encoded_fac, columns=enc_names, index=df.index)

        # Assemble features
        num_df = raw[NUMERICAL_FEATURE_COLS].copy()
        if self.scale_features:
            scaled_num = self.scaler.transform(num_df.values)
            num_df = pd.DataFrame(scaled_num, columns=NUMERICAL_FEATURE_COLS, index=df.index)

        cat_df = raw[CATEGORICAL_FEATURE_COLS].copy()
        out_df = pd.concat([num_df, cat_df, enc_df], axis=1)

        # Ensure correct column ordering
        return out_df[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits on df and returns transformed feature matrix."""
        return self.fit(df).transform(df)
