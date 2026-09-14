"""Sub-pixel fire temperature retrieval from dual-band brightness temperatures.

WHY THIS EXISTS
---------------
Every feature the classifier currently leans on -- `inside_industrial`,
`is_exact_match`, `n_30d`, `frp`, `z_frp` -- is an input to the weak-labelling
rule that produced its training labels. The circularity audit measures the
consequence: ablate those five and macro F1 falls from 0.999 to 0.729. The model
is substantially reproducing its own rule.

Adding more detections cannot fix that, because more rows of the same features
carry the same dependency. What fixes it is a feature derived from physics the
labelling rule never consults.

Absolute combustion temperature is that feature. Industrial gas flares burn
methane at roughly 1600-2500 K. Biomass smoulders and flames between about
600-1000 K. That separation is a property of the fuel, not of a map layer or a
recurrence counter, and it is exactly the discriminator the problem statement's
own research calls for when it specifies VIIRS Nightfire Planck-curve fitting.

THE METHOD
----------
A 375 m pixel containing a 10 m flare is overwhelmingly background. The observed
radiance is a mixture:

    L(lambda) = p * B(lambda, T_fire) + (1 - p) * B(lambda, T_background)

with B the Planck function, p the fraction of the pixel occupied by fire. Two
unknowns, so two channels suffice -- the bi-spectral method of Dozier (1981).
FIRMS supplies a mid-wave channel (VIIRS I-4 at 3.74 um, MODIS T21 at 3.96 um)
and a thermal channel (VIIRS I-5 at 11.45 um, MODIS T31 at 11.03 um). The
mid-wave channel is acutely sensitive to small hot areas; the thermal channel is
dominated by background. Their disagreement is what carries the temperature.

Brightness temperatures are inverted to radiance, then solved for (T_fire, p) by
vectorised bisection -- 2M detections at once rather than a root-find per row.

VALIDATION RESULT: THIS MODULE IS NOT WIRED IN
----------------------------------------------
Run against the verified events on 2026-09-13, the retrieval failed the only
test that matters -- telling combustion regimes apart:

    Reliance Jamnagar flare      443 K   (expected 1700-2000 K)
    JSW Vijayanagar furnace      457 K   (expected 1500-1900 K)
    Visakhapatnam Steel          487 K   (expected 1500-1900 K)
    Punjab crop burning          522 K   (expected  800-1000 K)
    Khavda SOLAR PARK            466 K   (expected: no combustion)

Every regime collapses into a 443-522 K band, so the retrieval does not
separate a refinery flare from a wheat field -- which is the entire reason to
compute temperature. Worse, it returns a confident 466 K for a photovoltaic
array, where the correct answer is that there is no fire.

The cause is limitation 2 below, and it is more severe than that note claims:
the estimated background biases solutions toward the low bound for strong fires
as well as weak ones. A two-channel Dozier retrieval cannot resolve sub-pixel
temperature from FIRMS brightness alone. The fix is the multi-band VIIRS
Nightfire fit (limitation 3), which needs VNF/VNP46 products rather than the
active-fire product.

The physics and the solver are correct and tested; the inputs are insufficient.
It stays unwired rather than shipping a number that looks like a measurement.

HONEST LIMITATIONS
------------------
1. **Saturation.** VIIRS I-4 clips at 367 K. 3.0% of detections sit at that
   ceiling and cannot be retrieved -- and they skew hot, so the very fires most
   worth measuring are the likeliest to be lost (12.1% of ACCIDENTAL_FIRE
   detections are saturated). MODIS T21 clips near 500 K and is affected far
   less, so MODIS partially covers the gap.
2. **Background temperature is estimated, not observed.** The retrieval needs
   the ambient temperature of the pixel's surroundings, which the FIRMS active
   fire product does not carry. It is estimated here from the low percentile of
   the thermal channel within each (month, day/night) stratum. Retrievals for
   weak fires are sensitive to this; strong fires are not. Every result carries
   a confidence flag reflecting that.
3. **This is not the full VIIRS Nightfire algorithm.** VNF fits M7/M8/M10/M12/M13
   simultaneously and is more robust. Those bands are not in the FIRMS active
   fire product, so the two-channel retrieval is what this data supports.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("thermal_physics")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Physical constants (SI)
PLANCK_H = 6.62607015e-34      # J s
BOLTZMANN_K = 1.380649e-23     # J / K
LIGHT_C = 2.99792458e8         # m / s

# Effective band centres, metres.
VIIRS_I4_UM = 3.74     # 3.55-3.93 um
VIIRS_I5_UM = 11.45    # 10.50-12.40 um
MODIS_T21_UM = 3.96
MODIS_T31_UM = 11.03

# VIIRS I-4 clips here; MODIS T21 near 500 K.
VIIRS_I4_SATURATION_K = 366.9
MODIS_T21_SATURATION_K = 499.0

# Physically admissible combustion range. The lower bound is below smouldering
# peat; the upper is above an adiabatic methane flame, so a retrieval outside it
# indicates a failed solve rather than an exotic fire.
MIN_FIRE_TEMP_K = 400.0
MAX_FIRE_TEMP_K = 2800.0

# Literature separation between fuel regimes, used only for labelling the
# retrieved temperature, never for the retrieval itself.
BIOMASS_MAX_K = 1000.0
FLARE_MIN_K = 1600.0

BISECTION_ITERATIONS = 80


def _numeric_col(df: pd.DataFrame, name: str, default: float) -> np.ndarray:
    """Fetches a column as a numeric array, tolerating its absence.

    `df.get(name, default)` returns the bare scalar when the column is missing,
    which then fails on any Series method. The same trap was fixed in
    feature_engineering; it survived here because this module was never wired
    into the pipeline and so was never run on a frame missing a column.
    """
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy()
    return np.full(len(df), float(default), dtype=float)


def planck_radiance(wavelength_um: float, temperature_k: np.ndarray) -> np.ndarray:
    """Spectral radiance of a blackbody, W m^-2 sr^-1 um^-1.

    B(lambda, T) = 2hc^2 / (lambda^5 * (exp(hc / (lambda k T)) - 1))
    """
    lam = wavelength_um * 1e-6
    t = np.asarray(temperature_k, dtype=float)
    t = np.clip(t, 1.0, None)

    c1 = 2.0 * PLANCK_H * LIGHT_C ** 2
    c2 = PLANCK_H * LIGHT_C / (BOLTZMANN_K * lam)

    # Clip the exponent to avoid overflow at low temperatures; the resulting
    # radiance underflows to ~0, which is physically correct.
    expo = np.clip(c2 / t, None, 700.0)
    radiance = c1 / (lam ** 5 * (np.expm1(expo)))
    return radiance * 1e-6  # per metre -> per micrometre


def dozier_bispectral(
    t_mwir: np.ndarray,
    t_tir: np.ndarray,
    t_background: np.ndarray,
    lam_mwir_um: float,
    lam_tir_um: float,
    iterations: int = BISECTION_ITERATIONS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solves the two-channel mixture for (fire temperature, area fraction).

    Vectorised bisection over all detections simultaneously. A per-row root-find
    would be correct but hopeless at 2M rows; the residual is monotonic in
    T_fire across the admissible range, so bisection converges reliably.

    Returns:
        (t_fire_k, area_fraction). Both NaN where no admissible solution exists.
    """
    t_mwir = np.asarray(t_mwir, dtype=float)
    t_tir = np.asarray(t_tir, dtype=float)
    t_bg = np.asarray(t_background, dtype=float)

    # Observed radiances implied by the reported brightness temperatures.
    l_mwir_obs = planck_radiance(lam_mwir_um, t_mwir)
    l_tir_obs = planck_radiance(lam_tir_um, t_tir)
    l_mwir_bg = planck_radiance(lam_mwir_um, t_bg)
    l_tir_bg = planck_radiance(lam_tir_um, t_bg)

    def residual(t_fire: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Thermal-channel mismatch for a candidate fire temperature."""
        l_mwir_f = planck_radiance(lam_mwir_um, t_fire)
        l_tir_f = planck_radiance(lam_tir_um, t_fire)

        denom = l_mwir_f - l_mwir_bg
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(np.abs(denom) > 1e-12, (l_mwir_obs - l_mwir_bg) / denom, np.nan)
        p = np.clip(p, 0.0, 1.0)

        l_tir_model = p * l_tir_f + (1.0 - p) * l_tir_bg
        return l_tir_model - l_tir_obs, p

    lo = np.full_like(t_mwir, MIN_FIRE_TEMP_K)
    hi = np.full_like(t_mwir, MAX_FIRE_TEMP_K)

    res_lo, _ = residual(lo)
    res_hi, _ = residual(hi)

    # A bracketed root requires a sign change across the interval.
    bracketed = np.sign(res_lo) != np.sign(res_hi)
    bracketed &= np.isfinite(res_lo) & np.isfinite(res_hi)

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        res_mid, _ = residual(mid)
        go_high = np.sign(res_mid) == np.sign(res_lo)
        lo = np.where(go_high, mid, lo)
        hi = np.where(go_high, hi, mid)
        res_lo = np.where(go_high, res_mid, res_lo)

    t_fire = 0.5 * (lo + hi)
    _, area_fraction = residual(t_fire)

    valid = (
        bracketed
        & np.isfinite(t_fire)
        & (t_fire > MIN_FIRE_TEMP_K + 1.0)
        & (t_fire < MAX_FIRE_TEMP_K - 1.0)
        & np.isfinite(area_fraction)
        & (area_fraction > 0.0)
        & (area_fraction < 1.0)
    )

    return np.where(valid, t_fire, np.nan), np.where(valid, area_fraction, np.nan)


def estimate_background_temperature(
    df: pd.DataFrame,
    tir_col: str,
    percentile: float = 10.0,
) -> np.ndarray:
    """Estimates ambient background temperature per (month, day/night) stratum.

    The FIRMS active fire product reports the fire pixel only, not the
    surrounding scene, so background must be inferred. For sub-pixel fires the
    thermal channel is dominated by background, so a low percentile of that
    channel within a stratum approximates the ambient surface temperature while
    excluding the strongly contaminated pixels.

    Stratifying by month and day/night matters: a 300 K daytime assumption
    applied to a pre-dawn overpass would inflate every retrieval.
    """
    tir = pd.to_numeric(df[tir_col], errors="coerce")

    if "timestamp_utc" in df.columns:
        month = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce").dt.month
    elif "acq_date" in df.columns:
        month = pd.to_datetime(df["acq_date"], errors="coerce").dt.month
    else:
        month = pd.Series(0, index=df.index)
    month = month.fillna(0).astype(int)

    daynight = df.get("daynight", pd.Series("D", index=df.index)).astype(str).str.upper()

    strata = month.astype(str) + "_" + daynight
    bg = tir.groupby(strata).transform(lambda s: np.nanpercentile(s, percentile))

    # Fall back to a global low percentile where a stratum is degenerate.
    global_bg = float(np.nanpercentile(tir.dropna(), percentile)) if tir.notna().any() else 290.0
    bg = bg.fillna(global_bg)

    # Keep within physically sane surface temperatures for the region.
    return np.clip(bg.to_numpy(dtype=float), 250.0, 330.0)


def classify_combustion_regime(t_fire: np.ndarray) -> np.ndarray:
    """Labels a retrieved temperature by fuel regime.

    Descriptive only -- the retrieval itself never uses these boundaries.
    """
    out = np.full(len(t_fire), "UNKNOWN", dtype=object)
    finite = np.isfinite(t_fire)
    out[finite & (t_fire < BIOMASS_MAX_K)] = "BIOMASS"
    out[finite & (t_fire >= BIOMASS_MAX_K) & (t_fire < FLARE_MIN_K)] = "INTERMEDIATE"
    out[finite & (t_fire >= FLARE_MIN_K)] = "HIGH_TEMP_COMBUSTION"
    return out


def add_fire_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds retrieved fire temperature, area fraction and confidence to a frame.

    VIIRS and MODIS carry different band pairs and different saturation ceilings,
    so each instrument is solved with its own wavelengths and then recombined.

    Adds:
        t_fire_k            - retrieved sub-pixel combustion temperature (K)
        fire_area_fraction  - fraction of the pixel occupied by fire
        fire_area_m2        - that fraction times the pixel footprint
        combustion_regime   - BIOMASS / INTERMEDIATE / HIGH_TEMP_COMBUSTION
        t_fire_confidence   - HIGH / MODERATE / SATURATED / FAILED
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        for c, v in (("t_fire_k", np.nan), ("fire_area_fraction", np.nan),
                     ("fire_area_m2", np.nan), ("combustion_regime", "UNKNOWN"),
                     ("t_fire_confidence", "FAILED")):
            out[c] = v
        return out

    t_fire = np.full(n, np.nan)
    area_frac = np.full(n, np.nan)
    saturated = np.zeros(n, dtype=bool)

    instrument = out.get("instrument", pd.Series("VIIRS", index=out.index)).astype(str).str.upper()

    configs = [
        ("VIIRS", "bright_ti4", "bright_ti5", VIIRS_I4_UM, VIIRS_I5_UM, VIIRS_I4_SATURATION_K),
        ("MODIS", "brightness", "bright_t31", MODIS_T21_UM, MODIS_T31_UM, MODIS_T21_SATURATION_K),
    ]

    for name, mwir_col, tir_col, lam_m, lam_t, sat_k in configs:
        mask = (instrument == name).to_numpy()
        if not mask.any() or mwir_col not in out.columns or tir_col not in out.columns:
            continue

        sub = out.loc[mask]
        t_m = pd.to_numeric(sub[mwir_col], errors="coerce").to_numpy(dtype=float)
        t_t = pd.to_numeric(sub[tir_col], errors="coerce").to_numpy(dtype=float)
        t_bg = estimate_background_temperature(sub, tir_col)

        sat = t_m >= sat_k
        saturated[mask] = sat

        # A retrieval needs the mid-wave channel meaningfully above background;
        # otherwise the mixture is unconstrained.
        solvable = np.isfinite(t_m) & np.isfinite(t_t) & (t_m > t_bg + 2.0) & ~sat

        tf = np.full(len(sub), np.nan)
        af = np.full(len(sub), np.nan)
        if solvable.any():
            tf_s, af_s = dozier_bispectral(
                t_m[solvable], t_t[solvable], t_bg[solvable], lam_m, lam_t
            )
            tf[solvable] = tf_s
            af[solvable] = af_s

        t_fire[mask] = tf
        area_frac[mask] = af
        logger.info(
            "%s: %d/%d retrieved (%.1f%%), %d saturated, %d unsolvable.",
            name, int(np.isfinite(tf).sum()), len(sub),
            100.0 * np.isfinite(tf).sum() / max(len(sub), 1),
            int(sat.sum()), int((~solvable & ~sat).sum()),
        )

    out["t_fire_k"] = np.round(t_fire, 1)
    out["fire_area_fraction"] = area_frac

    scan = _numeric_col(out, "scan", 0.375)
    track = _numeric_col(out, "track", 0.375)
    out["fire_area_m2"] = np.round(area_frac * scan * track * 1e6, 1)

    out["combustion_regime"] = classify_combustion_regime(t_fire)

    confidence = np.full(n, "FAILED", dtype=object)
    confidence[saturated] = "SATURATED"
    retrieved = np.isfinite(t_fire)
    # A larger apparent fire fills more of the pixel, so the mixture is better
    # constrained and less sensitive to the background estimate.
    confidence[retrieved] = "MODERATE"
    confidence[retrieved & (area_frac > 1e-4)] = "HIGH"
    out["t_fire_confidence"] = confidence

    logger.info(
        "Fire temperature retrieved for %d/%d detections (%.1f%%). Regimes: %s",
        int(retrieved.sum()), n, 100.0 * retrieved.sum() / n,
        dict(pd.Series(out["combustion_regime"]).value_counts()),
    )
    return out
