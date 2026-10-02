# ruff: noqa: INP001
"""Synthetic injection test of the region-by-region EP search.

Injects one pulsar into each of two coarse regions with different nbins, in Gaussian
noise, searches with a threshold set by a false-alarm probability over the trials
measured on noise-only searches, and exits non-zero unless both pulsars are detected,
nothing else is, and their harmonics are flagged.

Build this branch first, e.g. from the repository root with
``pip install --no-build-isolation --no-deps --target build/site .``; the demo imports
loki from build/site when it exists.

    python scripts/ep_sweep_regions_demo.py [outdir]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from ep_calibration import (
    detection_threshold,
    effective_trials,
    injection_test,
    noise_trials,
)

logger = logging.getLogger(__name__)

SEED = 1  # noise realization
NOISE_SEED = 2  # noise-only realizations that measure the trials
N_NOISE = 8  # noise-only searches: N_eff to about 1 / sqrt(8)
NSAMPS = 2**16
TSAMP = 64e-6
NSEGMENTS = 8  # EP segments
FAP = 1e-3  # false-alarm probability over the whole search
# One pulsar per region: [100, 200] Hz folds with 32 bins, [50, 100] Hz with 64
INJECTIONS = [{"freq": 120.0}, {"freq": 70.0}]
DUTY = 0.05
# Ideal S/N, far above the threshold: checks the pipeline, not its sensitivity
INJECTED_SNR = 25.0
CFG = {
    "nsamps": NSAMPS,
    "tsamp": TSAMP,
    "nbins": 32,
    "eta": 1.0,
    "param_limits": [[-10.0, 10.0], [50.0, 200.0]],  # accel (m/s^2), freq (Hz)
    "ducy_max": 0.3,
    "wtsp": 1.5,
    "use_fourier": False,
    "nthreads": 8,
    "octave_scale": 2.0,
    "nbins_max": 1024,
    "bseg_brute": 1024,
    "bseg_ffa": NSAMPS // NSEGMENTS,
    "prune_poly_order": 2,
}
# One EP run from each segment as the reference: a pulsar lost from one can survive
# from another (n_runs takes precedence over any ref_segs)
SWEEP = {"show_progress": False, "n_runs": NSEGMENTS}


def main(outdir: Path) -> int:
    rng = np.random.default_rng(SEED)
    ts_e = rng.normal(size=NSAMPS)
    ts_v = np.ones(NSAMPS)
    # A threshold scheme designed where noise peaks once keeps pulsars at the threshold
    cfg = {**CFG, "snr_min": detection_threshold(effective_trials(CFG), 1)}
    site = Path(__file__).resolve().parents[1] / "build" / "site"
    loki_site = site if site.is_dir() else None
    # effective_trials counts neither pruning nor the runs from several segments
    n_eff, _ = noise_trials(
        cfg,
        N_NOISE,
        outdir,
        "noise",
        seed=NOISE_SEED,
        loki_site=loki_site,
        sweep_kwargs=SWEEP,
    )
    snr_threshold = detection_threshold(n_eff, FAP)
    logger.info(
        f"N_eff {n_eff:.3g}: threshold {snr_threshold:.2f} at FAP {FAP}, "
        f"snr_min {cfg['snr_min']:.2f}",
    )
    results, groups = injection_test(
        ts_e,
        ts_v,
        cfg,
        INJECTIONS,
        outdir,
        "demo",
        duty=DUTY,
        snr=INJECTED_SNR,
        snr_threshold=snr_threshold,
        loki_site=loki_site,
        sweep_kwargs=SWEEP,
    )
    for row in results.itertuples():
        found = (
            f"recovered at {row.group_freq:.4f} Hz, S/N {row.score:.1f}"
            if row.recovered
            else "NOT recovered"
        )
        logger.info(f"Injected {row.freq} Hz: {found}")
    # Above the threshold, the search itself flags harmonics of stronger groups;
    # the injection test also explains detections at harmonics of an injection
    flagged = groups[(groups["score"] >= snr_threshold) & groups["harmonic_of"].notna()]
    for row in flagged.itertuples():
        logger.info(
            f"Flagged harmonic {row.freq:.4f} Hz = {row.ratio} x "
            f"{row.harmonic_of:.4f} Hz, S/N {row.score:.1f}",
        )
    of_injection = groups[groups["detected"] & (groups["relation"].str.len() > 0)]
    for row in of_injection[of_injection["relation"] != "1/1"].itertuples():
        logger.info(
            f"Detected {row.freq:.4f} Hz = {row.relation} x injection "
            f"{INJECTIONS[row.injection]['freq']} Hz, S/N {row.score:.1f}",
        )
    false_alarms = groups[groups["detected"] & (groups["injection"] < 0)]
    for row in false_alarms.itertuples():
        logger.info(f"False alarm {row.freq:.4f} Hz, S/N {row.score:.1f}")
    return 0 if results["recovered"].all() and false_alarms.empty else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ep_sweep_demo")))
