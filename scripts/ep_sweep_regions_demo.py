# ruff: noqa: INP001
"""Synthetic demo of ep_sweep_regions: two pulsars in two regions with different nbins.

Build this branch first, e.g. from the repository root with
``pip install --no-build-isolation --no-deps --target build/site .``; the demo imports
loki from build/site when it exists. Exits non-zero unless each region's best candidate
is its injected pulsar, and the strong independent groups are exactly those pulsars.

    python scripts/ep_sweep_regions_demo.py [outdir]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from ep_sweep_regions import ep_sweep_by_region, group_candidates, load_candidates

logger = logging.getLogger(__name__)

NSAMPS = 2**16
TSAMP = 64e-6
# One pulsar in each region: [100, 200] Hz folds with 32 bins, [50, 100] Hz with 64
F_INJ = (120.0, 70.0)


def main(outdir: Path) -> int:
    rng = np.random.default_rng(1)
    t = np.arange(NSAMPS) * TSAMP
    ts_e = rng.normal(size=NSAMPS)
    for f_inj in F_INJ:
        ts_e += 0.5 * (((t * f_inj) % 1.0) < 0.05)
    ts_v = np.ones(NSAMPS)
    cfg_kwargs = {
        "nsamps": NSAMPS,
        "tsamp": TSAMP,
        "nbins": 32,
        "eta": 1.0,
        "param_limits": [[-10.0, 10.0], [50.0, 200.0]],  # accel (m/s^2), freq (Hz)
        "ducy_max": 0.3,
        "use_fourier": False,
        "nthreads": 8,
        "octave_scale": 2.0,
        "bseg_brute": 1024,
        "bseg_ffa": NSAMPS // 8,
        "snr_min": 5.0,
        "prune_poly_order": 2,
    }
    site = Path(__file__).resolve().parents[1] / "build" / "site"
    paths = ep_sweep_by_region(
        ts_e,
        ts_v,
        cfg_kwargs,
        outdir,
        "demo",
        loki_site=site if site.is_dir() else None,
        sweep_kwargs={"show_progress": False, "n_runs": 1, "ref_segs": [4]},
    )
    cands = load_candidates(paths)
    ok = True
    for region, f_inj in enumerate(F_INJ):
        best = cands[cands["region"] == region].nlargest(1, "score").iloc[0]
        found = abs(best["freq"] - f_inj) <= best["dfreq"]
        logger.info(
            f"Region {region}: best {best['freq']:.4f} +- {best['dfreq']:.4f} Hz, "
            f"accel {best['accel']:.2f}, S/N {best['score']:.1f} "
            f"(injected {f_inj} Hz: {'found' if found else 'NOT found'})",
        )
        ok &= found

    # Grouped, the strong independent signals are the injected pulsars alone; their
    # harmonics (e.g. 140 Hz = 2 x 70 Hz) are flagged
    groups = group_candidates(cands)
    strong = groups[groups["score"] >= 10]
    for row in strong.itertuples():
        relation = f", {row.ratio} x {row.harmonic_of:.4f} Hz" if row.ratio else ""
        logger.info(
            f"Group {row.freq:.4f} Hz: S/N {row.score:.1f}, {row.n} candidates"
            f"{relation}",
        )
    signals = np.sort(strong.loc[strong["harmonic_of"].isna(), "freq"].to_numpy())
    ok &= len(signals) == len(F_INJ) and bool(
        np.all(np.abs(signals - np.sort(F_INJ)) <= strong["dfreq"].max()),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ep_sweep_demo")))
