# ruff: noqa: INP001
"""Detection thresholds for EPFreqSweep searches, from the size of the search.

    from ep_calibration import detection_threshold, effective_trials, target_snr

    n_eff = effective_trials(cfg_kwargs)
    snr_threshold = detection_threshold(n_eff, fap)  # report groups above this
    snr_min = target_snr(snr_threshold, completeness)  # use as cfg_kwargs["snr_min"]

``fap`` is the false-alarm probability over the whole search and ``completeness`` the
probability of reporting a pulsar whose S/N is ``snr_min``; both are the user's choice.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from ep_sweep_regions import drift_ranges, region_configs
from scipy.constants import speed_of_light
from scipy.stats import norm


def box_widths(nbins: int, ducy_max: float, wtsp: float) -> list[int]:
    """Boxcar widths (bins) loki scores a fold with: generate_box_width_trials."""
    wmax = int(max(1.0, ducy_max * nbins))
    widths = [1]
    while widths[-1] < wmax:
        next_width = max(widths[-1] + 1, int(wtsp * widths[-1]))
        if next_width > wmax:
            break
        widths.append(next_width)
    return widths


def _drift_cells_integral(
    f_lo: float,
    f_hi: float,
    tobs: float,
    drifts: list[tuple[int, float]],
) -> float:
    """Integral over frequency of the number of independent drift cells.

    A velocity derivative of order k adds the phase term f d t^(k+1) / ((k+1)! c).
    Lower-order terms absorb what they can of it: the best such fit to t^(k+1) on
    [0, T] leaves a peak-to-peak residual of 2^(-2k) T^(k+1) (Chebyshev). A range of
    width w then spans alpha f cycles of residual phase, with
    alpha = w T^(k+1) 2^(-2k) / ((k+1)! c), so max(1, alpha f) independent cells.
    The product over drift parameters is a power of f between the points where
    each alpha f crosses 1, and is integrated exactly piece by piece.
    """
    alphas = [
        width
        * tobs ** (order + 1)
        / (4**order * math.factorial(order + 1) * speed_of_light)
        for order, width in drifts
    ]
    edges = sorted(
        {f_lo, f_hi, *(1 / a for a in alphas if a > 0 and f_lo < 1 / a < f_hi)},
    )
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        # No alpha f crosses 1 inside a piece, so its midpoint decides for all of it
        active = [a for a in alphas if a * (lo + hi) / 2 > 1]
        power = len(active)
        total += (
            math.prod(active) * (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)
        )
    return total


def effective_trials(cfg_kwargs: dict[str, Any]) -> float:
    """Estimate the number of independent trials of an EPFreqSweep search.

    Per coarse region: independent frequencies (one per 1/T), times independent drift
    cells (see _drift_cells_integral), times boxcar templates per fold (a width-w
    boxcar has nbins / w non-overlapping positions, summed over loki's widths).
    Correlations between neighbouring widths make the template count high, while
    oversampled parameter grids find maxima between independent cells, so treat this
    as an estimate and check it with noise-only runs.
    """
    missing = [
        key for key in ("nsamps", "tsamp", "ducy_max", "wtsp") if key not in cfg_kwargs
    ]
    if missing:
        msg = f"cfg_kwargs must set {missing} explicitly"
        raise KeyError(msg)
    tobs = cfg_kwargs["nsamps"] * cfg_kwargs["tsamp"]
    drifts = drift_ranges(cfg_kwargs)
    total = 0.0
    for region, _ in region_configs(cfg_kwargs):
        nbins = int(region["nbins"])
        widths = box_widths(nbins, cfg_kwargs["ducy_max"], cfg_kwargs["wtsp"])
        templates = sum(nbins / w for w in widths)
        cells = tobs * _drift_cells_integral(
            region["f_start"],
            region["f_end"],
            tobs,
            drifts,
        )
        total += templates * cells
    return total


def detection_threshold(n_eff: float, fap: float) -> float:
    """S/N that noise exceeds with probability fap over n_eff independent trials.

    Each trial's boxcar S/N on white noise is a unit Gaussian.
    """
    return float(norm.isf(fap / n_eff))


def target_snr(snr_threshold: float, completeness: float) -> float:
    """S/N of a pulsar reported above snr_threshold with probability completeness.

    A measured S/N scatters about the true one with unit variance.
    """
    return float(snr_threshold + norm.ppf(completeness))
