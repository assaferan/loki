# ruff: noqa: INP001
"""Detection thresholds for EPFreqSweep searches, from the size of the search.

    from ep_calibration import detection_threshold, effective_trials

    n_eff = effective_trials(cfg_kwargs)
    snr_threshold = detection_threshold(n_eff, fap)  # report groups above this
    cfg_kwargs["snr_min"] = detection_threshold(n_eff, 1)  # where noise peaks once

``fap`` is the false-alarm probability over the whole search, the user's choice.
snr_min only designs EP's threshold scheme, which keeps a pulsar at snr_min a fraction
of the time and one a few S/N above it most of the time: designed where noise peaks
about once, it keeps pulsars at snr_threshold, and noise_trials can measure the trials
with the search's own pruning. target_snr(snr_threshold, completeness) is the S/N of a
pulsar reported with probability completeness. injection_test injects pulsars into a
time series, searches it with search_timeseries, and reports which were recovered.
noise_trials measures the effective number of trials on noise-only searches, to check
effective_trials.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from ep_sweep_regions import (
    drift_ranges,
    ep_sweep_by_region,
    freq_tolerance,
    harmonic_parent,
    harmonic_windows,
    load_candidates,
    region_configs,
    search_timeseries,
)
from scipy.constants import speed_of_light
from scipy.stats import norm

if TYPE_CHECKING:
    from pathlib import Path


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


def inject_pulsars(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    tsamp: float,
    injections: list[dict[str, Any]],
    *,
    duty: float,
    snr: float,
) -> np.ndarray:
    """Return a copy of ts_e with boxcar pulse trains added.

    Each injection gives "freq" (Hz, at the first sample) and optionally "phase"
    (cycles), "drift" (velocity derivatives, in the order of param_limits' drift
    rows: highest order first) and "snr" (in place of snr). As in loki, the observed
    frequency is freq (1 - v / c) for a line-of-sight velocity v away from us, so a
    positive acceleration lowers it over time and a pulse is on while
    (phase + freq * (t - sum_k d_k t^(k+1) / ((k+1)! c))) mod 1 < duty. Under the
    inverse-variance weighting of (ts_e, ts_v), a pulse of constant physical amplitude
    adds A * ts_v to ts_e, and A is set so that each train's ideal matched-filter S/N,
    A * sqrt(duty * (1 - duty) * sum(ts_v)), is snr.
    """
    ts_v = np.asarray(ts_v, dtype=np.float64)
    t = np.arange(len(ts_e)) * tsamp
    ideal_snr_per_amplitude = math.sqrt(duty * (1 - duty) * ts_v.sum())
    out = np.array(ts_e, dtype=np.float64)
    for inj in injections:
        drift = inj.get("drift", ())
        delay = t.copy()
        for i, d in enumerate(drift):
            order = len(drift) - i
            delay -= d * t ** (order + 1) / (math.factorial(order + 1) * speed_of_light)
        phase = (inj.get("phase", 0.0) + inj["freq"] * delay) % 1.0
        amplitude = inj.get("snr", snr) / ideal_snr_per_amplitude
        out += amplitude * ts_v * (phase < duty)
    return out


def injection_test(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    cfg_kwargs: dict[str, Any],
    injections: list[dict[str, Any]],
    outdir: str | Path,
    prefix: str,
    *,
    duty: float,
    snr: float,
    snr_threshold: float,
    loki_site: str | Path | None = None,
    sweep_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject pulsars into a time series, search it, and check what is recovered.

    An injection is recovered when a detected group (search_timeseries) lies within
    freq_tolerance of its frequency. A detected group can also be a harmonic of an
    injection, if harmonic_parent can tell it from chance and its S/N is possible for
    a harmonic of the injection's S/N (its "snr", or snr), with
    p_max = 1 / (number of detected groups). Detected groups that are neither are
    false alarms, or real signals already in the data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Per injection: its freq, recovered, and the matching group's freq and score.
        The groups, with "injection" the index of the injection each detected group
        matches (-1 if none) and "relation" its ratio to it ("1/1" for the
        fundamental).
    """
    injected = inject_pulsars(
        ts_e,
        ts_v,
        cfg_kwargs["tsamp"],
        injections,
        duty=duty,
        snr=snr,
    )
    groups = search_timeseries(
        injected,
        ts_v,
        cfg_kwargs,
        outdir,
        prefix,
        snr_threshold=snr_threshold,
        loki_site=loki_site,
        sweep_kwargs=sweep_kwargs,
    )
    tobs = cfg_kwargs["nsamps"] * cfg_kwargs["tsamp"]
    drifts = drift_ranges(cfg_kwargs)
    match = np.full(len(groups), -1)
    relation = [""] * len(groups)
    rows = []
    for i, inj in enumerate(injections):
        tol_inj = freq_tolerance(inj["freq"], tobs, drifts)
        near = groups["detected"] & ((groups["freq"] - inj["freq"]).abs() <= tol_inj)
        for k in np.flatnonzero(near.to_numpy()):
            match[k], relation[k] = i, "1/1"
        best = groups[near].nlargest(1, "score")
        rows.append(
            {
                "freq": inj["freq"],
                "recovered": not best.empty,
                "group_freq": best["freq"].iloc[0] if not best.empty else np.nan,
                "score": best["score"].iloc[0] if not best.empty else np.nan,
            },
        )
    band = tuple(cfg_kwargs["param_limits"][-1])
    p_max = 1 / max(int(groups["detected"].sum()), 1)
    injected_snrs = [inj.get("snr", snr) for inj in injections]
    windows = [
        harmonic_windows(
            inj["freq"],
            tobs=tobs,
            drifts=drifts,
            band=band,
            p_max=p_max,
            parent_score=inj_snr,
            snr_threshold=snr_threshold,
        )
        for inj, inj_snr in zip(injections, injected_snrs, strict=True)
    ]
    for k in np.flatnonzero(groups["detected"].to_numpy() & (match < 0)):
        found = harmonic_parent(
            groups["freq"].iloc[k],
            windows,
            band=band,
            p_max=p_max,
            score=groups["score"].iloc[k],
            parent_scores=injected_snrs,
            snr_threshold=snr_threshold,
        )
        if found is not None:
            match[k] = found[0]
            relation[k] = f"{found[1].numerator}/{found[1].denominator}"
    return pd.DataFrame(rows), groups.assign(injection=match, relation=relation)


def noise_trials(
    cfg_kwargs: dict[str, Any],
    n_realizations: int,
    outdir: str | Path,
    prefix: str,
    *,
    seed: int | None = None,
    loki_site: str | Path | None = None,
    sweep_kwargs: dict[str, Any] | None = None,
) -> tuple[float, np.ndarray]:
    """Measure a search's effective number of trials on noise-only series.

    Searches n_realizations series of unit Gaussian noise (ts_v = 1) with the config
    and takes each one's maximum candidate score. For a maximum over N independent
    unit Gaussian trials, P(max <= m) = Phi(m)^N, so the maximum-likelihood N from
    maxima m_i is n / -sum(log Phi(m_i)); -N log Phi(max) is exponentially
    distributed, so its relative uncertainty is about 1 / sqrt(n). Unlike
    effective_trials, this includes the pipeline's pruning. Pruning also hides noise
    below the final thresholds, so use an snr_min low enough that every realization
    keeps candidates, such as detection_threshold(effective_trials(cfg_kwargs), 1):
    the S/N noise reaches about once in the search.

    Returns
    -------
    tuple[float, np.ndarray]
        The estimated number of trials, for detection_threshold, and the maxima.
    """
    rng = np.random.default_rng(seed)
    nsamps = cfg_kwargs["nsamps"]
    maxima = []
    for i in range(n_realizations):
        paths = ep_sweep_by_region(
            rng.normal(size=nsamps),
            np.ones(nsamps),
            cfg_kwargs,
            outdir,
            f"{prefix}_{i:03d}",
            loki_site=loki_site,
            sweep_kwargs=sweep_kwargs,
        )
        cands = load_candidates(paths)
        if cands.empty:
            msg = (
                f"Noise realization {i} kept no candidates, so its maximum is hidden "
                "by pruning; lower snr_min for this calibration"
            )
            raise RuntimeError(msg)
        maxima.append(cands["score"].max())
    maxima = np.asarray(maxima, dtype=np.float64)
    return float(len(maxima) / -norm.logcdf(maxima).sum()), maxima
