# ruff: noqa: INP001
"""Run EPFreqSweep one coarse FFA region at a time, each region in a fresh process.

Workaround for a crash in EPFreqSweep on multi-region searches: DynamicThresholdScheme
keeps per-thread scratch sized for the first region's nbins, so planning a second region
with more bins in the same process overruns it. A fresh process per region gets fresh
scratch. Once that is fixed, call EPFreqSweep on the whole frequency range instead.

Usage::

    from ep_sweep_regions import ep_sweep_by_region, group_candidates, load_candidates

    paths = ep_sweep_by_region(ts_e, ts_v, cfg_kwargs, "ep_out", "test",
                               sweep_kwargs={"n_runs": 1, "ref_segs": [4]})
    cands = load_candidates(paths)
    groups = group_candidates(cands)  # one row per signal, harmonics flagged

``cfg_kwargs`` are PulsarSearchConfig keyword arguments, with ``param_limits`` given
as a list of ``[min, max]`` rows ending with the frequency row (Hz). The calling
process never imports loki; each region's process imports it from ``loki_site`` when
given (e.g. a build of this branch made with
``pip install --no-deps --target build/site .``), or else from the environment.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.constants import speed_of_light

logger = logging.getLogger(__name__)


def ffa_regions(
    p_min: float,
    p_max: float,
    tsamp: float,
    nbins_min: int,
    eta_min: float,
    octave_scale: float,
    nbins_max: int,
) -> list[dict[str, float]]:
    """Plan the coarse FFA regions as generate_ffa_regions in lib/regions.cpp does.

    The Python binding of generate_ffa_regions can't convert its return value, hence
    this port.

    Returns
    -------
    list[dict[str, float]]
        Regions with keys f_start, f_end (Hz), nbins and eta, highest frequency first.
    """
    t_w = max(p_min / nbins_min, tsamp)
    rho = eta_min / nbins_min
    regions = []
    p_cur_low = p_min
    while p_cur_low < p_max:
        # std::round rounds halves away from zero
        nbins_k = math.floor(min(p_cur_low / t_w, nbins_max) + 0.5)
        nbins_k = min(max(nbins_k, 2), nbins_max)
        if nbins_k >= nbins_max:
            regions.append(
                {
                    "f_start": 1 / p_max,
                    "f_end": 1 / p_cur_low,
                    "nbins": nbins_max,
                    "eta": rho * nbins_max,
                },
            )
            break
        p_cur_high = min(p_cur_low * octave_scale, p_max)
        regions.append(
            {
                "f_start": 1 / p_cur_high,
                "f_end": 1 / p_cur_low,
                "nbins": nbins_k,
                "eta": rho * nbins_k,
            },
        )
        p_cur_low = p_cur_high
    return regions


def _regions_of(cfg_kwargs: dict[str, Any]) -> list[dict[str, float]]:
    # No fallbacks: copies of loki's defaults here could silently drift from them
    missing = [
        key
        for key in ("tsamp", "nbins", "eta", "octave_scale", "nbins_max")
        if key not in cfg_kwargs
    ]
    if missing:
        msg = f"cfg_kwargs must set {missing} explicitly"
        raise KeyError(msg)
    f_min, f_max = cfg_kwargs["param_limits"][-1]
    return ffa_regions(
        1 / f_max,
        1 / f_min,
        cfg_kwargs["tsamp"],
        cfg_kwargs["nbins"],
        cfg_kwargs["eta"],
        cfg_kwargs["octave_scale"],
        cfg_kwargs["nbins_max"],
    )


def region_configs(
    cfg_kwargs: dict[str, Any],
) -> list[tuple[dict[str, float], dict[str, Any]]]:
    """Split a search config into one config per coarse region.

    Each config covers its region's frequency range with that region's nbins and eta,
    and plans as exactly one region.

    Returns
    -------
    list[tuple[dict[str, float], dict[str, Any]]]
        Pairs of (region, config keyword arguments), highest frequency first.
    """
    out = []
    for region in _regions_of(cfg_kwargs):
        rcfg = dict(cfg_kwargs)
        rcfg["param_limits"] = [list(row) for row in cfg_kwargs["param_limits"][:-1]]
        rcfg["param_limits"].append([region["f_start"], region["f_end"]])
        rcfg["nbins"] = region["nbins"]
        rcfg["eta"] = region["eta"]
        # The region's own period ratio, stepped up by the least amount that keeps
        # rounding in 1/f from splitting off a sliver as a second region
        rcfg["octave_scale"] = region["f_end"] / region["f_start"]
        while len(_regions_of(rcfg)) > 1:
            rcfg["octave_scale"] = math.nextafter(rcfg["octave_scale"], math.inf)
        replanned = _regions_of(rcfg)
        if (
            len(replanned) != 1
            or replanned[0]["nbins"] != region["nbins"]
            or not math.isclose(replanned[0]["eta"], region["eta"])
        ):
            msg = f"Region {region} does not replan as one region: {replanned}"
            raise RuntimeError(msg)
        out.append((region, rcfg))
    return out


def ep_sweep_by_region(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    cfg_kwargs: dict[str, Any],
    outdir: str | Path,
    prefix: str,
    *,
    loki_site: str | Path | None = None,
    sweep_kwargs: dict[str, Any] | None = None,
) -> list[Path]:
    """Run EPFreqSweep on each coarse region in its own process.

    Parameters
    ----------
    ts_e, ts_v : np.ndarray
        The time series to search.
    cfg_kwargs : dict[str, Any]
        PulsarSearchConfig keyword arguments for the whole search.
    outdir : str | Path
        Directory for the results, logs and region specs. The copy of the time
        series the region processes read is removed once they finish.
    prefix : str
        File prefix; region i writes ``{prefix}_r{i:02d}_ep_results.h5``.
    loki_site : str | Path | None, optional
        Directory to import loki from in the region processes.
    sweep_kwargs : dict[str, Any] | None, optional
        EPFreqSweep keyword arguments; n_runs or ref_segs is required.
        plan_cache_file can't be passed from Python yet (its binding lacks pybind11's
        filesystem caster).

    Returns
    -------
    list[Path]
        The result files, one per region, highest frequency first.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts_e_path = outdir / f"{prefix}_ts_e.npy"
    ts_v_path = outdir / f"{prefix}_ts_v.npy"
    np.save(ts_e_path, np.asarray(ts_e, dtype=np.float32))
    np.save(ts_v_path, np.asarray(ts_v, dtype=np.float32))
    paths = []
    try:
        for i, (region, rcfg) in enumerate(region_configs(cfg_kwargs)):
            rprefix = f"{prefix}_r{i:02d}"
            spec = {
                "loki_site": None if loki_site is None else str(loki_site),
                "ts_e": str(ts_e_path),
                "ts_v": str(ts_v_path),
                "cfg": rcfg,
                "sweep": sweep_kwargs or {},
                "outdir": str(outdir),
                "prefix": rprefix,
            }
            spec_path = outdir / f"{rprefix}_spec.json"
            spec_path.write_text(json.dumps(spec, indent=1))
            logger.info(
                f"Region {i}: f=[{region['f_start']:.3f}, {region['f_end']:.3f}] Hz, "
                f"nbins={region['nbins']}, eta={region['eta']:.3f}",
            )
            log_path = outdir / f"{rprefix}.log"
            with log_path.open("w") as log:
                proc = subprocess.run(  # noqa: S603 - runs this file with our own spec
                    [sys.executable, __file__, str(spec_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if proc.returncode != 0:
                msg = f"Region {i} failed (exit code {proc.returncode}), see {log_path}"
                raise RuntimeError(msg)
            paths.append(outdir / f"{rprefix}_ep_results.h5")
    finally:
        # Only the region processes read the series copy
        ts_e_path.unlink(missing_ok=True)
        ts_v_path.unlink(missing_ok=True)
    return paths


def load_candidates(paths: list[Path]) -> pd.DataFrame:
    """Collect the candidates of EPFreqSweep result files.

    Returns
    -------
    pd.DataFrame
        One row per candidate: each parameter and its uncertainty (``d`` prefix),
        score, score_ep, and the region, chunk and run it came from. The observation
        length (s) and the parameter names, frequency last, are kept in
        ``attrs["tobs"]`` and ``attrs["param_names"]``.
    """
    frames = []
    tobs = None
    names = []
    for region, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            tobs = float(f.attrs["tobs"])
            names = [str(name) for name in f.attrs["param_names"]]
            for chunk_name, chunk in f["chunks"].items():
                for run_name, run in chunk["runs"].items():
                    param_sets = run["param_sets"][:]
                    cols = {}
                    for k, name in enumerate(names):
                        cols[name] = param_sets[:, k, 0]
                        cols[f"d{name}"] = param_sets[:, k, 1]
                    cols["score"] = run["scores"][:]
                    cols["score_ep"] = run["scores_ep"][:]
                    run_frame = pd.DataFrame(cols)
                    run_frame["region"] = region
                    run_frame["chunk"] = chunk_name
                    run_frame["run"] = run_name
                    frames.append(run_frame)
    cands = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cands.attrs["tobs"] = tobs
    cands.attrs["param_names"] = names
    return cands


def drift_ranges(cfg_kwargs: dict[str, Any]) -> list[tuple[int, float]]:
    """Drift parameters searched by a config, as (order, width of range) pairs.

    param_limits rows end with frequency; the row before it is the first derivative
    of velocity (acceleration, order 1), the one before that the second (jerk, order
    2), and so on.
    """
    rows = cfg_kwargs["param_limits"][:-1]
    return [(len(rows) - i, hi - lo) for i, (lo, hi) in enumerate(rows)]


def _observed_drift_ranges(cands: pd.DataFrame) -> list[tuple[int, float]]:
    drifts = cands.attrs["param_names"][:-1]
    return [
        (len(drifts) - i, float(cands[name].max() - cands[name].min()))
        for i, name in enumerate(drifts)
    ]


def freq_tolerance(
    freq: float | np.ndarray,
    tobs: float,
    drifts: list[tuple[int, float]],
) -> float | np.ndarray:
    """Frequency spread (Hz) of one signal's candidates near ``freq``.

    One resolution element, 1 / T, plus the largest change in apparent frequency over
    the observation between drift trials across the searched ranges: a difference in
    the order-k velocity derivative of width w shifts the frequency by up to
    ``freq * w * T**k / (k! * c)``.
    """
    return 1 / tobs + sum(
        freq * width * tobs**order / (math.factorial(order) * speed_of_light)
        for order, width in drifts
    )


def harmonic_windows(
    parent_freq: float,
    *,
    tobs: float,
    drifts: list[tuple[int, float]],
    band: tuple[float, float],
    p_max: float,
) -> dict[int, list[tuple[Fraction, float, float]]]:
    """Harmonic windows of parent_freq, by ratio complexity.

    Ratios a/b (coprime, a != b) inside the band, each with the window
    |freq - a/b parent_freq| <= freq_tolerance(a/b parent_freq) + a/b
    freq_tolerance(parent_freq), grouped by their complexity a b. An unrelated
    frequency, uniform over the band, falls in a given window with probability its
    width over the band's. Levels are added until this parent's windows alone exceed
    p_max: no match beyond that level can count (see harmonic_parent).

    Returns
    -------
    dict[int, list[tuple[Fraction, float, float]]]
        For each level a b: (ratio, frequency, window half-width) per ratio.
    """
    f_lo, f_hi = band
    tol_parent = freq_tolerance(parent_freq, tobs, drifts)
    levels: dict[int, list[tuple[Fraction, float, float]]] = {}
    chance = 0.0
    level = 1
    while chance <= p_max:
        level += 1
        for a in range(1, level + 1):
            b, rem = divmod(level, a)
            if rem or a == b or math.gcd(a, b) != 1:
                continue
            target = a * parent_freq / b
            if f_lo <= target <= f_hi:
                half = freq_tolerance(target, tobs, drifts) + a / b * tol_parent
                levels.setdefault(level, []).append((Fraction(a, b), target, half))
                chance += 2 * half / (f_hi - f_lo)
    return levels


def harmonic_parent(
    freq: float,
    parents: list[dict[int, list[tuple[Fraction, float, float]]]],
    *,
    band: tuple[float, float],
    p_max: float,
) -> tuple[int, Fraction] | None:
    """Explain freq as a harmonic of one of several parents, if chance can't.

    Goes through ratio complexities a b in increasing order, adding up the windows of
    every parent's ratios at that level (from harmonic_windows). The first level with
    a window holding freq gives the match, if the chance of an unrelated frequency
    falling in some window up to that level is at most p_max; the strongest parent
    wins ties.

    Returns
    -------
    tuple[int, Fraction] | None
        Index of the parent in parents and the ratio, or None.
    """
    width = band[1] - band[0]
    chance = 0.0
    top = max((max(levels) for levels in parents if levels), default=1)
    for level in range(2, top + 1):
        found = None
        for j, levels in enumerate(parents):
            for ratio, target, half in levels.get(level, []):
                chance += 2 * half / width
                if found is None and abs(freq - target) <= half:
                    found = (j, ratio)
        if chance > p_max:
            return None
        if found is not None:
            return found
    return None


def group_candidates(
    cands: pd.DataFrame,
    *,
    drifts: list[tuple[int, float]] | None = None,
    band: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Collapse candidates into frequency groups and flag harmonically related ones.

    A strong signal survives in many neighbouring trials. Candidates whose frequency
    gap is within freq_tolerance form one group, represented by its best candidate by
    score. Going from the strongest group down, a group is a harmonic of a stronger,
    independent group when harmonic_parent can tell it from chance, with
    p_max = 1 / (number of groups): fewer than one chance attribution expected over
    the whole list.

    Parameters
    ----------
    cands : pd.DataFrame
        Candidates from load_candidates.
    drifts : list[tuple[int, float]] | None, optional
        Searched drift ranges, as from drift_ranges(cfg_kwargs). Defaults to the
        spread of the candidates' own drift parameters.
    band : tuple[float, float] | None, optional
        Searched frequency range (Hz). Defaults to the candidates' frequency range.

    Returns
    -------
    pd.DataFrame
        One row per group, strongest first: the best candidate's columns, plus n (the
        group's size), f_lo and f_hi (its frequency span), harmonic_of (the frequency
        of the group it is a harmonic of, NaN if independent) and ratio ("a/b").
    """
    if cands.empty:
        return cands.copy()
    tobs = cands.attrs["tobs"]
    drifts = _observed_drift_ranges(cands) if drifts is None else drifts
    band = (cands["freq"].min(), cands["freq"].max()) if band is None else band

    by_freq = cands.sort_values("freq", ignore_index=True)
    tol = freq_tolerance(by_freq["freq"], tobs, drifts)
    group_id = (by_freq["freq"].diff() > tol).cumsum()
    spans = by_freq.groupby(group_id)["freq"].agg(["size", "min", "max"])
    groups = by_freq.loc[by_freq.groupby(group_id)["score"].idxmax()]
    groups = groups.assign(
        n=spans["size"].to_numpy(),
        f_lo=spans["min"].to_numpy(),
        f_hi=spans["max"].to_numpy(),
    ).sort_values("score", ascending=False, ignore_index=True)

    p_max = 1 / len(groups)
    harmonic_of = np.full(len(groups), np.nan)
    ratio = [""] * len(groups)
    parent_freqs: list[float] = []  # independent groups so far, strongest first
    parent_windows: list[dict[int, list[tuple[Fraction, float, float]]]] = []
    for i, freq in enumerate(groups["freq"]):
        match = harmonic_parent(freq, parent_windows, band=band, p_max=p_max)
        if match is not None:
            harmonic_of[i] = parent_freqs[match[0]]
            ratio[i] = f"{match[1].numerator}/{match[1].denominator}"
        else:
            parent_freqs.append(freq)
            parent_windows.append(
                harmonic_windows(
                    freq,
                    tobs=tobs,
                    drifts=drifts,
                    band=band,
                    p_max=p_max,
                ),
            )
    return groups.assign(harmonic_of=harmonic_of, ratio=ratio)


def search_timeseries(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    cfg_kwargs: dict[str, Any],
    outdir: str | Path,
    prefix: str,
    *,
    snr_threshold: float,
    loki_site: str | Path | None = None,
    sweep_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Search a time series and report its signals.

    Runs ep_sweep_by_region, groups the candidates with the config's drift ranges,
    and marks as detected each independent group (not a harmonic of a stronger one)
    whose score reaches snr_threshold (see ep_calibration.detection_threshold).

    Returns
    -------
    pd.DataFrame
        group_candidates' groups, strongest first, with a boolean "detected" column.
    """
    paths = ep_sweep_by_region(
        ts_e,
        ts_v,
        cfg_kwargs,
        outdir,
        prefix,
        loki_site=loki_site,
        sweep_kwargs=sweep_kwargs,
    )
    groups = group_candidates(
        load_candidates(paths),
        drifts=drift_ranges(cfg_kwargs),
        band=tuple(cfg_kwargs["param_limits"][-1]),
    )
    if groups.empty:
        return groups.assign(detected=pd.Series(dtype=bool))
    return groups.assign(
        detected=(groups["score"] >= snr_threshold) & groups["harmonic_of"].isna(),
    )


def _run_region(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    if spec["loki_site"]:
        # Prefer the given build over an editable install's redirecting finder
        sys.meta_path[:] = [
            finder
            for finder in sys.meta_path
            if "ScikitBuild" not in type(finder).__name__
        ]
        sys.path.insert(0, spec["loki_site"])
    from loki import libloki  # noqa: PLC0415 - after choosing which build to import

    ts_e = np.load(spec["ts_e"], mmap_mode="r")
    ts_v = np.load(spec["ts_v"], mmap_mode="r")
    cfg_kwargs = dict(spec["cfg"])
    cfg_kwargs["param_limits"] = np.array(cfg_kwargs["param_limits"], dtype=np.float64)
    cfg = libloki.configs.PulsarSearchConfig(**cfg_kwargs)
    sweep = libloki.prune.EPFreqSweep(cfg, **spec["sweep"])
    sweep.execute(
        np.ascontiguousarray(ts_e),
        np.ascontiguousarray(ts_v),
        outdir=spec["outdir"],
        file_prefix=spec["prefix"],
    )


if __name__ == "__main__":
    _run_region(sys.argv[1])
