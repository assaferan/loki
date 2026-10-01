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
        Directory for the results, logs, region specs and the time series copy.
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
    return paths


def load_candidates(paths: list[Path]) -> pd.DataFrame:
    """Collect the candidates of EPFreqSweep result files.

    Returns
    -------
    pd.DataFrame
        One row per candidate: each parameter and its uncertainty (``d`` prefix),
        score, score_ep, and the region, chunk and run it came from. The observation
        length is kept in ``attrs["tobs"]`` (s).
    """
    frames = []
    tobs = None
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
    return cands


def group_candidates(
    cands: pd.DataFrame,
    *,
    freq_tol: float | None = None,
    harmonic_tol: float | None = None,
    max_harmonic: int = 4,
) -> pd.DataFrame:
    """Collapse candidates into frequency groups and flag harmonically related ones.

    A strong signal survives in many neighbouring trials. Candidates within
    ``freq_tol`` of a neighbour in frequency form one group, represented by its best
    candidate by score. A group is a harmonic when its frequency is a/b times that of
    a stronger, independent group (a, b <= max_harmonic), to within
    ``harmonic_tol * (1 + a/b)``.

    Parameters
    ----------
    cands : pd.DataFrame
        Candidates from load_candidates.
    freq_tol : float | None, optional
        Largest frequency gap (Hz) inside a group. Defaults to 40 / T, with T the
        observation length: neighbouring trials of a signal at S/N ~15 can be ~25 / T
        apart.
    harmonic_tol : float | None, optional
        Frequency tolerance (Hz) of a harmonic match. Defaults to 2 / T.
    max_harmonic : int, optional
        Largest numerator and denominator of the harmonic ratios tried.

    Returns
    -------
    pd.DataFrame
        One row per group, strongest first: the best candidate's columns, plus n (the
        group's size), f_lo and f_hi (its frequency span), harmonic_of (the frequency
        of the group it is a harmonic of, NaN if independent) and ratio ("a/b").
    """
    if cands.empty:
        return cands.copy()
    tobs = cands.attrs.get("tobs")
    if (freq_tol is None or harmonic_tol is None) and tobs is None:
        msg = "Pass freq_tol and harmonic_tol, or candidates from load_candidates"
        raise ValueError(msg)
    freq_tol = 40 / tobs if freq_tol is None else freq_tol
    harmonic_tol = 2 / tobs if harmonic_tol is None else harmonic_tol

    by_freq = cands.sort_values("freq", ignore_index=True)
    group_id = (by_freq["freq"].diff() > freq_tol).cumsum()
    spans = by_freq.groupby(group_id)["freq"].agg(["size", "min", "max"])
    groups = by_freq.loc[by_freq.groupby(group_id)["score"].idxmax()]
    groups = groups.assign(
        n=spans["size"].to_numpy(),
        f_lo=spans["min"].to_numpy(),
        f_hi=spans["max"].to_numpy(),
    ).sort_values("score", ascending=False, ignore_index=True)

    ratios = sorted(
        {
            Fraction(a, b)
            for a in range(1, max_harmonic + 1)
            for b in range(1, max_harmonic + 1)
            if a != b
        },
    )
    harmonic_of = np.full(len(groups), np.nan)
    ratio = [""] * len(groups)
    independent = []
    for i, freq in enumerate(groups["freq"]):
        for parent in independent:
            match = next(
                (r for r in ratios if abs(freq - r * parent) <= harmonic_tol * (1 + r)),
                None,
            )
            if match is not None:
                harmonic_of[i] = parent
                ratio[i] = f"{match.numerator}/{match.denominator}"
                break
        else:
            independent.append(freq)
    return groups.assign(harmonic_of=harmonic_of, ratio=ratio)


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
