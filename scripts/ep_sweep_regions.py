# ruff: noqa: INP001
"""Run EPFreqSweep one coarse FFA region at a time, each region in a fresh process.

Workaround for a crash in EPFreqSweep on multi-region searches: DynamicThresholdScheme
keeps per-thread scratch sized for the first region's nbins, so planning a second region
with more bins in the same process overruns it. A fresh process per region gets fresh
scratch. Once that is fixed, call EPFreqSweep on the whole frequency range instead.

Usage::

    from ep_sweep_regions import ep_sweep_by_region, load_candidates

    paths = ep_sweep_by_region(ts_e, ts_v, cfg_kwargs, "ep_out", "test",
                               sweep_kwargs={"n_runs": 1, "ref_segs": [4]})
    cands = load_candidates(paths)

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
    f_min, f_max = cfg_kwargs["param_limits"][-1]
    return ffa_regions(
        1 / f_max,
        1 / f_min,
        cfg_kwargs["tsamp"],
        cfg_kwargs["nbins"],
        cfg_kwargs["eta"],
        cfg_kwargs.get("octave_scale", 2.0),
        cfg_kwargs.get("nbins_max", 1024),
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
        # Wide enough that rounding in 1/f can't split off a sliver as a second region
        rcfg["octave_scale"] = 2 * max(
            region["f_end"] / region["f_start"],
            cfg_kwargs.get("octave_scale", 2.0),
        )
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
        score, score_ep, and the region, chunk and run it came from.
    """
    frames = []
    for region, path in enumerate(paths):
        with h5py.File(path, "r") as f:
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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
