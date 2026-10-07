"""EPMultiPass.execute_pruning: pruning a precomputed FFA fold matches execute."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from loki import libloki
from pyloki.periodogram import ScatteredPeriodogram

if TYPE_CHECKING:
    from pathlib import Path

NSAMPS = 2**20
TSAMP = 0.000064
NSEGMENTS = 32


def _config() -> libloki.configs.PulsarSearchConfig:
    bseg_ffa = NSAMPS // NSEGMENTS
    return libloki.configs.PulsarSearchConfig(
        nsamps=NSAMPS,
        tsamp=TSAMP,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [143.0, 144.0]], dtype=np.float64),
        use_fourier=True,
        nthreads=2,
        bseg_brute=bseg_ffa // 16,
        bseg_ffa=bseg_ffa,
        prune_poly_order=2,
    )


def _series() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261007)
    t = np.arange(NSAMPS) * TSAMP
    phase = (143.5 * t) % 1.0
    pulse = np.exp(-0.5 * ((phase - 0.5) / 0.03) ** 2)
    ts_e = (rng.standard_normal(NSAMPS) + 0.05 * pulse).astype(np.float32)
    return ts_e, np.ones(NSAMPS, dtype=np.float32)


def _ep(cfg: libloki.configs.PulsarSearchConfig) -> libloki.prune.EPMultiPassFourier:
    thresholds = np.linspace(1.0, 4.0, NSEGMENTS - 1).astype(np.float32).tolist()
    return libloki.prune.EPMultiPassFourier(
        cfg,
        thresholds,
        n_runs=2,
        show_progress=False,
    )


def _candidates(outdir: Path, prefix: str) -> np.ndarray:
    result = outdir / f"{prefix}_pruning_nstages_{NSEGMENTS}_results.h5"
    data = ScatteredPeriodogram.load(str(result)).data
    data = data.drop(columns=["run_id"]).sort_values(list(data.columns.drop("run_id")))
    return data.to_numpy(dtype=np.float64)


def test_execute_pruning_matches_execute(tmp_path: Path) -> None:
    cfg = _config()
    ts_e, ts_v = _series()
    _ep(cfg).execute(ts_e, ts_v, str(tmp_path / "full"), "full")

    fold, _ = libloki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
    np.save(tmp_path / "fold.npy", fold)
    shared = np.load(tmp_path / "fold.npy", mmap_mode="r")
    _ep(cfg).execute_pruning(shared, str(tmp_path / "pruned"), "pruned")

    full = _candidates(tmp_path / "full", "full")
    pruned = _candidates(tmp_path / "pruned", "pruned")
    assert full.shape[0] > 0
    np.testing.assert_allclose(pruned, full, rtol=1e-6, atol=1e-9)


def test_execute_pruning_rejects_wrong_size(tmp_path: Path) -> None:
    cfg = _config()
    ts_e, ts_v = _series()
    fold, _ = libloki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
    with pytest.raises(RuntimeError, match="final fold"):
        _ep(cfg).execute_pruning(fold[:-1], str(tmp_path), "short")


def test_execute_pruning_refuses_to_copy(tmp_path: Path) -> None:
    cfg = _config()
    ts_e, ts_v = _series()
    fold, _ = libloki.ffa.compute_ffa_fourier(ts_e, ts_v, cfg, quiet=True)
    with pytest.raises(TypeError, match="used in place"):
        _ep(cfg).execute_pruning(fold.astype(np.complex128), str(tmp_path), "dtype")
    strided = np.repeat(fold, 2)[::2]
    with pytest.raises(TypeError, match="used in place"):
        _ep(cfg).execute_pruning(strided, str(tmp_path), "strided")


def test_execute_twice_reallocates_the_ffa(tmp_path: Path) -> None:
    # execute() releases its FFA buffers after the FFA; a second call on the
    # same object must allocate them again and give the same candidates.
    cfg = _config()
    ts_e, ts_v = _series()
    ep = _ep(cfg)
    ep.execute(ts_e, ts_v, str(tmp_path / "first"), "first")
    ep.execute(ts_e, ts_v, str(tmp_path / "second"), "second")
    first = _candidates(tmp_path / "first", "first")
    second = _candidates(tmp_path / "second", "second")
    assert first.shape[0] > 0
    np.testing.assert_allclose(second, first, rtol=1e-6, atol=1e-9)
