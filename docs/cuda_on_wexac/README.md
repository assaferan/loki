# CUDA on wexac: build LOKI's CUDA backend and test our fixes on real GPUs

Branch `cuda-on-wexac`, cut from `upstream/main` 94c1eaa (pravirkr/loki, 2026-10-11).
Worktree `/Users/eranasaf/GitHub/loki-wt/cuda-on-wexac`. Set up by the pyloki overview
session (`pyloki-79`) at assaferan's request, 2026-10-11.

This is a **working branch**: it carries this README and notes. A PR is never opened from it;
cut a clean branch per fix holding only the files that fix needs (see "Upstream" below).

## Why

LOKI's CUDA code has never been exercised by us: there was no GPU. Upstream CI builds with
`-DLOKI_CUDA=OFF` (`.github/workflows/ci.yml`), so nothing CUDA is tested there either. Pravir's
CUDA parity tests (a69f3bc, "Add GPU snr_boxcar_2d, CUDA parity tests and GPU-host fixes") need a
GPU to run. wexac (Weizmann LSF cluster) has about 700 GPUs, so now we can.

## The three items, in priority order

1. **Branch overflow on the CUDA path: not fixed at all.** pravirkr/loki PR #23
   (merge 4f93103; fix 56a0146, tests 4a0978f and 2ec69cc on `fix-branch-overflow`) made the CPU
   path never write branched leaves past the workspace, and retry an oversize batch by halving.
   Its commit message says: "The CUDA path (prune_cuda.cu) has the same post-hoc check and is not
   changed here." Upstream: `lib/cuda/prune_cuda.cu`. On a GPU an oversize batch would write past
   its buffer before anything checks. Production 2005 runs use the CPU path, so this blocks nothing
   running, but it is a real memory-safety defect for any GPU user.
   Port: the same capacity-guarded writes + halving retry, and a GPU counterpart of
   `tests/cpp/internal/branch_capacity_t.cpp` / `tests/python/test_ep_branch_overflow.py`.

2. **`WorldTreeCUDA::add_initial` rejects a seed batch larger than it copies.** CPU fix merged as
   PR #10 (merge aa70cfa). The CUDA fix exists on our fork's branch `fix-add-initial-chunks-cuda`
   (4cc0180), written against the pre-refactor layout (`lib/world_tree_cuda.cu`) and never
   compiled. Upstream still has the strict check in `lib/cuda/world_tree_cuda.cu` (around line
   611: `check_equal(slots_to_write, scores_batch.size(), ...)`). The fix is 3 lines:
   `check_greater_equal(scores_batch.size(), slots_to_write, ...)`. Port it to the new path.

3. **FFAFreqSweepCUDA per-chunk widths / score buffer.** Our fork's `fix-ffa-freq-sweep-cuda`
   (da6bb4d, f8a11de; CPU side was our PR #6, closed) may be superseded by Pravir's b009dce
   ("fix(ffa): bound the frequency-sweep candidate accumulator, decode per-chunk widths") and the
   later pipeline rewrite (3839436, 907f354). Do not port blindly: first test upstream's
   `lib/cuda/ffa_freq_sweep_cuda.cu` on a multi-region sweep on GPU against the CPU result. Port
   only what still fails.

## Order of work

1. Build LOKI with CUDA on wexac in a **new** environment (below).
2. Run upstream main's own CUDA tests and the CPU/CUDA parity tests **before touching code**. These
   are the cases the build must accept; if they fail, the build is wrong, not LOKI. Record which
   GPU model each result came from.
3. For each item: a regression test that **fails on upstream main** on a GPU and passes with the fix.
   A test that cannot fail proves nothing (see the shared memory facts on checkers).
4. Commit locally, push to the `assaferan` fork, and stop. assaferan reads before anything goes to
   pravirkr/loki.

## wexac facts (measured 2026-10-11 by pyloki-79)

- Access: `ssh wexac` from the laptop (key set up). marvin also has a `wexac` host entry.
- Project dir `/home/projects/barakz/eranasaf` (LSF uses it as the job's home and working dir).
  43 TB free on the project filesystem; quota fine per assaferan.
- **Do not touch production.** The micromamba env `loki`
  (`/home/projects/barakz/eranasaf/micromamba/envs/loki`), the clone `~/src/loki` (branch
  `rerun-build-0f181dc-overflowfix`, tree f0ef8e3b) and everything under `~/ter5/` belong to the
  2005 rerun (loki overview's and pyloki-79's). Make your own env (e.g. `loki-cuda`) and your own
  clone (e.g. `~/src/loki-cuda`).
- Toolchain: RHEL 9.4, system gcc 11.4, **no system cmake**. The production env got compilers,
  cmake and ninja from conda-forge via micromamba (`$HOME/micromamba/bin/micromamba`); the
  build recipe is `~/src/build_loki.sh` (CPU; uses `x86_64-conda-linux-gnu-g++`, `CPM_SOURCE_CACHE`,
  `-march=x86-64-v3`). Get the CUDA toolkit from conda (nvidia / conda-forge `cuda-toolkit`)
  matching the driver; modules also exist (`module avail cuda`: CUDAcore 11.x, CUDA-Samples 12.5 /
  13.2) but are older or GCC-13-bound.
- Driver 595.58 on the GPU hosts. GPU compute capabilities on LSF hosts: 8.6 (414 GPUs), 8.9 (184),
  8.0 (57, A100 80 GB), 9.0 (40, H100). Check `lshosts -gpu -w` for models.
- **cuRANDDx is compiled for one SM** (`cmake/FindMathDX_SM.cmake`: MATHDX_SM, >= 700, from the
  first LOKI_CUDA_ARCHITECTURES entry). Build for one architecture and pin jobs to that GPU model,
  or build once per architecture. Do not assume a fat binary.
- GPU queues: `short-gpu` (JL/U 800), `long-gpu` (500), `interactive-gpu` / `gpu-interactive` (30).
  Read `bqueues -l short-gpu` for its RES_REQ and run limit before writing bsub lines; check
  `bsub -gpu` syntax there.
- **LSF memory reservation is a hard kill** (TERM_MEMLIMIT, exit 137, verified). Reserve enough;
  `rusage[mem=]` is per slot. The general queues add `select[mem>15360]`.
- Compute nodes have internet (CPM/pip fetches work in jobs).
- Gotchas hit already: `/usr/bin/time` does not exist on compute nodes (LSF's job report gives
  Max Memory); a job script with `set -e` that sources `/etc/profile` aborts before doing anything.
- CPU builds compile in ~30 min on 8 cores, peak 3.9 GB; budget more for nvcc.

## Rules that apply here

- LOKI's `AGENTS.md`: read `docs/architecture.md` first; `scripts/check_architecture.sh`,
  `pre-commit run --files ...`, and the dev preset build + ctest must pass. Do not change kernel
  math, RNG streams or output formats as a side effect.
- No Claude attribution in commits or PRs (no `Co-Authored-By: Claude`, no "Generated with
  Claude Code").
- Never use GitHub issues or PR threads to talk to other sessions; use `SendMessage` (run
  `ListAgents` first). Tracker comments are for the maintainers and carry only defect/fix content.
- Upstream: one clean branch per fix, from `upstream/main`, holding only that fix's files. Check
  `git diff --stat upstream/main...HEAD` and `gh pr view N --json files` — the file list, not just
  the code. Closing keywords: repeat the verb per issue (`Fixes #8, fixes #9`), never in the title.
- Report to `pyloki-79` when the build works, and for each item when its test fails on main and
  passes with the fix.

## State (2026-10-11, cuda-on-wexac session)

**Build: works.** Env `loki-cuda` (conda-forge gcc 13.4, cuda-nvcc 12.6.85 + cudart/cufft/curand/
cccl/compute-sanitizer, cmake 4.4, hdf5 2.2, fftw, python 3.12), clone `~/src/loki-cuda` at
94c1eaa (remotes: `origin` = pravirkr, `fork` = assaferan), build dir `build-cuda` from
`cmake --preset cuda -DLOKI_CUDA=AUTO -DLOKI_CUDA_ARCHITECTURES=86 -DLOKI_ENABLE_NATIVE_ARCH=OFF
-DCMAKE_CXX_FLAGS=-march=x86-64-v3 -DLOKI_BUILD_BENCHMARKS=OFF`. All 233 compile/link steps
pass in ~10 min on 8 cores (peak 4.4 GB). Scripts in `~/src`: `loki_cuda_env.sh`,
`build_loki_cuda.sh` (CPU `short` queue), `test_loki_cuda.sh` (GPU: finishes the build, then
ctest + pytest), `verify_patch.sh <patch> <ctest-regex>` (tests of a patch against the unpatched
library, then with it), `run_patch_tests.sh`. Patches in `~/src/patches/`.

Gotchas found: (a) `AUTO` + an explicit architecture needs no `nvidia-smi`, so the compile can
run on a CPU queue; (b) Catch2's `catch_discover_tests` runs the test binaries at build time and
they need `libcuda.so.1`, which CPU nodes lack: the last build steps must run in the GPU job;
(c) the `short` queue is preemptable and jobs are rerunnable: a preemption mid-configure left a
half-extracted CPM package and broke the rerun. Configure once on the login node first.

**Fixes, one worktree/branch each under `/Users/eranasaf/GitHub/loki-wt/`, from upstream/main:**

1. `fix-branch-overflow-cuda` (c48bcff, pushed, pravirkr/loki PR #27): host check of the branched total against the output
   capacity before the mask memset and the materialize kernel in taylor/chebyshev/circular;
   capacity-bounded crackle append; per-parameter overflow flag (`kBranchFlagOverflow` in
   `CUBScratchArena::d_reduce_out`, the circular crackle flag is `kBranchFlagCrackle`); the
   device point generator never writes past its slot. Test
   `tests/cpp/internal/branch_capacity_cuda_t.cpp` (GPU twin of `branch_capacity_t.cpp`, plus
   CPU-count parity). `test_ep_branch_overflow.py` parametrized over backend. Note: upstream's
   e98c294 already sizes `branch_max` from the worst simulated leaf on both backends, so the
   halving retry of 56a0146 is gone; only the guards were missing on CUDA.
2. `fix-add-initial-cuda` (b768f7b, pushed, PR #28): the 3-line
   `check_greater_equal` port + `tests/cpp/internal/world_tree_cuda_t.cpp`.
3. `ffa-sweep-cuda-parity` (80dbd97, pushed, PR #29): upstream b009dce already re-uploads `m_widths_d` per
   chunk and bounds the accumulator, i.e. our `fix-ffa-freq-sweep-cuda` is superseded.
   `tests/cpp/api/ffa_sweep_parity_t.cpp` checks CPU vs CUDA on a 50-500 Hz multi-chunk sweep.

**GPU results (short-gpu, NVIDIA A40, sm_86, driver 595.58, CUDA 12.6.85):**

- Upstream main 94c1eaa: ctest 269/269 (job 692077), pytest 60/60 (692184). Note
  `CUDA_VISIBLE_DEVICES` listed all 8 GPUs of the host despite `-gpu num=1`; the tests use
  device 0. The Python package reads its version from installed metadata: a metadata-only
  `loki-0.0.3.dist-info` in the env's site-packages makes `PYTHONPATH=src` imports work.
- Item 2 (692079): the new test fails on main with "slots_to_write must match batch_scores
  size (25 == 40)", passes with the fix; full ctest 271/271. Upstream's own CUDA sweep tests
  pass on main because every nbins group there has one chunk. A sweep-level reproduction
  (`~/src/ep_sweep_groups_repro.py`, jobs 692239..693124) tried ~30 configurations (70-145 /
  70-300 Hz, 2^16..2^22 samples, accel boxes to +-1000 and a jerk box, budgets 2..16 GB, FFA
  segments 1/8 to 1/4 of the data): the CUDA planner always kept one chunk per nbins region
  (up to 701559 coords in one chunk at 8 GB) or refused the budget on the fixed buffers. So
  the failure needs a region the planner cuts into chunks of different ncoords, which these
  sizes never produce; the evidence stays the unit test and the code path (`ep_chunking`:
  `GroupMaxima::absorb` takes the max ncoords of the group; `EPMultiPassCudaCore::initialize`
  passes the whole `seed_scores_d` with the chunk's own count).
- Item 1 (692803): on main, 4 leaves x 216 children against a 32-slot span wrote 832 leaf
  slots and 832 validation-mask bytes past the span and nothing threw; the per-parameter
  overflow case did not throw either; the batch that fits matches the CPU count, origins and
  leaves. With the fix: 34/34 CUDA-tagged tests, full ctest 272/272, pytest overflow test on
  both backends. Earlier runs (692080, 692307, 692737) were my test's fault: internal kernels
  need an active `cuda_utils::CudaSetDeviceGuard` (thread-local device id) or the launch
  checks fail with cudaErrorInvalidDevice.
- Item 3 (693077): the parity test passes on upstream main: 3 chunks (64/128/256 bins), 64490
  candidates on both backends, S/N within 1e-3 relative. Our `fix-ffa-freq-sweep-cuda` is
  superseded. Two findings for the test: the two backends decode the same grid cell to
  frequencies differing in the last bits (match by nearest f0), and a strong candidate can
  sit on a neighbouring boxcar width on the GPU (compare widths as a set, S/N as the cell's
  best over widths).

Commit messages with the measured numbers: `commit_messages.md` next to this file (as
committed). assaferan approved commit, push and PRs on 2026-10-11 after reading the results.
Logs: `~/src/loki-cuda-test.692077.log`, `~/src/pytest-main.692184.log`,
`~/src/verify-add-initial.692079.log`, `~/src/verify-branch-overflow.692803.log`,
`~/src/ffa-sweep-parity.693077.log`, `~/src/repro-item2.*.log`.
