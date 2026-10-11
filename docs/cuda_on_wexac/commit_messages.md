# Commit messages of the CUDA fix branches (2026-10-11), as committed

## fix-branch-overflow-cuda c48bcff, PR #27

```
fix(cuda): never write branched leaves past the workspace; reject the batch first

The CUDA counterpart of PR #23 (56a0146, e98c294). The CPU branch kernels
throw before a parent's Cartesian write when its children do not fit, and
branch_param_padded throws when one parameter branches past branch_max.
The CUDA kernels had neither guard: the analyze kernel wrote a parameter's
points past its branch_max slot, the host memset the validation mask and
launched the materialize kernel for the full child count whatever the
output size, and the circular hole expansion appended through an atomic
counter with no bound. The size check in execute_iteration_batched ran
only after branch() returned.

- kernel_utils.cuh: branch_param_generate_points_device writes at most
  out_size points and keeps the true count. flag_branch_overflow_device
  raises kBranchFlagOverflow in the batch's flag word
  (CUBScratchArena::d_reduce_out) when any count passed branch_max; the
  circular crackle flag is bit kBranchFlagCrackle of the same word.
  branch_capacity and check_branch_fits are the host-side checks.
- taylor, chebyshev, circular: the host zeroes the flag word before the
  analyze kernel, reads it back with the batch total in the existing sync,
  and throws before the validation mask or any child is written: on the
  overflow flag, or when the total exceeds the smallest of the leaves,
  origins and mask views. kernel_expand_crackle_holes stops appending at
  capacity (the count keeps growing) and the host rejects a final count
  past it before the mask memset. The crackle test "has_crackle == 0"
  compared a pointer and always launched kernel 3; it now tests the flag,
  which is what the kernel recomputes per child anyway.
- tests/cpp/internal/branch_capacity_cuda_t.cpp, the GPU counterpart of
  branch_capacity_t.cpp: a batch that fits matches the CPU count, origins
  and leaves; ~216 children per leaf against 8 slots throws with the
  canaries behind the leaves, origins and mask intact; a parameter past
  branch_max throws on both backends. On upstream main (A40, sm_86, CUDA
  12.6.85) the 4 x 216 children against a 32-slot span wrote 832 leaf
  slots and 832 validation-mask bytes past the span and nothing threw.
- tests/python/test_ep_branch_overflow.py runs the overflow configuration
  on both backends.

Verified on wexac (NVIDIA A40, CUDA 12.6.85, gcc 13.4): with the fix,
ctest 272/272 (all CUDA EP sweep and parity tests included) and the
Python overflow test on both backends.
```

## fix-add-initial-cuda b768f7b, PR #28

```
fix(world_tree_cuda): accept a seed batch larger than the seeds it copies

The CUDA counterpart of aa70cfa (PR #10). EPFreqSweep sizes a group's
EPWorkspaceCUDA seed buffers for the largest chunk of the group
(ep_chunking: group.ncoords is the max over its chunks), and
EPMultiPassCudaCore::initialize passes the whole seed_scores_d buffer with
the chunk's own leaf count. add_initial required the two sizes to be equal,
so a chunk smaller than the largest of its group threw "slots_to_write
must match batch_scores size". It now only requires the batch to hold the
seeds it copies, as the CPU WorldTree does.

tests/cpp/internal/world_tree_cuda_t.cpp: add_initial of 25 seeds from a
40-seed batch keeps the leading 25 (leaves, folds, scores, score_max), and
a batch shorter than the count is still rejected. On upstream main (A40,
sm_86, CUDA 12.6.85) the test fails with "slots_to_write must match
batch_scores size (25 == 40)"; with the fix, ctest is 271/271. The
sweep tests do not reach it: at their sizes every nbins group holds one
chunk.
```

## ffa-sweep-cuda-parity 80dbd97, PR #29

```
test(ffa): CPU vs CUDA parity of FFAFreqSweep across chunks

The CUDA sweep scores and decodes each chunk with its own widths since
b009dce. This checks it end to end: a 50-200 Hz sweep with three chunks
(64, 128, 256 bins) on both backends produces the same (nbins, width)
cells, the same candidate count, and for the 50 strongest CPU candidates
the GPU has the same frequency cell with its best S/N within 1%. The two
backends decode a cell's frequency with last-bit differences, so cells are
matched by nearest frequency (relative 1e-9), not by equality.

On an A40 (sm_86, CUDA 12.6.85): 64490 candidates on both backends, S/N
differences below 1e-3 relative.
```
