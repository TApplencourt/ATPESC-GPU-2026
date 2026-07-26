# GPU programming-model demos (Aurora / Intel PVC)

Compile on a **login** node, run on a **compute** node. `icpx`/`ifx` are already
in your PATH.

## Compile

One flag set per programming model:

| Model | Flags |
|-------|-------|
| SYCL | `icpx -fsycl` |
| OpenMP target | `icpx -fiopenmp -fopenmp-targets=spir64` |
| `std::par` | `icpx -fsycl -fsycl-pstl-offload=gpu` |
| `do concurrent` | `ifx -fiopenmp -fopenmp-targets=spir64 -fopenmp-target-do-concurrent` |
| Kokkos | `module load kokkos/5.0.1-sycl` then `icpx -fsycl -fiopenmp … -lkokkoscore -lkokkoscontainers` |
| Python (dpnp) | `module load frameworks/2025.3.1` (nothing to compile) |

The Kokkos module sets the include/lib paths for you, so no `-I`/`-L` is needed.
`-fiopenmp` is required for Kokkos here because this build also bundles the
OpenMP host backend (the default *device* execution space is still SYCL/GPU).

Or just let `make` do it — each binary is built next to its source:

```sh
make            # builds everything
```

## Run

On a compute node. Pin one tile with `ZE_AFFINITY_MASK=0.0`:

```sh
make run                                    # runs everything
# or one at a time:
export ZE_AFFINITY_MASK=0.0
./triad_galery/sycl_triad
```

For Kokkos, `module load kokkos/5.0.1-sycl` first. For Python, `module load
frameworks/2025.3.1` and pin one tile like this (the frameworks stack uses a
flat device hierarchy, so `ZE_AFFINITY_MASK=0.0` alone hides every device):

```sh
module load frameworks/2025.3.1; ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0.0 python triad_galery/py_triad.py
```

`make run` does all of this for you.

## The gallery — same triad, every model

`triad_galery/` is the identical `z = x + a·y` (x=2, y=1, a=2 → z=4) written once
per model, each with a min-timing loop and a `sum (z-4)²` residual (0 = correct).
Measured on one PVC tile (N = 2²⁰, min over 1000 steps):

| Model | min (ms) | notes |
|-------|---------:|-------|
| SYCL (shared USM, resident) | 0.013 | data stays on the GPU |
| Kokkos (`View`, resident) | 0.017 | SYCL backend, resident |
| `dpnp` (Python) | 0.15 | resident device arrays |
| OpenMP (`map()` every step) | 0.75 | re-migrates x/y/z each step |
| `std::par` | 0.83 | host `std::vector`, re-migrates |
| `do concurrent` (Fortran) | 1.31 | host allocatables, re-migrates |

Same math everywhere — the spread is the **memory model**, not the language. The
resident-data models (SYCL/Kokkos/dpnp) are ~50× faster than the ones that keep
data in plain host allocations and re-migrate it every step. That gap is exactly
what `bench_to_fix/exp1` dissects.

## Layout

- `triad_galery/` — the same `z = x + a·y` in each model (the gallery)
- `characterisation/` — `peak_sol` (BW/FLOP roofs) and `flops` (compute roof)
- `bench_to_fix/` — two rigged benchmarks + their honest fixes
- `thomas/` — a 40-line single-source GPU layer over OpenCL + SPIR-V
  (THOMAS: Tiny Heterogeneous Offload Macro, ATPESC + SPIR-V). The "not magic"
  demo: what SYCL/Kokkos do, hand-rolled. Has its own `Makefile` (see its README).
