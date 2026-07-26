# THOMAS — a 40-line single-source GPU layer

**T**iny **H**eterogeneous **O**penCL **M**acro, **A**TPESC + **S**PIR-V.

This is the "not magic" demo: what SYCL/Kokkos do for you, hand-rolled over
OpenCL + SPIR-V in ~40 lines. Write the kernel **once** as a C++ lambda; it runs
on the PVC GPU.

The trick: `triad.cpp` is compiled **twice** from the same source —

- **device pass** — `icpx -x cl -cl-std=clc++ -DTHOMAS_DEVICE` turns the lambda
  body into a `__kernel` and lowers it to SPIR-V (embedded as a C array);
- **host pass** — plain `icpx` keeps only the kernel *name*, loads the SPIR-V,
  JITs it for the device, and launches it.

The `KERNEL(name, params, body)` macro is the whole shim — one invocation, two
expansions. `gptr<T>` is a `__global` pointer on the device pass and a plain
`T*` on the host pass, so one signature parses in both. See `thomas.hpp`.

## Build & run

Build on a **login** node, run on a **compute** node (GPU only):

```sh
make                 # compiles the device pass -> SPIR-V, then the host program
make run             # ZE_AFFINITY_MASK=0.0 ./triad   (one PVC tile)
```

## The triad — same benchmark as the gallery

`triad.cpp` runs the identical `z = x + a·y` (x=2, y=1, a=2 → z=4) as
`triad_galery/`, min-timed over 1000 steps with a `sum(z-4)²` residual (0 =
correct). Because the data stays resident in USM and the kernel is real SPIR-V,
THOMAS lands right on SYCL — it *is* the same thing, minus the sugar:

```
thomas: running on Intel(R) Data Center GPU Max 1550
THOMAS    N=1048576  min_ms=0.013  residual=0
SYCL      N=1048576  min_ms=0.013  residual=0     # triad_galery/sycl_triad
```

## Files

- `thomas.hpp` — the tiny runtime + the `KERNEL` macro (device & host views)
- `triad.cpp` — single-source example: the triad kernel written once + `main()`
- `Makefile` — the two-pass build (device → SPIR-V → host)
