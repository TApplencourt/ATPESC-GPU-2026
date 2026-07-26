"""Minimal GPU triad (z = x + a*y) in Python using dpnp.

dpnp is a NumPy-drop-in that runs on Intel GPUs. It does NOT reimplement the
GPU stack: its arrays live in Level Zero / SYCL USM memory and its kernels are
dispatched through the same oneAPI (SYCL / Level Zero) runtime that a C++ SYCL
program uses. Same runtime underneath, Python API on top.
"""

import time

import dpnp

N = 1 << 20
STEPS = 1000
a = 2.0

x = dpnp.full(N, 2.0, dtype=dpnp.float32)  # allocated on the GPU
y = dpnp.full(N, 1.0, dtype=dpnp.float32)

best_ms = float("inf")
for _ in range(STEPS):
    t0 = time.perf_counter()
    z = x + a * y                          # triad kernel runs on the GPU
    best_ms = min(best_ms, (time.perf_counter() - t0) * 1e3)

# Reduction: residual error sum (z-4)^2, also on the GPU.
err = float(dpnp.sum((z - 4.0) ** 2))
print(f"dpnp      N={N}  min_ms={best_ms:.3f}  residual={err:g}")
