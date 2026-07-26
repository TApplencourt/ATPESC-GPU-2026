// OpenMP target: triad z = x + a*y
// build: icpx -fiopenmp -fopenmp-targets=spir64 omp_triad.cpp -o omp_triad
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <omp.h>

int main() {
  // omp_is_initial_device routine returns true if the current task is executing on the host device; 
  // otherwise, it returns false. 
  int on_gpu = 0;
  #pragma omp target map(from:on_gpu)
  on_gpu = !omp_is_initial_device();
  std::printf("OpenMP on: %s\n", on_gpu ? "GPU (offload device)" : "host (no offload!)");

  const size_t N = 1 << 20;
  const int STEPS = 1000;
  const float a = 2.0f;
  float *x = new float[N], *y = new float[N], *z = new float[N];
  for (size_t i = 0; i < N; ++i) { x[i] = 2.0f; y[i] = 1.0f; z[i] = 0.0f; }

  // Timing loop: run the triad STEPS times, keep the fastest (min drops jitter).
  // Note the EXPLICIT loop — OpenMP is descriptive: you write the loop, the
  // pragma says "offload and parallelize it". map() moves the data.
  double best_ms = std::numeric_limits<double>::max();
  for (int s = 0; s < STEPS; ++s) {
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp target teams distribute parallel for map(to:x[0:N], y[0:N]) map(from:z[0:N])
    for (size_t i = 0; i < N; ++i)
      z[i] = x[i] + a * y[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  // Reduction: residual error sum (z-4)^2, accumulated on the GPU.
  double err = 0.0;
  #pragma omp target teams distribute parallel for map(to:z[0:N]) reduction(+:err)
  for (size_t i = 0; i < N; ++i) {
    const double d = z[i] - 4.0;
    err += d * d;
  }

  std::printf("OpenMP    N=%zu  min_ms=%.3f  residual=%g\n", N, best_ms, err);

  delete[] x; delete[] y; delete[] z;
}
