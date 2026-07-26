// BANDWIDTH roof — OpenMP target, STREAM triad z = x + a*y (x=2, y=1, a=2 -> z=4).
// Pure memory traffic: 3 arrays/step (read x, read y, write z), ~no compute, so
// it measures the HBM ceiling. Pass the iteration space N on the command line
// (default 1); grow it until the achieved GB/s plateaus --- the bandwidth roof.
//
// build: icpx -fiopenmp -fopenmp-targets=spir64 -O3 triad.cpp -o triad
// run:   ZE_AFFINITY_MASK=0.0 ./triad 67108864
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <omp.h>
#include <string>

using T = float;

int main(int argc, char **argv) {
  const int64_t N = (argc > 1) ? std::stoll(argv[1]) : 1;   // iteration space
  const int STEPS = 100;
  const T a = 2.0f;

  T *x = new T[N], *y = new T[N], *z = new T[N];
  for (int64_t i = 0; i < N; ++i) { x[i] = 2.0f; y[i] = 1.0f; z[i] = 0.0f; }
  #pragma omp target enter data map(to:x[0:N], y[0:N]) map(alloc:z[0:N])

  double best = std::numeric_limits<double>::max();          // min drops jitter
  for (int s = 0; s < STEPS; ++s) {
    const double t0 = omp_get_wtime();
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < N; ++i)
      z[i] = x[i] + a * y[i];
    best = std::min(best, omp_get_wtime() - t0);
  }
  #pragma omp target exit data map(from:z[0:N])

  double err = 0.0;
  for (int64_t i = 0; i < N; ++i) { const double d = z[i] - 4.0; err += d * d; }

  // 3 arrays moved per step (read x, read y, write z); BW = bytes / time.
  const double gbs = 3.0 * N * sizeof(T) / best * 1e-9;
  std::printf("TRIAD   N=%lld  min_ms=%.3f  %.1f GB/s  residual=%g\n",
              (long long)N, best * 1e3, gbs, err);

  delete[] x; delete[] y; delete[] z;
}
