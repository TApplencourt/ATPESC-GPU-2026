// triad.cpp — SINGLE SOURCE, same triad as triad_galery/: z = x + a*y
// (x=2, y=1, a=2 -> z=4). Written ONCE as a lambda, run on the PVC GPU through
// THOMAS. Min-timed over STEPS like the gallery, residual sum(z-4)^2 (0 = ok).
#include "thomas.hpp"

// The kernel, written ONCE. gptr<T> is a __global pointer on the device pass
// and a plain T* on the host pass — one signature, both compilations.
KERNEL(triad, (float a, gptr<const float> x, gptr<const float> y, gptr<float> z),
{
  z[i] = x[i] + a * y[i];
})

#ifndef THOMAS_DEVICE
#include <algorithm>
#include <chrono>
#include <limits>

int main() {
  thomas::init();

  const size_t N = 1 << 20;
  const int STEPS = 1000;
  const float a = 2.0f;
  float *x = thomas::alloc<float>(N);
  float *y = thomas::alloc<float>(N);
  float *z = thomas::alloc<float>(N);
  for (size_t i = 0; i < N; ++i) { x[i] = 2.0f; y[i] = 1.0f; z[i] = 0.0f; }

  // Data stays resident in USM; parallel_for blocks (clFinish) each step, so we
  // time the whole launch+finish like the gallery's .wait() loop.
  double best_ms = std::numeric_limits<double>::max();
  for (int s = 0; s < STEPS; ++s) {
    auto t0 = std::chrono::high_resolution_clock::now();
    thomas::parallel_for(N, triad, a, x, y, z);
    auto t1 = std::chrono::high_resolution_clock::now();
    best_ms = std::min(best_ms,
                       std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  double err = 0;
  for (size_t i = 0; i < N; ++i) err += (z[i] - 4.0) * (z[i] - 4.0);

  // Triad moves 3 arrays/step (read x, read y, write z); BW = bytes / time.
  const double gbps = 3.0 * N * sizeof(float) / (best_ms * 1e-3) / 1e9;
  std::printf("THOMAS    N=%zu  min_ms=%.3f  BW=%.0f GB/s  residual=%g\n",
              N, best_ms, gbps, err);

  thomas::free(x); thomas::free(y); thomas::free(z);
  return 0;
}
#endif
