// std::par — standard C++ parallel algorithms, offloaded to the GPU.
// Intel path lowers par_unseq through oneDPL/SYCL.
// build: icpx -fsycl -fsycl-pstl-offload=gpu stdpar_triad.cpp -o stdpar_triad
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <execution>
#include <limits>
#include <numeric>
#include <vector>

int main() {
  const size_t N = 1 << 20;
  const int STEPS = 1000;
  const float a = 2.0f;
  std::vector<float> x(N, 2.0f), y(N, 1.0f), z(N, 0.0f);

  std::vector<size_t> idx(N);
  std::iota(idx.begin(), idx.end(), 0);

  double best_ms = std::numeric_limits<double>::max();
  for (int s = 0; s < STEPS; ++s) {
    auto t0 = std::chrono::high_resolution_clock::now();
    // No pragma, no lambda-to-kernel keyword — just ISO C++. The compiler/runtime
    // decides it runs on the GPU.
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                  [=, xp = x.data(), yp = y.data(), zp = z.data()](size_t i) {
                    zp[i] = xp[i] + a * yp[i];
                  });
    auto t1 = std::chrono::high_resolution_clock::now();
    best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  // Reduction: residual error sum (z-4)^2, also a standard algorithm.
  double err = std::transform_reduce(
      std::execution::par_unseq, z.begin(), z.end(), 0.0, std::plus<double>(),
      [](float v) { const double d = v - 4.0; return d * d; });
  std::printf("std::par  N=%zu  min_ms=%.3f  residual=%g\n", N, best_ms, err);
}
