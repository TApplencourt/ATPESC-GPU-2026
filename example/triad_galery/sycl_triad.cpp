// SYCL: triad z = x + a*y   —   build: icpx -fsycl sycl_triad.cpp -o sycl_triad
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <limits>

int main() {
  sycl::queue q;
  std::printf("SYCL on: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());

  const size_t N = 1 << 20;
  const int STEPS = 1000;
  const float a = 2.0f;
  float *x = sycl::malloc_shared<float>(N, q);
  float *y = sycl::malloc_shared<float>(N, q);
  float *z = sycl::malloc_shared<float>(N, q);
  for (size_t i = 0; i < N; ++i) { x[i] = 2.0f; y[i] = 1.0f; z[i] = 0.0f; }

  double best_ms = std::numeric_limits<double>::max();
  for (int s = 0; s < STEPS; ++s) {
    auto t0 = std::chrono::high_resolution_clock::now();
    q.parallel_for(N, [=](sycl::id<1> i) { z[i] = x[i] + a * y[i]; }).wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  // Reduction: residual error sum (z-4)^2 on the GPU.
  double *err = sycl::malloc_shared<double>(1, q);
  *err = 0.0;
  q.parallel_for(N, sycl::reduction(err, sycl::plus<double>()),
                 [=](sycl::id<1> i, auto &acc) {
                   const double d = z[i] - 4.0;
                   acc += d * d;
                 }).wait();
  std::printf("SYCL      N=%zu  min_ms=%.3f  residual=%g\n", N, best_ms, *err);

  sycl::free(x, q); sycl::free(y, q); sycl::free(z, q); sycl::free(err, q);
}
