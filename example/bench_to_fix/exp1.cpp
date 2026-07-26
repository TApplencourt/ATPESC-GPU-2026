// EXPERIMENT 1 -- "SYCL is faster than OpenMP" ... or is it?
#include "bench.hpp"
#include <sycl/sycl.hpp>
#include <format>
#include <iostream>
#include <vector>

int main() {
  const size_t N = 1 << 20;
  const int STEPS = 1000;
  const float a = 2.0f;

  std::vector<float> z_ref(N, 2.0f + a * 1.0f);   // x=2, y=1  ->  z = 4

  unsigned long omp_ns = 0, sycl_ns = 0;

  // ---- OpenMP target ----
  {
    std::vector<float> x(N, 2.0f), y(N, 1.0f), z(N, 0.0f);
    auto f = [&]() {
      float *xp = x.data(), *yp = y.data(), *zp = z.data();  // locals: map() can't attach to a capture
      #pragma omp target teams distribute parallel for map(to:xp[0:N], yp[0:N]) map(from:zp[0:N])
      for (size_t i = 0; i < N; ++i)
        zp[i] = xp[i] + a * yp[i];
    };
    omp_ns = bench_min_ns(STEPS, f);
    assert(z == z_ref);
    std::cout << std::format("openmp   N={} STEPS={} min_ms={:.3f}", N, STEPS, omp_ns / 1e6) << '\n';
  }

  // ---- SYCL, shared USM ----
  {
    sycl::queue q{sycl::property::queue::in_order()};

    sycl::usm_allocator<float, sycl::usm::alloc::shared> alloc(q);
    std::vector<float, decltype(alloc)> x(N, 2.0f, alloc), y(N, 1.0f, alloc), z(N, 0.0f, alloc);
    auto f = [&]() {
      float *xp = x.data(), *yp = y.data(), *zp = z.data();
      q.parallel_for(N, [=](sycl::id<1> i) { zp[i] = xp[i] + a * yp[i]; });
    };
    sycl_ns = bench_min_ns(STEPS, f);
    q.wait();
    assert(z == z_ref);
    std::cout << std::format("sycl     N={} STEPS={} min_ms={:.3f}", N, STEPS, sycl_ns / 1e6) << '\n';
  }

  std::cout << std::format(">>> SYCL {:.1f}x faster than OpenMP !!", (double)omp_ns / sycl_ns) << '\n';
}
