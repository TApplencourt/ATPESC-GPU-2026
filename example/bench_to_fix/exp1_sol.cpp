// EXPERIMENT 1 -- FIXED. Same result as exp1.cpp, honest comparison.
//
// The two traps from exp1.cpp are removed:
//   1. Data motion. OpenMP keeps x/y/z resident with `omp target enter/exit
//      data`; the kernel inside the loop no longer maps anything, so nothing
//      crosses the link per launch.
//   2. What is timed. The SYCL f() now WAITS per launch, so bench_min_ns clocks
//      real GPU compute for both models, not enqueue latency.
//
// Both results are still checked against z_ref. The two min_ms numbers now land
// in the same ballpark -- it was the benchmark, not the language.
//
// build: icpx -fsycl -fiopenmp -fopenmp-targets=spir64 -O2 -std=c++23 exp1_sol.cpp -o exp1_sol
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

  // ---- OpenMP target
  {
    std::vector<float> x(N, 2.0f), y(N, 1.0f), z(N, 0.0f);
    auto f = [&]() {
      float *xp = x.data(), *yp = y.data(), *zp = z.data();
      #pragma omp target teams distribute parallel for
      for (size_t i = 0; i < N; ++i)
        zp[i] = xp[i] + a * yp[i];
    };

    float *xp = x.data(), *yp = y.data(), *zp = z.data();
    #pragma omp target enter data map(to:xp[0:N], yp[0:N], zp[0:N])
    omp_ns = bench_min_ns(STEPS, f);
    #pragma omp target exit data map(from:zp[0:N])
    
    assert(z == z_ref);
    std::cout << std::format("openmp   N={} STEPS={} min_ms={:.3f}", N, STEPS, omp_ns / 1e6) << '\n';
  }

  // ---- SYCL ----
  {
    sycl::queue q{sycl::property::queue::in_order()};
    
    sycl::usm_allocator<float, sycl::usm::alloc::shared> alloc(q);
    std::vector<float, decltype(alloc)> x(N, 2.0f, alloc), y(N, 1.0f, alloc), z(N, 0.0f, alloc);
    auto f = [&]() {
      float *xp = x.data(), *yp = y.data(), *zp = z.data();
      q.parallel_for(N, [=](sycl::id<1> i) { zp[i] = xp[i] + a * yp[i]; }).wait();
    };
    sycl_ns = bench_min_ns(STEPS, f);
    assert(z == z_ref);
    std::cout << std::format("sycl     N={} STEPS={} min_ms={:.3f}", N, STEPS, sycl_ns / 1e6) << '\n';
  }

  std::cout << std::format(">>> SYCL {:.1f}x vs OpenMP", (double)omp_ns / sycl_ns) << '\n';
}
