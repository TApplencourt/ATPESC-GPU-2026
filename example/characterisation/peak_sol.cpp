// Reach both memory and Flops roof.

// Build: icpx -fsycl -O3 -std=c++23 peak_sol.cpp -o peak_sol
// Run:   ZE_AFFINITY_MASK=0.0 ./peak_sol

#include <sycl/sycl.hpp>
#include "mdspan.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <format>
#include <iostream>
#include <type_traits>

namespace md = std; // We use kokkos and not std, so trick to be portable
using T = float;
constexpr const char *TYPE = std::is_same_v<T, double> ? "double (DP)" : "float (SP)";

// Fastest run, in nanoseconds. 
template <class Fn> static double best_ns(Fn kern, int reps = 20) {
  double best = std::numeric_limits<double>::max();  // min over reps drops warm-up
  for (int t = 0; t < reps; t++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    kern();
    auto t1 = std::chrono::high_resolution_clock::now();
    best = std::min(best,
      std::chrono::duration<double, std::nano>(t1 - t0).count());
  }
  return best;
}

// Achieved bandwidth (GB/s) and compute (GFLOP/s) of a run.
struct Rate { double gbs, gflops; };

// Run the kernel, print its rates, and return them. 
template <int TILE_SIZE, int N_FMA>
static Rate run(sycl::queue &q, const char *tag, T *a, T *b, T *c, size_t N) {
  const size_t TILE_NUM = N / TILE_SIZE;
  // 2D row-major views: view[i, tid] == p[i*TILE_NUM + tid] (coalesced access).
  md::mdspan<T, md::dextents<size_t, 2>> av(a, TILE_SIZE, TILE_NUM),
                                         bv(b, TILE_SIZE, TILE_NUM),
                                         cv(c, TILE_SIZE, TILE_NUM);
  double ns = best_ns([&] {
    q.parallel_for(sycl::range<1>(TILE_NUM), [=](sycl::id<1> id_raw) {
       const size_t id = id_raw;   // mdspan operator[] needs an integer, not sycl::id
       std::array<T, TILE_SIZE> x, y;
       for (int i = 0; i < TILE_SIZE; i++) { x[i] = bv[i, id]; y[i] = cv[i, id]; }
       for (int j = 0; j < N_FMA; j++)
         for (int i = 0; i < TILE_SIZE; i++) x[i] = y[i]*x[i] + y[i];
       for (int i = 0; i < TILE_SIZE; i++) av[i, id] = x[i];
     }).wait();
  });
  Rate r{ (3.0 * sizeof(T)) * N / ns, (2.0 * N_FMA) * N / ns };
  std::cout << std::format("{:<10} | {:>6} | {:>8.1f} | {:>9.1f}", tag, N_FMA, r.gbs, r.gflops) << '\n';
  return r;
}

int main(int argc, char **argv) {
  const size_t N = (argc > 1) ? std::stoul(argv[1]) : 240'000'000;
  sycl::queue q;
  std::cout << std::format("Device: {} [N={}, type={}]",
               q.get_device().get_info<sycl::info::device::name>(), N, TYPE) << '\n';

  T *a = sycl::malloc_device<T>(N, q);
  T *b = sycl::malloc_device<T>(N, q);
  T *c = sycl::malloc_device<T>(N, q);
  q.fill(a, T(1.1), N); q.fill(b, T(2.2), N); q.fill(c, T(3.3), N);
  q.wait();

  // TILE_SIZE must divide N (TILE_NUM = N / TILE_SIZE).
  assert(N % 12 == 0);

  std::cout << std::format("{:<10} | {:>6} | {:>8} | {:>9}", "Name", "N_FMA", "GB/s", "GFLOP/s") << '\n';
  Rate bw   = run<6, 1>(q, "BW roof",  a, b, c, N); // triad, no FMA (TILE=6 to saturate HBM)
  Rate fma  = run<12, 2048>(q, "FMA roof", a, b, c, N); // Lot of FMA, a few tile for ILP
  Rate roof = run<6, 120>(q, "ROOFER",   a, b, c, N); // ridge for float (SP)

  // ROOFER as a fraction of each roof
  std::cout << std::format("ROOFER reaches {:.0f}% of BW roof and {:.0f}% of FMA roof at once",
               100.0 * roof.gbs / bw.gbs, 100.0 * roof.gflops / fma.gflops) << '\n';

  sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
  return 0;
}
