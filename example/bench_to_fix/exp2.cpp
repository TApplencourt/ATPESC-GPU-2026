// EXPERIMENT 2 -- "SYCL is faster than Kokkos" ... or is it?
// build: module load kokkos/5.0.1-sycl   (sets the include/lib paths)
//        icpx -fsycl -fiopenmp -O2 -std=c++23 exp2.cpp -lkokkoscore -lkokkoscontainers -o exp2
//   -fiopenmp is required: this Kokkos build bundles the OpenMP host backend.
#include "bench.hpp"
#include <Kokkos_Core.hpp>
#include <sycl/sycl.hpp>
#include <format>
#include <iostream>
#include <vector>

#define MAD_4(x, y)  x = y*x + y; y = x*y + x; x = y*x + y; y = x*y + x;
#define MAD_16(x, y) MAD_4(x,y) MAD_4(x,y) MAD_4(x,y) MAD_4(x,y)
#define MAD_64(x, y) MAD_16(x,y) MAD_16(x,y) MAD_16(x,y) MAD_16(x,y)

static double chain() {                       // host reference: same chain, on CPU
  double x = 1.3, y = 2.0;
  for (int r = 0; r < 4096; ++r) { MAD_64(x, y); }
  return y;
}

int main(int argc, char *argv[]) {
  const int K = 5;
  const int REP = 4096;
  const int STEPS = 500;

  std::vector<double> y_ref(K, chain());

  unsigned long sycl_ns = 0, kokkos_ns = 0;

  // ---- SYCL ----
  {
    sycl::queue q;
    double *out = sycl::malloc_shared<double>(K, q);
    auto f = [&]() {
      for (int k = 0; k < K; ++k) {
        double *o = out + k;
        q.parallel_for(1, [=](sycl::id<1>) {
          double x = 1.3, y = 2.0;
          for (int r = 0; r < REP; ++r) { MAD_64(x, y); }
          o[0] = y;
        });
      }
      q.wait();
    };
    sycl_ns = bench_min_ns(STEPS, f);
    std::vector<double> y(out, out + K);
    assert(y == y_ref);
    std::cout << std::format("sycl     K={} STEPS={} min_ms={:.3f}", K, STEPS, sycl_ns / 1e6) << '\n';
    sycl::free(out, q);
  }

  // ---- Kokkos ----
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double *> out("out", K);
    auto f = [&]() {
      for (int k = 0; k < K; ++k) {
        auto o = out;
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const size_t) {
          double x = 1.3, y = 2.0;
          for (int r = 0; r < REP; ++r) { MAD_64(x, y); }
          o(k) = y;
        });
      }
      Kokkos::fence();
    };
    kokkos_ns = bench_min_ns(STEPS, f);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    std::vector<double> y(h.data(), h.data() + K);
    assert(y == y_ref);
    std::cout << std::format("kokkos   K={} STEPS={} min_ms={:.3f}", K, STEPS, kokkos_ns / 1e6) << '\n';
  }
  Kokkos::finalize();

  std::cout << std::format(">>> SYCL {:.1f}x faster than Kokkos !!", (double)kokkos_ns / sycl_ns) << '\n';
}
