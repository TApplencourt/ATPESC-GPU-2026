// EXPERIMENT 2 -- FIXED. Same result as exp2.cpp, honest comparison.
//
// The trap in exp2.cpp was the QUEUE semantics, not the language:
//   * SYCL's default queue is OUT-OF-ORDER, so the K independent kernels overlap
//     and the timed loop pays ~one chain latency.
//   * Kokkos' default execution space is a single IN-ORDER stream, so the same K
//     kernels serialize and pay ~K chain latencies.
//
// The fair fix here is to make KOKKOS overlap too (rather than slow SYCL down):
// Kokkos::Experimental::partition_space hands out K independent execution-space
// instances -- K separate streams. Launch kernel k on instance[k] and the K
// chains run concurrently, exactly like the out-of-order SYCL queue. Both now
// pay ~one chain latency, both compute the same y, and the two min_ms numbers
// match -- it was the benchmark, not the language.
//
// (The other honest fix is the reverse: give the SYCL queue the in_order
//  property so IT serializes too. Either way the point stands -- compare like
//  with like. Both defaults are fine; you just have to know what they are.)
//
// build: module load kokkos/5.0.1-sycl   (sets the include/lib paths)
//        icpx -fsycl -fiopenmp -O2 -std=c++23 exp2_sol.cpp -lkokkoscore -lkokkoscontainers -o exp2_sol
#include "bench.hpp"
#include <Kokkos_Core.hpp>
#include <sycl/sycl.hpp>
#include <format>
#include <iostream>
#include <vector>

#define MAD_4(x, y)  x = y*x + y; y = x*y + x; x = y*x + y; y = x*y + x;
#define MAD_16(x, y) MAD_4(x,y) MAD_4(x,y) MAD_4(x,y) MAD_4(x,y)
#define MAD_64(x, y) MAD_16(x,y) MAD_16(x,y) MAD_16(x,y) MAD_16(x,y)

static double chain() {
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

  // ---- SYCL, out-of-order queue (unchanged: it already overlaps) ----
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

  // ---- Kokkos, one execution-space instance per kernel (the fix) ----
  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::DefaultExecutionSpace;
    Kokkos::View<double *> out("out", K);
    auto inst = Kokkos::Experimental::partition_space(Exec(), std::vector<int>(K, 1));
    auto f = [&]() {
      for (int k = 0; k < K; ++k) {
        auto o = out;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Exec>(inst[k], 0, 1), KOKKOS_LAMBDA(const size_t) {
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

  std::cout << std::format(">>> SYCL {:.1f}x vs Kokkos", (double)kokkos_ns / sycl_ns) << '\n';
}
