// FLOPS roof — OpenMP target, register-resident FMA chain (no memory traffic).
// Naive port of the clpeak idea (cf. Aurora user-guides flops.cpp), MPI removed.
//
// Each work-item runs a long chain of fused multiply-adds on registers only, so
// the kernel measures the pure compute ceiling. Pass the iteration space N on
// the command line (default 1); grow it until every hardware thread is busy ---
// the roof: wall time barely moves while N (and GFLOP/s) climbs.
//
// build: icpx -fiopenmp -fopenmp-targets=spir64 -O3 flops.cpp -o flops
// run:   ZE_AFFINITY_MASK=0.0 ./flops 67108864
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <omp.h>
#include <string>
#include <vector>

#define MAD_4(x, y)  x = y*x + y; y = x*y + x; x = y*x + y; y = x*y + x;
#define MAD_16(x, y) MAD_4(x,y) MAD_4(x,y) MAD_4(x,y) MAD_4(x,y)
#define MAD_64(x, y) MAD_16(x,y) MAD_16(x,y) MAD_16(x,y) MAD_16(x,y)

using T = float;
const int    INNER = 128;                      // MAD_64 blocks per work-item
const double FLOP_PER_WI = INNER * 64 * 2.0;   // 64 FMAs/block, 2 FLOP/FMA

int main(int argc, char **argv) {
  const int64_t N = (argc > 1) ? std::stoll(argv[1]) : 1;   // iteration space
  const int STEPS = 100;

  std::vector<T> A(N, T(1.1));
  T *Aptr = A.data();
  #pragma omp target enter data map(to:Aptr[0:N])

  double best = std::numeric_limits<double>::max();          // min drops warm-up
  for (int s = 0; s < STEPS; ++s) {
    const double t0 = omp_get_wtime();
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < N; ++i) {
      T x = Aptr[i], y = -x;
      for (int j = 0; j < INNER; ++j) { MAD_64(x, y); }
      Aptr[i] = y;
    }
    best = std::min(best, omp_get_wtime() - t0);
  }
  #pragma omp target exit data map(from:Aptr[0:N])

  const double gflops = FLOP_PER_WI * N / best * 1e-9;
  std::printf("FLOPS   N=%lld  min_ms=%.3f  %.1f GFLOP/s\n",
              (long long)N, best * 1e3, gflops);
}
