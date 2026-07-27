// FLOPS roof + physical occupancy map --- OpenMP target on Intel GPU (PVC).
// build: icpx -fiopenmp -fopenmp-targets=spir64 -O3 flops_with_probe_sol.cpp -o flops_with_probe_sol
// run:   ZE_AFFINITY_MASK=0.0 ./flops_with_probe_sol 67108864
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <omp.h>
#include <string>
#include <vector>

// Intel GPU location built-ins. IGC resolves these at JIT time; we only declare
// them for the device pass. 
// See https://github.com/Shamrock-code/Shamrock/blob/main/src/shambackends/include/shambackends/intrisics/get_device_clock.hpp 
// for other HW
#pragma omp begin declare target
extern "C++" {
unsigned __attribute__((overloadable)) intel_get_tile_id(void);
unsigned __attribute__((overloadable)) intel_get_slice_id(void);
unsigned __attribute__((overloadable)) intel_get_subslice_id(void);
unsigned __attribute__((overloadable)) intel_get_dual_subslice_id(void);
unsigned __attribute__((overloadable)) intel_get_eu_id(void);
unsigned __attribute__((overloadable)) intel_get_hw_thread_id(void);
}
#pragma omp end declare target

// Host fallback: only the device pass (__SPIR64__) has the real built-ins, but
// the runtime still links a host copy of the target region. These never run.
#ifndef __SPIR64__
unsigned intel_get_tile_id(void) { return 0; }
unsigned intel_get_slice_id(void) { return 0; }
unsigned intel_get_subslice_id(void) { return 0; }
unsigned intel_get_dual_subslice_id(void) { return 0; }
unsigned intel_get_eu_id(void) { return 0; }
unsigned intel_get_hw_thread_id(void) { return 0; }
#endif

#define MAD_4(x, y)  x = y*x + y; y = x*y + x; x = y*x + y; y = x*y + x;
#define MAD_16(x, y) MAD_4(x,y) MAD_4(x,y) MAD_4(x,y) MAD_4(x,y)
#define MAD_64(x, y) MAD_16(x,y) MAD_16(x,y) MAD_16(x,y) MAD_16(x,y)

using T = float;
const int    INNER = 128;                      // MAD_64 blocks per work-item
const double FLOP_PER_WI = INNER * 64 * 2.0;   // 64 FMAs/block, 2 FLOP/FMA

// Counter capacities, generous for one PVC tile (real ranges are 8 / 8 / 8 / 8).
enum { TILE_CAP = 8, SLICE_CAP = 8, SS_CAP = 8, EU_CAP = 8, THR = 8, HT_CAP = 8192 };
using u64 = unsigned long long;

// Shade a 0..1 fraction as a heat cell (idle -> hot).
static const char *shade(double f) {
  static const char *ramp[] = {"·", "░", "▒", "▓", "█"};
  if (f <= 0.0) return " ";
  int i = (int)(f * 4.999);
  return ramp[i < 0 ? 0 : (i > 4 ? 4 : i)];
}

int main(int argc, char **argv) {
  const int64_t N = (argc > 1) ? std::stoll(argv[1]) : 1;   // iteration space
  const int STEPS = 100;

  std::vector<T> A(N, T(1.1));
  T *Aptr = A.data();
  #pragma omp target enter data map(to:Aptr[0:N])

  // ---- timing loop: pure FMA, no atomics, so GFLOP/s stays clean ------------
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

  // ---- census pass: the SAME FMA kernel, re-run once with a location tally ----
  // We do NOT tally inside the timing loop (atomics would pollute GFLOP/s). But a
  // separate empty kernel might be scheduled differently, so instead we run the
  // identical FMA body one more time and tag each work-item by where it landed ---
  // this map reflects how the benchmarked kernel itself was spread over the die.
  std::vector<u64> tile(TILE_CAP, 0), slice(SLICE_CAP, 0), ss(SS_CAP, 0),
                   dss(SS_CAP, 0), eu(EU_CAP, 0), ht(HT_CAP, 0);
  u64 *pt = tile.data(), *psl = slice.data(), *pss = ss.data(),
      *pds = dss.data(), *peu = eu.data(), *pht = ht.data();
  #pragma omp target enter data map(to:Aptr[0:N])
  #pragma omp target teams distribute parallel for \
      map(tofrom:pt[0:TILE_CAP],psl[0:SLICE_CAP],pss[0:SS_CAP],pds[0:SS_CAP],\
                 peu[0:EU_CAP],pht[0:HT_CAP])
  for (int64_t i = 0; i < N; ++i) {
    T x = Aptr[i], y = -x;                        // identical FMA work
    for (int j = 0; j < INNER; ++j) { MAD_64(x, y); }
    Aptr[i] = y;
    unsigned t = intel_get_tile_id()          % TILE_CAP;
    unsigned s = intel_get_slice_id()         % SLICE_CAP;
    unsigned u = intel_get_subslice_id()      % SS_CAP;
    unsigned d = intel_get_dual_subslice_id() % SS_CAP;
    unsigned e = intel_get_eu_id()            % EU_CAP;
    unsigned h = intel_get_hw_thread_id()     % HT_CAP;
    #pragma omp atomic
    pt[t]++;
    #pragma omp atomic
    psl[s]++;
    #pragma omp atomic
    pss[u]++;
    #pragma omp atomic
    pds[d]++;
    #pragma omp atomic
    peu[e]++;
    #pragma omp atomic
    pht[h]++;
  }
  #pragma omp target exit data map(release:Aptr[0:N])

  // ---- summary: how much of each level of the hierarchy was touched ---------
  auto used = [](const std::vector<u64> &v) {
    int n = 0; for (u64 c : v) if (c) n++; return n; };
  // Cumulative over the whole pass: N >> 4096 threads, so each thread runs many
  // work-items in turn --- this is "ever ran here", not a point-in-time snapshot.
  std::printf("\nOccupancy census (%lld work-items, cumulative over the pass):\n",
              (long long)N);
  std::printf("  tiles      %d/%d\n", used(tile),  TILE_CAP);
  std::printf("  slices     %d/%d\n", used(slice), SLICE_CAP);
  std::printf("  subslices  %d/%d\n", used(ss),    SS_CAP);
  std::printf("  dual-ss    %d/%d\n", used(dss),   SS_CAP);
  std::printf("  EUs        %d/%d  (per subslice)\n", used(eu), EU_CAP);
  std::printf("  hw-threads %d/%d  (%.0f%% of the tile ever ran a work-item)\n",
              used(ht), SLICE_CAP*SS_CAP*EU_CAP*THR,
              100.0 * used(ht) / (SLICE_CAP*SS_CAP*EU_CAP*THR));

  // ---- physical map: decode hw_thread -> (slice, subslice, EU) ---------------
  // Each EU owns 8 hw-threads. Shade its cell by how many of those 8 ever ran a
  // work-item, which directly shows the idle pockets. Rows = slice, block =
  // subslice, char = EU. ' ' = EU never used, '█' = all 8 threads used.
  int active[SLICE_CAP][SS_CAP][EU_CAP] = {};
  for (unsigned h = 0; h < HT_CAP; ++h) {
    if (!ht[h]) continue;
    unsigned s = (h / 512) % SLICE_CAP, u = (h / 64) % SS_CAP, e = (h / 8) % EU_CAP;
    active[s][u][e]++;
  }
  std::printf("\nPhysical map of tile 0  (row = slice, block = subslice, char = EU;"
              " shade = threads used /8)\n");
  std::printf("         ");                                  // 9 = width of row label
  for (unsigned u = 0; u < SS_CAP; ++u) std::printf("ss%-7u", u);  // 8-char EU block + 1 gap
  std::printf("\n");
  for (unsigned s = 0; s < SLICE_CAP; ++s) {
    std::printf("slice %u  ", s);
    for (unsigned u = 0; u < SS_CAP; ++u) {
      for (unsigned e = 0; e < EU_CAP; ++e)
        std::printf("%s", shade(active[s][u][e] / (double)THR));
      std::printf(" ");                                      // gap between subslices
    }
    std::printf("\n");
  }
}
