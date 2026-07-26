// PCIe roof — SYCL, CONCURRENT host<->device bandwidth over the bus.
// The bus is full-duplex: a H2D copy and a D2H copy can run at the same time,
// so we submit both before a single wait() and count bytes in BOTH directions.
// Host buffers are PINNED (sycl::malloc_host) --- pageable memory caps the link
// far below its roof; pinned lets DMA reach it. Expect ~76 GB/s on one PVC tile.
//
// Pass the iteration space N (elements) on the command line (default 1).
//
// build: icpx -fsycl -O3 pcie.cpp -o pcie
// run:   ZE_AFFINITY_MASK=0.0 ./pcie 67108864
#include <sycl/sycl.hpp>
#include <array>
#include <chrono>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

using T = int;

// Fill every buffer (host or device) with the same random bytes via the queue.
static void fill_randomly(sycl::queue &Q, size_t N, std::initializer_list<T *> ptrs) {
  std::vector<T> tmp(N);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<T> dist;
  for (auto &v : tmp) v = dist(rng);
  for (T *p : ptrs) Q.memcpy(p, tmp.data(), N * sizeof(T));
  Q.wait();
}

int main(int argc, char **argv) {
  const size_t N = (argc > 1) ? std::stoul(argv[1]) : 1;   // elements per buffer
  const size_t N_byte = N * sizeof(T);
  const int STEPS = 100;

  sycl::queue Q;
  std::printf("Device: %s [N=%zu]\n",
              Q.get_device().get_info<sycl::info::device::name>().c_str(), N);

  T *a_cpu = sycl::malloc_host<T>(N, Q);   // pinned host memory
  T *b_cpu = sycl::malloc_host<T>(N, Q);
  T *a_gpu = sycl::malloc_device<T>(N, Q);
  T *b_gpu = sycl::malloc_device<T>(N, Q);
  fill_randomly(Q, N, {a_cpu, b_cpu, a_gpu, b_gpu});

  // Two copies, opposite directions, issued together so the bus runs both ways:
  //   H2D: a_cpu -> a_gpu     D2H: b_gpu -> b_cpu
  std::array<std::pair<T *, T *>, 2> ptrs = {{ {a_gpu, a_cpu}, {b_cpu, b_gpu} }};

  double best = std::numeric_limits<double>::max();          // min drops jitter
  for (int s = 0; s < STEPS; ++s) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto [dest, src] : ptrs)
      Q.memcpy(dest, src, N_byte);
    Q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }

  // Both directions move N_byte each -> 2*N_byte per step.
  const double gbs = 2.0 * N_byte / best * 1e-9;
  std::printf("PCIE    N=%zu  min_ms=%.3f  %.1f GB/s (H2D+D2H concurrent)\n",
              N, best * 1e3, gbs);

  sycl::free(a_cpu, Q); sycl::free(b_cpu, Q);
  sycl::free(a_gpu, Q); sycl::free(b_gpu, Q);
}
