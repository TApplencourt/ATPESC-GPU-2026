#pragma once
#include <cassert>
#include <chrono>
#include <climits>
#include <cstddef>
#include <functional>
#include <vector>

// Compare two std::vectors that may use DIFFERENT allocators (e.g. a plain
// reference vector vs. a sycl::usm_allocator result vector). The standard
// operator== only matches identical allocator types, so `z == z_ref` would not
// compile across a USM and a default allocator; this element-wise version lets
// the experiments keep the terse `assert(z == z_ref)`.
template <class T, class A1, class A2>
inline bool operator==(const std::vector<T, A1> &a, const std::vector<T, A2> &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return false;
  return true;
}

// Minimum-time timing.
//
// bench_min_ns(steps, f) runs `f` `steps` times and returns the MINIMUM time of
// a single call, in ns.
//
// `f` is one unit of work (one kernel launch). The step loop lives HERE, not in
// the caller -- the caller never writes its own `for (s < STEPS)`. `f` must be
// blocking (an OpenMP `target` region already is; a SYCL submit needs .wait()),
// otherwise the timer stops before the GPU finishes.
//
// Why the minimum: for a fixed workload the true cost is a floor; noise (OS
// scheduling, DVFS, contention) only makes a run slower. Why no warmup: the cold
// first call (JIT, first touch) is never the fastest, so the min drops it.

static inline unsigned long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

inline unsigned long bench_min_ns(int steps, const std::function<void()> &f) {
  unsigned long min_time = ULONG_MAX;
  for (int s = 0; s < steps; ++s) {
    const unsigned long start = now_ns();
    f();
    const unsigned long time = now_ns() - start;
    if (time < min_time)
      min_time = time;
  }
  return min_time;
}
