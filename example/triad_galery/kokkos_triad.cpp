// Kokkos: triad z = x + a*y   —   SYCL backend, Intel PVC
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <limits>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Kokkos on: %s\n", Kokkos::DefaultExecutionSpace::name());

    const size_t N = 1 << 20;
    const int STEPS = 1000;
    const float a = 2.0f;

    Kokkos::View<float*> x("x", N), y("y", N), z("z", N);
    Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(const size_t i) {
      x(i) = 2.0f; y(i) = 1.0f; z(i) = 0.0f;
    });

    double best_ms = std::numeric_limits<double>::max();
    for (int s = 0; s < STEPS; ++s) {
      auto t0 = std::chrono::high_resolution_clock::now();
      Kokkos::parallel_for("triad", N, KOKKOS_LAMBDA(const size_t i) {
        z(i) = x(i) + a * y(i);
      });
      Kokkos::fence();
      auto t1 = std::chrono::high_resolution_clock::now();
      best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Reduction: residual error sum (z-4)^2.
    double err = 0.0;
    Kokkos::parallel_reduce("residual", N, KOKKOS_LAMBDA(const size_t i, double& acc) {
      const double d = z(i) - 4.0;
      acc += d * d;
    }, err);

    std::printf("Kokkos    N=%zu  min_ms=%.3f  residual=%g\n", N, best_ms, err);
  }
  Kokkos::finalize();
}
