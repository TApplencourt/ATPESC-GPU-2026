# GPU programming-model demos for Aurora (Intel PVC).
#   make        build everything  (on a login node)
#   make run    run everything    (on a compute node)
# No cd needed: use `make -C /path/to/example [run]`.
#
# icpx/ifx are already in PATH. The Kokkos module sets the include/lib paths
# (CPATH, LIBRARY_PATH, LD_LIBRARY_PATH), so Kokkos needs no -I/-L/-rpath.

SHELL := /bin/bash
KOKKOS := module load kokkos/5.0.1-sycl   # sets Kokkos include + lib paths
RUN    := ZE_AFFINITY_MASK=0.0            # pin one PVC tile

BINS := characterisation/flops characterisation/peak_sol \
        triad_galery/sycl_triad triad_galery/stdpar_triad triad_galery/omp_triad \
        triad_galery/kokkos_triad triad_galery/dc_triad \
        bench_to_fix/exp1 bench_to_fix/exp1_sol bench_to_fix/exp2 bench_to_fix/exp2_sol

.PHONY: all run clean
all: $(BINS)

characterisation/flops:      characterisation/flops.cpp
	icpx -fiopenmp -fopenmp-targets=spir64 -O3 $< -o $@
characterisation/peak_sol:   characterisation/peak_sol.cpp mdspan.hpp
	icpx -fsycl -O3 -std=c++23 -I. $< -o $@   # C++23: mdspan a[i,j] subscript

triad_galery/sycl_triad:     triad_galery/sycl_triad.cpp
	icpx -fsycl -O3 $< -o $@
triad_galery/stdpar_triad:   triad_galery/stdpar_triad.cpp
	icpx -fsycl -fsycl-pstl-offload=gpu -O3 -std=c++23 $< -o $@
triad_galery/omp_triad:      triad_galery/omp_triad.cpp
	icpx -fiopenmp -fopenmp-targets=spir64 -O3 $< -o $@
triad_galery/kokkos_triad:   triad_galery/kokkos_triad.cpp
	$(KOKKOS); icpx -fsycl -fiopenmp -O2 -std=c++23 $< -lkokkoscore -lkokkoscontainers -o $@
triad_galery/dc_triad:       triad_galery/dc_triad.f90
	ifx -fiopenmp -fopenmp-targets=spir64 -fopenmp-target-do-concurrent $< -o $@

# exp1/exp2 = two models in one binary (SYCL+OpenMP / SYCL+Kokkos)
bench_to_fix/exp1 bench_to_fix/exp1_sol: bench_to_fix/%: bench_to_fix/%.cpp bench_to_fix/bench.hpp
	icpx -fsycl -fiopenmp -fopenmp-targets=spir64 -O2 -std=c++23 -Ibench_to_fix $< -o $@
bench_to_fix/exp2 bench_to_fix/exp2_sol: bench_to_fix/%: bench_to_fix/%.cpp bench_to_fix/bench.hpp
	$(KOKKOS); icpx -fsycl -fiopenmp -O2 -std=c++23 -Ibench_to_fix $< -lkokkoscore -lkokkoscontainers -o $@

run: all
	$(RUN) ./characterisation/flops
	$(RUN) ./characterisation/peak_sol
	$(RUN) ./triad_galery/sycl_triad
	$(RUN) ./triad_galery/stdpar_triad
	$(RUN) ./triad_galery/omp_triad
	$(KOKKOS); $(RUN) ./triad_galery/kokkos_triad
	$(RUN) ./triad_galery/dc_triad
	module load frameworks/2025.3.1; ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE $(RUN) python triad_galery/py_triad.py
	$(RUN) ./bench_to_fix/exp1
	$(RUN) ./bench_to_fix/exp1_sol
	$(KOKKOS); $(RUN) ./bench_to_fix/exp2
	$(KOKKOS); $(RUN) ./bench_to_fix/exp2_sol

clean:
	rm -f $(BINS)
