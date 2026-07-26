# Characterisation micro-benchmarks: the three roofs.
#   triad -> HBM bandwidth roof      (memory-bound, no compute)
#   flops -> compute roof            (register FMA chain, no memory)
#   pcie  -> host<->device bus roof  (transfer only)
#   peak_sol -> BW + FMA at once (SYCL) (the roofline ridge)
# Each takes the iteration space N as argv[1] (default 1).
ICPX     := icpx
OMPFLAGS := -fiopenmp -fopenmp-targets=spir64 -O3
SYCLFLAGS := -fsycl -O3 -std=c++23

OMP_BINS := triad flops flops_with_probe_sol

all: $(OMP_BINS) pcie peak_sol

$(OMP_BINS): %: %.cpp
	$(ICPX) $(OMPFLAGS) $< -o $@

pcie: pcie.cpp
	$(ICPX) $(SYCLFLAGS) $< -o $@

peak_sol: peak_sol.cpp mdspan.hpp
	$(ICPX) $(SYCLFLAGS) $< -o $@

mdspan.hpp:
	ln -sf ../mdspan.hpp .

clean:
	rm -f $(OMP_BINS) pcie peak_sol

.PHONY: all clean
