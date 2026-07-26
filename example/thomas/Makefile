# THOMAS — SINGLE SOURCE, compiled TWICE. This Makefile is the lesson.
#   Tiny Heterogeneous OpenCL Macro, ATPESC + SPIR-V.
# triad.cpp is compiled once as a DEVICE kernel (-> SPIR-V) and once as the
# HOST program that embeds and launches it. thomas.hpp hides the OpenCL.
ICPX       := icpx
LLVM_SPIRV := /opt/aurora/26.181.0/oneapi/2026.1/bin/compiler/llvm-spirv
CLINC      := /opt/aurora/26.181.0/oneapi/2026.1/include
CLLIB      := /opt/aurora/26.181.0/oneapi/2026.1/lib

SRC := triad.cpp

all: triad

# --- DEVICE PASS: the kernel lambda in $(SRC) -> SPIR-V ---
kernel.bc: $(SRC) thomas.hpp
	$(ICPX) -c -emit-llvm -target spir64-unknown-unknown \
	        -x cl -cl-std=clc++ -DTHOMAS_DEVICE $(SRC) -o $@

kernel.spv: kernel.bc
	$(LLVM_SPIRV) $< -o $@

thomas_spv.h: kernel.spv
	xxd -i -n thomas_spv $< > $@

# --- HOST PASS: the C++ side of the SAME file, embedding the SPIR-V ---
triad: $(SRC) thomas.hpp thomas_spv.h
	$(ICPX) -O2 -DCL_TARGET_OPENCL_VERSION=300 -I$(CLINC) $(SRC) \
	        -o $@ -L$(CLLIB) -lOpenCL

run: triad
	ZE_AFFINITY_MASK=0.0 ./triad   # pin one PVC tile

clean:
	rm -f triad kernel.bc kernel.spv thomas_spv.h

.PHONY: all run clean
