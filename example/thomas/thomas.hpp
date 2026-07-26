// thomas.hpp — THOMAS: Tiny Heterogeneous OpenCL Macro, ATPESC + SPIR-V.
// A single-source "mini-Kokkos": write a kernel ONCE as a C++ lambda, run it on
// the GPU. No <sycl/sycl.hpp>, no -fsycl. We hand-roll the one thing -fsycl does
// for us: get the SAME lambda body into two compilations.
//
// The file that uses this header is compiled TWICE:
//   DEVICE pass:  icpx -x cl -cl-std=clc++ -DTHOMAS_DEVICE ...  (lambda -> SPIR-V)
//   HOST   pass:  icpx (C++) ...                                (drives OpenCL)
//
// KERNEL(name, params, body) is the shim. One invocation, two expansions:
//   * device: a __kernel that runs `body` as a lambda over get_global_id(0)
//   * host:   a thomas::Kernel object that knows the kernel's name so
//             thomas::parallel_for(N, name, args...) can bind args and launch.
#pragma once

#ifdef THOMAS_DEVICE
// ===================== DEVICE VIEW (compiled as C++ for OpenCL) =====================
// gptr<T> is a __global pointer. OpenCL kernel pointer args must name their
// address space; gptr hides that keyword so the SAME signature also parses on
// the host, where __global isn't a C++ keyword (see the #else branch).
template <class T> using gptr = __global T *;

#define KERNEL(name, params, body)                                             \
  __kernel void name params {                                                  \
    auto _k = [&](unsigned long i) body;                                       \
    _k(get_global_id(0));                                                      \
  }

#else
// =============================== HOST VIEW (C++) ===============================
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cstdio>
#include <cstdlib>

// On the host, the kernel body/params are irrelevant — we only need the name to
// fish the compiled kernel out of the embedded SPIR-V. So KERNEL discards them.
// `__global` isn't a C++ keyword, so make it vanish; gptr<T> is a plain pointer.
#define __global
template <class T> using gptr = T *;
#define KERNEL(name, params, body) ::thomas::Kernel name{#name};

namespace thomas {

inline void die(const char *what, cl_int e) {
  std::fprintf(stderr, "%s -> OpenCL error %d\n", what, e);
  std::exit(1);
}
#define THOMAS_CK(x)                                                             \
  do { cl_int _e = (x); if (_e) ::thomas::die(#x, _e); } while (0)

// The SPIR-V produced by the device pass, embedded as a C array (xxd -i).
// Provides: unsigned char thomas_spv[]; unsigned int thomas_spv_len;
#include "thomas_spv.h"

// ---- one global runtime context: device, queue, program, + USM function ptrs ----
struct Runtime {
  cl_device_id dev;
  cl_context ctx;
  cl_command_queue q;
  cl_program prog;
  // USM extension entry points (loaded at runtime for this platform).
  void *(*Shared)(cl_context, cl_device_id, const cl_ulong *, size_t, cl_uint, cl_int *);
  cl_int (*Free)(cl_context, void *);
  cl_int (*SetArgPtr)(cl_kernel, cl_uint, const void *);
};

inline Runtime &rt() {
  static Runtime r = [] {
    Runtime r{};
    // Device selection: THOMAS_DEV=gpu|cpu (default gpu, fall back to cpu).
    const char *want = std::getenv("THOMAS_DEV");
    cl_device_type type = (want && want[0] == 'c') ? CL_DEVICE_TYPE_CPU
                                                   : CL_DEVICE_TYPE_GPU;
    cl_uint np = 0;
    clGetPlatformIDs(0, nullptr, &np);
    cl_platform_id plats[16];
    clGetPlatformIDs(np, plats, nullptr);
    cl_platform_id chosen = nullptr;
    for (cl_uint p = 0; p < np && !r.dev; ++p) {
      cl_int e = clGetDeviceIDs(plats[p], type, 1, &r.dev, nullptr);
      if (e == CL_SUCCESS) chosen = plats[p];
    }
    if (!r.dev) { // fall back to any device
      for (cl_uint p = 0; p < np && !r.dev; ++p)
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_DEFAULT, 1, &r.dev,
                           nullptr) == CL_SUCCESS)
          chosen = plats[p];
    }
    if (!r.dev) die("no OpenCL device", -1);

    cl_int e;
    r.ctx = clCreateContext(nullptr, 1, &r.dev, nullptr, nullptr, &e);
    THOMAS_CK(e);
    r.q = clCreateCommandQueueWithProperties(r.ctx, r.dev, nullptr, &e);
    THOMAS_CK(e);

    // THE key step: load our SPIR-V and JIT it for this device.
    r.prog = clCreateProgramWithIL(r.ctx, thomas_spv, thomas_spv_len, &e);
    THOMAS_CK(e);
    THOMAS_CK(clBuildProgram(r.prog, 1, &r.dev, nullptr, nullptr, nullptr));

    // Load the USM extension entry points for this platform.
    auto load = [&](const char *n) {
      return clGetExtensionFunctionAddressForPlatform(chosen, n);
    };
    *(void **)&r.Shared = load("clSharedMemAllocINTEL");
    *(void **)&r.Free = load("clMemFreeINTEL");
    *(void **)&r.SetArgPtr = load("clSetKernelArgMemPointerINTEL");
    if (!r.Shared || !r.SetArgPtr) die("USM extension unavailable", -1);
    return r;
  }();
  return r;
}

inline void init() { rt(); } // force device selection / build now

// A "View": USM shared memory — one pointer usable on host AND device.
template <class T> T *alloc(size_t n) {
  cl_int e;
  T *p = (T *)rt().Shared(rt().ctx, rt().dev, nullptr, n * sizeof(T), 0, &e);
  THOMAS_CK(e);
  return p;
}
template <class T> void free(T *p) { rt().Free(rt().ctx, p); }

// Host-side handle produced by the KERNEL macro. Lazily grabs the cl_kernel.
struct Kernel {
  const char *name;
  cl_kernel k = nullptr;
  cl_kernel get() {
    if (!k) { cl_int e; k = clCreateKernel(rt().prog, name, &e); THOMAS_CK(e); }
    return k;
  }
};

// Bind one argument: a USM pointer goes through the INTEL entry point, a scalar
// (float/int/...) goes through the plain clSetKernelArg with its byte size.
template <class T> void set_arg(cl_kernel k, cl_uint i, T *ptr) {
  THOMAS_CK(rt().SetArgPtr(k, i, ptr));
}
template <class T> void set_arg(cl_kernel k, cl_uint i, T scalar) {
  THOMAS_CK(clSetKernelArg(k, i, sizeof(T), &scalar));
}

// parallel_for: bind args positionally (scalars + USM pointers), launch N items.
template <class... Args>
void parallel_for(size_t n, Kernel &kern, Args... args) {
  cl_kernel k = kern.get();
  cl_uint idx = 0;
  (set_arg(k, idx++, args), ...); // fold over the argument pack
  size_t global = n;
  THOMAS_CK(clEnqueueNDRangeKernel(rt().q, k, 1, nullptr, &global, nullptr, 0,
                                 nullptr, nullptr));
  THOMAS_CK(clFinish(rt().q));
}

} // namespace thomas
#endif
