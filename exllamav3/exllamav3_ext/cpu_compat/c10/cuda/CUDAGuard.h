#pragma once

// CPU-only extension build: shadow of torch's c10/cuda/CUDAGuard.h (see cuda_runtime.h in this
// directory). stloader.cpp includes it for the GPU builds but references nothing from it; the
// real header needs the CUDA toolkit, which a CPU-only host does not have.
