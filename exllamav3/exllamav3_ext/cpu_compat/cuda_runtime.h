#pragma once

// CPU-only extension build (cpu_extension_sources in util/cuda_flags.py): stand-in for the CUDA
// toolkit's cuda_runtime.h. stloader.cpp keeps its CUDA load engine compiled in the CPU build so
// the file builds unchanged next to the GPU builds; only the declarations below are needed for
// that, and the definitions (cpu_compat/cuda_stub.cpp) fail cleanly if a CUDA path is ever taken
// on a CPU-only host. This directory precedes torch's own include paths (extra_include_paths are
// -I, torch's are -isystem), so it also shadows the header for the c10/ATen CUDA headers included
// next to it.

#include <cstddef>

typedef enum cudaError
{
    cudaSuccess = 0,
    cudaErrorNotSupported = 801,
}
cudaError_t;

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
};

enum
{
    cudaStreamDefault = 0,
    cudaStreamNonBlocking = 1,
    cudaEventDefault = 0,
    cudaEventBlockingSync = 1,
    cudaEventDisableTiming = 2,
};

const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMemcpyAsync
(
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream
);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* pEvent, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
