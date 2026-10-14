// CPU-only extension build: link-time definitions for the CUDA runtime entry points declared in
// cuda_runtime.h (this directory) and the two conversion helpers declared in stloader_cu.cuh.
// stloader.cpp keeps its CUDA engine compiled in, so these must exist; every CUDA entry point
// returns cudaErrorNotSupported, which the loader reports as a normal load failure. The CPU-side
// bf16 -> fp16 conversion is reimplemented here because its definition lives in stloader_cu.cu,
// which is not part of the CPU build.

#include "cuda_runtime.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

static const char* cuda_stub_error =
    "CUDA is not available in this CPU-only build of exllamav3_ext";

void inplace_bf16_to_fp16_cpu
(
    void* buffer,
    const size_t numel
)
{
    // Same math as stloader_cu.cu: bf16 -> fp32 (exact) -> fp16 round-to-nearest-even, forward
    // in place, so a destination element is never overwritten before it is read
    const c10::BFloat16* rd = static_cast<const c10::BFloat16*>(buffer);
    c10::Half* wr = static_cast<c10::Half*>(buffer);
    for (size_t i = 0; i < numel; ++i)
        wr[i] = c10::Half(static_cast<float>(rd[i]));
}

cudaError_t inplace_bf16_to_fp16_cuda
(
    void* buffer,
    const size_t numel,
    cudaStream_t stream
)
{
    (void) buffer; (void) numel; (void) stream;
    return cudaErrorNotSupported;
}

const char* cudaGetErrorString(cudaError_t error)
{
    (void) error;
    return cuda_stub_error;
}

cudaError_t cudaGetDevice(int* device) { (void) device; return cudaErrorNotSupported; }
cudaError_t cudaSetDevice(int device) { (void) device; return cudaErrorNotSupported; }
cudaError_t cudaDeviceSynchronize(void) { return cudaErrorNotSupported; }
cudaError_t cudaMallocHost(void** ptr, size_t size) { (void) ptr; (void) size; return cudaErrorNotSupported; }
cudaError_t cudaFreeHost(void* ptr) { (void) ptr; return cudaErrorNotSupported; }

cudaError_t cudaMemcpyAsync
(
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream
)
{
    (void) dst; (void) src; (void) count; (void) kind; (void) stream;
    return cudaErrorNotSupported;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
{
    (void) pStream; (void) flags;
    return cudaErrorNotSupported;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void) stream; return cudaErrorNotSupported; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* pEvent, unsigned int flags)
{
    (void) pEvent; (void) flags;
    return cudaErrorNotSupported;
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) { (void) event; (void) stream; return cudaErrorNotSupported; }
cudaError_t cudaEventSynchronize(cudaEvent_t event) { (void) event; return cudaErrorNotSupported; }
cudaError_t cudaEventDestroy(cudaEvent_t event) { (void) event; return cudaErrorNotSupported; }
