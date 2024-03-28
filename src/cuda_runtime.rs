use crate::Str;
pub use nvtx_sys::ffi::{cudaEvent_t, cudaStream_t};

/// Enum for all CUDA Runtime types
pub enum CudaRuntimeResource {
    /// device (integer)
    Device(i32),
    /// event
    Event(cudaEvent_t),
    /// stream
    Stream(cudaStream_t),
}

impl From<i32> for CudaRuntimeResource {
    fn from(value: i32) -> Self {
        CudaRuntimeResource::Device(value)
    }
}

impl From<cudaEvent_t> for CudaRuntimeResource {
    fn from(value: cudaEvent_t) -> Self {
        CudaRuntimeResource::Event(value)
    }
}

impl From<cudaStream_t> for CudaRuntimeResource {
    fn from(value: cudaStream_t) -> Self {
        CudaRuntimeResource::Stream(value)
    }
}

/// Name a CUDA Runtime Resource (one of: Device, Event, or Stream)
///
/// ```
/// nvtx::name_cudart_resource(nvtx::CudaRuntimeResource::Device(0), "GPU 0");
/// /// or implicitly:
/// nvtx::name_cudart_resource(0, "GPU 0");
/// ```
pub fn name_cudart_resource(resource: impl Into<CudaRuntimeResource>, name: impl Into<Str>) {
    match resource.into() {
        CudaRuntimeResource::Device(device) => match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCudaDeviceA(device, s.as_ptr()) },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxNameCudaDeviceW(device, s.as_ptr().cast())
            },
        },
        CudaRuntimeResource::Event(event) => match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCudaEventA(event, s.as_ptr()) },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxNameCudaEventW(event, s.as_ptr().cast())
            },
        },
        CudaRuntimeResource::Stream(stream) => match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCudaStreamA(stream, s.as_ptr()) },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxNameCudaStreamW(stream, s.as_ptr().cast())
            },
        },
    }
}
