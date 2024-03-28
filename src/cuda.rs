use crate::Str;
pub use nvtx_sys::ffi::{CUcontext, CUdevice, CUevent, CUstream};

/// Enum for all CUDA types
pub enum CudaResource {
    /// CUdevice
    Device(CUdevice),
    /// CUcontext
    Context(CUcontext),
    /// CUevent
    Event(CUevent),
    /// CUstream
    Stream(CUstream),
}

impl From<CUdevice> for CudaResource {
    fn from(value: CUdevice) -> Self {
        CudaResource::Device(value)
    }
}

impl From<CUcontext> for CudaResource {
    fn from(value: CUcontext) -> Self {
        CudaResource::Context(value)
    }
}

impl From<CUevent> for CudaResource {
    fn from(value: CUevent) -> Self {
        CudaResource::Event(value)
    }
}

impl From<CUstream> for CudaResource {
    fn from(value: CUstream) -> Self {
        CudaResource::Stream(value)
    }
}

/// Name a CUDA Resource (one of: Device, Context, Event, or Stream)
///
/// ```
/// nvtx::name_cuda_resource(nvtx::CudaResource::Device(0), "GPU 0");
/// /// or implicitly:
/// nvtx::name_cuda_resource(0, "GPU 0");
/// ```
pub fn name_cuda_resource(resource: impl Into<CudaResource>, name: impl Into<Str>) {
    match resource.into() {
        CudaResource::Context(context) => match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCuContextA(context, s.as_ptr()) },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxNameCuContextW(context, s.as_ptr().cast())
            },
        },
        CudaResource::Device(device) => match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCuDeviceA(device, s.as_ptr()) },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxNameCuDeviceW(device, s.as_ptr().cast())
            },
        },
        CudaResource::Event(event) => match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCuEventA(event, s.as_ptr()) },
            Str::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxNameCuEventW(event, s.as_ptr().cast()) },
        },
        CudaResource::Stream(stream) => match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCuStreamA(stream, s.as_ptr()) },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxNameCuStreamW(stream, s.as_ptr().cast())
            },
        },
    }
}
