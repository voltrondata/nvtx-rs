use super::Identifier;
use crate::{
    native_types::{CUcontext, CUdevice, CUevent, CUstream},
    TypeValueEncodable,
};

/// Identifiers used for CUDA resources
pub enum CudaIdentifier {
    /// device
    Device(CUdevice),
    /// context
    Context(CUcontext),
    /// event
    Event(CUevent),
    /// stream
    Stream(CUstream),
}

impl From<CudaIdentifier> for Identifier {
    fn from(value: CudaIdentifier) -> Self {
        Identifier::Cuda(value)
    }
}

impl TypeValueEncodable for CudaIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ffi::nvtxResourceAttributes_v0_identifier_t;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            CudaIdentifier::Device(id) => (
                nvtx_sys::ffi::nvtxResourceCUDAType_t::NVTX_RESOURCE_TYPE_CUDA_DEVICE as u32,
                Self::Value {
                    ullValue: *id as u64,
                },
            ),
            CudaIdentifier::Context(id) => (
                nvtx_sys::ffi::nvtxResourceCUDAType_t::NVTX_RESOURCE_TYPE_CUDA_CONTEXT as u32,
                Self::Value { pValue: id.cast() },
            ),
            CudaIdentifier::Event(id) => (
                nvtx_sys::ffi::nvtxResourceCUDAType_t::NVTX_RESOURCE_TYPE_CUDA_EVENT as u32,
                Self::Value { pValue: id.cast() },
            ),
            CudaIdentifier::Stream(id) => (
                nvtx_sys::ffi::nvtxResourceCUDAType_t::NVTX_RESOURCE_TYPE_CUDA_STREAM as u32,
                Self::Value { pValue: id.cast() },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (
            nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_UNKNOWN as u32,
            Self::Value { ullValue: 0 },
        )
    }
}
