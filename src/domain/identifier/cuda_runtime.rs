use super::Identifier;
use crate::TypeValueEncodable;

/// Identifiers used for CUDA resources
pub enum CudaRuntimeIdentifier {
    /// device
    Device(i32),
    /// event
    Event(crate::cudaEvent_t),
    /// stream
    Stream(crate::cudaStream_t),
}

impl From<CudaRuntimeIdentifier> for Identifier {
    fn from(value: CudaRuntimeIdentifier) -> Self {
        Identifier::CudaRuntime(value)
    }
}

impl TypeValueEncodable for CudaRuntimeIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ffi::nvtxResourceAttributes_v0_identifier_t;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            CudaRuntimeIdentifier::Device(id) => (
                nvtx_sys::ffi::nvtxResourceCUDARTType_t::NVTX_RESOURCE_TYPE_CUDART_DEVICE as u32,
                Self::Value {
                    ullValue: *id as u64,
                },
            ),
            CudaRuntimeIdentifier::Event(id) => (
                nvtx_sys::ffi::nvtxResourceCUDARTType_t::NVTX_RESOURCE_TYPE_CUDART_EVENT as u32,
                Self::Value { pValue: id.cast() },
            ),
            CudaRuntimeIdentifier::Stream(id) => (
                nvtx_sys::ffi::nvtxResourceCUDARTType_t::NVTX_RESOURCE_TYPE_CUDART_STREAM as u32,
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
