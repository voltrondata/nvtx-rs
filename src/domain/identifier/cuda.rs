use super::Identifier;
use crate::TypeValueEncodable;

/// Identifiers used for CUDA resources
pub enum CudaIdentifier {
    /// device
    Device(nvtx_sys::CuDevice),
    /// context
    Context(nvtx_sys::CuContext),
    /// event
    Event(nvtx_sys::CuEvent),
    /// stream
    Stream(nvtx_sys::CuStream),
}

impl From<CudaIdentifier> for Identifier {
    fn from(value: CudaIdentifier) -> Self {
        Self::Cuda(value)
    }
}

impl TypeValueEncodable for CudaIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ResourceAttributesId;

    fn encode(&self) -> (Self::Type, Self::Value) {
        use nvtx_sys::resource_type::*;
        match self {
            Self::Device(id) => (
                NVTX_RESOURCE_TYPE_CUDA_DEVICE,
                Self::Value {
                    ullValue: *id as u64,
                },
            ),
            Self::Context(id) => (
                NVTX_RESOURCE_TYPE_CUDA_CONTEXT,
                Self::Value { pValue: id.cast() },
            ),
            Self::Event(id) => (
                NVTX_RESOURCE_TYPE_CUDA_EVENT,
                Self::Value { pValue: id.cast() },
            ),
            Self::Stream(id) => (
                NVTX_RESOURCE_TYPE_CUDA_STREAM,
                Self::Value { pValue: id.cast() },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        Identifier::default_encoding()
    }
}
