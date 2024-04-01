use super::Identifier;
use crate::TypeValueEncodable;

/// Identifiers used for CUDA resources
pub enum CudaIdentifier {
    /// Device
    Device(nvtx_sys::CuDevice),
    /// Context
    Context(nvtx_sys::CuContext),
    /// Event
    Event(nvtx_sys::CuEvent),
    /// Stream
    Stream(nvtx_sys::CuStream),
}

impl From<CudaIdentifier> for Identifier {
    fn from(value: CudaIdentifier) -> Self {
        Self::Cuda(value)
    }
}

impl TypeValueEncodable for CudaIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ResourceAttributesIdentifier;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            Self::Device(id) => (
                nvtx_sys::resource_type::CUDA_DEVICE,
                Self::Value {
                    ullValue: *id as u64,
                },
            ),
            Self::Context(id) => (
                nvtx_sys::resource_type::CUDA_CONTEXT,
                Self::Value { pValue: id.cast() },
            ),
            Self::Event(id) => (
                nvtx_sys::resource_type::CUDA_EVENT,
                Self::Value { pValue: id.cast() },
            ),
            Self::Stream(id) => (
                nvtx_sys::resource_type::CUDA_STREAM,
                Self::Value { pValue: id.cast() },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        Identifier::default_encoding()
    }
}
