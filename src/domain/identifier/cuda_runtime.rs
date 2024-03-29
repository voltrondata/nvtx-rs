use super::Identifier;
use crate::TypeValueEncodable;

/// Identifiers used for CUDA resources
pub enum CudaRuntimeIdentifier {
    /// device
    Device(i32),
    /// event
    Event(nvtx_sys::CudaEvent),
    /// stream
    Stream(nvtx_sys::CudaStream),
}

impl From<CudaRuntimeIdentifier> for Identifier {
    fn from(value: CudaRuntimeIdentifier) -> Self {
        Self::CudaRuntime(value)
    }
}

impl TypeValueEncodable for CudaRuntimeIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ResourceAttributesIdentifier;

    fn encode(&self) -> (Self::Type, Self::Value) {
        use nvtx_sys::resource_type::*;
        match self {
            Self::Device(id) => (
                CUDART_DEVICE,
                Self::Value {
                    ullValue: *id as u64,
                },
            ),
            Self::Event(id) => (CUDART_EVENT, Self::Value { pValue: id.cast() }),
            Self::Stream(id) => (CUDART_STREAM, Self::Value { pValue: id.cast() }),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        Identifier::default_encoding()
    }
}
