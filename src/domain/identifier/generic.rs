use super::Identifier;
use crate::TypeValueEncodable;

/// Identifiers used for Generic resources
pub enum GenericIdentifier {
    /// generic pointer
    Pointer(*const ::std::os::raw::c_void),
    /// generic handle
    Handle(u64),
    /// generic thread native
    NativeThread(u64),
    /// generic thread posix
    PosixThread(u64),
}

impl From<GenericIdentifier> for Identifier {
    fn from(value: GenericIdentifier) -> Self {
        Self::Generic(value)
    }
}

impl TypeValueEncodable for GenericIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ResourceAttributesId;

    fn encode(&self) -> (Self::Type, Self::Value) {
        use nvtx_sys::resource_type::*;
        match self {
            Self::Pointer(p) => (
                NVTX_RESOURCE_TYPE_GENERIC_POINTER,
                Self::Value { pValue: *p },
            ),
            Self::Handle(h) => (
                NVTX_RESOURCE_TYPE_GENERIC_HANDLE,
                Self::Value { ullValue: *h },
            ),
            Self::NativeThread(t) => (
                NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE,
                Self::Value { ullValue: *t },
            ),
            Self::PosixThread(t) => (
                NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX,
                Self::Value { ullValue: *t },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        Identifier::default_encoding()
    }
}
