use super::Identifier;
use crate::TypeValueEncodable;

/// Identifiers used for Generic resources
pub enum GenericIdentifier {
    /// Generic pointer
    Pointer(*const ::std::os::raw::c_void),
    /// Generic handle
    Handle(u64),
    /// Generic thread native
    NativeThread(u64),
    /// Generic thread posix
    PosixThread(u64),
}

impl From<GenericIdentifier> for Identifier {
    fn from(value: GenericIdentifier) -> Self {
        Self::Generic(value)
    }
}

impl TypeValueEncodable for GenericIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ResourceAttributesIdentifier;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            Self::Pointer(p) => (
                nvtx_sys::resource_type::GENERIC_POINTER,
                Self::Value { pValue: *p },
            ),
            Self::Handle(h) => (
                nvtx_sys::resource_type::GENERIC_HANDLE,
                Self::Value { ullValue: *h },
            ),
            Self::NativeThread(t) => (
                nvtx_sys::resource_type::GENERIC_THREAD_NATIVE,
                Self::Value { ullValue: *t },
            ),
            Self::PosixThread(t) => (
                nvtx_sys::resource_type::GENERIC_THREAD_POSIX,
                Self::Value { ullValue: *t },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        Identifier::default_encoding()
    }
}
