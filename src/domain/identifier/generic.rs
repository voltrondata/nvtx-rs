use crate::TypeValueEncodable;

use super::Identifier;

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
        Identifier::Generic(value)
    }
}

impl TypeValueEncodable for GenericIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ffi::nvtxResourceAttributes_v0_identifier_t;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            GenericIdentifier::Pointer(p) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_POINTER as u32,
                Self::Value { pValue: *p },
            ),
            GenericIdentifier::Handle(h) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_HANDLE as u32,
                Self::Value { ullValue: *h },
            ),
            GenericIdentifier::NativeThread(t) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE
                    as u32,
                Self::Value { ullValue: *t },
            ),
            GenericIdentifier::PosixThread(t) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX
                    as u32,
                Self::Value { ullValue: *t },
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
