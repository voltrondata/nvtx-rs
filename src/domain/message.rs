use std::ffi::{CStr, CString};

use widestring::WideCString;

use crate::TypeValueEncodable;

use super::registered_string::RegisteredString;


/// Represents a message for use within events and ranges
#[derive(Debug, Clone)]
pub enum Message<'a> {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
    /// discriminant for a registered string belonging to a domain
    Registered(&'a RegisteredString<'a>),
}

impl<'a> From<&'a RegisteredString<'a>> for Message<'a> {
    fn from(v: &'a RegisteredString) -> Self {
        Self::Registered(v)
    }
}

impl From<String> for Message<'_> {
    fn from(v: String) -> Self {
        Self::Unicode(WideCString::from_str(v.as_str()).expect("Could not convert to wide string"))
    }
}

impl From<&str> for Message<'_> {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for Message<'_> {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for Message<'_> {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}

impl<'a> TypeValueEncodable for Message<'a> {
    type Type = nvtx_sys::ffi::nvtxMessageType_t;
    type Value = nvtx_sys::ffi::nvtxMessageValue_t;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match &self {
            Message::Ascii(s) => (
                nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII,
                Self::Value { ascii: s.as_ptr() },
            ),
            Message::Unicode(s) => (
                nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_UNICODE,
                Self::Value {
                    unicode: s.as_ptr().cast(),
                },
            ),
            Message::Registered(r) => (
                nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_REGISTERED,
                Self::Value {
                    registered: r.handle,
                },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (
            nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_UNKNOWN,
            Self::Value {
                ascii: std::ptr::null(),
            },
        )
    }
}
