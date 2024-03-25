use crate::TypeValueEncodable;
use std::ffi::{CStr, CString};
use widestring::WideCString;

/// Represents a message for use within events and ranges
#[derive(Debug, Clone)]
pub enum Message {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
}

impl From<String> for Message {
    fn from(v: String) -> Self {
        Self::Unicode(WideCString::from_str(v.as_str()).expect("Could not convert to wide string"))
    }
}

impl From<&str> for Message {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for Message {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for Message {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}

impl TypeValueEncodable for Message {
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
