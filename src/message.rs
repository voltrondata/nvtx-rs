use crate::{Str, TypeValueEncodable};
use std::ffi::CString;
use widestring::WideCString;

/// Represents a message for use within events and ranges
///
/// * [`Message::Ascii`] is the discriminator for C strings
/// * [`Message::Unicode`] is the discriminator for Rust strings
#[derive(Debug, Clone)]
pub enum Message {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
}

impl<T: Into<Str>> From<T> for Message {
    fn from(value: T) -> Self {
        match value.into() {
            Str::Ascii(s) => Message::Ascii(s),
            Str::Unicode(s) => Message::Unicode(s),
        }
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
