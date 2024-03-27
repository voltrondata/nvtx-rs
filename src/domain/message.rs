use super::RegisteredString;
use crate::{Str, TypeValueEncodable};
use std::ffi::CString;
use widestring::WideCString;

/// Represents a message for use within events and ranges
///
/// * [`Message::Ascii`] is the discriminator for ASCII C strings
/// * [`Message::Unicode`] is the discriminator for Rust strings and wide C strings
/// * [`Message::Registered`] is the discriminator for nvtx domain-registered strings
#[derive(Debug, Clone)]
pub enum Message<'a> {
    /// discriminator for an owned ASCII string
    Ascii(CString),
    /// discriminator for an owned Unicode string
    Unicode(WideCString),
    /// discriminator for a registered string belonging to a domain
    Registered(&'a RegisteredString<'a>),
}

impl<'a> From<&'a RegisteredString<'a>> for Message<'a> {
    fn from(v: &'a RegisteredString) -> Self {
        Self::Registered(v)
    }
}

impl<'a, T: Into<Str>> From<T> for Message<'a> {
    fn from(value: T) -> Self {
        match value.into() {
            Str::Ascii(s) => Message::Ascii(s),
            Str::Unicode(s) => Message::Unicode(s),
        }
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
