use super::RegisteredString;
use crate::{Str, TypeValueEncodable};
use std::ffi::CString;
use widestring::WideCString;

/// Represents a message for use within events and ranges
///
/// * [`Message::Ascii`] is the discriminator for ASCII C strings
/// * [`Message::Unicode`] is the discriminator for Rust strings and wide C strings
/// * [`Message::Registered`] is the discriminator for NVTX domain-registered strings
#[derive(Debug, Clone)]
pub enum Message<'a> {
    /// An owned ASCII string.
    Ascii(CString),
    /// An owned Unicode string.
    Unicode(WideCString),
    /// A registered string handle belonging to a domain.
    Registered(RegisteredString<'a>),
}

impl<'a> From<RegisteredString<'a>> for Message<'a> {
    fn from(v: RegisteredString<'a>) -> Self {
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
    type Type = nvtx_sys::MessageType;
    type Value = nvtx_sys::MessageValue;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match &self {
            Message::Ascii(s) => (
                Self::Type::NVTX_MESSAGE_TYPE_ASCII,
                Self::Value { ascii: s.as_ptr() },
            ),
            Message::Unicode(s) => (
                Self::Type::NVTX_MESSAGE_TYPE_UNICODE,
                Self::Value {
                    unicode: s.as_ptr().cast(),
                },
            ),
            Message::Registered(r) => (
                Self::Type::NVTX_MESSAGE_TYPE_REGISTERED,
                Self::Value {
                    registered: r.handle.into(),
                },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (
            Self::Type::NVTX_MESSAGE_UNKNOWN,
            Self::Value {
                ascii: std::ptr::null(),
            },
        )
    }
}
