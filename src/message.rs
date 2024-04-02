use crate::{Str, TypeValueEncodable};
use std::ffi::CString;
use widestring::WideCString;

/// Represents a message for use within events and ranges.
///
/// * [`Message::Ascii`] is the discriminator for ASCII C strings
/// * [`Message::Unicode`] is the discriminator for Rust strings and wide C strings
#[derive(Debug, Clone)]
pub enum Message {
    /// An owned ASCII string.
    Ascii(CString),
    /// An owned Unicode string.
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
    type Type = nvtx_sys::MessageType;
    type Value = nvtx_sys::MessageValue;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
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

#[cfg(test)]
mod tests {
    use super::Message;
    use crate::TypeValueEncodable;
    use std::ffi::{CStr, CString};
    use widestring::{WideCStr, WideCString};

    #[test]
    fn test_message_ascii() {
        let cstr = CString::new("hello").unwrap();
        let m = Message::Ascii(cstr.clone());
        assert!(matches!(m, Message::Ascii(s) if s == cstr));
    }

    #[test]
    fn test_message_unicode() {
        let s = "hello";
        let wstr = WideCString::from_str(s).unwrap();
        let m = Message::Unicode(wstr.clone());
        assert!(matches!(m, Message::Unicode(s) if s == wstr));
    }

    #[test]
    fn test_encode_ascii() {
        let cstr = CString::new("hello").unwrap();
        let m = Message::Ascii(cstr.clone());
        let (t, v) = m.encode();
        assert_eq!(t, nvtx_sys::MessageType::NVTX_MESSAGE_TYPE_ASCII);
        unsafe {
            assert!(
                matches!(v, nvtx_sys::MessageValue{ ascii: p } if CStr::from_ptr(p) == cstr.as_c_str())
            );
        }
    }

    #[test]
    fn test_encode_unicode() {
        let s = "hello";
        let wstr = WideCString::from_str(s).unwrap();
        let m = Message::Unicode(wstr.clone());
        let (t, v) = m.encode();
        assert_eq!(t, nvtx_sys::MessageType::NVTX_MESSAGE_TYPE_UNICODE);
        unsafe {
            assert!(
                matches!(v, nvtx_sys::MessageValue{ unicode: p } if WideCStr::from_ptr_str(p.cast()) == wstr)
            )
        };
    }
}
