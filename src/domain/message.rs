use super::RegisteredString;
use crate::common::GenericMessage;
use crate::TypeValueEncodable;

/// Represents a message for use within events and ranges
///
/// * [`Message::Ascii`] is the discriminator for ASCII C strings
/// * [`Message::Unicode`] is the discriminator for Rust strings and wide C strings
/// * [`Message::Registered`] is the discriminator for NVTX domain-registered strings
pub type Message<'a> = GenericMessage<RegisteredString<'a>>;

impl<'a> From<RegisteredString<'a>> for Message<'a> {
    fn from(v: RegisteredString<'a>) -> Self {
        Self::Registered(v)
    }
}

impl TypeValueEncodable for Message<'_> {
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
            Message::Registered(r) => (
                Self::Type::NVTX_MESSAGE_TYPE_REGISTERED,
                Self::Value {
                    registered: r.handle().into(),
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
    use crate::{common::TestUtils, TypeValueEncodable};
    use std::ffi::CString;
    use widestring::WideCString;

    use super::{super::Domain, Message};

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
    fn test_message_registered() {
        let d = Domain::new("d");
        let reg = d.register_string("test");
        let m = Message::Registered(reg);
        assert!(matches!(m, Message::Registered(s) if s == reg));
    }

    #[test]
    fn test_encode_ascii() {
        let cstr = CString::new("hello").unwrap();
        let m = Message::Ascii(cstr.clone());
        TestUtils::assert_message_ascii_encoding(&m, "hello");
    }

    #[test]
    fn test_encode_unicode() {
        let s = "hello";
        let wstr = WideCString::from_str(s).unwrap();
        let m = Message::Unicode(wstr.clone());
        TestUtils::assert_message_unicode_encoding(&m, "hello");
    }

    #[test]
    fn test_encode_registered() {
        let d = Domain::new("d");
        let reg = d.register_string("test");
        let m = Message::Registered(reg);
        let (t, v) = m.encode();
        assert_eq!(t, nvtx_sys::MessageType::NVTX_MESSAGE_TYPE_REGISTERED);
        unsafe {
            assert!(
                matches!(v, nvtx_sys::MessageValue{ registered: r } if r == nvtx_sys::ffi::nvtxStringHandle_t::from(reg.handle()))
            );
        }
    }
}
