use crate::common::GenericMessage;
use crate::TypeValueEncodable;

/// Represents a message for use within events and ranges.
///
/// * [`Message::Ascii`] is the discriminator for ASCII C strings
/// * [`Message::Unicode`] is the discriminator for Rust strings and wide C strings
pub type Message = GenericMessage<()>;

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
            Message::Registered(_) => {
                unreachable!("Registered strings are not valid in the global context")
            }
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
    use crate::common::TestUtils;
    use std::ffi::CString;
    use widestring::WideCString;

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
        TestUtils::assert_message_ascii_encoding(&m, "hello");
    }

    #[test]
    fn test_encode_unicode() {
        let s = "hello";
        let wstr = WideCString::from_str(s).unwrap();
        let m = Message::Unicode(wstr.clone());
        TestUtils::assert_message_unicode_encoding(&m, "hello");
    }
}
