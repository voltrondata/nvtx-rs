use crate::{Str, TypeValueEncodable};
use std::ffi::{CStr, CString};
use widestring::{WideCStr, WideCString};

/// Generic message type that can be used in both global and domain contexts
#[derive(Debug, Clone)]
pub enum GenericMessage<T = ()> {
    /// An owned ASCII string.
    Ascii(CString),
    /// An owned Unicode string.
    Unicode(WideCString),
    /// A registered string handle (only available in domain context).
    Registered(T),
}

impl<T> GenericMessage<T> {
    /// Create an ASCII message
    pub fn ascii(s: CString) -> Self {
        Self::Ascii(s)
    }

    /// Create a Unicode message
    pub fn unicode(s: WideCString) -> Self {
        Self::Unicode(s)
    }

    /// Create a registered message (domain context only)
    pub fn registered(r: T) -> Self {
        Self::Registered(r)
    }
}

impl<T> From<Str> for GenericMessage<T> {
    fn from(value: Str) -> Self {
        match value {
            Str::Ascii(s) => Self::Ascii(s),
            Str::Unicode(s) => Self::Unicode(s),
        }
    }
}

impl<T> From<String> for GenericMessage<T> {
    fn from(value: String) -> Self {
        GenericMessage::from(Str::from(value))
    }
}

impl<T> From<&str> for GenericMessage<T> {
    fn from(value: &str) -> Self {
        GenericMessage::from(Str::from(value))
    }
}

impl<T> From<CString> for GenericMessage<T> {
    fn from(value: CString) -> Self {
        Self::Ascii(value)
    }
}

impl<T> From<WideCString> for GenericMessage<T> {
    fn from(value: WideCString) -> Self {
        Self::Unicode(value)
    }
}

impl<T> From<&CStr> for GenericMessage<T> {
    fn from(value: &CStr) -> Self {
        Self::Ascii(value.to_owned())
    }
}

impl<T> From<&WideCStr> for GenericMessage<T> {
    fn from(value: &WideCStr) -> Self {
        Self::Unicode(value.to_owned())
    }
}

impl<T> TypeValueEncodable for GenericMessage<T>
where
    T: TypeValueEncodable<Type = nvtx_sys::MessageType, Value = nvtx_sys::MessageValue>,
{
    type Type = nvtx_sys::MessageType;
    type Value = nvtx_sys::MessageValue;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            GenericMessage::Ascii(s) => (
                Self::Type::NVTX_MESSAGE_TYPE_ASCII,
                Self::Value { ascii: s.as_ptr() },
            ),
            GenericMessage::Unicode(s) => (
                Self::Type::NVTX_MESSAGE_TYPE_UNICODE,
                Self::Value {
                    unicode: s.as_ptr().cast(),
                },
            ),
            GenericMessage::Registered(r) => r.encode(),
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
