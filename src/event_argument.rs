use crate::{EventAttributes, Message, Str};
use std::ffi::{CStr, CString};
use widestring::WideCString;

/// Convenience wrapper for all valid argument types to ranges and marks
///
/// * If a Rust string, it will automatically be converted into [`EventArgument::Unicode`]
/// * If a C string, it will automatically be converted into [`EventArgument::Ascii`]
/// * If a [`Str`], it will automatically be converted to its underlying active discriminant
/// * If an [`EventArgument::EventAttribute`] only specifies a [`Message`], it will automatically be converted into the message's underlying active discriminant
#[derive(Debug, Clone)]
pub enum EventArgument {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
    /// discriminant for a detailed Attribute
    EventAttribute(EventAttributes),
}

impl From<EventAttributes> for EventArgument {
    fn from(value: EventAttributes) -> Self {
        match value {
            EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(Message::Ascii(s)),
            } => EventArgument::Ascii(s),
            EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(Message::Unicode(s)),
            } => EventArgument::Unicode(s),
            attr => EventArgument::EventAttribute(attr),
        }
    }
}

impl From<&str> for EventArgument {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<String> for EventArgument {
    fn from(v: String) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for EventArgument {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for EventArgument {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}

impl From<Str> for EventArgument {
    fn from(value: Str) -> Self {
        match value {
            Str::Ascii(s) => Self::Ascii(s),
            Str::Unicode(s) => Self::Unicode(s),
        }
    }
}
