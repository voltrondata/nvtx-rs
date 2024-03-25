use std::ffi::{CStr, CString};

use widestring::WideCString;

use crate::{EventAttributes, Message};


/// Convenience wrapper for all valid argument types
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
            attr => EventArgument::EventAttribute(attr.into()),
        }
    }
}

impl From<&str> for EventArgument {
    fn from(v: &str) -> Self {
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
