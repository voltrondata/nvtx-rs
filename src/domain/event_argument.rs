use std::ffi::{CStr, CString};

use widestring::WideCString;

use super::{event_attributes::EventAttributes, message::Message};


/// Convenience wrapper for all valid argument types
#[derive(Debug, Clone)]
pub enum EventArgument<'a> {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
    /// discriminant for a detailed Attribute
    EventAttribute(EventAttributes<'a>),
}

impl<'a> From<EventAttributes<'a>> for EventArgument<'a> {
    fn from(value: EventAttributes<'a>) -> Self {
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

impl From<&str> for EventArgument<'_> {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for EventArgument<'_> {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for EventArgument<'_> {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}
