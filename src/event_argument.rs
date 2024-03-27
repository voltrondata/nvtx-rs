use crate::EventAttributes;
use std::ffi::CString;
use widestring::WideCString;

/// Convenience wrapper for all valid argument types to ranges and marks
///
/// * Any string type will be translated to [`EventArgument::Ascii`] or [`EventArgument::Unicode`] depending on its type.
/// * If [`EventArgument::EventAttribute`] is the active discriminator and its held [`EventAttributes`] only specifies a message, it will automatically be converted into the message's underlying active discriminant. Otherwise, the existing [`EventAttributes`] will be used for the event.
#[derive(Debug, Clone)]
pub enum EventArgument {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
    /// discriminant for a detailed Attribute
    EventAttribute(EventAttributes),
}

impl<T: Into<EventAttributes>> From<T> for EventArgument {
    fn from(value: T) -> Self {
        match value.into() {
            EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(m),
            } => m.into(),
            attr => EventArgument::EventAttribute(attr),
        }
    }
}
