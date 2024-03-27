use super::{EventAttributes, Message, RegisteredString};
use std::ffi::CString;
use widestring::WideCString;

/// Convenience wrapper for all valid argument types to ranges and marks
///
/// * Any string type will be translated to [`EventArgument::Ascii`], [`EventArgument::Unicode`], or [`EventArgument::Registered`] depending on its type.
/// * If [`EventArgument::EventAttribute`] is the active discriminator:
///   - The if its held [`EventAttributes`] only specifies a message, it will be converted into the message's active discriminator.
///   - Otherwise, the existing [`EventAttributes`] will be used for the event.
#[derive(Debug, Clone)]
pub enum EventArgument<'a> {
    /// discriminator for an owned ASCII string
    Ascii(CString),
    /// discriminator for an owned Unicode string
    Unicode(WideCString),
    /// discriminator for a referenced registered string
    Registered(&'a RegisteredString<'a>),
    /// discriminator for a detailed Attribute
    EventAttribute(EventAttributes<'a>),
}

impl<'a, T: Into<EventAttributes<'a>>> From<T> for EventArgument<'a> {
    fn from(value: T) -> Self {
        match value.into() {
            EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(m),
            } => match m {
                Message::Ascii(s) => EventArgument::Ascii(s),
                Message::Unicode(s) => EventArgument::Unicode(s),
                Message::Registered(s) => EventArgument::Registered(s),
            },
            attr => EventArgument::EventAttribute(attr),
        }
    }
}
