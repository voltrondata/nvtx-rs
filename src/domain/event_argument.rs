use super::{EventAttributes, Message};

/// Convenience wrapper for all valid argument types to ranges and marks.
///
/// * Any string type will be translated to [`EventArgument::Message`].
/// * If [`EventArgument::Attributes`] is the active discriminator:
///   - If its [`EventAttributes`] only specifies a message, then message will be used.
///   - Otherwise, the existing [`EventAttributes`] will be used for the event.
#[derive(Debug, Clone)]
pub enum EventArgument<'a> {
    /// discriminator for a Message
    Message(Message<'a>),
    /// discriminator for an EventAttributes
    Attributes(EventAttributes<'a>),
}

impl<'a, T: Into<EventAttributes<'a>>> From<T> for EventArgument<'a> {
    fn from(value: T) -> Self {
        match value.into() {
            EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(m),
            } => EventArgument::Message(m),
            attr => EventArgument::Attributes(attr),
        }
    }
}
