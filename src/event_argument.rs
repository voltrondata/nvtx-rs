use crate::{EventAttributes, Message};

/// Convenience wrapper for all valid argument types to ranges and marks
///
/// * Any string type will be translated to [`EventArgument::Message`].
/// * If [`EventArgument::Attributes`] is the active discriminator:
///   - Then if its held [`EventAttributes`] only specifies a message, then it's Message will be used
///   - Otherwise, the existing [`EventAttributes`] will be used for the event.
#[derive(Debug, Clone)]
pub enum EventArgument {
    /// discriminator for a Message
    Message(Message),
    /// discriminator for an EventAttributes
    Attributes(EventAttributes),
}

impl<T: Into<EventAttributes>> From<T> for EventArgument {
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
