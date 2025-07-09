use super::{EventAttributes, Message};
use crate::common::GenericEventArgument;

/// Convenience wrapper for all valid argument types to ranges and marks.
///
/// * Any string type will be translated to [`EventArgument::Message`].
/// * If [`EventArgument::Attributes`] is the active discriminator:
///   - If its [`EventAttributes`] only specifies a message, then message will be used.
///   - Otherwise, the existing [`EventAttributes`] will be used for the event.
pub type EventArgument<'a> = GenericEventArgument<Message<'a>, EventAttributes<'a>>;

impl<'a, T: Into<EventAttributes<'a>>> From<T> for EventArgument<'a> {
    fn from(value: T) -> Self {
        match value.into() {
            EventAttributes {
                category: None,
                color: None,
                message: Some(m),
                payload: None,
            } => EventArgument::Message(m),
            attr => EventArgument::Attributes(attr),
        }
    }
}
