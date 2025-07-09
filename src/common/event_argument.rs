/// Generic event argument type that can be used in both global and domain contexts
#[derive(Debug, Clone)]
pub enum GenericEventArgument<M, A> {
    /// Holds a Message.
    Message(M),
    /// Holds an EventAttributes.
    Attributes(A),
}

impl<M, A> GenericEventArgument<M, A> {
    /// Create a new event argument from a message
    pub fn message(message: M) -> Self {
        Self::Message(message)
    }

    /// Create a new event argument from attributes
    pub fn attributes(attributes: A) -> Self {
        Self::Attributes(attributes)
    }

    /// Check if this is a message variant
    pub fn is_message(&self) -> bool {
        matches!(self, Self::Message(_))
    }

    /// Check if this is an attributes variant
    pub fn is_attributes(&self) -> bool {
        matches!(self, Self::Attributes(_))
    }
}
