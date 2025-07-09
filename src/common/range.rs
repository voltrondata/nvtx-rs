use crate::EventArgument;

/// Common trait for range-like types that can be created from event arguments
pub trait RangeLike {
    type Id;

    /// Create a new range from an event argument
    fn new_from_arg(arg: impl Into<EventArgument>) -> Self;
}

/// Common trait for local range-like types
pub trait LocalRangeLike {
    /// Create a new local range from an event argument
    fn new_from_arg(arg: impl Into<EventArgument>) -> Self;
}
