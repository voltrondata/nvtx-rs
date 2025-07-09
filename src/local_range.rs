use crate::{common::LocalRangeLike, EventArgument, Message};
use std::marker::PhantomData;

/// A RAII-like object for modeling callstack (thread-local) Ranges.
#[derive(Debug)]
pub struct LocalRange {
    // prevent Sync + Send
    _phantom: PhantomData<*mut i32>,
}

impl LocalRange {
    /// Create an RAII-friendly range type which (1) cannot be moved across thread
    /// boundaries and (2) automatically ended when dropped.
    ///
    /// ```
    /// // creation from Rust string
    /// let range = nvtx::LocalRange::new("simple name");
    ///
    /// // creation from C string (since 1.77)
    /// let range = nvtx::LocalRange::new(c"simple name");
    ///
    /// // creation from EventAttributes
    /// let attr = nvtx::EventAttributes::builder()
    ///     .payload(1)
    ///     .message("complex range")
    ///     .build();
    /// let range = nvtx::LocalRange::new(attr);
    ///
    /// // explicitly end a range
    /// drop(range)
    /// ```
    pub fn new(arg: impl Into<EventArgument>) -> LocalRange {
        LocalRange::new_from_arg(arg)
    }
}

impl LocalRangeLike for LocalRange {
    fn new_from_arg(arg: impl Into<EventArgument>) -> Self {
        match arg.into() {
            EventArgument::Message(Message::Ascii(s)) => nvtx_sys::range_push_ascii(&s),
            EventArgument::Message(Message::Unicode(s)) => nvtx_sys::range_push_unicode(&s),
            EventArgument::Message(Message::Registered(_)) => {
                unreachable!("Registered strings are not valid in the global context")
            }
            EventArgument::Attributes(a) => nvtx_sys::range_push_ex(&a.encode()),
        };
        LocalRange {
            _phantom: PhantomData,
        }
    }
}

impl Drop for LocalRange {
    fn drop(&mut self) {
        nvtx_sys::range_pop();
    }
}
