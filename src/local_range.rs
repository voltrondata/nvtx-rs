use crate::{EventArgument, Message};
use std::marker::PhantomData;

/// A RAII-like object for modeling callstack Ranges
#[derive(Debug)]
pub struct LocalRange {
    // prevent Sync + Send
    _phantom: PhantomData<*mut i32>,
}

impl LocalRange {
    /// Create an RAII-friendly range type which (1) cannot be moved across thread boundaries and (2) automatically ended when dropped.
    ///
    /// ```
    /// // creation from Rust string
    /// let range = nvtx::LocalRange::new("simple name");
    ///
    /// // creation from C string (since 1.77)
    /// let range = nvtx::LocalRange::new(c"simple name");
    ///
    /// // creation from EventAttributes
    /// let attr = nvtx::EventAttributesBuilder::default().payload(1).message("complex range").build();
    /// let range = nvtx::LocalRange::new(attr);
    ///
    /// // explicitly end a range
    /// drop(range)
    /// ```
    pub fn new(arg: impl Into<EventArgument>) -> LocalRange {
        match arg.into() {
            EventArgument::Message(m) => match &m {
                Message::Ascii(s) => nvtx_sys::nvtxRangePushA(s),
                Message::Unicode(s) => nvtx_sys::nvtxRangePushW(s),
            },
            EventArgument::Attributes(a) => nvtx_sys::nvtxRangePushEx(&a.encode()),
        };
        LocalRange {
            _phantom: PhantomData,
        }
    }
}

impl Drop for LocalRange {
    fn drop(&mut self) {
        nvtx_sys::nvtxRangePop();
    }
}
