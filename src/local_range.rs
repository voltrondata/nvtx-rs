use std::marker::PhantomData;

use crate::event_argument::EventArgument;

/// A RAII-like object for modeling callstack Ranges
#[derive(Debug)]
pub struct LocalRange {
    level: i32,
    // prevent Sync + Send
    _phantom: PhantomData<*mut i32>,
}

impl LocalRange {
    /// Create an RAII-friendly range type which (1) cannot be moved across thread boundaries and (2) automatically ended when dropped. Panics on drop() if the opening level doesn't match the closing level (since it must model a perfect stack).
    ///
    /// ```
    /// // creation from a unicode string
    /// let range = nvtx::LocalRange::new("simple name");
    ///
    /// // creation from a c string (from rust 1.77+)
    /// let range = nvtx::LocalRange::new(c"simple name");
    ///
    /// // creation from an EventAttribute
    /// let attr = nvtx::EventAttributeBuilder::default().payload(1).message("complex range").build();
    /// let range = nvtx::LocalRange::new(attr);
    ///
    /// // explicitly end a range
    /// drop(range)
    /// ```
    pub fn new(arg: impl Into<EventArgument>) -> LocalRange {
        let argument = arg.into();
        let level = match &argument {
            EventArgument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangePushA(s.as_ptr()) },
            EventArgument::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxRangePushW(s.as_ptr().cast())
            },
            EventArgument::EventAttribute(a) => unsafe {
                nvtx_sys::ffi::nvtxRangePushEx(&a.encode())
            },
        };
        LocalRange {
            level,
            _phantom: PhantomData,
        }
    }
}

impl Drop for LocalRange {
    fn drop(&mut self) {
        let end_level = unsafe { nvtx_sys::ffi::nvtxRangePop() };
        assert_eq!(self.level, end_level, "Mismatch on levels for LocalRange");
    }
}
