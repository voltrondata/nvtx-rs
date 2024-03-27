use std::marker::PhantomData;

use crate::EventArgument;

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
            EventArgument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangePushA(s.as_ptr()) },
            EventArgument::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxRangePushW(s.as_ptr().cast())
            },
            EventArgument::EventAttribute(a) => unsafe {
                nvtx_sys::ffi::nvtxRangePushEx(&a.encode())
            },
        };
        LocalRange {
            _phantom: PhantomData,
        }
    }
}

impl Drop for LocalRange {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxRangePop() };
    }
}
