use crate::EventArgument;

/// A RAII-like object for modeling start/end Ranges
#[derive(Debug)]
pub struct Range {
    id: nvtx_sys::ffi::nvtxRangeId_t,
}

impl Range {
    /// Create an RAII-friendly range type which (1) can be moved across thread boundaries and (2) automatically ended when dropped
    ///
    /// ```
    /// // creation from a unicode string
    /// let range = nvtx::Range::new("simple name");
    ///
    /// // creation from a c string (from rust 1.77+)
    /// let range = nvtx::Range::new(c"simple name");
    ///
    /// // creation from EventAttributes
    /// let attr = nvtx::EventAttributesBuilder::default().payload(1).message("complex range").build();
    /// let range = nvtx::Range::new(attr);
    ///
    /// // explicitly end a range
    /// drop(range)
    /// ```
    pub fn new(arg: impl Into<EventArgument>) -> Range {
        let argument = arg.into();
        let id = match &argument {
            EventArgument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartA(s.as_ptr()) },
            EventArgument::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxRangeStartW(s.as_ptr().cast())
            },
            EventArgument::EventAttribute(a) => unsafe {
                nvtx_sys::ffi::nvtxRangeStartEx(&a.encode())
            },
        };
        Range { id }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxRangeEnd(self.id) }
    }
}

unsafe impl Send for Range {}

unsafe impl Sync for Range {}
