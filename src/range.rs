use crate::event_argument::EventArgument;

/// A RAII-like object for modeling start/end Ranges
#[derive(Debug)]
pub struct Range {
    id: nvtx_sys::ffi::nvtxRangeId_t,
}

impl Range {
    /// Create an RAII-friendly range type which (1) can be moved across thread boundaries and (2) automatically ended when dropped
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
