use super::{Domain, EventArgument};
use std::marker::PhantomData;

/// A RAII-like object for modeling callstack Ranges within a Domain
#[derive(Debug)]
pub struct LocalRange<'a> {
    pub(super) domain: &'a Domain,
    // prevent Sync + Send
    _phantom: PhantomData<*mut i32>,
}

impl<'a> LocalRange<'a> {
    pub(super) fn new(arg: impl Into<EventArgument<'a>>, domain: &'a Domain) -> LocalRange<'a> {
        let arg = match arg.into() {
            EventArgument::Attributes(attr) => attr,
            EventArgument::Message(m) => domain.event_attributes_builder().message(m).build(),
        };
        unsafe { nvtx_sys::ffi::nvtxDomainRangePushEx(domain.handle, &arg.encode()) };
        LocalRange {
            domain,
            _phantom: PhantomData,
        }
    }
}

impl<'a> Drop for LocalRange<'a> {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxDomainRangePop(self.domain.handle) };
    }
}
