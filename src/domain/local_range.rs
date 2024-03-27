use super::{Domain, EventArgument};
use std::marker::PhantomData;

/// A RAII-like object for modeling callstack Ranges within a Domain
#[derive(Debug)]
pub struct LocalRange<'a> {
    pub(super) level: i32,
    pub(super) domain: &'a Domain,
    // prevent Sync + Send
    _phantom: PhantomData<*mut i32>,
}

impl<'a> LocalRange<'a> {
    pub(super) fn new(arg: impl Into<EventArgument<'a>>, domain: &'a Domain) -> LocalRange<'a> {
        let argument = arg.into();
        let arg = match argument {
            EventArgument::EventAttribute(attr) => attr,
            EventArgument::Ascii(s) => domain.event_attributes_builder().message(s).build(),
            EventArgument::Unicode(s) => domain.event_attributes_builder().message(s).build(),
            EventArgument::Registered(s) => domain.event_attributes_builder().message(s).build(),
        };
        let level = unsafe { nvtx_sys::ffi::nvtxDomainRangePushEx(domain.handle, &arg.encode()) };
        LocalRange {
            level,
            domain,
            _phantom: PhantomData,
        }
    }
}

impl<'a> Drop for LocalRange<'a> {
    fn drop(&mut self) {
        let end_level = unsafe { nvtx_sys::ffi::nvtxDomainRangePop(self.domain.handle) };
        assert_eq!(
            self.level, end_level,
            "Mismatch on levels for domain::LocalRange"
        );
    }
}
