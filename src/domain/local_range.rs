use super::EventArgument;
use crate::Domain;
use std::marker::PhantomData;

/// A RAII-like object for modeling callstack (thread-local) Ranges within a Domain.
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
        nvtx_sys::domain_range_push_ex(domain.handle, &arg.encode());
        LocalRange {
            domain,
            _phantom: PhantomData,
        }
    }
}

impl Drop for LocalRange<'_> {
    fn drop(&mut self) {
        nvtx_sys::domain_range_pop(self.domain.handle);
    }
}
