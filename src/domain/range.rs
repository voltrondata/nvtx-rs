use super::EventArgument;
use crate::Domain;

/// A RAII-like object for modeling start/end Ranges within a Domain
#[derive(Debug)]
pub struct Range<'a> {
    pub(super) id: nvtx_sys::RangeId,
    pub(super) domain: &'a Domain,
}

impl<'a> Range<'a> {
    pub(super) fn new(arg: impl Into<EventArgument<'a>>, domain: &'a Domain) -> Range<'a> {
        let arg = match arg.into() {
            EventArgument::Attributes(attr) => attr,
            EventArgument::Message(m) => m.into(),
        };
        let id = nvtx_sys::domain_range_start_ex(domain.handle, &arg.encode());
        Range { id, domain }
    }
}

impl<'a> Drop for Range<'a> {
    fn drop(&mut self) {
        nvtx_sys::domain_range_end(self.domain.handle, self.id)
    }
}

unsafe impl<'a> Send for Range<'a> {}

unsafe impl<'a> Sync for Range<'a> {}
