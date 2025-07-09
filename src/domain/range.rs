use super::EventArgument;
use crate::Domain;

/// A RAII-like object for modeling process-wide Ranges within a Domain.
#[derive(Debug)]
pub struct Range<'a> {
    pub(super) id: nvtx_sys::RangeId,
    pub(super) domain: &'a Domain,
}

impl<'a> Range<'a> {
    pub(super) fn new(arg: impl Into<EventArgument<'a>>, domain: &'a Domain) -> Range<'a> {
        Range::new_from_arg_with_domain(arg, domain)
    }

    fn new_from_arg_with_domain(arg: impl Into<EventArgument<'a>>, domain: &'a Domain) -> Self {
        Range {
            id: domain.range_start(arg),
            domain,
        }
    }
}

impl Drop for Range<'_> {
    fn drop(&mut self) {
        self.domain.range_end(self.id);
    }
}
