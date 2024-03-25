use super::{Domain,event_argument::EventArgument, event_attributes::EventAttributes, message::Message};


/// A RAII-like object for modeling start/end Ranges within a Domain
#[derive(Debug)]
pub struct Range<'a> {
    pub(super) id: nvtx_sys::ffi::nvtxRangeId_t,
    pub(super) domain: &'a Domain,
}

impl<'a> Range<'a> {
    pub(super) fn new<'domain: 'a>(
        arg: impl Into<EventArgument<'domain>>,
        domain: &'domain Domain,
    ) -> Range<'a> {
        let argument = arg.into();
        let arg = match argument {
            EventArgument::EventAttribute(attr) => attr,
            EventArgument::Ascii(s) => EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(Message::Ascii(s)),
            },
            EventArgument::Unicode(s) => EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(Message::Unicode(s)),
            },
        };
        let id = unsafe { nvtx_sys::ffi::nvtxDomainRangeStartEx(domain.handle, &arg.encode()) };
        Range { id, domain }
    }
}

impl<'a> Drop for Range<'a> {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxDomainRangeEnd(self.domain.handle, self.id) }
    }
}
