use crate::Domain;

/// Handle for retrieving a registered string.
///
/// See [`Domain::register_string`] and [`Domain::register_strings`]
#[derive(Debug, Clone, Copy)]
pub struct RegisteredString<'a> {
    handle: nvtx_sys::StringHandle,
    domain: &'a Domain,
}

impl<'a> RegisteredString<'a> {
    pub(super) fn new(handle: nvtx_sys::StringHandle, domain: &'a Domain) -> RegisteredString<'a> {
        RegisteredString { handle, domain }
    }

    pub(super) fn handle(&self) -> nvtx_sys::StringHandle {
        self.handle
    }

    pub(super) fn domain(&self) -> &Domain {
        self.domain
    }
}
