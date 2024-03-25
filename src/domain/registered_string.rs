use super::Domain;

/// Handle for retrieving a registered string. See [`Domain::register_string`] and [`Domain::register_strings`]
#[derive(Debug, Clone, Copy)]
pub struct RegisteredString<'str> {
    pub(super) handle: nvtx_sys::ffi::nvtxStringHandle_t,
    pub(super) domain: &'str Domain,
}
