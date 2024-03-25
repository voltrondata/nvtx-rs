use std::marker::PhantomData;

/// Named resource handle
pub struct Resource<'a> {
    pub(super) handle: nvtx_sys::ffi::nvtxResourceHandle_t,
    pub(super) _lifetime: PhantomData<&'a ()>,
}

impl<'a> Drop for Resource<'a> {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxDomainResourceDestroy(self.handle) }
    }
}
