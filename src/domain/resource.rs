use std::marker::PhantomData;

/// Named resource handle
pub struct Resource<'a> {
    pub(super) handle: nvtx_sys::ResourceHandle,
    pub(super) _lifetime: PhantomData<&'a ()>,
}

impl<'a> Drop for Resource<'a> {
    fn drop(&mut self) {
        nvtx_sys::domain_resource_destroy(self.handle)
    }
}
