#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// NVTX version compiled
pub const NVTX_VERSION: i16 = ffi::NVTX_VERSION as i16;

/// A unique range identifier
pub type RangeId = ffi::nvtxRangeId_t;

/// Struct representing all possible Event attributes
pub type EventAttributes = ffi::nvtxEventAttributes_t;

/// Struct representing all possible Resource attributes
pub type ResourceAttributes = ffi::nvtxResourceAttributes_t;

/// Struct representing all possible User-defined Synchronization attributes
pub type SyncUserAttributes = ffi::nvtxSyncUserAttributes_t;

pub type ResourceAttributesId = ffi::nvtxResourceAttributes_v0_identifier_t;
pub type ColorType = ffi::nvtxColorType_t;
pub type MessageType = ffi::nvtxMessageType_t;
pub type MessageValue = ffi::nvtxMessageValue_t;
pub type PayloadType = ffi::nvtxPayloadType_t;
pub type PayloadValue = ffi::nvtxEventAttributes_v2_payload_t;

#[derive(Debug, Clone, Copy)]
pub struct DomainHandle {
    handle: ffi::nvtxDomainHandle_t,
}

/// Unique handle for a registered resource
#[derive(Clone, Copy)]
pub struct ResourceHandle {
    handle: ffi::nvtxResourceHandle_t,
}

/// Unique handle for a registered string
#[derive(Debug, Clone, Copy)]
pub struct StringHandle {
    handle: ffi::nvtxStringHandle_t,
}

impl StringHandle {
    pub fn get_handle(&self) -> ffi::nvtxStringHandle_t {
        self.handle
    }
}

/// Unique handle for a registered user-defined synchronization object
#[derive(Debug, Clone, Copy)]
pub struct SyncUserHandle {
    handle: ffi::nvtxSyncUser_t,
}

#[cfg(feature = "cuda_runtime")]
pub type CudaEvent = ffi::cudaEvent_t;
#[cfg(feature = "cuda_runtime")]
pub type CudaStream = ffi::cudaStream_t;

#[cfg(feature = "cuda")]
pub type CuContext = ffi::CUcontext;
#[cfg(feature = "cuda")]
pub type CuDevice = ffi::CUdevice;
#[cfg(feature = "cuda")]
pub type CuEvent = ffi::CUevent;
#[cfg(feature = "cuda")]
pub type CuStream = ffi::CUstream;

pub mod resource_type {
    #![allow(ambiguous_glob_reexports)]
    #[cfg(feature = "cuda_runtime")]
    pub use super::ffi::nvtxResourceCUDARTType_t::*;
    #[cfg(feature = "cuda")]
    pub use super::ffi::nvtxResourceCUDAType_t::*;
    pub use super::ffi::nvtxResourceGenericType_t::*;
    #[cfg(target_family = "unix")]
    pub use super::ffi::nvtxResourceSyncPosixThreadType_t::*;
}

use std::ffi::CStr;
use widestring::WideCStr;

pub fn nvtxDomainMarkEx(domain: DomainHandle, eventAttrib: &EventAttributes) {
    unsafe { crate::ffi::nvtxDomainMarkEx(domain.handle, eventAttrib) }
}

pub fn nvtxMarkEx(eventAttrib: &EventAttributes) {
    unsafe { crate::ffi::nvtxMarkEx(eventAttrib) }
}

pub fn nvtxMarkA(message: &CStr) {
    unsafe { crate::ffi::nvtxMarkA(message.as_ptr()) }
}

pub fn nvtxMarkW(message: &WideCStr) {
    unsafe { crate::ffi::nvtxMarkW(message.as_ptr().cast()) }
}

pub fn nvtxDomainRangeStartEx(domain: DomainHandle, eventAttrib: &EventAttributes) -> RangeId {
    unsafe { crate::ffi::nvtxDomainRangeStartEx(domain.handle, eventAttrib) }
}

pub fn nvtxRangeStartEx(eventAttrib: &EventAttributes) -> RangeId {
    unsafe { crate::ffi::nvtxRangeStartEx(eventAttrib) }
}

pub fn nvtxRangeStartA(message: &CStr) -> RangeId {
    unsafe { crate::ffi::nvtxRangeStartA(message.as_ptr()) }
}

pub fn nvtxRangeStartW(message: &WideCStr) -> RangeId {
    unsafe { crate::ffi::nvtxRangeStartW(message.as_ptr().cast()) }
}

pub fn nvtxDomainRangeEnd(domain: DomainHandle, id: RangeId) {
    unsafe { crate::ffi::nvtxDomainRangeEnd(domain.handle, id) }
}

pub fn nvtxRangeEnd(id: RangeId) {
    unsafe { crate::ffi::nvtxRangeEnd(id) }
}

pub fn nvtxDomainRangePushEx(domain: DomainHandle, eventAttrib: &EventAttributes) -> i32 {
    unsafe { crate::ffi::nvtxDomainRangePushEx(domain.handle, eventAttrib) }
}

pub fn nvtxRangePushEx(eventAttrib: &EventAttributes) -> i32 {
    unsafe { crate::ffi::nvtxRangePushEx(eventAttrib) }
}

pub fn nvtxRangePushA(message: &CStr) -> i32 {
    unsafe { crate::ffi::nvtxRangePushA(message.as_ptr()) }
}

pub fn nvtxRangePushW(message: &WideCStr) -> i32 {
    unsafe { crate::ffi::nvtxRangePushW(message.as_ptr().cast()) }
}

pub fn nvtxDomainRangePop(domain: DomainHandle) -> i32 {
    unsafe { crate::ffi::nvtxDomainRangePop(domain.handle) }
}

pub fn nvtxRangePop() -> i32 {
    unsafe { crate::ffi::nvtxRangePop() }
}

pub fn nvtxDomainResourceCreate(
    domain: DomainHandle,
    attribs: ResourceAttributes,
) -> ResourceHandle {
    ResourceHandle {
        handle: unsafe {
            crate::ffi::nvtxDomainResourceCreate(
                domain.handle,
                std::ptr::addr_of!(attribs).cast_mut(),
            )
        },
    }
}

pub fn nvtxDomainResourceDestroy(resource: ResourceHandle) {
    unsafe { crate::ffi::nvtxDomainResourceDestroy(resource.handle) }
}

pub fn nvtxDomainNameCategoryA(domain: DomainHandle, category: u32, name: &CStr) {
    unsafe { crate::ffi::nvtxDomainNameCategoryA(domain.handle, category, name.as_ptr()) }
}

pub fn nvtxDomainNameCategoryW(domain: DomainHandle, category: u32, name: &WideCStr) {
    unsafe { crate::ffi::nvtxDomainNameCategoryW(domain.handle, category, name.as_ptr().cast()) }
}

pub fn nvtxNameCategoryA(category: u32, name: &CStr) {
    unsafe { crate::ffi::nvtxNameCategoryA(category, name.as_ptr()) }
}

pub fn nvtxNameCategoryW(category: u32, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameCategoryW(category, name.as_ptr().cast()) }
}

pub fn nvtxNameOsThreadA(threadId: u32, name: &CStr) {
    unsafe { crate::ffi::nvtxNameOsThreadA(threadId, name.as_ptr()) }
}

pub fn nvtxNameOsThreadW(threadId: u32, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameOsThreadW(threadId, name.as_ptr().cast()) }
}

#[must_use]
pub fn nvtxDomainRegisterStringA(domain: DomainHandle, string: &CStr) -> StringHandle {
    StringHandle {
        handle: unsafe { crate::ffi::nvtxDomainRegisterStringA(domain.handle, string.as_ptr()) },
    }
}

#[must_use]
pub fn nvtxDomainRegisterStringW(domain: DomainHandle, string: &WideCStr) -> StringHandle {
    StringHandle {
        handle: unsafe {
            crate::ffi::nvtxDomainRegisterStringW(domain.handle, string.as_ptr().cast())
        },
    }
}

#[must_use]
pub fn nvtxDomainCreateA(name: &CStr) -> DomainHandle {
    DomainHandle {
        handle: unsafe { crate::ffi::nvtxDomainCreateA(name.as_ptr()) },
    }
}

#[must_use]
pub fn nvtxDomainCreateW(name: &WideCStr) -> DomainHandle {
    DomainHandle {
        handle: unsafe { crate::ffi::nvtxDomainCreateW(name.as_ptr().cast()) },
    }
}

pub fn nvtxDomainDestroy(domain: DomainHandle) {
    unsafe { crate::ffi::nvtxDomainDestroy(domain.handle) }
}

#[cfg(feature = "cuda")]
pub fn nvtxNameCuDeviceA(device: CuDevice, name: &CStr) {
    unsafe { crate::ffi::nvtxNameCuDeviceA(device, name.as_ptr()) }
}

#[cfg(feature = "cuda")]
pub fn nvtxNameCuDeviceW(device: CuDevice, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameCuDeviceW(device, name.as_ptr().cast()) }
}

#[cfg(feature = "cuda")]
/// # Safety
pub unsafe fn nvtxNameCuContextA(context: CuContext, name: &CStr) {
    unsafe { crate::ffi::nvtxNameCuContextA(context, name.as_ptr()) }
}

#[cfg(feature = "cuda")]
/// # Safety
pub unsafe fn nvtxNameCuContextW(context: CuContext, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameCuContextW(context, name.as_ptr().cast()) }
}

#[cfg(feature = "cuda")]
/// # Safety
pub unsafe fn nvtxNameCuStreamA(stream: CuStream, name: &CStr) {
    unsafe { crate::ffi::nvtxNameCuStreamA(stream, name.as_ptr()) }
}

#[cfg(feature = "cuda")]
/// # Safety
pub unsafe fn nvtxNameCuStreamW(stream: CuStream, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameCuStreamW(stream, name.as_ptr().cast()) }
}

#[cfg(feature = "cuda")]
/// # Safety
pub unsafe fn nvtxNameCuEventA(event: CuEvent, name: &CStr) {
    unsafe { crate::ffi::nvtxNameCuEventA(event, name.as_ptr()) }
}

#[cfg(feature = "cuda")]
/// # Safety
pub unsafe fn nvtxNameCuEventW(event: CuEvent, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameCuEventW(event, name.as_ptr().cast()) }
}

#[cfg(feature = "cuda_runtime")]
pub fn nvtxNameCudaDeviceA(device: i32, name: &CStr) {
    unsafe { crate::ffi::nvtxNameCudaDeviceA(device, name.as_ptr()) }
}

#[cfg(feature = "cuda_runtime")]
pub fn nvtxNameCudaDeviceW(device: i32, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameCudaDeviceW(device, name.as_ptr().cast()) }
}

#[cfg(feature = "cuda_runtime")]
/// # Safety
pub unsafe fn nvtxNameCudaStreamA(stream: CudaStream, name: &CStr) {
    unsafe { crate::ffi::nvtxNameCudaStreamA(stream, name.as_ptr()) }
}

#[cfg(feature = "cuda_runtime")]
/// # Safety
pub unsafe fn nvtxNameCudaStreamW(stream: CudaStream, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameCudaStreamW(stream, name.as_ptr().cast()) }
}

#[cfg(feature = "cuda_runtime")]
/// # Safety
pub unsafe fn nvtxNameCudaEventA(event: CudaEvent, name: &CStr) {
    unsafe { crate::ffi::nvtxNameCudaEventA(event, name.as_ptr()) }
}

#[cfg(feature = "cuda_runtime")]
/// # Safety
pub unsafe fn nvtxNameCudaEventW(event: CudaEvent, name: &WideCStr) {
    unsafe { crate::ffi::nvtxNameCudaEventW(event, name.as_ptr().cast()) }
}

#[must_use]
pub fn nvtxDomainSyncUserCreate(
    domain: DomainHandle,
    attribs: SyncUserAttributes,
) -> SyncUserHandle {
    SyncUserHandle {
        handle: unsafe {
            crate::ffi::nvtxDomainSyncUserCreate(
                domain.handle,
                std::ptr::addr_of!(attribs).cast_mut(),
            )
        },
    }
}

pub fn nvtxDomainSyncUserDestroy(handle: SyncUserHandle) {
    unsafe { crate::ffi::nvtxDomainSyncUserDestroy(handle.handle) }
}

pub fn nvtxDomainSyncUserAcquireStart(handle: SyncUserHandle) {
    unsafe { crate::ffi::nvtxDomainSyncUserAcquireStart(handle.handle) }
}

pub fn nvtxDomainSyncUserAcquireFailed(handle: SyncUserHandle) {
    unsafe { crate::ffi::nvtxDomainSyncUserAcquireFailed(handle.handle) }
}

pub fn nvtxDomainSyncUserAcquireSuccess(handle: SyncUserHandle) {
    unsafe { crate::ffi::nvtxDomainSyncUserAcquireSuccess(handle.handle) }
}

pub fn nvtxDomainSyncUserReleasing(handle: SyncUserHandle) {
    unsafe { crate::ffi::nvtxDomainSyncUserReleasing(handle.handle) }
}
