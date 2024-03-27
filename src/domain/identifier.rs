use crate::TypeValueEncodable;

#[cfg(target_family = "unix")]
use libc::{
    pthread_barrier_t, pthread_cond_t, pthread_mutex_t, pthread_once_t, pthread_rwlock_t,
    pthread_spinlock_t,
};

/// Identifiers used for PThread resources
#[cfg(target_family = "unix")]
pub enum PThreadIdentifier<'a> {
    /// pthread mutex
    Mutex(&'a pthread_mutex_t),
    /// pthread condition_variable
    Condition(&'a pthread_cond_t),
    /// pthread rwlock
    RWLock(&'a pthread_rwlock_t),
    /// pthread barrier
    Barrier(&'a pthread_barrier_t),
    /// pthread spinlock
    Spinlock(&'a pthread_spinlock_t),
    /// pthread once
    Once(&'a pthread_once_t),
}

/// Identifiers used for Generic resources
pub enum GenericIdentifier {
    /// generic pointer
    Pointer(*const ::std::os::raw::c_void),
    /// generic handle
    Handle(u64),
    /// generic thread native
    NativeThread(u64),
    /// generic thread posix
    PosixThread(u64),
}

/// Identifier used for supported resource types
pub enum Identifier<'a> {
    /// generic identifier
    Generic(GenericIdentifier),
    /// pthread-specific identifier
    #[cfg(target_family = "unix")]
    PThread(PThreadIdentifier<'a>),
}

impl<'a> From<GenericIdentifier> for Identifier<'a> {
    fn from(value: GenericIdentifier) -> Self {
        Identifier::Generic(value)
    }
}

#[cfg(target_family = "unix")]
impl<'a> From<PThreadIdentifier<'a>> for Identifier<'a> {
    fn from(value: PThreadIdentifier<'a>) -> Self {
        Identifier::PThread(value)
    }
}

impl<'a> TypeValueEncodable for Identifier<'a> {
    type Type = u32;
    type Value = nvtx_sys::ffi::nvtxResourceAttributes_v0_identifier_t;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            Identifier::Generic(g) => match g {
            GenericIdentifier::Pointer(p) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_POINTER as u32,
                Self::Value { pValue: *p },
            ),
            GenericIdentifier::Handle(h) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_HANDLE as u32,
                Self::Value { ullValue: *h },
            ),
            GenericIdentifier::NativeThread(t) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE
                    as u32,
                Self::Value { ullValue: *t },
            ),
            GenericIdentifier::PosixThread(t) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX
                    as u32,
                Self::Value { ullValue: *t },
            ),
        }
            #[cfg(target_family = "unix")]
            Identifier::PThread(p) => match p {
                PThreadIdentifier::Mutex(m) => (
                    nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_MUTEX as u32,
                    Self::Value { pValue: std::ptr::from_ref(*m).cast() }
                ),
                PThreadIdentifier::Condition(cv) =>  (
                    nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_CONDITION as u32,
                    Self::Value { pValue: std::ptr::from_ref(*cv).cast() }
                ),
                PThreadIdentifier::RWLock(rwl) =>  (
                    nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_RWLOCK as u32,
                    Self::Value { pValue: std::ptr::from_ref(*rwl).cast() }
                ),
                PThreadIdentifier::Barrier(bar) =>  (
                    nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_BARRIER as u32,
                    Self::Value { pValue: std::ptr::from_ref(*bar).cast() }
                ),
                PThreadIdentifier::Spinlock(s) =>  (
                    nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_SPINLOCK as u32,
                    Self::Value { pValue: std::ptr::from_ref(*s).cast() }
                ),
                PThreadIdentifier::Once(o) =>  (
                    nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_ONCE as u32,
                    Self::Value { pValue: std::ptr::from_ref(*o).cast() }
                ),
            },
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (
            nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_UNKNOWN as u32,
            Self::Value { ullValue: 0 },
        )
    }
}
