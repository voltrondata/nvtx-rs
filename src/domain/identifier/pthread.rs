use super::Identifier;
use crate::{
    native_types::{
        pthread_barrier_t, pthread_cond_t, pthread_mutex_t, pthread_once_t, pthread_rwlock_t,
        pthread_spinlock_t,
    },
    TypeValueEncodable,
};

/// Identifiers used for PThread resources
pub enum PThreadIdentifier {
    /// pthread mutex
    Mutex(*const pthread_mutex_t),
    /// pthread condition_variable
    Condition(*const pthread_cond_t),
    /// pthread rwlock
    RWLock(*const pthread_rwlock_t),
    /// pthread barrier
    Barrier(*const pthread_barrier_t),
    /// pthread spinlock
    Spinlock(*const pthread_spinlock_t),
    /// pthread once
    Once(*const pthread_once_t),
}

impl From<PThreadIdentifier> for Identifier {
    fn from(value: PThreadIdentifier) -> Self {
        Identifier::PThread(value)
    }
}

impl TypeValueEncodable for PThreadIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ffi::nvtxResourceAttributes_v0_identifier_t;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            PThreadIdentifier::Mutex(m) => (
                nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_MUTEX as u32,
                Self::Value { pValue: m.cast() }
            ),
            PThreadIdentifier::Condition(cv) =>  (
                nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_CONDITION as u32,
                Self::Value { pValue: cv.cast() }
            ),
            PThreadIdentifier::RWLock(rwl) =>  (
                nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_RWLOCK as u32,
                Self::Value { pValue: rwl.cast() }
            ),
            PThreadIdentifier::Barrier(bar) =>  (
                nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_BARRIER as u32,
                Self::Value { pValue: bar.cast() }
            ),
            PThreadIdentifier::Spinlock(s) =>  (
                nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_SPINLOCK as u32,
                Self::Value { pValue: s.cast() }
            ),
            PThreadIdentifier::Once(o) =>  (
                nvtx_sys::ffi::nvtxResourceSyncPosixThreadType_t::NVTX_RESOURCE_TYPE_SYNC_PTHREAD_ONCE as u32,
                Self::Value { pValue: o.cast() }
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (
            nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_UNKNOWN as u32,
            Self::Value { ullValue: 0 },
        )
    }
}
