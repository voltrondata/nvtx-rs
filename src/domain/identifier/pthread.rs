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
        Self::PThread(value)
    }
}

impl TypeValueEncodable for PThreadIdentifier {
    type Type = u32;
    type Value = nvtx_sys::ResourceAttributesId;

    fn encode(&self) -> (Self::Type, Self::Value) {
        use nvtx_sys::resource_type::*;
        match self {
            Self::Mutex(m) => (
                NVTX_RESOURCE_TYPE_SYNC_PTHREAD_MUTEX,
                Self::Value { pValue: m.cast() },
            ),
            Self::Condition(cv) => (
                NVTX_RESOURCE_TYPE_SYNC_PTHREAD_CONDITION,
                Self::Value { pValue: cv.cast() },
            ),
            Self::RWLock(rwl) => (
                NVTX_RESOURCE_TYPE_SYNC_PTHREAD_RWLOCK,
                Self::Value { pValue: rwl.cast() },
            ),
            Self::Barrier(bar) => (
                NVTX_RESOURCE_TYPE_SYNC_PTHREAD_BARRIER,
                Self::Value { pValue: bar.cast() },
            ),
            Self::Spinlock(s) => (
                NVTX_RESOURCE_TYPE_SYNC_PTHREAD_SPINLOCK,
                Self::Value { pValue: s.cast() },
            ),
            Self::Once(o) => (
                NVTX_RESOURCE_TYPE_SYNC_PTHREAD_ONCE,
                Self::Value { pValue: o.cast() },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        Identifier::default_encoding()
    }
}
