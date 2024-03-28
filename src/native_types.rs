#[cfg(feature = "cuda")]
pub use nvtx_sys::ffi::{CUcontext, CUdevice, CUevent, CUstream};

#[cfg(feature = "cuda_runtime")]
pub use nvtx_sys::ffi::{cudaEvent_t, cudaStream_t};

#[cfg(target_family = "unix")]
pub use libc::{
    pthread_barrier_t, pthread_cond_t, pthread_mutex_t, pthread_once_t, pthread_rwlock_t,
    pthread_spinlock_t,
};
