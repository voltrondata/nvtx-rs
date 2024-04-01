#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg, doc_cfg))]

//! Crate for interfacing with NVIDIA's NVTX API
//!
//! When not running within NSight tools, the calls will dispatch to
//! empty method stubs, thus enabling low-overhead profiling.
//!
//! * All events are fully supported:
//!   * process ranges [`Range`] and [`domain::Range`]
//!   * thread ranges [`LocalRange`] and [`domain::LocalRange`]
//!   * marks [`mark`] and [`Domain::mark`]
//! * Naming threads is fully supported (See [`name_thread`] and [`name_current_thread`]).
//! * Domain, category, and registered strings are fully supported.
//! * The user-defined synchronization API is implemented.
//! * The user-defined resource naming API is implemented for the following platforms:
//!   * Pthreads (on unix-like platforms)
//!   * CUDA
//!   * CUDA Runtime
//!
//! ## Features
//!
//! This crate defines a few features which provide opt-in behavior. By default, all
//! features are enabled.
//!
//! * **color-names** -
//!   When enabled, [`color`] is populated with many human-readable color names.
//!
//! * **name-current-thread** -
//!   When enabled, [`name_current_thread`] is added to the crate. This may be preferred
//!   to manually determining the current OS-native thread ID.
//!
//! * **cuda** -
//!   When enabled, [`name_cuda_resource`] is added to the crate. This enables the naming
//!   of CUDA resources such as Devices, Contexts, Events, and Streams. The feature also
//!   adds [`domain::CudaIdentifier`] to provide an alternative naming mechanism via
//!   [`Domain::name_resource`].
//!
//! * **cuda_runtime** -
//!   When enabled, [`name_cudart_resource`] is added to the crate. This enables the
//!   naming of CUDA runtime resources such as Devices, Events, and Streams. The feature
//!   also adds [`domain::CudaRuntimeIdentifier`] to provide an alternative naming
//!   mechanism via [`Domain::name_resource`].
//!
//! * **tracing** -
//!   When enabled, a tracing `Layer` is provided which consumes tracing spans and events
//!   which will yield NVTX ranges and marks, respectively. Only a subset of
//!   functionality is supported.
//!
//! ## Platform-specific types
//!
//! * **PThread Resource Naming** -
//!   `PThreadIdentifier` is added to the [`domain`] module on UNIX-like platforms. This
//!   enables the naming of Pthread-specific entities such as mutexes, semaphores,
//!   condition variables, and read-write-locks.

mod category;
pub use category::Category;

/// Support for colors.
pub mod color;
/// Represents a color in use for controlling appearance within NSight profilers.
pub type Color = color::Color;

#[cfg(feature = "cuda")]
/// Support for CUDA-related APIs.
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "cuda_runtime")]
/// Support for CUDA runtime related APIs.
mod cuda_runtime;
#[cfg(feature = "cuda_runtime")]
pub use cuda_runtime::*;

/// Specialized types for use within a domain context.
pub mod domain;
/// Represents a domain for high-level grouping within NSight profilers.
pub type Domain = domain::Domain;

/// Internal type used for efficient dispatch
mod event_argument;
pub use event_argument::EventArgument;

/// Support for constructing detailed annotations for Ranges and Marks.
mod event_attributes;
pub use event_attributes::{EventAttributes, EventAttributesBuilder};

/// Support for thread-local ranges.
mod local_range;
pub use local_range::LocalRange;

/// Support for ASCII and Unicode strings.
mod message;
pub use message::Message;

/// Platform-native types.
pub mod native_types;

/// Support for payload information for Ranges and Marks.
mod payload;
pub use payload::Payload;

/// Support for process-wide ranges.
mod range;
pub use range::Range;

/// Support for transparent string types (ASCII or Unicode).
mod str;
pub use crate::str::Str;

#[cfg(feature = "tracing")]
/// Support for tracing.
pub mod tracing;

/// Trait used to encode a type to a struct type and value.
trait TypeValueEncodable {
    type Type;
    type Value;

    /// Analyze the current state and yield a type-value tuple.
    fn encode(&self) -> (Self::Type, Self::Value);

    /// Yield a default type-value tuple.
    fn default_encoding() -> (Self::Type, Self::Value);
}

/// Marks an instantaneous event in the application.
///
/// A marker can contain a text message or specify additional information using the event
/// attributes structure. These attributes include a text message, color, category, and a
/// payload. Each of the attributes is optional.
///
/// ```
/// nvtx::mark("Sample mark");
///
/// nvtx::mark(c"Another example");
///
/// nvtx::mark(
///   nvtx::EventAttributesBuilder::default()
///     .message("Interesting example")
///     .color([255, 0, 0])
///     .build());
/// ```
pub fn mark(argument: impl Into<EventArgument>) {
    match argument.into() {
        EventArgument::Message(m) => match &m {
            Message::Ascii(s) => nvtx_sys::mark_ascii(s),
            Message::Unicode(s) => nvtx_sys::mark_unicode(s),
        },
        EventArgument::Attributes(a) => nvtx_sys::mark_ex(&a.encode()),
    }
}

/// Name an active thread of the current process.
///
/// If an invalid thread ID is provided or a thread ID from a different process is used
/// the behavior of the tool is implementation dependent. This requires the OS-specific
/// thread id to be passed in.
///
/// See [`Str`] for valid conversions.
///
/// Note: getting the native TID is not necessarily easy. Prefer [`name_current_thread`]
/// if you are trying to name the current thread.
///
/// ```
/// nvtx::name_thread(12345, "My custom name");
/// ```
pub fn name_thread(native_tid: u32, name: impl Into<Str>) {
    match &name.into() {
        Str::Ascii(s) => nvtx_sys::name_os_thread_ascii(native_tid, s),
        Str::Unicode(s) => nvtx_sys::name_os_thread_unicode(native_tid, s),
    }
}

#[cfg(feature = "name-current-thread")]
/// Name the current thread of the current process.
///
/// See [`Str`] for valid conversions.
///
/// ```
/// nvtx::name_current_thread("Main thread");
/// ```
pub fn name_current_thread(name: impl Into<Str>) {
    name_thread(gettid::gettid() as u32, name);
}

/// Register a new category within the default (global) scope.
///
/// See [`Str`] for valid conversions.
/// ```
/// let cat_a = nvtx::register_category("Category A");
/// ```
pub fn register_category(name: impl Into<Str>) -> Category {
    Category::new(name)
}

/// Register many categories within the default (global) scope.
///
/// See [`Str`] for valid conversions
/// ```
/// let [cat_a, cat_b] = nvtx::register_categories(["Category A", "Category B"]);
/// ```
pub fn register_categories<const C: usize>(names: [impl Into<Str>; C]) -> [Category; C] {
    names.map(register_category)
}
