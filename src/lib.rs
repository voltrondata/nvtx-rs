#![deny(missing_docs)]

//! crate for interfacing with NVIDIA's nvtx API

pub use crate::{
    category::Category,
    color::Color,
    domain::Domain,
    event_argument::EventArgument,
    event_attributes::{EventAttributes, EventAttributesBuilder},
    local_range::LocalRange,
    message::Message,
    payload::Payload,
    range::Range,
    str::Str,
};

/// color support
pub mod color;
/// specialized types for use within a domain context
pub mod domain;
/// user-defined synchronization objects
pub mod sync;

mod category;
mod event_argument;
mod event_attributes;
mod local_range;
mod message;
mod payload;
mod range;
mod str;

trait TypeValueEncodable {
    type Type;
    type Value;
    fn encode(&self) -> (Self::Type, Self::Value);
    fn default_encoding() -> (Self::Type, Self::Value);
}

/// Marks an instantaneous event in the application.
///
/// A marker can contain a text message or specify additional information using the event attributes structure. These attributes include a text message, color, category, and a payload. Each of the attributes is optional.
/// ```
/// nvtx::mark("Sample mark");
///
/// nvtx::mark(c"Another example");
///
/// nvtx::mark(nvtx::EventAttributesBuilder::default().message("Interesting example").color(nvtx::color::red).build());
/// ```
pub fn mark(argument: impl Into<EventArgument>) {
    match argument.into() {
        EventArgument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxMarkA(s.as_ptr()) },
        EventArgument::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxMarkW(s.as_ptr().cast()) },
        EventArgument::EventAttribute(a) => unsafe { nvtx_sys::ffi::nvtxMarkEx(&a.encode()) },
    }
}

/// Name an active thread of the current process.
///
/// If an invalid thread ID is provided or a thread ID from a different process is used the behavior of the tool is implementation dependent.
///
/// See [`Str`] for valid conversions
///
/// Note: getting the native TID is not necessarily simple. If you are trying to name the current thread, please use [`name_current_thread`]
/// ```
/// nvtx::name_thread(12345, "My custom name");
/// ```
pub fn name_thread(native_tid: u32, name: impl Into<Str>) {
    match name.into() {
        Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameOsThreadA(native_tid, s.as_ptr()) },
        Str::Unicode(s) => unsafe {
            nvtx_sys::ffi::nvtxNameOsThreadW(native_tid, s.as_ptr().cast())
        },
    }
}

#[cfg(feature = "name-current-thread")]
/// Name the current thread of the current process
///
/// See [`Str`] for valid conversions
/// ```
/// nvtx::name_current_thread("Main thread");
/// ```
pub fn name_current_thread(name: impl Into<Str>) {
    let tid = gettid::gettid() as u32;
    match name.into() {
        Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameOsThreadA(tid, s.as_ptr()) },
        Str::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxNameOsThreadW(tid, s.as_ptr().cast()) },
    }
}

/// Register a new category within the default (global) scope. Categories are used to group sets of events.
///
/// See [`Str`] for valid conversions
/// ```
/// let cat_a = nvtx::register_category("Category A");
/// ```
pub fn register_category(name: impl Into<Str>) -> Category {
    Category::new(name)
}

/// Register many categories within the default (global) scope. Categories are used to group sets of events.
///
/// See [`Str`] for valid conversions
/// ```
/// let [cat_a, cat_b] = nvtx::register_categories(["Category A", "Category B"]);
/// ```
pub fn register_categories<const C: usize>(names: [impl Into<Str>; C]) -> [Category; C] {
    names.map(register_category)
}
