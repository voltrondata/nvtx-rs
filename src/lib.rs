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

/// Marks an instantaneous event in the application. A marker can contain a text message or specify additional information using the event attributes structure. These attributes include a text message, color, category, and a payload. Each of the attributes is optional.
pub fn mark(argument: impl Into<EventArgument>) {
    match argument.into() {
        EventArgument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxMarkA(s.as_ptr()) },
        EventArgument::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxMarkW(s.as_ptr().cast()) },
        EventArgument::EventAttribute(a) => unsafe { nvtx_sys::ffi::nvtxMarkEx(&a.encode()) },
    }
}

/// Allows the user to name an active thread of the current process. If an invalid thread ID is provided or a thread ID from a different process is used the behavior of the tool is implementation dependent.
pub fn name_thread(thread_id: u32, name: impl Into<Str>) {
    match name.into() {
        Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameOsThreadA(thread_id, s.as_ptr()) },
        Str::Unicode(s) => unsafe {
            nvtx_sys::ffi::nvtxNameOsThreadA(thread_id, s.as_ptr().cast())
        },
    }
}

/// Register a new category within the default (global) scope. Categories are used to group sets of events.
pub fn register_category(name: impl Into<Str>) -> Category {
    Category::new(name)
}

/// Register many categories within the default (global) scope. Categories are used to group sets of events.
pub fn register_categories<const C: usize>(names: [impl Into<Str>; C]) -> [Category; C] {
    names.map(register_category)
}
