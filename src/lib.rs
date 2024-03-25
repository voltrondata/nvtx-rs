#![deny(missing_docs)]

//! crate for interfacing with NVIDIA's nvtx API

/// all domain-specific entities
pub mod domain;
/// user-defined synchronization object types
pub mod sync;
/// color support
pub mod color;
/// transparent payload abstractions
pub mod payload;
/// transparent string abstractions
pub mod str;

use color::Color;
use domain::Domain;
use payload::Payload;
use str::Str;
use std::{
    ffi::{CStr, CString},
    sync::atomic::{AtomicU32, Ordering},
};
use widestring::WideCString;

trait TypeValueEncodable {
    type Type;
    type Value;
    fn encode(&self) -> (Self::Type, Self::Value);
    fn default_encoding() -> (Self::Type, Self::Value);
}

/// Represents a category for use with event and range grouping. See [`register_category`], [`register_categories`]
#[derive(Debug, Clone, Copy)]
pub struct Category {
    id: u32,
}

impl Category {
    /// Create a new category not affiliated with any domain
    fn new(name: impl Into<Str>) -> Category {
        static COUNT: AtomicU32 = AtomicU32::new(0);
        let id: u32 = 1 + COUNT.fetch_add(1, Ordering::SeqCst);
        match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCategoryA(id, s.as_ptr()) },
            Str::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxNameCategoryW(id, s.as_ptr().cast()) },
        }
        Category { id }
    }
}

/// Represents a message for use within events and ranges
#[derive(Debug, Clone)]
pub enum Message {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
}

impl From<String> for Message {
    fn from(v: String) -> Self {
        Self::Unicode(WideCString::from_str(v.as_str()).expect("Could not convert to wide string"))
    }
}

impl From<&str> for Message {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for Message {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for Message {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}

impl TypeValueEncodable for Message {
    type Type = nvtx_sys::ffi::nvtxMessageType_t;
    type Value = nvtx_sys::ffi::nvtxMessageValue_t;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match &self {
            Message::Ascii(s) => (
                nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII,
                Self::Value { ascii: s.as_ptr() },
            ),
            Message::Unicode(s) => (
                nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_UNICODE,
                Self::Value {
                    unicode: s.as_ptr().cast(),
                },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (
            nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_UNKNOWN,
            Self::Value {
                ascii: std::ptr::null(),
            },
        )
    }
}

/// All attributes that are associated with marks and ranges
#[derive(Debug, Clone)]
pub struct EventAttributes {
    category: Option<Category>,
    color: Option<Color>,
    payload: Option<Payload>,
    message: Option<Message>,
}

impl EventAttributes {
    fn encode(&self) -> nvtx_sys::ffi::nvtxEventAttributes_t {
        let (color_type, color_value) = self
            .color
            .as_ref()
            .map(Color::encode)
            .unwrap_or_else(Color::default_encoding);
        let (payload_type, payload_value) = self
            .payload
            .as_ref()
            .map(Payload::encode)
            .unwrap_or_else(Payload::default_encoding);
        let cat = self.category.as_ref().map(|c| c.id).unwrap_or(0);
        let emit = |(t, v)| nvtx_sys::ffi::nvtxEventAttributes_t {
            version: nvtx_sys::ffi::NVTX_VERSION as u16,
            size: 48,
            category: cat,
            colorType: color_type as i32,
            color: color_value,
            payloadType: payload_type as i32,
            reserved0: 0,
            payload: payload_value,
            messageType: t as i32,
            message: v,
        };
        // this is separated as a callable since we need encode() to outlive the call to emit
        emit(
            self.message
                .as_ref()
                .map(Message::encode)
                .unwrap_or_else(Message::default_encoding),
        )
    }
}

/// Builder to facilitate easier construction of [`EventAttributes`]
#[derive(Debug, Clone)]
pub struct EventAttributesBuilder<'a> {
    category: Option<&'a Category>,
    color: Option<Color>,
    payload: Option<Payload>,
    message: Option<Message>,
}

impl<'a> EventAttributesBuilder<'a> {
    /// update the attribute's category
    pub fn category(mut self, category: &'a Category) -> EventAttributesBuilder<'a> {
        self.category = Some(category);
        self
    }

    /// update the attribute's color
    pub fn color(mut self, color: impl Into<Color>) -> EventAttributesBuilder<'a> {
        self.color = Some(color.into());
        self
    }

    /// update the attribute's payload
    pub fn payload(mut self, payload: impl Into<Payload>) -> EventAttributesBuilder<'a> {
        self.payload = Some(payload.into());
        self
    }

    /// update the attribute's message
    pub fn message(mut self, message: impl Into<Message>) -> EventAttributesBuilder<'a> {
        self.message = Some(message.into());
        self
    }

    /// build the attribute from the builder's state
    pub fn build(self) -> EventAttributes {
        EventAttributes {
            category: self.category.copied(),
            color: self.color,
            payload: self.payload,
            message: self.message,
        }
    }
}

/// Convenience wrapper for all valid argument types
#[derive(Debug, Clone)]
pub enum EventArgument {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
    /// discriminant for a detailed Attribute
    EventAttribute(EventAttributes),
}

impl From<EventAttributes> for EventArgument {
    fn from(value: EventAttributes) -> Self {
        match value {
            EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(Message::Ascii(s)),
            } => EventArgument::Ascii(s),
            EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(Message::Unicode(s)),
            } => EventArgument::Unicode(s),
            attr => EventArgument::EventAttribute(attr.into()),
        }
    }
}

impl From<&str> for EventArgument {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for EventArgument {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for EventArgument {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}

/// A RAII-like object for modeling start/end Ranges
#[derive(Debug)]
pub struct Range {
    id: nvtx_sys::ffi::nvtxRangeId_t,
}

impl Range {
    fn new(arg: impl Into<EventArgument>) -> Range {
        let argument = arg.into();
        let id = match &argument {
            EventArgument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartA(s.as_ptr()) },
            EventArgument::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxRangeStartW(s.as_ptr().cast())
            },
            EventArgument::EventAttribute(a) => unsafe {
                nvtx_sys::ffi::nvtxRangeStartEx(&a.encode())
            },
        };
        Range { id }
    }
}

impl<'a> Drop for Range {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxRangeEnd(self.id) }
    }
}

/// Gets a new builder instance for event attribute construction in the default (global) scope
pub fn event_attributes_builder<'a>() -> EventAttributesBuilder<'a> {
    EventAttributesBuilder {
        category: None,
        color: None,
        payload: None,
        message: None,
    }
}

/// Marks an instantaneous event in the application. A marker can contain a text message or specify additional information using the event attributes structure. These attributes include a text message, color, category, and a payload. Each of the attributes is optional.
pub fn mark(argument: impl Into<EventArgument>) {
    match argument.into() {
        EventArgument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxMarkA(s.as_ptr()) },
        EventArgument::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxMarkW(s.as_ptr().cast()) },
        EventArgument::EventAttribute(a) => unsafe { nvtx_sys::ffi::nvtxMarkEx(&a.encode()) },
    }
}

/// Create a new domain
pub fn domain(name: impl Into<Str>) -> Domain {
    Domain::new(name)
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

/// Create an RAII-friendly range type which can (1) be moved across thread boundaries and (2) automatically ended when dropped
pub fn range(arg: impl Into<EventArgument>) -> Range {
    Range::new(arg)
}

/// Register a new category within the default (global) scope. Categories are used to group sets of events.
pub fn register_category(name: impl Into<Str>) -> Category {
    Category::new(name)
}

/// Register many categories within the default (global) scope. Categories are used to group sets of events.
pub fn register_categories<const C: usize>(names: [impl Into<Str>; C]) -> [Category; C] {
    names.map(register_category)
}
