use std::{ffi::{CStr, CString}, marker::PhantomData, sync::atomic::{AtomicU32, Ordering}};

use nvtx_sys::ffi::NVTX_VERSION;
use widestring::WideCString;

use crate::{sync, color::Color, payload::Payload, Str, TypeValueEncodable};

/// Represents a domain for high-level grouping
#[derive(Debug)]
pub struct Domain {
    handle: nvtx_sys::ffi::nvtxDomainHandle_t,
    registered_categories: AtomicU32,
}

impl Domain {
    /// Register a NVTX domain
    pub fn new(name: impl Into<Str>) -> Self {
        Domain {
            handle: match name.into() {
                Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxDomainCreateA(s.as_ptr()) },
                Str::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxDomainCreateW(s.as_ptr().cast()) },
            },
            registered_categories: AtomicU32::new(0),
        }
    }

    /// Gets a new builder instance for event attribute construction in the current domain
    pub fn event_attributes_builder(&self) -> EventAttributesBuilder<'_> {
        EventAttributesBuilder {
            domain: self,
            category: None,
            color: None,
            payload: None,
            message: None,
        }
    }

    /// Registers an immutable string within the current domain
    pub fn register_string(&self, string: impl Into<Str>) -> RegisteredString<'_> {
        let handle = match string.into() {
            Str::Ascii(s) => unsafe {
                nvtx_sys::ffi::nvtxDomainRegisterStringA(self.handle, s.as_ptr())
            },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxDomainRegisterStringW(self.handle, s.as_ptr().cast())
            },
        };
        RegisteredString {
            handle,
            domain: self,
        }
    }

    /// Register many immutable strings within the current domain
    pub fn register_strings<const N: usize>(
        &self,
        strings: [impl Into<Str>; N],
    ) -> [RegisteredString<'_>; N] {
        strings.map(|string| self.register_string(string))
    }

    /// Register a new category within the domain. Categories are used to group sets of events.
    pub fn register_category(&self, name: impl Into<Str>) -> Category<'_> {
        let id = 1 + self.registered_categories.fetch_add(1, Ordering::SeqCst);
        match name.into() {
            Str::Ascii(s) => unsafe {
                nvtx_sys::ffi::nvtxDomainNameCategoryA(self.handle, id, s.as_ptr())
            },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxDomainNameCategoryW(self.handle, id, s.as_ptr().cast())
            },
        }
        Category { id, domain: self }
    }

    /// Register new categories within the domain. Categories are used to group sets of events.
    pub fn register_categories<const N: usize>(
        &self,
        names: [impl Into<Str>; N],
    ) -> [Category<'_>; N] {
        names.map(|name| self.register_category(name))
    }

    /// Marks an instantaneous event in the application. A marker can contain a text message or specify additional information using the event attributes structure. These attributes include a text message, color, category, and a payload. Each of the attributes is optional.
    pub fn mark<'a>(&'a self, arg: impl Into<EventArgument<'a>>) {
        let attribute = match arg.into() {
            EventArgument::EventAttribute(attr) => attr,
            EventArgument::Ascii(s) => self.event_attributes_builder().message(s).build(),
            EventArgument::Unicode(s) => self
                .event_attributes_builder()
                .message(Message::Unicode(s))
                .build(),
        };
        let encoded = attribute.encode();
        unsafe { nvtx_sys::ffi::nvtxDomainMarkEx(self.handle, &encoded) }
    }

    /// Create an RAII-friendly, domain-owned range type which can (1) be moved across thread boundaries and (2) automatically ended when dropped
    pub fn range<'a>(&'a self, arg: impl Into<EventArgument<'a>>) -> Range<'a> {
        Range::new(arg, self)
    }

    /// Name a resource
    pub fn name_resource<'a>(
        &'a self,
        identifier: Identifier,
        name: impl Into<Message<'a>>,
    ) -> Resource<'a> {
        let materialized_name = name.into();
        let (msg_type, msg_value) = materialized_name.encode();
        let (id_type, id_value) = identifier.encode();
        let mut attrs = nvtx_sys::ffi::nvtxResourceAttributes_t {
            version: NVTX_VERSION as u16,
            size: 32,
            identifierType: id_type as i32,
            identifier: id_value,
            messageType: msg_type as i32,
            message: msg_value,
        };
        let ptr: *mut nvtx_sys::ffi::nvtxResourceAttributes_v0 = &mut attrs;
        let handle = unsafe { nvtx_sys::ffi::nvtxDomainResourceCreate(self.handle, ptr) };
        Resource {
            handle,
            _lifetime: PhantomData,
        }
    }

    /// Create a user defined synchronization object This is used to track non-OS synchronization working with spinlocks and atomics.
    pub fn user_sync<'a>(&'a self, name: impl Into<Message<'a>>) -> sync::UserSync<'a> {
        let message = name.into();
        let (msg_type, msg_value) = message.encode();
        let attrs = nvtx_sys::ffi::nvtxSyncUserAttributes_t {
            version: NVTX_VERSION as u16,
            size: 16,
            messageType: msg_type as i32,
            message: msg_value,
        };
        let handle = unsafe { nvtx_sys::ffi::nvtxDomainSyncUserCreate(self.handle, &attrs) };
        sync::UserSync {
            handle,
            _lifetime: PhantomData,
        }
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxDomainDestroy(self.handle) }
    }
}

/// Handle for retrieving a registered string. See [`Domain::register_string`] and [`Domain::register_strings`]
#[derive(Debug, Clone, Copy)]
pub struct RegisteredString<'str> {
    handle: nvtx_sys::ffi::nvtxStringHandle_t,
    domain: &'str Domain,
}

/// Represents a category for use with event and range grouping. See [`Domain::register_category`], [`Domain::register_categories`]
#[derive(Debug, Clone, Copy)]
pub struct Category<'cat> {
    id: u32,
    domain: &'cat Domain,
}

/// Represents a message for use within events and ranges
#[derive(Debug, Clone)]
pub enum Message<'a> {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
    /// discriminant for a registered string belonging to a domain
    Registered(&'a RegisteredString<'a>),
}

impl<'a> From<&'a RegisteredString<'a>> for Message<'a> {
    fn from(v: &'a RegisteredString) -> Self {
        Self::Registered(v)
    }
}

impl From<String> for Message<'_> {
    fn from(v: String) -> Self {
        Self::Unicode(WideCString::from_str(v.as_str()).expect("Could not convert to wide string"))
    }
}

impl From<&str> for Message<'_> {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for Message<'_> {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for Message<'_> {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}

impl<'a> TypeValueEncodable for Message<'a> {
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
            Message::Registered(r) => (
                nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_REGISTERED,
                Self::Value {
                    registered: r.handle,
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
pub struct EventAttributes<'a> {
    category: Option<Category<'a>>,
    color: Option<Color>,
    payload: Option<Payload>,
    message: Option<Message<'a>>,
}

impl<'a> EventAttributes<'a> {
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
    domain: &'a Domain,
    category: Option<&'a Category<'a>>,
    color: Option<Color>,
    payload: Option<Payload>,
    message: Option<Message<'a>>,
}

impl<'a> EventAttributesBuilder<'a> {
    /// Update the attribute's category. An assertion will be thrown if a Category is passed in whose domain is not the same as this builder
    pub fn category(mut self, category: &'a Category<'a>) -> EventAttributesBuilder<'a> {
        assert!(
            std::ptr::eq(category.domain, self.domain),
            "Builder's Domain differs from Category's Domain"
        );
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

    /// Update the attribute's message. An assertion will be thrown if a RegisteredString is passed in whose domain is not the same as this builder
    pub fn message(mut self, message: impl Into<Message<'a>>) -> EventAttributesBuilder<'a> {
        let msg: Message = message.into();
        if let Message::Registered(r) = &msg {
            assert!(
                std::ptr::eq(r.domain, self.domain),
                "Builder's Domain differs from domain::RegisteredString's Domain"
            )
        }
        self.message = Some(msg);
        self
    }

    /// build the attribute from the builder's state
    pub fn build(self) -> EventAttributes<'a> {
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
pub enum EventArgument<'a> {
    /// discriminant for an owned ASCII string
    Ascii(CString),
    /// discriminant for an owned Unicode string
    Unicode(WideCString),
    /// discriminant for a detailed Attribute
    EventAttribute(EventAttributes<'a>),
}

impl<'a> From<EventAttributes<'a>> for EventArgument<'a> {
    fn from(value: EventAttributes<'a>) -> Self {
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

impl From<&str> for EventArgument<'_> {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for EventArgument<'_> {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for EventArgument<'_> {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}

/// A RAII-like object for modeling start/end Ranges within a Domain
#[derive(Debug)]
pub struct Range<'a> {
    id: nvtx_sys::ffi::nvtxRangeId_t,
    domain: &'a Domain,
}

impl<'a> Range<'a> {
    fn new<'domain: 'a>(
        arg: impl Into<EventArgument<'domain>>,
        domain: &'domain Domain,
    ) -> Range<'a> {
        let argument = arg.into();
        let arg = match argument {
            EventArgument::EventAttribute(attr) => attr,
            EventArgument::Ascii(s) => EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(Message::Ascii(s)),
            },
            EventArgument::Unicode(s) => EventAttributes {
                category: None,
                color: None,
                payload: None,
                message: Some(Message::Unicode(s)),
            },
        };
        let id = unsafe { nvtx_sys::ffi::nvtxDomainRangeStartEx(domain.handle, &arg.encode()) };
        Range { id, domain }
    }
}

impl<'a> Drop for Range<'a> {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxDomainRangeEnd(self.domain.handle, self.id) }
    }
}
/// Identifier used for Resource
pub enum Identifier {
    /// generic pointer
    Pointer(*const ::std::os::raw::c_void),
    /// generic handle
    Handle(u64),
    /// generic thread native
    NativeThread(u64),
    /// generic thread posix
    PosixThread(u64),
}

impl TypeValueEncodable for Identifier {
    type Type = u32;
    type Value = nvtx_sys::ffi::nvtxResourceAttributes_v0_identifier_t;

    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            Identifier::Pointer(p) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_POINTER as u32,
                Self::Value { pValue: p.clone() },
            ),
            Identifier::Handle(h) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_HANDLE as u32,
                Self::Value {
                    ullValue: h.clone(),
                },
            ),
            Identifier::NativeThread(t) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE
                    as u32,
                Self::Value {
                    ullValue: t.clone(),
                },
            ),
            Identifier::PosixThread(t) => (
                nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX
                    as u32,
                Self::Value {
                    ullValue: t.clone(),
                },
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

/// Named resource handle
pub struct Resource<'a> {
    handle: nvtx_sys::ffi::nvtxResourceHandle_t,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> Drop for Resource<'a> {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxDomainResourceDestroy(self.handle) }
    }
}
