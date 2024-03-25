#![deny(missing_docs)]

//! NVTX crate for interfacing with NVIDIA's nvtx API

/// the module for nvtx API
pub mod nvtx {

    pub use color_name::colors;
    use nvtx_sys::ffi::NVTX_VERSION;
    use std::{
        ffi::{CStr, CString},
        marker::PhantomData,
        sync::atomic::{AtomicU32, Ordering},
    };
    use widestring::WideCString;

    trait TypeValueEncodable {
        type Type;
        type Value;
        fn encode(&self) -> (Self::Type, Self::Value);
        fn default_encoding() -> (Self::Type, Self::Value);
    }

    /// Represents a color in use for controlling appearance within NSight Systems
    #[derive(Debug, Clone, Copy)]
    pub struct Color {
        /// alpha channel
        a: u8,
        /// red channel
        r: u8,
        /// green channel
        g: u8,
        /// blue channel
        b: u8,
    }

    impl From<[u8; 3]> for Color {
        fn from(value: [u8; 3]) -> Self {
            Color {
                a: 255,
                r: value[0],
                g: value[1],
                b: value[2],
            }
        }
    }

    impl Color {
        /// Create a new color from specified channels
        pub fn new(a: u8, r: u8, g: u8, b: u8) -> Self {
            Self { a, r, g, b }
        }

        /// Change the transparency channel for the color
        pub fn transparency(mut self, value: u8) -> Self {
            self.a = value;
            self
        }

        /// Change the red channel for the color
        pub fn red(mut self, value: u8) -> Self {
            self.r = value;
            self
        }

        /// Change the green channel for the color
        pub fn green(mut self, value: u8) -> Self {
            self.g = value;
            self
        }

        /// Change the blue channel for the color
        pub fn blue(mut self, value: u8) -> Self {
            self.b = value;
            self
        }
    }

    impl TypeValueEncodable for Color {
        type Type = nvtx_sys::ffi::nvtxColorType_t;
        type Value = u32;

        fn encode(&self) -> (Self::Type, Self::Value) {
            let as_u32 = (self.a as u32) << 24
                | (self.r as u32) << 16
                | (self.g as u32) << 8
                | (self.b as u32);
            (nvtx_sys::ffi::nvtxColorType_t::NVTX_COLOR_ARGB, as_u32)
        }

        fn default_encoding() -> (Self::Type, Self::Value) {
            (nvtx_sys::ffi::nvtxColorType_t::NVTX_COLOR_UNKNOWN, 0)
        }
    }

    /// Represents a payload value for use within event attributes
    #[derive(Debug, Clone, Copy)]
    pub enum Payload {
        /// the payload shall hold a 32-bit floating-point value
        Float(f32),
        /// the payload shall hold a 64-bit floating-point value
        Double(f64),
        /// the payload shall hold a 32-bit integral value
        Int32(i32),
        /// the payload shall hold a 64-bit integral value
        Int64(i64),
        /// the payload shall hold a 32-bit unsigned integral value
        Uint32(u32),
        /// the payload shall hold a 64-bit unsigned integral value
        Uint64(u64),
    }

    impl From<u64> for Payload {
        fn from(v: u64) -> Self {
            Self::Uint64(v)
        }
    }

    impl From<u32> for Payload {
        fn from(v: u32) -> Self {
            Self::Uint32(v)
        }
    }

    impl From<i64> for Payload {
        fn from(v: i64) -> Self {
            Self::Int64(v)
        }
    }

    impl From<i32> for Payload {
        fn from(v: i32) -> Self {
            Self::Int32(v)
        }
    }

    impl From<f64> for Payload {
        fn from(v: f64) -> Self {
            Self::Double(v)
        }
    }

    impl From<f32> for Payload {
        fn from(v: f32) -> Self {
            Self::Float(v)
        }
    }

    impl TypeValueEncodable for Payload {
        type Type = nvtx_sys::ffi::nvtxPayloadType_t;
        type Value = nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t;
        fn encode(&self) -> (Self::Type, Self::Value) {
            match self {
                Payload::Float(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_FLOAT,
                    Self::Value { fValue: *x },
                ),
                Payload::Double(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_DOUBLE,
                    Self::Value { dValue: *x },
                ),
                Payload::Int32(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT32,
                    Self::Value { iValue: *x },
                ),
                Payload::Int64(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT64,
                    Self::Value { llValue: *x },
                ),
                Payload::Uint32(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
                    Self::Value { uiValue: *x },
                ),
                Payload::Uint64(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
                    Self::Value { ullValue: *x },
                ),
            }
        }

        fn default_encoding() -> (Self::Type, Self::Value) {
            (
                nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_UNKNOWN,
                Self::Value { ullValue: 0 },
            )
        }
    }

    /// Handle for retrieving a registered string. See [`Domain::register_string`] and [`Domain::register_strings`]
    #[derive(Debug, Clone, Copy)]
    pub struct DomainRegisteredString<'str> {
        handle: nvtx_sys::ffi::nvtxStringHandle_t,
        domain: &'str Domain,
    }

    /// Represents a category for use with event and range grouping. See [`Domain::register_category`], [`Domain::register_categories`]
    #[derive(Debug, Clone, Copy)]
    pub struct DomainCategory<'cat> {
        id: u32,
        domain: &'cat Domain,
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
                Str::Unicode(s) => unsafe {
                    nvtx_sys::ffi::nvtxNameCategoryW(id, s.as_ptr().cast())
                },
            }
            Category { id }
        }
    }

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
                    Str::Unicode(s) => unsafe {
                        nvtx_sys::ffi::nvtxDomainCreateW(s.as_ptr().cast())
                    },
                },
                registered_categories: AtomicU32::new(0),
            }
        }

        /// Gets a new builder instance for event attribute construction in the current domain
        pub fn event_attributes_builder(&self) -> DomainEventAttributesBuilder<'_> {
            DomainEventAttributesBuilder {
                domain: self,
                category: None,
                color: None,
                payload: None,
                message: None,
            }
        }

        /// Registers an immutable string within the current domain
        pub fn register_string(&self, string: impl Into<Str>) -> DomainRegisteredString<'_> {
            let handle = match string.into() {
                Str::Ascii(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainRegisterStringA(self.handle, s.as_ptr())
                },
                Str::Unicode(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainRegisterStringW(self.handle, s.as_ptr().cast())
                },
            };
            DomainRegisteredString {
                handle,
                domain: self,
            }
        }

        /// Register many immutable strings within the current domain
        pub fn register_strings<const N: usize>(
            &self,
            strings: [impl Into<Str>; N],
        ) -> [DomainRegisteredString<'_>; N] {
            strings.map(|string| self.register_string(string))
        }

        /// Register a new category within the domain. Categories are used to group sets of events.
        pub fn register_category(&self, name: impl Into<Str>) -> DomainCategory<'_> {
            let id = 1 + self.registered_categories.fetch_add(1, Ordering::SeqCst);
            match name.into() {
                Str::Ascii(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainNameCategoryA(self.handle, id, s.as_ptr())
                },
                Str::Unicode(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainNameCategoryW(self.handle, id, s.as_ptr().cast())
                },
            }
            DomainCategory { id, domain: self }
        }

        /// Register new categories within the domain. Categories are used to group sets of events.
        pub fn register_categories<const N: usize>(
            &self,
            names: [impl Into<Str>; N],
        ) -> [DomainCategory<'_>; N] {
            names.map(|name| self.register_category(name))
        }

        /// Marks an instantaneous event in the application. A marker can contain a text message or specify additional information using the event attributes structure. These attributes include a text message, color, category, and a payload. Each of the attributes is optional.
        pub fn mark<'a>(&'a self, arg: impl Into<DomainEventArgument<'a>>) {
            let attribute = match arg.into() {
                DomainEventArgument::EventAttribute(attr) => attr,
                DomainEventArgument::Ascii(s) => self.event_attributes_builder().message(s).build(),
                DomainEventArgument::Unicode(s) => self
                    .event_attributes_builder()
                    .message(DomainMessage::Unicode(s))
                    .build(),
            };
            let encoded = attribute.encode();
            unsafe { nvtx_sys::ffi::nvtxDomainMarkEx(self.handle, &encoded) }
        }

        /// RAII-friendly domain-owned range type which can (1) be moved across thread boundaries and (2) automatically ended when dropped
        pub fn range<'a>(&'a self, arg: impl Into<DomainEventArgument<'a>>) -> DomainRange<'a> {
            DomainRange::new(arg, self)
        }

        /// Name a resource
        pub fn name_resource<'a>(
            &'a self,
            identifier: DomainIdentifier,
            name: impl Into<DomainMessage<'a>>,
        ) -> DomainResource<'a> {
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
            DomainResource {
                handle,
                _lifetime: PhantomData,
            }
        }

        /// Create a user defined synchronization object This is used to track non-OS synchronization working with spinlocks and atomics.
        pub fn user_sync<'a>(&'a self, name: impl Into<DomainMessage<'a>>) -> sync::UserSync<'a> {
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

    /// A convenience wrapper for various string types
    #[derive(Debug, Clone)]
    pub enum Str {
        /// Represents an ASCII friendly string
        Ascii(CString),
        /// Represents a Unicode string
        Unicode(WideCString),
    }

    impl From<String> for Str {
        fn from(v: String) -> Self {
            Self::Unicode(
                WideCString::from_str(v.as_str()).expect("Could not convert to wide string"),
            )
        }
    }

    impl From<&str> for Str {
        fn from(v: &str) -> Self {
            Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
        }
    }

    impl From<CString> for Str {
        fn from(v: CString) -> Self {
            Self::Ascii(v)
        }
    }

    impl From<&CStr> for Str {
        fn from(v: &CStr) -> Self {
            Self::Ascii(CString::from(v))
        }
    }

    /// Represents a message for use within events and ranges
    #[derive(Debug, Clone)]
    pub enum DomainMessage<'a> {
        /// discriminant for an owned ASCII string
        Ascii(CString),
        /// discriminant for an owned Unicode string
        Unicode(WideCString),
        /// discriminant for a registered string belonging to a domain
        Registered(&'a DomainRegisteredString<'a>),
    }

    impl<'a> From<&'a DomainRegisteredString<'a>> for DomainMessage<'a> {
        fn from(v: &'a DomainRegisteredString) -> Self {
            Self::Registered(v)
        }
    }

    impl From<String> for DomainMessage<'_> {
        fn from(v: String) -> Self {
            Self::Unicode(
                WideCString::from_str(v.as_str()).expect("Could not convert to wide string"),
            )
        }
    }

    impl From<&str> for DomainMessage<'_> {
        fn from(v: &str) -> Self {
            Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
        }
    }

    impl From<CString> for DomainMessage<'_> {
        fn from(v: CString) -> Self {
            Self::Ascii(v)
        }
    }

    impl From<&CStr> for DomainMessage<'_> {
        fn from(v: &CStr) -> Self {
            Self::Ascii(CString::from(v))
        }
    }

    impl<'a> TypeValueEncodable for DomainMessage<'a> {
        type Type = nvtx_sys::ffi::nvtxMessageType_t;
        type Value = nvtx_sys::ffi::nvtxMessageValue_t;

        fn encode(&self) -> (Self::Type, Self::Value) {
            match &self {
                DomainMessage::Ascii(s) => (
                    nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII,
                    Self::Value { ascii: s.as_ptr() },
                ),
                DomainMessage::Unicode(s) => (
                    nvtx_sys::ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_UNICODE,
                    Self::Value {
                        unicode: s.as_ptr().cast(),
                    },
                ),
                DomainMessage::Registered(r) => (
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
            Self::Unicode(
                WideCString::from_str(v.as_str()).expect("Could not convert to wide string"),
            )
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

    impl<'a> TypeValueEncodable for Message {
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

    impl<'a> EventAttributes {
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

    /// All attributes that are associated with marks and ranges
    #[derive(Debug, Clone)]
    pub struct DomainEventAttributes<'a> {
        category: Option<DomainCategory<'a>>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<DomainMessage<'a>>,
    }

    impl<'a> DomainEventAttributes<'a> {
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
                    .map(DomainMessage::encode)
                    .unwrap_or_else(Message::default_encoding),
            )
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

    impl<'a> From<EventAttributes> for EventArgument {
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

    /// Convenience wrapper for all valid argument types
    #[derive(Debug, Clone)]
    pub enum DomainEventArgument<'a> {
        /// discriminant for an owned ASCII string
        Ascii(CString),
        /// discriminant for an owned Unicode string
        Unicode(WideCString),
        /// discriminant for a detailed Attribute
        EventAttribute(DomainEventAttributes<'a>),
    }

    impl<'a> From<DomainEventAttributes<'a>> for DomainEventArgument<'a> {
        fn from(value: DomainEventAttributes<'a>) -> Self {
            match value {
                DomainEventAttributes {
                    category: None,
                    color: None,
                    payload: None,
                    message: Some(DomainMessage::Ascii(s)),
                } => DomainEventArgument::Ascii(s),
                DomainEventAttributes {
                    category: None,
                    color: None,
                    payload: None,
                    message: Some(DomainMessage::Unicode(s)),
                } => DomainEventArgument::Unicode(s),
                attr => DomainEventArgument::EventAttribute(attr.into()),
            }
        }
    }

    impl From<&str> for DomainEventArgument<'_> {
        fn from(v: &str) -> Self {
            Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
        }
    }

    impl From<CString> for DomainEventArgument<'_> {
        fn from(v: CString) -> Self {
            Self::Ascii(v)
        }
    }

    impl From<&CStr> for DomainEventArgument<'_> {
        fn from(v: &CStr) -> Self {
            Self::Ascii(CString::from(v))
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

    /// Builder to facilitate easier construction of [`DomainEventAttributes`]
    #[derive(Debug, Clone)]
    pub struct DomainEventAttributesBuilder<'a> {
        domain: &'a Domain,
        category: Option<&'a DomainCategory<'a>>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<DomainMessage<'a>>,
    }

    impl<'a> DomainEventAttributesBuilder<'a> {
        /// Update the attribute's category. An assertion will be thrown if a DomainCategory is passed in whose domain is not the same as this builder
        pub fn category(
            mut self,
            category: &'a DomainCategory<'a>,
        ) -> DomainEventAttributesBuilder<'a> {
            assert!(
                std::ptr::eq(category.domain, self.domain),
                "Builder's Domain differs from Category's Domain"
            );
            self.category = Some(category);
            self
        }

        /// update the attribute's color
        pub fn color(mut self, color: impl Into<Color>) -> DomainEventAttributesBuilder<'a> {
            self.color = Some(color.into());
            self
        }

        /// update the attribute's payload
        pub fn payload(mut self, payload: impl Into<Payload>) -> DomainEventAttributesBuilder<'a> {
            self.payload = Some(payload.into());
            self
        }

        /// Update the attribute's message. An assertion will be thrown if a DomainRegisteredString is passed in whose domain is not the same as this builder
        pub fn message(
            mut self,
            message: impl Into<DomainMessage<'a>>,
        ) -> DomainEventAttributesBuilder<'a> {
            let msg: DomainMessage = message.into();
            if let DomainMessage::Registered(r) = &msg {
                assert!(
                    std::ptr::eq(r.domain, self.domain),
                    "Builder's Domain differs from DomainRegisteredString's Domain"
                )
            }
            self.message = Some(msg);
            self
        }

        /// build the attribute from the builder's state
        pub fn build(self) -> DomainEventAttributes<'a> {
            DomainEventAttributes {
                category: self.category.copied(),
                color: self.color,
                payload: self.payload,
                message: self.message,
            }
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

    /// A RAII-like object for modeling start/end Ranges within a Domain
    #[derive(Debug)]
    pub struct DomainRange<'a> {
        id: nvtx_sys::ffi::nvtxRangeId_t,
        domain: &'a Domain,
    }

    impl<'a> DomainRange<'a> {
        fn new<'arg, 'domain: 'a + 'arg>(
            arg: impl Into<DomainEventArgument<'arg>>,
            domain: &'domain Domain,
        ) -> DomainRange<'a> {
            let argument = arg.into();
            let arg = match argument {
                DomainEventArgument::EventAttribute(attr) => attr,
                DomainEventArgument::Ascii(s) => DomainEventAttributes {
                    category: None,
                    color: None,
                    payload: None,
                    message: Some(DomainMessage::Ascii(s)),
                },
                DomainEventArgument::Unicode(s) => DomainEventAttributes {
                    category: None,
                    color: None,
                    payload: None,
                    message: Some(DomainMessage::Unicode(s)),
                },
            };
            let id = unsafe { nvtx_sys::ffi::nvtxDomainRangeStartEx(domain.handle, &arg.encode()) };
            DomainRange { id, domain }
        }
    }

    impl<'a> Drop for DomainRange<'a> {
        fn drop(&mut self) {
            unsafe { nvtx_sys::ffi::nvtxDomainRangeEnd(self.domain.handle, self.id) }
        }
    }

    /// module for User Synchronization state machine types
    pub mod sync {
        use std::marker::PhantomData;

        /// User Defined Synchronization Object
        pub struct UserSync<'a> {
            pub(super) handle: nvtx_sys::ffi::nvtxSyncUser_t,
            pub(super) _lifetime: PhantomData<&'a ()>,
        }

        impl<'a> UserSync<'a> {
            /// Signal to tools that an attempt to acquire a user defined synchronization object.
            #[must_use = "Dropping the return will violate the state machine"]
            pub fn acquire(self) -> UserSyncAcquireStart<'a> {
                unsafe { nvtx_sys::ffi::nvtxDomainSyncUserAcquireStart(self.handle) }
                UserSyncAcquireStart { sync_object: self }
            }
        }

        impl<'a> Drop for UserSync<'a> {
            fn drop(&mut self) {
                unsafe { nvtx_sys::ffi::nvtxDomainSyncUserDestroy(self.handle) }
            }
        }

        /// State modeling the start of acquiring the synchronization object.
        pub struct UserSyncAcquireStart<'a> {
            sync_object: UserSync<'a>,
        }

        impl<'a> UserSyncAcquireStart<'a> {
            /// Signal to tools of failure in acquiring a user defined synchronization object.
            #[must_use = "Dropping the return will result in the Synchronization Object being destroyed"]
            pub fn failed(self) -> UserSync<'a> {
                unsafe { nvtx_sys::ffi::nvtxDomainSyncUserAcquireFailed(self.sync_object.handle) }
                self.sync_object
            }

            /// Signal to tools of success in acquiring a user defined synchronization object.
            #[must_use = "Dropping the return will violate the state machine"]
            pub fn success(self) -> UserSyncSuccess<'a> {
                unsafe { nvtx_sys::ffi::nvtxDomainSyncUserAcquireSuccess(self.sync_object.handle) }
                UserSyncSuccess {
                    sync_object: self.sync_object,
                }
            }
        }

        /// State modeling the success of acquiring the synchronization object.
        pub struct UserSyncSuccess<'a> {
            sync_object: UserSync<'a>,
        }

        impl<'a> UserSyncSuccess<'a> {
            /// Signal to tools of releasing a reservation on user defined synchronization object.
            #[must_use = "Dropping the return will result in the Synchronization Object being destroyed"]
            pub fn releasing(self) -> UserSync<'a> {
                unsafe { nvtx_sys::ffi::nvtxDomainSyncUserReleasing(self.sync_object.handle) }
                self.sync_object
            }
        }
    }

    /// DomainIdentifier used for DomainResource
    pub enum DomainIdentifier {
        /// generic pointer
        Pointer(*const ::std::os::raw::c_void),
        /// generic handle
        Handle(u64),
        /// generic thread native
        NativeThread(u64),
        /// generic thread posix
        PosixThread(u64),
    }

    impl TypeValueEncodable for DomainIdentifier {
        type Type = u32;
        type Value = nvtx_sys::ffi::nvtxResourceAttributes_v0_identifier_t;

        fn encode(&self) -> (Self::Type, Self::Value) {
            match self {
                DomainIdentifier::Pointer(p) => (nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_POINTER as u32, Self::Value{pValue: p.clone()}),
                DomainIdentifier::Handle(h) => (nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_HANDLE as u32, Self::Value{ullValue: h.clone()}),
                DomainIdentifier::NativeThread(t) => (nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE as u32, Self::Value{ullValue: t.clone()}),
                DomainIdentifier::PosixThread(t) => (nvtx_sys::ffi::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX as u32, Self::Value{ullValue: t.clone()}),
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
    pub struct DomainResource<'a> {
        handle: nvtx_sys::ffi::nvtxResourceHandle_t,
        _lifetime: PhantomData<&'a ()>,
    }

    impl<'a> Drop for DomainResource<'a> {
        fn drop(&mut self) {
            unsafe { nvtx_sys::ffi::nvtxDomainResourceDestroy(self.handle) }
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

    /// Allows the user to name an active thread of the current process. If an invalid thread ID is provided or a thread ID from a different process is used the behavior of the tool is implementation dependent.
    pub fn name_thread(thread_id: u32, name: impl Into<Str>) {
        match name.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameOsThreadA(thread_id, s.as_ptr()) },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxNameOsThreadA(thread_id, s.as_ptr().cast())
            },
        }
    }

    /// RAII-friendly range type which can (1) be moved across thread boundaries and (2) automatically ended when dropped
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
}
