#![deny(missing_docs)]

//! NVTX crate for interfacing with NVIDIA's nvtx API

/// the module for nvtx API
pub mod nvtx {

    pub use color_name::colors;
    use std::{
        ffi::{CStr, CString},
        marker::PhantomData,
        sync::atomic::{AtomicU32, Ordering},
    };
    use widestring::{WideCStr, WideCString};

    trait TypeValueEncodable {
        type Type;
        type Value;
        fn encode(&self) -> (Self::Type, Self::Value);
        fn default_encoding() -> (Self::Type, Self::Value);
    }

    /// Represents a color in use for controlling appearance within nsight
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
            (nvtx_sys::ffi::nvtxColorType_t_NVTX_COLOR_ARGB, as_u32)
        }

        fn default_encoding() -> (Self::Type, Self::Value) {
            (nvtx_sys::ffi::nvtxColorType_t_NVTX_COLOR_UNKNOWN, 0)
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
                    nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_FLOAT,
                    Self::Value { fValue: *x },
                ),
                Payload::Double(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_DOUBLE,
                    Self::Value { dValue: *x },
                ),
                Payload::Int32(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_INT32,
                    Self::Value { iValue: *x },
                ),
                Payload::Int64(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_INT64,
                    Self::Value { llValue: *x },
                ),
                Payload::Uint32(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
                    Self::Value { uiValue: *x },
                ),
                Payload::Uint64(x) => (
                    nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
                    Self::Value { ullValue: *x },
                ),
            }
        }

        fn default_encoding() -> (Self::Type, Self::Value) {
            (
                nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_UNKNOWN,
                Self::Value { ullValue: 0 },
            )
        }
    }

    /// Type for a registered nvtx string. See [`Domain::register_string`] and [`Domain::get_registered_string`]
    #[derive(Debug, Clone)]
    pub struct RegisteredString {
        handle: nvtx_sys::ffi::nvtxStringHandle_t,
    }

    /// Handle for retrieving a registered string. See [`Domain::register_string`] and [`Domain::get_registered_string`]
    #[derive(Debug, PartialEq, PartialOrd, Eq, Ord)]
    pub struct RegisteredStringHandle {
        id: usize,
    }

    /// Represents a category for use with event and range grouping.
    #[derive(Debug, Clone, Copy)]
    pub struct Category<'a> {
        id: u32,
        _lifetime: PhantomData<&'a ()>,
    }

    impl Category<'static> {
        /// Create a new category given something that can be converted into a supported string type
        pub fn new(name: impl Into<Str>) -> Category<'static> {
            static COUNT: AtomicU32 = AtomicU32::new(0);
            let id: u32 = 1 + COUNT.fetch_add(1, Ordering::SeqCst);
            match name.into() {
                Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxNameCategoryA(id, s.as_ptr()) },
                Str::Unicode(s) => unsafe {
                    nvtx_sys::ffi::nvtxNameCategoryW(id, s.as_ptr().cast())
                },
            }
            Category {
                id,
                _lifetime: PhantomData,
            }
        }
    }

    /// Handle for retrieving a maintained category
    #[derive(Debug, PartialEq, PartialOrd, Eq, Ord)]
    pub struct CategoryHandle {
        id: u32,
    }

    /// Represents a domain for high-level grouping
    #[derive(Debug)]
    pub struct Domain {
        handle: nvtx_sys::ffi::nvtxDomainHandle_t,
        registered_strings: Vec<RegisteredString>,
        registered_categories: u32,
    }

    impl Domain {
        /// Create a new domain given something that can be converted into a supported string type
        pub fn new(name: impl Into<Str>) -> Self {
            Domain {
                handle: match name.into() {
                    Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxDomainCreateA(s.as_ptr()) },
                    Str::Unicode(s) => unsafe {
                        nvtx_sys::ffi::nvtxDomainCreateW(s.as_ptr().cast())
                    },
                },
                registered_strings: vec![],
                registered_categories: 0,
            }
        }

        /// register a new string to the domain
        pub fn register_string(&mut self, string: impl Into<Str>) -> RegisteredStringHandle {
            let handle = match string.into() {
                Str::Ascii(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainRegisterStringA(self.handle, s.as_ptr())
                },
                Str::Unicode(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainRegisterStringW(self.handle, s.as_ptr().cast())
                },
            };
            self.registered_strings.push(RegisteredString { handle });
            RegisteredStringHandle {
                id: self.registered_strings.len() - 1,
            }
        }

        /// registers many strings to the domain
        pub fn register_strings<const N: usize>(
            &mut self,
            strings: [impl Into<Str>; N],
        ) -> [RegisteredStringHandle; N] {
            strings.map(|s| self.register_string(s))
        }

        /// Get a registered string
        /// 
        /// If the handle isn't valid, None shall be returned 
        pub fn get_registered_string(
            &self,
            handle: &RegisteredStringHandle,
        ) -> Option<&RegisteredString> {
            self.registered_strings.get(handle.id)
        }

        /// get many registered strings
        /// 
        /// For any handles that are invalid, None shall be returned for the corresponding array
        pub fn get_registered_strings<const N: usize>(
            &self,
            handles: [RegisteredStringHandle; N],
        ) -> [Option<&RegisteredString>; N] {
            handles.map(|h| self.get_registered_string(&h))
        }

        /// register a category to the domain
        pub fn register_category<'a>(&'a mut self, name: impl Into<Str>) -> CategoryHandle {
            self.registered_categories += 1;
            let id = self.registered_categories;
            match name.into() {
                Str::Ascii(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainNameCategoryA(self.handle, id, s.as_ptr())
                },
                Str::Unicode(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainNameCategoryW(self.handle, id, s.as_ptr().cast())
                },
            }
            CategoryHandle { id }
        }

        /// register many categories to the domain
        pub fn register_categories<'a, const N: usize>(
            &'a mut self,
            names: [impl Into<Str>; N],
        ) -> [CategoryHandle; N] {
            names.map(|name| self.register_category(name))
        }

        /// get a registered category
        pub fn get_registered_category(&self, handle: &CategoryHandle) -> Option<Category<'_>> {
            (handle.id <= self.registered_categories).then(|| Category {
                id: handle.id,
                _lifetime: PhantomData,
            })
        }

        /// get many registered categories
        /// 
        /// For any handles that are invalid, None shall be returned for the corresponding array
        pub fn get_registered_categories<const N: usize>(
            &self,
            handles: [CategoryHandle; N],
        ) -> [Option<Category<'_>>; N] {
            handles.map(|h| self.get_registered_category(&h))
        }

        /// Start a new range which can be moved across thread boundaries
        /// 
        /// returns a RAII-friendly type which is automatically ended when dropped
        pub fn range<'a>(&'a self, arg: impl Into<Argument<'a>>) -> Range<'a> {
            Range {
                handle: RangeHandle::new(arg, Some(self)),
            }
        }

        /// Start a new range which cannot be moved across thread boundaries
        /// Additionally, this must adhere to strict push/pop semantics, making
        /// this extremely useful for tracing call stacks.
        /// 
        /// returns a handle which can be ended via [`ScopedRangeHandle::pop()`]
        pub fn range_push<'a>(&'a self, arg: impl Into<Argument<'a>>) -> ScopedRangeHandle<'a> {
            ScopedRangeHandle::new(arg, Some(self))
        }

        /// yield a range handle from this domain which can be moved to other threads
        /// 
        /// returns a handle which can be ended via [`RangeHandle::end()`]
       pub fn range_start<'a>(&'a self, arg: impl Into<Argument<'a>>) -> RangeHandle<'a> {
            RangeHandle::new(arg, Some(self))
        }
    }

    impl<'a> Domain {
        /// yield a mark for this domain
        pub fn mark(self: &Self, arg: impl Into<Argument<'a>>) {
            let attribute = match arg.into() {
                Argument::EventAttribute(attr) => attr,
                Argument::Ascii(s) => Attribute::from(s).into(),
                Argument::Unicode(s) => Attribute::from(s).into(),
            };
            let encoded = attribute.encode();
            unsafe { nvtx_sys::ffi::nvtxDomainMarkEx(self.handle, &encoded) }
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

    impl From<&WideCStr> for Str {
        fn from(v: &WideCStr) -> Self {
            Self::Unicode(WideCString::from_ustr(v).expect("Could not convert to wide string"))
        }
    }

    impl From<WideCString> for Str {
        fn from(v: WideCString) -> Self {
            Self::Unicode(v)
        }
    }

    /// Represents a message for use within events and ranges
    #[derive(Debug, Clone)]
    pub enum Message<'a> {
        /// discriminant for an owned ASCII string
        Ascii(CString),
        /// discriminant for an owned Unicode string
        Unicode(WideCString),
        /// discriminant for a registered string belonging to a domain
        Registered(&'a RegisteredString),
    }

    impl<'a> From<&'a RegisteredString> for Message<'a> {
        fn from(v: &'a RegisteredString) -> Self {
            Self::Registered(v)
        }
    }

    impl From<String> for Message<'_> {
        fn from(v: String) -> Self {
            Self::Unicode(
                WideCString::from_str(v.as_str()).expect("Could not convert to wide string"),
            )
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

    impl From<&WideCStr> for Message<'_> {
        fn from(v: &WideCStr) -> Self {
            Self::Unicode(WideCString::from_ustr(v).expect("Could not convert to wide string"))
        }
    }

    impl From<WideCString> for Message<'_> {
        fn from(v: WideCString) -> Self {
            Self::Unicode(v)
        }
    }

    impl<'a> TypeValueEncodable for Message<'a> {
        type Type = nvtx_sys::ffi::nvtxMessageType_t;
        type Value = nvtx_sys::ffi::nvtxMessageValue_t;

        fn encode(&self) -> (Self::Type, Self::Value) {
            match &self {
                Message::Ascii(s) => (
                    nvtx_sys::ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_ASCII,
                    Self::Value { ascii: s.as_ptr() },
                ),
                Message::Unicode(s) => (
                    nvtx_sys::ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_UNICODE,
                    Self::Value {
                        unicode: s.as_ptr().cast(),
                    },
                ),
                Message::Registered(r) => (
                    nvtx_sys::ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_REGISTERED,
                    Self::Value {
                        registered: r.handle,
                    },
                ),
            }
        }

        fn default_encoding() -> (Self::Type, Self::Value) {
            (
                nvtx_sys::ffi::nvtxMessageType_t_NVTX_MESSAGE_UNKNOWN,
                Self::Value {
                    ascii: std::ptr::null(),
                },
            )
        }
    }

    /// Model all possible attributes that can be associated with events and ranges
    #[derive(Debug, Clone)]
    pub struct Attribute<'a> {
        category: Option<Category<'a>>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<Message<'a>>,
    }

    impl<'a> Attribute<'a> {
        fn encode(&self) -> nvtx_sys::ffi::nvtxEventAttributes_t {
            let (color_type, color_value) = self
                .color
                .as_ref()
                .map(|&c| c.encode())
                .unwrap_or_else(Color::default_encoding);
            let (payload_type, payload_value) = self
                .payload
                .as_ref()
                .map(|c| c.encode())
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
                    .map(|m| m.encode())
                    .unwrap_or_else(|| Message::default_encoding()),
            )
        }
    }

    impl From<CString> for Attribute<'_> {
        fn from(s: CString) -> Self {
            AttributeBuilder::default().message(s).build()
        }
    }

    impl From<WideCString> for Attribute<'_> {
        fn from(s: WideCString) -> Self {
            AttributeBuilder::default().message(s).build()
        }
    }

    impl<'a> From<Message<'a>> for Attribute<'a> {
        fn from(m: Message<'a>) -> Self {
            AttributeBuilder::default().message(m).build()
        }
    }

    /// Convenience wrapper for all valid argument types
    #[derive(Clone)]
    pub enum Argument<'a> {
        /// discriminant for an owned ASCII string
        Ascii(CString),
        /// discriminant for an owned Unicode string
        Unicode(WideCString),
        /// discriminant for a detailed Attribute
        EventAttribute(Attribute<'a>),
    }

    impl<'a> From<Attribute<'a>> for Argument<'a> {
        fn from(value: Attribute<'a>) -> Self {
            match value {
                Attribute {
                    category: None,
                    color: None,
                    payload: None,
                    message: Some(Message::Ascii(s)),
                } => Argument::Ascii(s),
                Attribute {
                    category: None,
                    color: None,
                    payload: None,
                    message: Some(Message::Unicode(s)),
                } => Argument::Unicode(s),
                attr => Argument::EventAttribute(attr.into()),
            }
        }
    }

    impl From<&str> for Argument<'_> {
        fn from(v: &str) -> Self {
            Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
        }
    }

    impl From<CString> for Argument<'_> {
        fn from(v: CString) -> Self {
            Self::Ascii(v)
        }
    }

    impl From<&CStr> for Argument<'_> {
        fn from(v: &CStr) -> Self {
            Self::Ascii(CString::from(v))
        }
    }

    impl From<&WideCStr> for Argument<'_> {
        fn from(v: &WideCStr) -> Self {
            Self::Unicode(WideCString::from_ustr(v).expect("Could not convert to wide string"))
        }
    }

    impl From<WideCString> for Argument<'_> {
        fn from(v: WideCString) -> Self {
            Self::Unicode(v)
        }
    }

    /// Builder to facilitate easier construction of [`Attribute`]
    #[derive(Default)]
    pub struct AttributeBuilder<'a> {
        category: Option<Category<'a>>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<Message<'a>>,
    }


    impl<'a> AttributeBuilder<'a> {
        /// update the attribute's category
        pub fn category(mut self, category: impl Into<Category<'a>>) -> AttributeBuilder<'a> {
            self.category = Some(category.into());
            self
        }

        /// update the attribute's color
        pub fn color(mut self, color: impl Into<Color>) -> AttributeBuilder<'a> {
            self.color = Some(color.into());
            self
        }

        /// update the attribute's payload
        pub fn payload(mut self, payload: impl Into<Payload>) -> AttributeBuilder<'a> {
            self.payload = Some(payload.into());
            self
        }

        /// update the attribute's message
        pub fn message(mut self, message: impl Into<Message<'a>>) -> AttributeBuilder<'a> {
            self.message = Some(message.into());
            self
        }

        /// build the attribute from the builder's state
        pub fn build(self) -> Attribute<'a> {
            Attribute {
                category: self.category,
                color: self.color,
                payload: self.payload,
                message: self.message,
            }
        }
    }

    /// Id returned from certain nvtx function calls
    #[derive(Debug, Clone)]
    pub struct RangeHandle<'a> {
        id: nvtx_sys::ffi::nvtxRangeId_t,
        domain: Option<&'a Domain>,
    }

    impl<'a> RangeHandle<'a> {
        fn new(arg: impl Into<Argument<'a>>, domain: Option<&'a Domain>) -> RangeHandle<'a> {
            let argument = arg.into();
            if let Some(d) = domain {
                let arg = match argument {
                    Argument::EventAttribute(a) => a,
                    Argument::Ascii(s) => AttributeBuilder::default().message(s).build(),
                    Argument::Unicode(s) => AttributeBuilder::default().message(s).build(),
                };
                let id = unsafe { nvtx_sys::ffi::nvtxDomainRangeStartEx(d.handle, &arg.encode()) };
                RangeHandle {
                    id,
                    domain: Some(d),
                }
            } else {
                let id = match &argument {
                    Argument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartA(s.as_ptr()) },
                    Argument::Unicode(s) => unsafe {
                        nvtx_sys::ffi::nvtxRangeStartW(s.as_ptr().cast())
                    },
                    Argument::EventAttribute(a) => unsafe {
                        nvtx_sys::ffi::nvtxRangeStartEx(&a.encode())
                    },
                };
                RangeHandle { id, domain: None }
            }
        }

        /// ends the current range handle
        pub fn end(self) {
            match &self.domain {
                Some(d) => unsafe { nvtx_sys::ffi::nvtxDomainRangeEnd(d.handle, self.id) },
                None => unsafe { nvtx_sys::ffi::nvtxRangeEnd(self.id) },
            }
        }
    }

    /// A RAII-like wrapper for range creation and destruction which can be moved across threads
    #[derive(Debug)]
    pub struct Range<'a> {
        handle: RangeHandle<'a>,
    }

    impl<'a> Drop for Range<'a> {
        fn drop(&mut self) {
            match self.handle.domain {
                Some(d) => unsafe { nvtx_sys::ffi::nvtxDomainRangeEnd(d.handle, self.handle.id) },
                None => unsafe { nvtx_sys::ffi::nvtxRangeEnd(self.handle.id) },
            }
        }
    }

    /// Opaque type for inspecting returned levels from Push/Pop
    #[derive(Debug, Clone)]
    pub struct ScopedRangeHandle<'a> {
        value: i32,
        domain: Option<&'a Domain>,
    }

    impl<'a> ScopedRangeHandle<'a> {
        fn new(arg: impl Into<Argument<'a>>, domain: Option<&'a Domain>) -> ScopedRangeHandle<'a> {
            let argument = arg.into();
            if let Some(d) = &domain {
                let attr = match argument {
                    Argument::Ascii(s) => AttributeBuilder::default().message(s).build(),
                    Argument::Unicode(s) => AttributeBuilder::default().message(s).build(),
                    Argument::EventAttribute(a) => a,
                };
                let value =
                    unsafe { nvtx_sys::ffi::nvtxDomainRangePushEx(d.handle, &attr.encode()) };
                ScopedRangeHandle { value, domain }
            } else {
                let value = match &argument {
                    Argument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangePushA(s.as_ptr()) },
                    Argument::Unicode(s) => unsafe {
                        nvtx_sys::ffi::nvtxRangePushW(s.as_ptr().cast())
                    },
                    Argument::EventAttribute(a) => unsafe {
                        nvtx_sys::ffi::nvtxRangePushEx(&a.encode())
                    },
                };
                ScopedRangeHandle {
                    value,
                    domain: None,
                }
            }
        }

        /// closes the current range
        /// 
        /// This method will throw if push/pop semantics are not preserved
        pub fn pop(self) {
            let id = match self.domain {
                Some(d) => unsafe { nvtx_sys::ffi::nvtxDomainRangePop(d.handle) },
                None => unsafe { nvtx_sys::ffi::nvtxRangePop() },
            };
            assert_eq!(self.value, id);
        }
    }

    /// yield a mark to be emitted
    pub fn mark<'a>(arg: impl Into<Argument<'a>>) {
        let argument = arg.into();
        match &argument {
            Argument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxMarkA(s.as_ptr()) },
            Argument::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxMarkW(s.as_ptr().cast()) },
            Argument::EventAttribute(a) => unsafe { nvtx_sys::ffi::nvtxMarkEx(&a.encode()) },
        }
    }

    /// Start a new range which can be moved across thread boundaries
    /// 
    /// returns a RAII-friendly type which is automatically ended when dropped
    pub fn range<'a>(arg: impl Into<Argument<'a>>) -> Range<'a> {
        Range {
            handle: RangeHandle::new(arg, None),
        }
    }


    /// Start a new range which cannot be moved across thread boundaries
    /// Additionally, this must adhere to strict push/pop semantics, making
    /// this extremely useful for tracing call stacks.
    /// 
    /// returns a handle which can be ended via [`ScopedRangeHandle::pop()`]
    pub fn range_push<'a>(arg: impl Into<Argument<'a>>) -> ScopedRangeHandle<'a> {
        ScopedRangeHandle::new(arg, None)
    }

    /// Start a new range which can be moved across thread boundaries
    /// 
    /// returns a handle which can be ended via [`RangeHandle::end()`]
    pub fn range_start<'a>(arg: impl Into<Argument<'a>>) -> RangeHandle<'a> {
        RangeHandle::new(arg, None)
    }
}
