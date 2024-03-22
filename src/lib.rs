pub mod nvtx {

    pub use color_name::colors;
    use std::{
        ffi::{CStr, CString},
        marker::PhantomData,
        sync::atomic::{AtomicU32, Ordering},
    };
    use widestring::{WideCStr, WideCString};

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

    /// Represents a payload value for use within event attributes
    #[derive(Debug, Clone, Copy)]
    pub enum Payload {
        Float(f32),
        Double(f64),
        Int32(i32),
        Int64(i64),
        Uint32(u32),
        Uint64(u64),
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

    /// A convenience wrapper for various string types
    #[derive(Debug, Clone)]
    pub enum Str {
        Ascii(CString),
        Unicode(WideCString),
    }

    /// Represents a message for use within events and ranges
    #[derive(Debug, Clone)]
    pub enum Message<'a> {
        Ascii(CString),
        Unicode(WideCString),
        Registered(&'a RegisteredString),
    }

    /// Model all possible attributes that can be associated with events and ranges
    #[derive(Debug, Clone)]
    pub struct Attribute<'a> {
        category: Option<Category<'a>>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<Message<'a>>,
    }

    /// Convenience wrapper for all valid argument types
    #[derive(Clone)]
    pub enum Argument<'a> {
        Ascii(CString),
        Unicode(WideCString),
        EventAttribute(Attribute<'a>),
    }

    /// Builder to facilitate easier construction of [`Attribute`]
    #[derive(Default)]
    pub struct AttributeBuilder<'a> {
        category: Option<Category<'a>>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<Message<'a>>,
    }

    /// Id returned from certain nvtx function calls
    #[derive(Debug, Copy, Clone)]
    pub struct RangeId {
        id: nvtx_sys::ffi::nvtxRangeId_t,
    }

    /// A RAII-like wrapper for range creation and destruction
    #[derive(Debug)]
    pub struct Range {
        id: RangeId,
    }

    /// Opaque type for inspecting returned levels from Push/Pop
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct RangeLevel {
        value: i32,
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
        pub fn new(a: u8, r: u8, g: u8, b: u8) -> Self {
            Self { a, r, g, b }
        }

        pub fn transparency(mut self, value: u8) -> Self {
            self.a = value;
            self
        }

        pub fn red(mut self, value: u8) -> Self {
            self.r = value;
            self
        }

        pub fn green(mut self, value: u8) -> Self {
            self.g = value;
            self
        }

        pub fn blue(mut self, value: u8) -> Self {
            self.b = value;
            self
        }
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

    impl Category<'static> {
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

    impl Domain {
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

        pub fn get_registered_string(
            &self,
            handle: RegisteredStringHandle,
        ) -> Option<&RegisteredString> {
            self.registered_strings.get(handle.id)
        }

        pub fn register_category(&mut self, name: impl Into<Str>) -> CategoryHandle {
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

        pub fn get_registered_category(&self, handle: CategoryHandle) -> Option<Category<'_>> {
            (handle.id <= self.registered_categories).then(|| Category {
                id: handle.id,
                _lifetime: PhantomData,
            })
        }
    }

    impl<'a> Domain {
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

    impl<'a> AttributeBuilder<'a> {
        pub fn category(mut self, category: impl Into<Category<'a>>) -> AttributeBuilder<'a> {
            self.category = Some(category.into());
            self
        }

        pub fn color(mut self, color: impl Into<Color>) -> AttributeBuilder<'a> {
            self.color = Some(color.into());
            self
        }

        pub fn payload(mut self, payload: impl Into<Payload>) -> AttributeBuilder<'a> {
            self.payload = Some(payload.into());
            self
        }

        pub fn message(mut self, message: impl Into<Message<'a>>) -> AttributeBuilder<'a> {
            self.message = Some(message.into());
            self
        }

        pub fn build(self) -> Attribute<'a> {
            Attribute {
                category: self.category,
                color: self.color,
                payload: self.payload,
                message: self.message,
            }
        }
    }

    impl<'a> Range {
        pub fn new(arg: impl Into<Argument<'a>>) -> Range {
            let id = range_start(arg);
            Range { id }
        }
    }

    impl Drop for Range {
        fn drop(&mut self) {
            range_end(self.id)
        }
    }

    trait TypeValueEncodable {
        type Type;
        type Value;
        fn encode(&self) -> (Self::Type, Self::Value);
        fn default_encoding() -> (Self::Type, Self::Value);
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

    trait ValueEncodable {
        type Value;
        fn encode(self) -> Self::Value;
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

    pub fn mark<'a>(arg: impl Into<Argument<'a>>) {
        let argument = arg.into();
        match &argument {
            Argument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxMarkA(s.as_ptr()) },
            Argument::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxMarkW(s.as_ptr().cast()) },
            Argument::EventAttribute(a) => unsafe { nvtx_sys::ffi::nvtxMarkEx(&a.encode()) },
        }
    }

    pub fn range_start<'a>(arg: impl Into<Argument<'a>>) -> RangeId {
        let argument = arg.into();
        let id = match &argument {
            Argument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartA(s.as_ptr()) },
            Argument::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartW(s.as_ptr().cast()) },
            Argument::EventAttribute(a) => unsafe { nvtx_sys::ffi::nvtxRangeStartEx(&a.encode()) },
        };
        RangeId { id }
    }

    pub fn range_end(range: RangeId) {
        unsafe { nvtx_sys::ffi::nvtxRangeEnd(range.id) }
    }

    pub fn range_push<'a>(arg: impl Into<Argument<'a>>) -> Option<RangeLevel> {
        let argument = arg.into();
        let value = match &argument {
            Argument::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangePushA(s.as_ptr()) },
            Argument::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxRangePushW(s.as_ptr().cast()) },
            Argument::EventAttribute(a) => unsafe { nvtx_sys::ffi::nvtxRangePushEx(&a.encode()) },
        };
        if value < 0 {
            None
        } else {
            Some(RangeLevel { value })
        }
    }

    pub fn range_pop() -> Option<RangeLevel> {
        let value = unsafe { nvtx_sys::ffi::nvtxRangePop() };
        if value < 0 {
            None
        } else {
            Some(RangeLevel { value })
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nvtx::{self, colors, AttributeBuilder, Category, Domain, Range};

    #[test]
    fn it_works() {
        let mut theseus = Domain::new("theseus");
        let category_handle = theseus.register_category("Kernel");
        let registered_string_handle = theseus.register_string("A registered string");
        let kernel = theseus.get_registered_category(category_handle).unwrap();

        theseus.mark(
            AttributeBuilder::default()
                .color(colors::blanchedalmond)
                .payload(3.14)
                .message(
                    theseus
                        .get_registered_string(registered_string_handle)
                        .unwrap(),
                )
                .build(),
        );

        let _ = Range::new(
            AttributeBuilder::default()
                .category(kernel)
                .message("test")
                .build(),
        );
        let _ = Range::new("hello");

        let c = Category::new("cool");

        let r = nvtx::range_push("Test");
        let s = nvtx::range_start(
            AttributeBuilder::default()
                .category(c)
                .message("Cool")
                .build(),
        );
        let rr = nvtx::range_pop();

        assert!(r == rr);
        nvtx::range_end(s);
    }
}
