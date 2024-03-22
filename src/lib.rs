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

    /// Handle for a registered nvtx string. See [`Domain::register_string`]
    #[derive(Debug, Clone)]
    pub struct RegisteredString {
        handle: nvtx_sys::ffi::nvtxStringHandle_t,
    }

    #[derive(Debug, PartialEq, PartialOrd, Eq, Ord)]
    pub struct RegisteredStringHandle {
        id: usize,
    }

    /// Represents a category for use with event and range grouping
    #[derive(Debug, Clone, Copy)]
    pub struct Category<'a> {
        id: u32,
        _lifetime: PhantomData<&'a ()>,
    }

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

    /// Builder to facilitate easier construction of [Attribute]
    #[derive(Default, Debug)]
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

        pub fn transparency(&mut self, value: u8) {
            self.a = value
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

        pub fn mark(self: &Self, attr: Attribute) {
            let attribute = attr.encode();
            unsafe { nvtx_sys::ffi::nvtxDomainMarkEx(self.handle, &attribute) }
        }
    }

    impl Drop for Domain {
        fn drop(&mut self) {
            unsafe { nvtx_sys::ffi::nvtxDomainDestroy(self.handle) }
        }
    }

    impl From<String> for Str {
        fn from(v: String) -> Self {
            if v.is_ascii() {
                Self::Ascii(CString::new(v).unwrap())
            } else {
                Self::Unicode(
                    WideCString::from_str(v.as_str()).expect("Could not convert to wide string"),
                )
            }
        }
    }

    impl<'a> From<&'a str> for Str {
        fn from(v: &'a str) -> Self {
            if v.is_ascii() {
                Self::Ascii(CString::new(v).unwrap())
            } else {
                Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
            }
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

    impl<'a> From<&'a RegisteredString> for Message<'a> {
        fn from(v: &'a RegisteredString) -> Self {
            Self::Registered(v)
        }
    }

    impl From<Str> for Message<'_> {
        fn from(v: Str) -> Self {
            match v {
                Str::Ascii(s) => Message::Ascii(s),
                Str::Unicode(s) => Message::Unicode(s),
            }
        }
    }

    impl<'a> Attribute<'a> {
        pub fn builder() -> AttributeBuilder<'a> {
            AttributeBuilder::new()
        }
    }

    impl<'a> AttributeBuilder<'a> {
        fn new() -> AttributeBuilder<'a> {
            AttributeBuilder::default()
        }

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

    impl Range {
        pub fn new(attr: Attribute) -> Range {
            let id = range_start(attr);
            Range { id }
        }

        pub fn new_simple(s: impl Into<Str>) -> Range {
            let id = range_start_simple(s.into());
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
            use nvtx_sys::ffi;
            match self {
                Message::Ascii(s) => (
                    ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_ASCII,
                    Self::Value { ascii: s.as_ptr() },
                ),
                Message::Unicode(s) => (
                    ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_UNICODE,
                    Self::Value {
                        unicode: s.as_ptr().cast(),
                    },
                ),
                Message::Registered(r) => (
                    ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_REGISTERED,
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
        fn encode(&self) -> Self::Value;
    }

    impl<'a> ValueEncodable for Attribute<'a> {
        type Value = nvtx_sys::ffi::nvtxEventAttributes_t;

        fn encode(&self) -> Self::Value {
            let (color_type, color_value) = self
                .color
                .as_ref()
                .map(|c| c.encode())
                .unwrap_or_else(Color::default_encoding);
            let (payload_type, payload_value) = self
                .payload
                .as_ref()
                .map(|c| c.encode())
                .unwrap_or_else(Payload::default_encoding);
            let (msg_type, msg_value) = self
                .message
                .as_ref()
                .map(|c| c.encode())
                .unwrap_or_else(Message::default_encoding);
            let cat = self.category.as_ref().map(|c| c.id).unwrap_or(0);
            Self::Value {
                version: nvtx_sys::ffi::NVTX_VERSION as u16,
                size: 48,
                category: cat,
                colorType: color_type as i32,
                color: color_value,
                payloadType: payload_type as i32,
                reserved0: 0,
                payload: payload_value,
                messageType: msg_type as i32,
                message: msg_value,
            }
        }
    }

    pub fn mark(attr: Attribute) {
        let attribute = attr.encode();
        unsafe { nvtx_sys::ffi::nvtxMarkEx(&attribute) }
    }

    pub fn mark_simple(s: impl Into<Str>) {
        match s.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxMarkA(s.as_ptr()) },
            Str::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxMarkW(s.as_ptr().cast()) },
        }
    }

    pub fn range_start(attr: Attribute) -> RangeId {
        let attribute = attr.encode();
        let id = unsafe { nvtx_sys::ffi::nvtxRangeStartEx(&attribute) };
        RangeId { id }
    }

    pub fn range_start_simple(s: impl Into<Str>) -> RangeId {
        let id = match s .into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartA(s.as_ptr()) },
            Str::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartW(s.as_ptr().cast()) },
        };
        RangeId { id }
    }

    pub fn range_end(range: RangeId) {
        unsafe { nvtx_sys::ffi::nvtxRangeEnd(range.id) }
    }

    pub fn range_push(attr: Attribute) -> Option<RangeLevel> {
        let attribute = attr.encode();
        let value = unsafe { nvtx_sys::ffi::nvtxRangePushEx(&attribute) };
        if value < 0 {
            None
        } else {
            Some(RangeLevel { value })
        }
    }

    pub fn range_push_simple(s: impl Into<Str>) -> Option<RangeLevel> {
        let value = match s.into() {
            Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangePushA(s.as_ptr()) },
            Str::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxRangePushW(s.as_ptr().cast()) },
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
    use crate::nvtx::{self, colors, Attribute, Domain, Range};

    #[test]
    fn it_works() {
        let mut theseus = Domain::new("theseus");
        let category_handle = theseus.register_category("Kernel");
        let registered_string_handle = theseus.register_string("A registered string");
        let kernel = theseus.get_registered_category(category_handle).unwrap();

        let msg = 
            theseus
                .get_registered_string(registered_string_handle)
                .unwrap();
        
        theseus.mark(
            Attribute::builder()
                .color(colors::blanchedalmond)
                .payload(3.14)
                .message(msg)
                .build(),
        );

        let _ = Range::new(Attribute::builder().category(kernel).message(msg).build());
        let _ = Range::new_simple("hello");

        let r = nvtx::range_push_simple("Test");
        let s = nvtx::range_start_simple("Test 2");
        let rr = nvtx::range_pop();
        assert!(r == rr);
        nvtx::range_end(s);
    }
}
