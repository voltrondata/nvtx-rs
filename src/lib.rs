pub mod nvtx {

    use std::{
        ffi::{CStr, CString},
        sync::atomic::{AtomicU32, Ordering},
    };

    pub use color_name::colors;

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

    /// Represents a category for use with event and range grouping
    #[derive(Debug, Clone, Copy)]
    pub struct Category {
        id: u32,
    }

    /// Represents a domain for high-level grouping
    #[derive(Debug)]
    pub struct Domain {
        handle: nvtx_sys::ffi::nvtxDomainHandle_t,
    }

    /// A convenience wrapper for various string types
    #[derive(Debug, Clone)]
    pub enum StrType {
        Ascii(CString),
        Unicode(CString),
    }

    /// A convenience wrapper for various string types
    #[derive(Debug, Clone)]
    pub enum Str<'a> {
        CLikeString(CString),
        RustString(String),
        CLikeStr(&'a CStr),
        RustStr(&'a str),
    }

    /// Represents a message for use within events and ranges
    #[derive(Debug, Clone)]
    pub enum Message {
        Ascii(CString),
        Unicode(CString),
        Registered(RegisteredString),
    }

    /// Model all possible attributes that can be associated with events and ranges
    #[derive(Debug, Clone)]
    pub struct Attribute {
        category: Option<Category>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<Message>,
    }

    /// Builder to facilitate easier construction of [Attribute]
    #[derive(Default, Debug)]
    pub struct AttributeBuilder {
        category: Option<Category>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<Message>,
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

    type RangeLevel = i32;

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

    impl Category {
        pub fn new(name: Str<'_>, domain: Option<&Domain>) -> Category {
            static COUNT: AtomicU32 = AtomicU32::new(0);
            let id: u32 = 1 + COUNT.fetch_add(1, Ordering::SeqCst);
            match domain {
                Some(d) => match Str::make_from(name) {
                    StrType::Ascii(s) => unsafe {
                        nvtx_sys::ffi::nvtxDomainNameCategoryA(d.handle, id, s.as_ptr())
                    },
                    StrType::Unicode(s) => unsafe {
                        nvtx_sys::ffi::nvtxDomainNameCategoryW(d.handle, id, s.as_ptr().cast())
                    },
                },
                None => match Str::make_from(name) {
                    StrType::Ascii(s) => unsafe {
                        nvtx_sys::ffi::nvtxNameCategoryA(id, s.as_ptr())
                    },
                    StrType::Unicode(s) => unsafe {
                        nvtx_sys::ffi::nvtxNameCategoryW(id, s.as_ptr().cast())
                    },
                },
            }
            Category { id }
        }
    }

    impl Domain {
        pub fn new(name: Str<'_>) -> Self {
            Domain {
                handle: match Str::make_from(name) {
                    StrType::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxDomainCreateA(s.as_ptr()) },
                    StrType::Unicode(s) => unsafe {
                        nvtx_sys::ffi::nvtxDomainCreateW(s.as_ptr().cast())
                    },
                },
            }
        }

        pub fn register_string(self: &Self, string: Str<'_>) -> RegisteredString {
            let handle = match Str::make_from(string) {
                StrType::Ascii(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainRegisterStringA(self.handle, s.as_ptr())
                },
                StrType::Unicode(s) => unsafe {
                    nvtx_sys::ffi::nvtxDomainRegisterStringA(self.handle, s.as_ptr())
                },
            };
            RegisteredString { handle }
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

    impl<'a> Str<'a> {
        fn make_from(value: Str<'a>) -> StrType {
            match value {
                Str::CLikeString(x) => StrType::Ascii(x),
                Str::RustString(x) => StrType::Unicode(CString::new(x).unwrap()),
                Str::CLikeStr(x) => StrType::Ascii(CString::new(x.to_str().unwrap()).unwrap()),
                Str::RustStr(x) => StrType::Unicode(CString::new(x).unwrap()),
            }
        }
    }

    impl From<CString> for Str<'_> {
        fn from(v: CString) -> Self {
            Self::CLikeString(v)
        }
    }

    impl From<String> for Str<'_> {
        fn from(v: String) -> Self {
            Self::RustString(v)
        }
    }

    impl<'a> From<&'a str> for Str<'a> {
        fn from(v: &'a str) -> Self {
            Self::RustStr(v)
        }
    }

    impl<'a> From<&'a CStr> for Str<'a> {
        fn from(v: &'a CStr) -> Self {
            Self::CLikeStr(v)
        }
    }

    impl From<RegisteredString> for Message {
        fn from(v: RegisteredString) -> Self {
            Self::Registered(v)
        }
    }

    impl<'a> From<Str<'a>> for Message {
        fn from(v: Str<'a>) -> Self {
            match Str::make_from(v) {
                StrType::Ascii(s) => Message::Ascii(s),
                StrType::Unicode(s) => Message::Unicode(s),
            }
        }
    }

    impl Attribute {
        pub fn builder() -> AttributeBuilder {
            AttributeBuilder::new()
        }
    }

    impl AttributeBuilder {
        fn new() -> AttributeBuilder {
            AttributeBuilder::default()
        }

        pub fn category(mut self, category: Category) -> AttributeBuilder {
            self.category = Some(category);
            self
        }

        pub fn color(mut self, color: Color) -> AttributeBuilder {
            self.color = Some(color);
            self
        }

        pub fn payload(mut self, payload: Payload) -> AttributeBuilder {
            self.payload = Some(payload);
            self
        }

        pub fn message(mut self, message: Message) -> AttributeBuilder {
            self.message = Some(message);
            self
        }

        pub fn build(self) -> Attribute {
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

        pub fn new_simple(s: Str) -> Range {
            let id = range_start_simple(s);
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
            use nvtx_sys::ffi;
            match self {
                Payload::Float(x) => (
                    ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_FLOAT,
                    ffi::nvtxEventAttributes_v2_payload_t { fValue: *x },
                ),
                Payload::Double(x) => (
                    ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_DOUBLE,
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { dValue: *x },
                ),
                Payload::Int32(x) => (
                    ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_INT32,
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { iValue: *x },
                ),
                Payload::Int64(x) => (
                    ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_INT64,
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { llValue: *x },
                ),
                Payload::Uint32(x) => (
                    ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { uiValue: *x },
                ),
                Payload::Uint64(x) => (
                    ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { ullValue: *x },
                ),
            }
        }

        fn default_encoding() -> (Self::Type, Self::Value) {
            (
                nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_UNKNOWN,
                nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { ullValue: 0 },
            )
        }
    }

    impl TypeValueEncodable for Message {
        type Type = nvtx_sys::ffi::nvtxMessageType_t;
        type Value = nvtx_sys::ffi::nvtxMessageValue_t;

        fn encode(&self) -> (Self::Type, Self::Value) {
            use nvtx_sys::ffi;
            match self {
                Message::Ascii(s) => (
                    ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_ASCII,
                    nvtx_sys::ffi::nvtxMessageValue_t { ascii: s.as_ptr() },
                ),
                Message::Unicode(s) => (
                    ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_UNICODE,
                    nvtx_sys::ffi::nvtxMessageValue_t { ascii: s.as_ptr() },
                ),
                Message::Registered(r) => (
                    ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_REGISTERED,
                    nvtx_sys::ffi::nvtxMessageValue_t {
                        registered: r.handle,
                    },
                ),
            }
        }

        fn default_encoding() -> (Self::Type, Self::Value) {
            (
                nvtx_sys::ffi::nvtxMessageType_t_NVTX_MESSAGE_UNKNOWN,
                nvtx_sys::ffi::nvtxMessageValue_t {
                    ascii: std::ptr::null(),
                },
            )
        }
    }

    trait ValueEncodable {
        type Value;
        fn encode(&self) -> Self::Value;
    }

    impl ValueEncodable for Attribute {
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

    pub fn mark_simple(s: Str) {
        match Str::make_from(s) {
            StrType::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxMarkA(s.as_ptr()) },
            StrType::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxMarkW(s.as_ptr().cast()) },
        }
    }

    pub fn range_start(attr: Attribute) -> RangeId {
        let attribute = attr.encode();
        let id = unsafe { nvtx_sys::ffi::nvtxRangeStartEx(&attribute) };
        RangeId { id }
    }

    pub fn range_start_simple(s: Str) -> RangeId {
        let id = match Str::make_from(s) {
            StrType::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartA(s.as_ptr()) },
            StrType::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxRangeStartW(s.as_ptr().cast()) },
        };
        RangeId { id }
    }

    pub fn range_end(range: RangeId) {
        unsafe { nvtx_sys::ffi::nvtxRangeEnd(range.id) }
    }

    pub fn range_push(attr: Attribute) -> Option<RangeLevel> {
        let attribute = attr.encode();
        let res = unsafe { nvtx_sys::ffi::nvtxRangePushEx(&attribute) };
        if res < 0 {
            None
        } else {
            Some(res)
        }
    }

    pub fn range_push_simple(s: Str) -> Option<RangeLevel> {
        let res = match Str::make_from(s) {
            StrType::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxRangePushA(s.as_ptr()) },
            StrType::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxRangePushW(s.as_ptr().cast()) },
        };
        if res < 0 {
            None
        } else {
            Some(res)
        }
    }

    pub fn range_pop() -> Option<RangeLevel> {
        let res = unsafe { nvtx_sys::ffi::nvtxRangePop() };
        if res < 0 {
            None
        } else {
            Some(res)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nvtx::{
        self, colors, Attribute, Category, Color, Domain, Message, Payload, Range, Str,
    };

    #[test]
    fn it_works() {
        let theseus = Domain::new(Str::from("theseus"));
        let kernel = Category::new(Str::from("Kernel"), Some(&theseus));
        let registered_string = theseus.register_string(Str::from("A registered string"));

        let msg = Message::from(registered_string);
        let val = Payload::from(3.14);
        let color = Color::from(colors::aliceblue);
        theseus.mark(
            Attribute::builder()
                .color(color)
                .payload(val)
                .message(msg.clone())
                .build(),
        );

        let _ = Range::new(Attribute::builder().category(kernel).message(msg).build());
        let _ = Range::new_simple(Str::from("hello"));

        let r = nvtx::range_push_simple(Str::from("Test")).unwrap();
        let s = nvtx::range_start_simple(Str::from("Test 2"));
        let rr = nvtx::range_pop().unwrap();
        assert!(r == rr);
        nvtx::range_end(s);
    }
}
