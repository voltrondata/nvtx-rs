pub mod nvtx {

    use std::{collections::HashMap, ffi::CString};

    pub use color_name::colors;

    #[derive(Debug)]
    pub enum Payload {
        Float(f32),
        Double(f64),
        Int32(i32),
        Int64(i64),
        Uint32(u32),
        Uint64(u64),
    }

    impl Payload {
        fn to_type(self: &Payload) -> nvtx_sys::ffi::nvtxPayloadType_t {
            use nvtx_sys::ffi;
            match self {
                Payload::Float(_) => ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_FLOAT,
                Payload::Double(_) => ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_DOUBLE,
                Payload::Int32(_) => ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_INT32,
                Payload::Int64(_) => ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_INT64,
                Payload::Uint32(_) => ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
                Payload::Uint64(_) => ffi::nvtxPayloadType_t_NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
            }
        }

        fn to_value(self: &Payload) -> nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t {
            match self {
                Payload::Float(x) => nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { fValue: *x },
                Payload::Double(x) => {
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { dValue: *x }
                }
                Payload::Int32(x) => nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { iValue: *x },
                Payload::Int64(x) => {
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { llValue: *x }
                }
                Payload::Uint32(x) => {
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { uiValue: *x }
                }
                Payload::Uint64(x) => {
                    nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { ullValue: *x }
                }
            }
        }

        fn to_pair(
            self: &Self,
        ) -> (
            nvtx_sys::ffi::nvtxPayloadType_t,
            nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t,
        ) {
            (self.to_type(), self.to_value())
        }
    }

    #[derive(Debug, Clone)]
    pub struct Color {
        a: u8,
        r: u8,
        g: u8,
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
        pub fn transparency(self: &mut Color, value: u8) {
            self.a = value
        }

        fn to_type(self: &Self) -> nvtx_sys::ffi::nvtxColorType_t {
            nvtx_sys::ffi::nvtxColorType_t_NVTX_COLOR_ARGB
        }
        fn to_value(self: &Self) -> u32 {
            (self.a as u32) << 24 | (self.r as u32) << 16 | (self.g as u32) << 8 | (self.b as u32)
        }
        fn to_pair(self: &Self) -> (nvtx_sys::ffi::nvtxColorType_t, u32) {
            (self.to_type(), self.to_value())
        }
    }

    #[derive(Debug)]
    pub struct RegisteredString {
        handle: nvtx_sys::ffi::nvtxStringHandle_t,
    }

    #[derive(Debug, Clone)]
    pub struct Category {
        id: u32,
        name: CString,
        domain: Option<Domain>,
    }

    #[derive(Default, Debug)]
    pub struct CategoryRegistry {
        registrations: u32,
        domain_registrations: HashMap<nvtx_sys::ffi::nvtxDomainHandle_t, u32>,
    }

    impl CategoryRegistry {
        pub fn create(self: &mut Self, name: CString, domain: Option<Domain>) -> Category {
            let id = match &domain {
                Some(d) => self.domain_registrations.entry(d.handle).or_insert(0),
                None => &mut self.registrations,
            };
            *id += 1;
            let cat = Category {
                id: *id,
                name,
                domain,
            };
            unsafe {
                match cat.domain {
                    Some(ref d) => {
                        nvtx_sys::ffi::nvtxDomainNameCategoryA(d.handle, cat.id, cat.name.as_ptr())
                    }
                    None => nvtx_sys::ffi::nvtxNameCategoryA(cat.id, cat.name.as_ptr()),
                }
            }
            cat
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Domain {
        handle: nvtx_sys::ffi::nvtxDomainHandle_t,
    }

    impl Domain {
        pub fn register_string(self: &Self, string: CString) -> RegisteredString {
            unsafe {
                let handle = nvtx_sys::ffi::nvtxDomainRegisterStringA(self.handle, string.as_ptr());
                RegisteredString { handle }
            }
        }

        pub fn mark(self: &Self, attr: EventAttribute) {
            unsafe {
                let attribute = attr.to_struct();
                nvtx_sys::ffi::nvtxDomainMarkEx(self.handle, &attribute)
            }
        }
    }

    #[derive(Default, Debug)]
    pub struct DomainRegistry {
        registrations: HashMap<CString, Domain>,
    }

    impl Drop for DomainRegistry {
        fn drop(&mut self) {
            self.registrations
                .values()
                .for_each(|d| unsafe { nvtx_sys::ffi::nvtxDomainDestroy(d.handle) })
        }
    }

    impl DomainRegistry {
        pub fn create(self: &mut Self, name: CString) -> Domain {
            let domain = Domain {
                handle: unsafe { nvtx_sys::ffi::nvtxDomainCreateA(name.as_ptr()) },
            };
            self.registrations.insert(name.clone(), domain.clone());
            domain
        }

        pub fn destroy(self: &mut Self, name: CString) {
            match self.registrations.get(&name) {
                Some(d) => unsafe { nvtx_sys::ffi::nvtxDomainDestroy(d.handle) },
                None => (),
            }
        }
    }

    #[derive(Debug)]
    pub enum Message {
        Ascii(CString),
        Registered(RegisteredString),
    }

    impl Message {
        fn to_type(self: &Self) -> nvtx_sys::ffi::nvtxMessageType_t {
            use nvtx_sys::ffi;
            match self {
                Message::Ascii(_) => ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_ASCII,
                Message::Registered(_) => ffi::nvtxMessageType_t_NVTX_MESSAGE_TYPE_REGISTERED,
            }
        }

        fn to_value(self: &Self) -> nvtx_sys::ffi::nvtxMessageValue_t {
            match self {
                Message::Ascii(s) => nvtx_sys::ffi::nvtxMessageValue_t { ascii: s.as_ptr() },
                Message::Registered(r) => nvtx_sys::ffi::nvtxMessageValue_t {
                    registered: r.handle,
                },
            }
        }

        fn to_pair(
            self: &Self,
        ) -> (
            nvtx_sys::ffi::nvtxMessageType_t,
            nvtx_sys::ffi::nvtxMessageValue_t,
        ) {
            (self.to_type(), self.to_value())
        }
    }

    pub struct EventAttribute {
        category: Option<Category>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<Message>,
    }

    impl EventAttribute {
        pub fn builder() -> EventAttributeBuilder {
            EventAttributeBuilder::new()
        }

        fn to_struct(self) -> nvtx_sys::ffi::nvtxEventAttributes_t {
            let (color_type, color_value) = self
                .color
                .map(|c| c.to_pair())
                .unwrap_or((nvtx_sys::ffi::nvtxColorType_t_NVTX_COLOR_UNKNOWN, 0));
            let (payload_type, payload_value) = self.payload.map(|c| c.to_pair()).unwrap_or((
                nvtx_sys::ffi::nvtxPayloadType_t_NVTX_PAYLOAD_UNKNOWN,
                nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t { ullValue: 0 },
            ));
            let (msg_type, msg_value) = self.message.map(|c| c.to_pair()).unwrap_or((
                nvtx_sys::ffi::nvtxMessageType_t_NVTX_MESSAGE_UNKNOWN,
                nvtx_sys::ffi::nvtxMessageValue_t {
                    ascii: std::ptr::null(),
                },
            ));

            nvtx_sys::ffi::nvtxEventAttributes_t {
                version: nvtx_sys::ffi::NVTX_VERSION as u16,
                size: 48,
                category: self.category.map(|c| c.id).unwrap_or(0),
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

    #[derive(Default, Debug)]
    pub struct EventAttributeBuilder {
        category: Option<Category>,
        color: Option<Color>,
        payload: Option<Payload>,
        message: Option<Message>,
    }

    impl EventAttributeBuilder {
        fn new() -> EventAttributeBuilder {
            EventAttributeBuilder {
                category: None,
                color: None,
                payload: None,
                message: None,
            }
        }

        pub fn category(mut self, category: Category) -> EventAttributeBuilder {
            self.category = Some(category);
            self
        }

        pub fn color(mut self, color: Color) -> EventAttributeBuilder {
            self.color = Some(color);
            self
        }

        pub fn payload(mut self, payload: Payload) -> EventAttributeBuilder {
            self.payload = Some(payload);
            self
        }

        pub fn message(mut self, message: Message) -> EventAttributeBuilder {
            self.message = Some(message);
            self
        }

        pub fn build(self) -> EventAttribute {
            EventAttribute {
                category: self.category,
                color: self.color,
                payload: self.payload,
                message: self.message,
            }
        }
    }

    pub fn str(s: &str) -> CString {
        string(s.to_string())
    }

    pub fn string(string: String) -> CString {
        CString::new(string).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::nvtx;


    #[test]
    fn it_works() {
        let mut cr = nvtx::CategoryRegistry::default();
        let mut dr = nvtx::DomainRegistry::default();

        let theseus = dr.create(nvtx::str("theseus"));
        let kernel = cr.create(nvtx::str("Kernel"), Some(theseus));
        let registered_string = theseus.register_string(nvtx::str("A registered string"));

        theseus.mark(
            nvtx::EventAttribute::builder()
                .color(nvtx::Color::from(nvtx::colors::aliceblue))
                .category(kernel)
                .payload(nvtx::Payload::Double(3.14))
                .message(nvtx::Message::Registered(registered_string))
                .build()
        )
    }
}
