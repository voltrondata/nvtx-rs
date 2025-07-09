use crate::{
    common::event_attributes::GenericEventAttributesBuilder,
    common::{encode_event_attributes, GenericEventAttributes},
    Category, Message,
};

/// All attributes that are associated with marks and ranges.
pub type EventAttributes = GenericEventAttributes<Category, Message>;

impl EventAttributes {
    pub(super) fn encode(&self) -> nvtx_sys::EventAttributes {
        encode_event_attributes(&self.category, &self.color, &self.message, &self.payload)
    }

    pub fn builder() -> GenericEventAttributesBuilder<Category, Message> {
        GenericEventAttributesBuilder::default()
    }
}

impl<T: Into<Message>> From<T> for EventAttributes {
    fn from(value: T) -> Self {
        EventAttributes {
            category: None,
            color: None,
            message: Some(value.into()),
            payload: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use crate::{register_category, Color, EventAttributes, Message, Payload};

    #[test]
    fn test_builder_color() {
        let builder = EventAttributes::builder();
        let color = Color::new(0x11, 0x22, 0x44, 0x88);
        let attr = builder.color(color).build();
        assert!(matches!(attr.color, Some(c) if c == color));
    }

    #[test]
    fn test_builder_category() {
        let builder = EventAttributes::builder();
        let cat = register_category("cat");
        let attr = builder.category(cat).build();
        assert!(matches!(attr.category, Some(c) if c == cat));
    }

    #[test]
    fn test_builder_payload() {
        let attr = EventAttributes::builder().payload(1_i32).build();
        assert!(matches!(attr.payload, Some(Payload::Int32(i)) if i == 1_i32));
        let attr = EventAttributes::builder().payload(2_u32).build();
        assert!(matches!(attr.payload, Some(Payload::Uint32(i)) if i == 2_u32));
        let attr = EventAttributes::builder().payload(1_i64).build();
        assert!(matches!(attr.payload, Some(Payload::Int64(i)) if i == 1_i64));
        let attr = EventAttributes::builder().payload(2_u64).build();
        assert!(matches!(attr.payload, Some(Payload::Uint64(i)) if i == 2_u64));
        let attr = EventAttributes::builder().payload(1.0_f32).build();
        assert!(matches!(attr.payload, Some(Payload::Float(i)) if i == 1.0_f32));
        let attr = EventAttributes::builder().payload(2.0_f64).build();
        assert!(matches!(attr.payload, Some(Payload::Double(i)) if i == 2.0_f64));
    }

    #[test]
    fn test_builder_message() {
        let builder = EventAttributes::builder();
        let string = "This is a message";
        let attr = builder.message(string).build();
        assert!(
            matches!(attr.message, Some(Message::Unicode(s)) if s.to_string().unwrap() == string)
        );

        let builder = EventAttributes::builder();
        let cstring = CString::new("This is a message").unwrap();
        let attr = builder.message(cstring.clone()).build();
        assert!(matches!(attr.message, Some(Message::Ascii(s)) if s == cstring));
    }
}
