use crate::{Category, Color, Message, Payload, TypeValueEncodable};

/// All attributes that are associated with marks and ranges.
#[derive(Debug, Clone)]
pub struct EventAttributes {
    pub(super) category: Option<Category>,
    pub(super) color: Option<Color>,
    pub(super) message: Option<Message>,
    pub(super) payload: Option<Payload>,
}

impl EventAttributes {
    pub(super) fn encode(&self) -> nvtx_sys::EventAttributes {
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
        let emit = |(t, v)| nvtx_sys::EventAttributes {
            version: nvtx_sys::NVTX_VERSION as u16,
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

/// Builder to facilitate easier construction of [`EventAttributes`].
///
/// ```
/// let cat = nvtx::Category::new("Category1");
///
/// let attr = nvtx::EventAttributesBuilder::default()
///                .category(cat)
///                .color([20, 192, 240])
///                .payload(3.141592)
///                .message("Hello")
///                .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct EventAttributesBuilder {
    pub(super) category: Option<Category>,
    pub(super) color: Option<Color>,
    pub(super) payload: Option<Payload>,
    pub(super) message: Option<Message>,
}

impl EventAttributesBuilder {
    /// Update the builder's held [`Category`].
    ///
    /// ```
    /// let cat = nvtx::Category::new("Category1");
    /// let builder = nvtx::EventAttributesBuilder::default();
    /// // ...
    /// let builder = builder.category(cat);
    /// ```
    pub fn category(mut self, category: Category) -> EventAttributesBuilder {
        self.category = Some(category);
        self
    }

    /// Update the builder's held [`Color`]. See [`Color`] for valid conversions.
    ///
    /// ```
    /// let builder = nvtx::EventAttributesBuilder::default();
    /// // ...
    /// let builder = builder.color([255, 255, 255]);
    /// ```
    pub fn color(mut self, color: impl Into<Color>) -> EventAttributesBuilder {
        self.color = Some(color.into());
        self
    }

    /// Update the builder's held [`Payload`]. See [`Payload`] for valid conversions.
    ///
    /// ```
    /// let builder = nvtx::EventAttributesBuilder::default();
    /// // ...
    /// let builder = builder.payload(3.1415926535);
    /// ```
    pub fn payload(mut self, payload: impl Into<Payload>) -> EventAttributesBuilder {
        self.payload = Some(payload.into());
        self
    }

    /// Update the the builder's held [`Message`]. See [`Message`] for valid conversions.
    ///
    /// ```
    /// let builder = nvtx::EventAttributesBuilder::default();
    /// // ...
    /// let builder = builder.message("test");
    /// ```
    pub fn message(mut self, message: impl Into<Message>) -> EventAttributesBuilder {
        self.message = Some(message.into());
        self
    }

    /// Construct an [`EventAttributes`] from the builder's held state.
    ///
    /// ```
    /// let cat = nvtx::Category::new("Category1");
    /// let attr = nvtx::EventAttributesBuilder::default()
    ///                 .message("Example Range")
    ///                 .color([224, 192, 128])
    ///                 .category(cat)
    ///                 .payload(1234567)
    ///                 .build();
    /// ```
    pub fn build(self) -> EventAttributes {
        EventAttributes {
            category: self.category,
            color: self.color,
            payload: self.payload,
            message: self.message,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use crate::{register_category, Color, EventAttributesBuilder, Message, Payload};

    #[test]
    fn test_builder_color() {
        let builder = EventAttributesBuilder::default();
        let color = Color::new(0x11, 0x22, 0x44, 0x88);
        let attr = builder.color(color).build();
        assert!(matches!(attr.color, Some(c) if c == color));
    }

    #[test]
    fn test_builder_category() {
        let builder = EventAttributesBuilder::default();
        let cat = register_category("cat");
        let attr = builder.category(cat).build();
        assert!(matches!(attr.category, Some(c) if c == cat));
    }

    #[test]
    fn test_builder_payload() {
        let builder = EventAttributesBuilder::default();
        let attr = builder.clone().payload(1_i32).build();
        assert!(matches!(attr.payload, Some(Payload::Int32(i)) if i == 1_i32));
        let attr = builder.clone().payload(2_u32).build();
        assert!(matches!(attr.payload, Some(Payload::Uint32(i)) if i == 2_u32));
        let attr = builder.clone().payload(1_i64).build();
        assert!(matches!(attr.payload, Some(Payload::Int64(i)) if i == 1_i64));
        let attr = builder.clone().payload(2_u64).build();
        assert!(matches!(attr.payload, Some(Payload::Uint64(i)) if i == 2_u64));
        let attr = builder.clone().payload(1.0_f32).build();
        assert!(matches!(attr.payload, Some(Payload::Float(i)) if i == 1.0_f32));
        let attr = builder.clone().payload(2.0_f64).build();
        assert!(matches!(attr.payload, Some(Payload::Double(i)) if i == 2.0_f64));
    }

    #[test]
    fn test_builder_message() {
        let builder = EventAttributesBuilder::default();
        let string = "This is a message";
        let attr = builder.message(string).build();
        assert!(
            matches!(attr.message, Some(Message::Unicode(s)) if s.to_string().unwrap() == string)
        );

        let builder = EventAttributesBuilder::default();
        let cstring = CString::new("This is a message").unwrap();
        let attr = builder.message(cstring.clone()).build();
        assert!(matches!(attr.message, Some(Message::Ascii(s)) if s == cstring));
    }
}
