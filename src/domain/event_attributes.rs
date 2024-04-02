use super::{Category, Message};
use crate::{Color, Domain, Payload, Str, TypeValueEncodable};

/// All attributes that are associated with marks and ranges.
#[derive(Debug, Clone)]
pub struct EventAttributes<'a> {
    pub(super) domain: Option<&'a Domain>,
    pub(super) category: Option<Category<'a>>,
    pub(super) color: Option<Color>,
    pub(super) payload: Option<Payload>,
    pub(super) message: Option<Message<'a>>,
}

impl<'a> EventAttributes<'a> {
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
        let cat = self.category.as_ref().map(Category::id).unwrap_or(0);
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

impl<'a, T: Into<Message<'a>>> From<T> for EventAttributes<'a> {
    fn from(value: T) -> Self {
        EventAttributes {
            domain: None,
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
#[derive(Debug, Clone)]
pub struct EventAttributesBuilder<'a> {
    pub(super) domain: &'a Domain,
    pub(super) category: Option<Category<'a>>,
    pub(super) color: Option<Color>,
    pub(super) payload: Option<Payload>,
    pub(super) message: Option<Message<'a>>,
}

impl<'a> EventAttributesBuilder<'a> {
    /// Update the attribute's category. An assertion will be thrown if a Category is
    /// passed in whose domain is not the same as this builder.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// let cat = domain.register_category("Category1");
    /// // ...
    /// let builder = domain.event_attributes_builder();
    /// // ...
    /// let builder = builder.category(cat);
    /// ```
    pub fn category(mut self, category: Category<'a>) -> EventAttributesBuilder<'a> {
        assert!(
            std::ptr::eq(category.domain(), self.domain),
            "EventAttributesBuilder's Domain differs from Category's Domain"
        );
        self.category = Some(category);
        self
    }
    /// Update the attribute's category. An assertion will be thrown if a Category is
    /// passed in whose domain is not the same as this builder.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// let builder = domain.event_attributes_builder();
    /// // ...
    /// let builder = builder.category_name("Category2");
    /// ```
    pub fn category_name(mut self, name: impl Into<Str>) -> EventAttributesBuilder<'a> {
        self.category = Some(self.domain.register_category(name));
        self
    }

    /// Update the builder's held [`Color`]. See [`Color`] for valid conversions.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// let builder = domain.event_attributes_builder();
    /// // ...
    /// let builder = builder.color([255, 255, 255]);
    /// ```
    pub fn color(mut self, color: impl Into<Color>) -> EventAttributesBuilder<'a> {
        self.color = Some(color.into());
        self
    }

    /// Update the builder's held [`Payload`]. See [`Payload`] for valid conversions.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// let builder = domain.event_attributes_builder();
    /// // ...
    /// let builder = builder.payload(3.1415926535);
    /// ```
    pub fn payload(mut self, payload: impl Into<Payload>) -> EventAttributesBuilder<'a> {
        self.payload = Some(payload.into());
        self
    }

    /// Update the attribute's message. An assertion will be thrown if a
    /// [`super::RegisteredString`] is passed in whose domain is not the same as this
    /// builder.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// let builder = domain.event_attributes_builder();
    /// // ...
    /// let builder = builder.message("test");
    /// ```
    pub fn message(mut self, message: impl Into<Message<'a>>) -> EventAttributesBuilder<'a> {
        // implementation optimization: always prefer registered strings
        let msg = match message.into() {
            Message::Ascii(s) => self.domain.register_string(s.to_str().unwrap().to_string()),
            Message::Unicode(s) => self.domain.register_string(s.to_string().unwrap()),
            Message::Registered(r) => r,
        };
        assert!(
            std::ptr::eq(msg.domain(), self.domain),
            "EventAttributesBuilder's Domain differs from RegisteredString's Domain"
        );
        self.message = Some(Message::Registered(msg));
        self
    }

    /// Construct an [`EventAttributes`] from the builder's held state.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// let cat = domain.register_category("Category1");
    /// let attr = domain.event_attributes_builder()
    ///                 .message("Example Range")
    ///                 .color([224, 192, 128])
    ///                 .category(cat)
    ///                 .payload(1234567)
    ///                 .build();
    /// ```
    pub fn build(self) -> EventAttributes<'a> {
        EventAttributes {
            domain: Some(self.domain),
            category: self.category,
            color: self.color,
            payload: self.payload,
            message: self.message,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::domain::Domain;

    #[test]
    #[should_panic(expected = "EventAttributes' Domain does not match current Domain")]
    fn test_unowned_category_panic_mark() {
        let d1 = Domain::new("Domain1");
        let c1 = d1.register_category("category");
        let d2 = Domain::new("Domain2");
        d2.mark(d1.event_attributes_builder().category(c1).build());
    }

    #[test]
    #[should_panic(expected = "EventAttributesBuilder's Domain differs from Category's Domain")]
    fn test_unowned_category_panic_in_builder_mark() {
        let d1 = Domain::new("Domain1");
        let c1 = d1.register_category("category");
        let d2 = Domain::new("Domain2");
        d2.mark(d2.event_attributes_builder().category(c1).build());
    }

    #[test]
    #[should_panic(expected = "EventAttributes' Domain does not match current Domain")]
    fn test_unowned_category_panic_range() {
        let d1 = Domain::new("Domain1");
        let c1 = d1.register_category("category");
        let d2 = Domain::new("Domain2");
        d2.range(d1.event_attributes_builder().category(c1).build());
    }

    #[test]
    #[should_panic(expected = "EventAttributesBuilder's Domain differs from Category's Domain")]
    fn test_unowned_category_panic_in_builder_range() {
        let d1 = Domain::new("Domain1");
        let c1 = d1.register_category("category");
        let d2 = Domain::new("Domain2");
        d2.range(d2.event_attributes_builder().category(c1).build());
    }

    #[test]
    #[should_panic(expected = "EventAttributes' Domain does not match current Domain")]
    fn test_unowned_string_panic_mark() {
        let d1 = Domain::new("Domain1");
        let s1 = d1.register_string("test string");
        let d2 = Domain::new("Domain2");
        d2.mark(d1.event_attributes_builder().message(s1).build());
    }

    #[test]
    #[should_panic(
        expected = "EventAttributesBuilder's Domain differs from RegisteredString's Domain"
    )]
    fn test_unowned_string_panic_in_builder_mark() {
        let d1 = Domain::new("Domain1");
        let s1 = d1.register_string("test string");
        let d2 = Domain::new("Domain2");
        d2.mark(d2.event_attributes_builder().message(s1).build());
    }

    #[test]
    #[should_panic(expected = "EventAttributes' Domain does not match current Domain")]
    fn test_unowned_string_panic_range() {
        let d1 = Domain::new("Domain1");
        let s1 = d1.register_string("test string");
        let d2 = Domain::new("Domain2");
        d2.range(d1.event_attributes_builder().message(s1).build());
    }

    #[test]
    #[should_panic(
        expected = "EventAttributesBuilder's Domain differs from RegisteredString's Domain"
    )]
    fn test_unowned_string_panic_in_builder_range() {
        let d1 = Domain::new("Domain1");
        let s1 = d1.register_string("test string");
        let d2 = Domain::new("Domain2");
        d2.range(d2.event_attributes_builder().message(s1).build());
    }
}
