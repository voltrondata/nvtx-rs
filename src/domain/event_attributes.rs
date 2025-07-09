use super::{Category, Message};
use crate::{
    common::event_attributes::{
        encode_event_attributes, GenericEventAttributes, GenericEventAttributesBuilder,
    },
    Color, Domain, Payload, Str,
};

/// All attributes that are associated with marks and ranges.
pub type EventAttributes<'a> = GenericEventAttributes<Category<'a>, Message<'a>>;

impl EventAttributes<'_> {
    pub(super) fn encode(&self) -> nvtx_sys::EventAttributes {
        encode_event_attributes(&self.category, &self.color, &self.message, &self.payload)
    }
}

impl<'a, T: Into<Message<'a>>> From<T> for EventAttributes<'a> {
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
/// let domain = nvtx::Domain::new("Domain");
/// let cat = domain.register_category("Category1");
///
/// let attr = domain.event_attributes_builder()
///                .category(cat)
///                .color([20, 192, 240])
///                .payload(3.141592)
///                .message("Hello")
///                .build();
/// ```
#[derive(Clone)]
pub struct EventAttributesBuilder<'a> {
    pub(super) domain: &'a Domain,
    pub(super) inner: GenericEventAttributesBuilder<Category<'a>, Message<'a>>,
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
        self.inner = self.inner.category(category);
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
        let category = self.domain.register_category(name);
        self.inner = self.inner.category(category);
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
        self.inner = self.inner.color(color);
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
        self.inner = self.inner.payload(payload);
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
        self.inner = self.inner.message(Message::Registered(msg));
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
        self.inner.build()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        domain::{Domain, Message},
        Color, Payload,
    };

    #[test]
    fn test_builder_color() {
        let d = Domain::new("d");
        let builder = d.event_attributes_builder();
        let color = Color::new(0x11, 0x22, 0x44, 0x88);
        let attr = builder.color(color).build();
        assert!(matches!(attr.color, Some(c) if c == color));
    }

    #[test]
    fn test_builder_category() {
        let d = Domain::new("d");
        let cat = d.register_category("cat");
        let builder = d.event_attributes_builder();
        let attr = builder.category(cat).build();
        assert!(matches!(attr.category, Some(c) if c == cat));
    }

    #[test]
    fn test_builder_category_name() {
        let d = Domain::new("d");
        let builder = d.event_attributes_builder();
        let attr = builder.category_name("cat").build();
        let cat = d.register_category("cat");
        assert!(matches!(attr.category, Some(c) if c == cat));
    }

    #[test]
    fn test_builder_payload() {
        let d = Domain::new("d");
        let builder = d.event_attributes_builder();
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
        let d = Domain::new("d");
        let builder = d.event_attributes_builder();
        let attr = builder.message("This is a message").build();
        let registered = d.register_string("This is a message");
        assert!(matches!(attr.message, Some(Message::Registered(r)) if r == registered));
    }

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

    #[test]
    #[should_panic(expected = "EventAttributes' Domain does not match current Domain")]
    fn test_simple_domain_validation() {
        let d1 = Domain::new("Domain1");
        let s1 = d1.register_string("test string");
        let d2 = Domain::new("Domain2");
        // Create attributes with a string from d1, then use them with d2
        let attr = d1.event_attributes_builder().message(s1).build();
        d2.mark(attr);
    }
}
