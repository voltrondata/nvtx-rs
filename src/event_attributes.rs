use crate::{Category, Color, Message, Payload, TypeValueEncodable};

/// All attributes that are associated with marks and ranges
#[derive(Debug, Clone)]
pub struct EventAttributes {
    pub(super) category: Option<Category>,
    pub(super) color: Option<Color>,
    pub(super) payload: Option<Payload>,
    pub(super) message: Option<Message>,
}

impl EventAttributes {
    pub(super) fn encode(&self) -> nvtx_sys::ffi::nvtxEventAttributes_t {
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


/// Builder to facilitate easier construction of [`EventAttributes`]
#[derive(Debug, Clone, Default)]
pub struct EventAttributesBuilder<'a> {
    pub(super) category: Option<&'a Category>,
    pub(super) color: Option<Color>,
    pub(super) payload: Option<Payload>,
    pub(super) message: Option<Message>,
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
