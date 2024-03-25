use crate::{Color, Payload, TypeValueEncodable};

use super::{Domain, category::Category, message::Message};


/// All attributes that are associated with marks and ranges
#[derive(Debug, Clone)]
pub struct EventAttributes<'a> {
    pub(super) category: Option<Category<'a>>,
    pub(super) color: Option<Color>,
    pub(super) payload: Option<Payload>,
    pub(super) message: Option<Message<'a>>,
}

impl<'a> EventAttributes<'a> {
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
#[derive(Debug, Clone)]
pub struct EventAttributesBuilder<'a> {
    pub(super) domain: &'a Domain,
    pub(super)category: Option<&'a Category<'a>>,
    pub(super)color: Option<Color>,
    pub(super)payload: Option<Payload>,
    pub(super)message: Option<Message<'a>>,
}

impl<'a> EventAttributesBuilder<'a> {
    /// Update the attribute's category. An assertion will be thrown if a Category is passed in whose domain is not the same as this builder
    pub fn category(mut self, category: &'a Category<'a>) -> EventAttributesBuilder<'a> {
        assert!(
            std::ptr::eq(category.domain, self.domain),
            "Builder's Domain differs from Category's Domain"
        );
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

    /// Update the attribute's message. An assertion will be thrown if a RegisteredString is passed in whose domain is not the same as this builder
    pub fn message(mut self, message: impl Into<Message<'a>>) -> EventAttributesBuilder<'a> {
        let msg: Message = message.into();
        if let Message::Registered(r) = &msg {
            assert!(
                std::ptr::eq(r.domain, self.domain),
                "Builder's Domain differs from domain::RegisteredString's Domain"
            )
        }
        self.message = Some(msg);
        self
    }

    /// build the attribute from the builder's state
    pub fn build(self) -> EventAttributes<'a> {
        EventAttributes {
            category: self.category.copied(),
            color: self.color,
            payload: self.payload,
            message: self.message,
        }
    }
}
