use crate::{Color, Payload, TypeValueEncodable};
use derive_builder::Builder;

/// Generic event attributes that can be used in both global and domain contexts.
///
/// This struct holds optional fields for category, color, message, and payload.
/// It is used to describe the attributes of an NVTX event.
#[derive(Default, Debug, Clone, Builder)]
#[builder(pattern = "owned", derive(Clone), build_fn(skip))]
pub struct GenericEventAttributes<C, M> {
    #[builder(setter(into, strip_option))]
    pub category: Option<C>,
    #[builder(setter(into, strip_option))]
    pub color: Option<Color>,
    #[builder(setter(into, strip_option))]
    pub message: Option<M>,
    #[builder(setter(into, strip_option))]
    pub payload: Option<Payload>,
}

impl<C, M> GenericEventAttributesBuilder<C, M> {
    /// Build the event attributes, allowing all fields to be optional.
    pub fn build(self) -> GenericEventAttributes<C, M> {
        GenericEventAttributes {
            category: self.category.flatten(),
            color: self.color.flatten(),
            message: self.message.flatten(),
            payload: self.payload.flatten(),
        }
    }
}

/// Common encoding implementation for event attributes.
///
/// This function encodes the event attributes into the NVTX FFI struct.
/// Used internally by both global and domain event attributes.
///
/// # Parameters
/// - `category`: Optional category value
/// - `color`: Optional color value
/// - `message`: Optional message value
/// - `payload`: Optional payload value
pub fn encode_event_attributes<C, M>(
    category: &Option<C>,
    color: &Option<Color>,
    message: &Option<M>,
    payload: &Option<Payload>,
) -> nvtx_sys::EventAttributes
where
    C: CategoryEncodable,
    M: TypeValueEncodable<Type = nvtx_sys::MessageType, Value = nvtx_sys::MessageValue>,
{
    let (color_type, color_value) = color
        .as_ref()
        .map(Color::encode)
        .unwrap_or_else(Color::default_encoding);
    let (payload_type, payload_value) = payload
        .as_ref()
        .map(Payload::encode)
        .unwrap_or_else(Payload::default_encoding);
    let cat = category
        .as_ref()
        .map(CategoryEncodable::encode_id)
        .unwrap_or(0);
    let (message_type, message_value) = message
        .as_ref()
        .map(M::encode)
        .unwrap_or_else(M::default_encoding);

    nvtx_sys::EventAttributes {
        version: nvtx_sys::NVTX_VERSION as u16,
        size: 48,
        category: cat,
        colorType: color_type as i32,
        color: color_value,
        payloadType: payload_type as i32,
        reserved0: 0,
        payload: payload_value,
        messageType: message_type as i32,
        message: message_value,
    }
}

/// Trait for encoding category IDs.
///
/// Used to abstract over global and domain category types.
pub trait CategoryEncodable {
    /// Encode the category as a u32 ID for NVTX.
    fn encode_id(&self) -> u32;
}
