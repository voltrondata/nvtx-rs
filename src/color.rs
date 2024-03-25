use crate::TypeValueEncodable;
pub use color_name::colors::*;

/// Represents a color in use for controlling appearance within NSight Systems
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
    /// Create a new color from specified channels
    pub fn new(a: u8, r: u8, g: u8, b: u8) -> Self {
        Self { a, r, g, b }
    }

    /// Change the transparency channel for the color
    pub fn transparency(mut self, value: u8) -> Self {
        self.a = value;
        self
    }

    /// Change the red channel for the color
    pub fn red(mut self, value: u8) -> Self {
        self.r = value;
        self
    }

    /// Change the green channel for the color
    pub fn green(mut self, value: u8) -> Self {
        self.g = value;
        self
    }

    /// Change the blue channel for the color
    pub fn blue(mut self, value: u8) -> Self {
        self.b = value;
        self
    }
}

impl TypeValueEncodable for Color {
    type Type = nvtx_sys::ffi::nvtxColorType_t;
    type Value = u32;

    fn encode(&self) -> (Self::Type, Self::Value) {
        let as_u32 =
            (self.a as u32) << 24 | (self.r as u32) << 16 | (self.g as u32) << 8 | (self.b as u32);
        (nvtx_sys::ffi::nvtxColorType_t::NVTX_COLOR_ARGB, as_u32)
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (nvtx_sys::ffi::nvtxColorType_t::NVTX_COLOR_UNKNOWN, 0)
    }
}
