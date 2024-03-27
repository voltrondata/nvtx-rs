use crate::TypeValueEncodable;

#[cfg(feature = "color-name")]
pub use color_name::colors::*;

/// Represents a color in use for controlling appearance within NSight Systems
///
/// ```
/// // creation of a color from specific channel values:
/// let translucent_orange = nvtx::Color::new(255, 192, 0, 128);
///
/// // creation of a color from pre-specified names:
/// #[cfg(feature = "color-name")]
/// let salmon : nvtx::Color = nvtx::color::salmon.into();
///
/// // modification of a color after creation:
///
/// let opaque_orange = translucent_orange.with_alpha(255);
///
/// #[cfg(feature = "color-name")]
/// let translucent_salmon = salmon.with_alpha(128);
///
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
    ///
    /// ```
    /// let nice_blue = nvtx::Color::new(0, 192, 255, 255);
    /// ```
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { a, r, g, b }
    }

    /// Change the alpha channel for the color and yield a new color
    ///
    /// ```
    /// let nice_blue = nvtx::Color::new(0, 192, 255, 255);
    /// let translucent_blue = nice_blue.with_alpha(128);
    /// ```
    pub fn with_alpha(&self, a: u8) -> Self {
        Color { a, ..*self }
    }

    /// Change the red channel for the color and yield a new color
    ///
    /// ```
    /// let dark_gray = nvtx::Color::new(32, 32, 32, 255);
    /// let dark_red = dark_gray.with_red(128);
    /// ```
    pub fn with_red(&self, r: u8) -> Self {
        Color { r, ..*self }
    }

    /// Change the green channel for the color and yield a new color
    ///
    /// ```
    /// let dark_gray = nvtx::Color::new(32, 32, 32, 255);
    /// let dark_green = dark_gray.with_green(128);
    /// ```
    pub fn with_green(&self, g: u8) -> Self {
        Color { g, ..*self }
    }

    /// Change the blue channel for the color and yield a new color
    ///
    /// ```
    /// let dark_gray = nvtx::Color::new(32, 32, 32, 255);
    /// let dark_blue = dark_gray.with_blue(128);
    /// ```
    pub fn with_blue(&self, b: u8) -> Self {
        Color { b, ..*self }
    }
}

impl TypeValueEncodable for Color {
    type Type = nvtx_sys::ffi::nvtxColorType_t;
    type Value = u32;

    fn encode(&self) -> (Self::Type, Self::Value) {
        let as_u32 = u32::from_ne_bytes([self.a, self.r, self.g, self.b]);
        (nvtx_sys::ffi::nvtxColorType_t::NVTX_COLOR_ARGB, as_u32)
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (nvtx_sys::ffi::nvtxColorType_t::NVTX_COLOR_UNKNOWN, 0)
    }
}
