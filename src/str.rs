use std::ffi::{CStr, CString};
use widestring::WideCString;

/// A convenience wrapper for various string types
///
/// * [`Str::Ascii`] is the discriminator for C string types
/// * [`Str::Unicode`] is the discriminator for Rust string types
#[derive(Debug, Clone)]
pub enum Str {
    /// Represents an ASCII friendly string
    Ascii(CString),
    /// Represents a Unicode string
    Unicode(WideCString),
}

impl From<String> for Str {
    fn from(v: String) -> Self {
        Self::Unicode(WideCString::from_str(v.as_str()).expect("Could not convert to wide string"))
    }
}

impl From<&str> for Str {
    fn from(v: &str) -> Self {
        Self::Unicode(WideCString::from_str(v).expect("Could not convert to wide string"))
    }
}

impl From<CString> for Str {
    fn from(v: CString) -> Self {
        Self::Ascii(v)
    }
}

impl From<&CStr> for Str {
    fn from(v: &CStr) -> Self {
        Self::Ascii(CString::from(v))
    }
}
