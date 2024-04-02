use crate::TypeValueEncodable;

/// Represents a payload value for use within [`crate::EventAttributes`] and
/// [`crate::domain::EventAttributes`].
///
/// * [`Payload::Float`] holds a 32-bit floating-point playload
/// * [`Payload::Double`] holds a 64-bit floating-point playload
/// * [`Payload::Int32`] holds a 32-bit integral playload
/// * [`Payload::Int64`] holds a 64-bit integral playload
/// * [`Payload::Uint32`] holds a 32-bit unsigned integral playload
/// * [`Payload::Uint64`] holds a 64-bit unsigned integral playload
#[derive(Debug, Clone, Copy)]
pub enum Payload {
    /// A 32-bit floating-point value.
    Float(f32),
    /// A 64-bit floating-point value.
    Double(f64),
    /// A 32-bit integral value.
    Int32(i32),
    /// A 64-bit integral value.
    Int64(i64),
    /// A 32-bit unsigned integral value.
    Uint32(u32),
    /// A 64-bit unsigned integral value.
    Uint64(u64),
}

impl From<u64> for Payload {
    fn from(v: u64) -> Self {
        Self::Uint64(v)
    }
}

impl From<u32> for Payload {
    fn from(v: u32) -> Self {
        Self::Uint32(v)
    }
}

impl From<i64> for Payload {
    fn from(v: i64) -> Self {
        Self::Int64(v)
    }
}

impl From<i32> for Payload {
    fn from(v: i32) -> Self {
        Self::Int32(v)
    }
}

impl From<f64> for Payload {
    fn from(v: f64) -> Self {
        Self::Double(v)
    }
}

impl From<f32> for Payload {
    fn from(v: f32) -> Self {
        Self::Float(v)
    }
}

impl TypeValueEncodable for Payload {
    type Type = nvtx_sys::PayloadType;
    type Value = nvtx_sys::PayloadValue;
    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            Payload::Float(x) => (
                Self::Type::NVTX_PAYLOAD_TYPE_FLOAT,
                Self::Value { fValue: *x },
            ),
            Payload::Double(x) => (
                Self::Type::NVTX_PAYLOAD_TYPE_DOUBLE,
                Self::Value { dValue: *x },
            ),
            Payload::Int32(x) => (
                Self::Type::NVTX_PAYLOAD_TYPE_INT32,
                Self::Value { iValue: *x },
            ),
            Payload::Int64(x) => (
                Self::Type::NVTX_PAYLOAD_TYPE_INT64,
                Self::Value { llValue: *x },
            ),
            Payload::Uint32(x) => (
                Self::Type::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
                Self::Value { uiValue: *x },
            ),
            Payload::Uint64(x) => (
                Self::Type::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
                Self::Value { ullValue: *x },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (
            Self::Type::NVTX_PAYLOAD_UNKNOWN,
            Self::Value { ullValue: 0 },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::TypeValueEncodable;

    use super::Payload;

    #[test]
    fn test_encode_i32() {
        let p = Payload::Int32(std::i32::MAX);
        let (t, v) = p.encode();
        assert_eq!(t, nvtx_sys::PayloadType::NVTX_PAYLOAD_TYPE_INT32);
        unsafe { assert!(matches!(v, nvtx_sys::PayloadValue { iValue: v } if v == std::i32::MAX)) };
    }

    #[test]
    fn test_encode_u32() {
        let p = Payload::Uint32(std::u32::MAX);
        let (t, v) = p.encode();
        assert_eq!(t, nvtx_sys::PayloadType::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32);
        unsafe {
            assert!(matches!(v, nvtx_sys::PayloadValue { uiValue: v } if v == std::u32::MAX))
        };
    }

    #[test]
    fn test_encode_i64() {
        let p = Payload::Int64(std::i64::MAX);
        let (t, v) = p.encode();
        assert_eq!(t, nvtx_sys::PayloadType::NVTX_PAYLOAD_TYPE_INT64);
        unsafe {
            assert!(matches!(v, nvtx_sys::PayloadValue { llValue: v } if v == std::i64::MAX))
        };
    }

    #[test]
    fn test_encode_u64() {
        let p = Payload::Uint64(std::u64::MAX);
        let (t, v) = p.encode();
        assert_eq!(t, nvtx_sys::PayloadType::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64);
        unsafe {
            assert!(matches!(v, nvtx_sys::PayloadValue { ullValue: v } if v == std::u64::MAX))
        };
    }

    #[test]
    fn test_encode_float() {
        let p = Payload::Float(std::f32::consts::PI);
        let (t, v) = p.encode();
        assert_eq!(t, nvtx_sys::PayloadType::NVTX_PAYLOAD_TYPE_FLOAT);
        unsafe {
            assert!(matches!(v, nvtx_sys::PayloadValue { fValue: v } if v == std::f32::consts::PI))
        };
    }

    #[test]
    fn test_encode_double() {
        let p = Payload::Double(std::f64::consts::E);
        let (t, v) = p.encode();
        assert_eq!(t, nvtx_sys::PayloadType::NVTX_PAYLOAD_TYPE_DOUBLE);
        unsafe {
            assert!(matches!(v, nvtx_sys::PayloadValue { dValue: v } if v == std::f64::consts::E))
        };
    }

    #[test]
    fn test_encode_defaults() {
        let (t, v) = Payload::default_encoding();
        assert_eq!(t, nvtx_sys::PayloadType::NVTX_PAYLOAD_UNKNOWN);
        unsafe { assert!(matches!(v, nvtx_sys::PayloadValue { ullValue: v } if v == 0)) };
    }
}
