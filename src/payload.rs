use crate::TypeValueEncodable;

/// Represents a payload value for use within event attributes
#[derive(Debug, Clone, Copy)]
pub enum Payload {
    /// the payload shall hold a 32-bit floating-point value
    Float(f32),
    /// the payload shall hold a 64-bit floating-point value
    Double(f64),
    /// the payload shall hold a 32-bit integral value
    Int32(i32),
    /// the payload shall hold a 64-bit integral value
    Int64(i64),
    /// the payload shall hold a 32-bit unsigned integral value
    Uint32(u32),
    /// the payload shall hold a 64-bit unsigned integral value
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
    type Type = nvtx_sys::ffi::nvtxPayloadType_t;
    type Value = nvtx_sys::ffi::nvtxEventAttributes_v2_payload_t;
    fn encode(&self) -> (Self::Type, Self::Value) {
        match self {
            Payload::Float(x) => (
                nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_FLOAT,
                Self::Value { fValue: *x },
            ),
            Payload::Double(x) => (
                nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_DOUBLE,
                Self::Value { dValue: *x },
            ),
            Payload::Int32(x) => (
                nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT32,
                Self::Value { iValue: *x },
            ),
            Payload::Int64(x) => (
                nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT64,
                Self::Value { llValue: *x },
            ),
            Payload::Uint32(x) => (
                nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
                Self::Value { uiValue: *x },
            ),
            Payload::Uint64(x) => (
                nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
                Self::Value { ullValue: *x },
            ),
        }
    }

    fn default_encoding() -> (Self::Type, Self::Value) {
        (
            nvtx_sys::ffi::nvtxPayloadType_t::NVTX_PAYLOAD_UNKNOWN,
            Self::Value { ullValue: 0 },
        )
    }
}
