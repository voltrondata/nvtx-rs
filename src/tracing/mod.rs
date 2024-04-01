use crate::{
    domain::{Category, EventAttributes, Message},
    Color, Domain, Payload, Str,
};
use std::marker::PhantomData;
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

/// The tracing layer for nvtx range and events.
///
/// Only a subset of nvtx domain features are available.
///
/// Unavailable functionality:
/// - registering strings for efficient reuse
/// - registering names for categories
///
/// **Supported fields**
/// * `message` (`&str`) for Marks only -- the span name is used as the message for Spans
/// * `color` (`&str`) -- the valid names align the names provided by the color names
///   defined within [`crate::color`]
/// * `payload` (one of: `f64`, `u64`, `i64`)
/// * `category` (`u32`) provides a numerical category
///
/// `instrument` example:
///
/// ```
/// use tracing::instrument;
///
/// #[instrument(fields(color = "salmon", category = 2, payload = k))]
/// fn baz (k : u64) {
///     std::thread::sleep(std::time::Duration::from_millis(10 * k));
/// }
/// ```
///
/// `mark` example:
///
/// ```
/// use tracing::info;
///
/// info!(message = "At the beginning of the program", color = "blue", category = 2);
/// ```
///
/// `span` example:
///
/// ```
/// use tracing::info_span;
///
/// let span = info_span!("Running an arbitrary block!", color = "red", payload = 3.1415);
/// span.in_scope(|| {
///    // do work inside the span...
/// });
/// ```
pub struct NvtxLayer {
    /// The held domain of the layer.
    domain: crate::Domain,
}

impl NvtxLayer {
    /// Create a new layer with a given domain name.
    pub fn new(name: impl Into<Str>) -> NvtxLayer {
        NvtxLayer {
            domain: Domain::new(name),
        }
    }

    /// Get the layer's domain.
    pub fn get_domain(&self) -> &Domain {
        &self.domain
    }
}

/// Data modeling [`EventAttributes`] without the need for lifetime management.
#[derive(Debug, Clone, Default)]
struct NvtxData {
    message: Option<String>,
    color: Option<Color>,
    category: Option<u32>,
    payload: Option<Payload>,
}

/// Wrapper around NVTX Range ids for span extension storage.
struct NvtxId(u64);

impl<S> Layer<S> for NvtxLayer
where
    S: tracing::Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut data = NvtxData::default();
        let mut visitor = NvtxVisitor::<'_, S>::new(&mut data);
        event.record(&mut visitor);
        let attr = EventAttributes {
            message: data.message.as_ref().map(|s| Message::from(s.clone())),
            category: data.category.map(|c| Category {
                id: c,
                domain: &self.domain,
            }),
            color: data.color,
            payload: data.payload,
        };
        self.domain.mark(attr);
    }

    fn on_new_span<'a>(&'a self, attrs: &Attributes<'a>, id: &Id, ctx: Context<'a, S>) {
        let span = ctx.span(id).unwrap();
        let mut data = NvtxData::default();
        let mut visitor = NvtxVisitor::<'_, S>::new(&mut data);
        attrs.record(&mut visitor);
        data.message = Some(attrs.metadata().name().to_string());
        span.extensions_mut().insert(data);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        match ctx.span(id).unwrap().extensions_mut().get_mut::<NvtxData>() {
            Some(data) => {
                let mut visitor = NvtxVisitor::<'_, S>::new(data);
                values.record(&mut visitor);
            }
            None => todo!(),
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(data) = ctx.span(id).unwrap().extensions().get::<NvtxData>() {
            let attr = EventAttributes {
                message: data.message.as_ref().map(|s| Message::from(s.clone())),
                category: data.category.map(|c| Category {
                    id: c,
                    domain: &self.domain,
                }),
                color: data.color,
                payload: data.payload,
            };
            ctx.span(id)
                .unwrap()
                .extensions_mut()
                .insert(NvtxId(self.domain.range_start(attr)));
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).unwrap();
        let maybe_id = span.extensions_mut().remove::<NvtxId>();
        if let Some(NvtxId(id)) = maybe_id {
            self.domain.range_end(id)
        }
    }
}

struct NvtxVisitor<'a, S>
where
    S: tracing::Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    data: &'a mut NvtxData,
    _traits: PhantomData<S>,
}

impl<'a, S> NvtxVisitor<'a, S>
where
    S: tracing::Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    /// Create a new NvtxVisitor given a mutable data reference.
    fn new(data: &'a mut NvtxData) -> NvtxVisitor<'a, S> {
        NvtxVisitor {
            data,
            _traits: PhantomData,
        }
    }
}

impl<'a, S> Visit for NvtxVisitor<'a, S>
where
    S: tracing::Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}

    fn record_f64(&mut self, field: &Field, value: f64) {
        if field.name() == "payload" {
            self.data.payload = Some(Payload::Double(value));
        }
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "payload" {
            self.data.payload = Some(Payload::Int64(value));
        } else if field.name() == "category" {
            self.data.category = Some(value as u32);
        }
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "payload" {
            self.data.payload = Some(Payload::Uint64(value));
        } else if field.name() == "category" {
            self.data.category = Some(value as u32);
        }
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        if field.name() == "payload" {
            self.data.payload = Some(Payload::Int32(value as i32));
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "color" {
            if let Ok([r, g, b]) = color_name::Color::val().by_string(value.to_string()) {
                self.data.color = Some(Color::new(r, g, b, 255));
            }
        } else if field.name() == "message" {
            self.data.message = Some(value.to_string());
        }
    }
}
