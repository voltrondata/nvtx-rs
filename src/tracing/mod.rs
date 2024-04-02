use crate::{Color, Domain, Payload};
use std::{collections::HashMap, marker::PhantomData, sync::Mutex};
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

/// The tracing layer for nvtx range and events.
///
/// **Supported fields**
/// * `domain` (`&str`) indicates a domain name (the default is `"NVTX"`)
/// * `category` (`&str`) provides a category within a domain (or the default domain)
/// * `color` (`&str`) -- the valid names align the names provided by the color names
///   defined within [`crate::color`]
/// * `payload` (one of: `f64`, `u64`, `i64`, or `bool`)
/// * `message` (`&str`) for Marks only -- the span name is used as the message for Spans
///
/// `instrument` example:
///
/// ```
/// use tracing::instrument;
///
/// #[instrument(fields(color = "salmon", category = "cool", payload = k))]
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
/// info!(message = "At the beginning of the program", color = "blue", domain = "test");
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
    /// The held domains of the layer.
    domains: Mutex<HashMap<String, crate::Domain>>,
}

static DEFAULT_DOMAIN_NAME: &str = "NVTX";

impl Default for NvtxLayer {
    /// Create a new layer with a given domain name.
    fn default() -> NvtxLayer {
        NvtxLayer {
            domains: Mutex::new(HashMap::from([(
                DEFAULT_DOMAIN_NAME.to_string(),
                Domain::new(DEFAULT_DOMAIN_NAME),
            )])),
        }
    }
}

/// Data modeling [`EventAttributes`] without the need for lifetime management.
#[derive(Debug, Clone, Default)]
struct NvtxData {
    domain: Option<String>,
    category: Option<String>,
    message: Option<String>,
    color: Option<Color>,
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
        let domain_name = data
            .domain
            .unwrap_or_else(|| DEFAULT_DOMAIN_NAME.to_string());
        let mut lock = self.domains.lock().unwrap();
        let domain = lock
            .entry(domain_name.clone())
            .or_insert_with(|| Domain::new(domain_name));
        let mut builder = domain.event_attributes_builder();
        if let Some(c) = data.category {
            builder = builder.category_name(c);
        }
        if let Some(s) = data.message {
            builder = builder.message(s);
        }
        if let Some(c) = data.color {
            builder = builder.color(c);
        }
        if let Some(p) = data.payload {
            builder = builder.payload(p);
        }
        domain.mark(builder.build());
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
        let mut range_id: Option<u64> = None;
        if let Some(data) = ctx.span(id).unwrap().extensions().get::<NvtxData>() {
            let domain_name = &data
                .domain
                .clone()
                .unwrap_or_else(|| DEFAULT_DOMAIN_NAME.to_string());
            let mut lock = self.domains.lock().unwrap();
            let domain = lock
                .entry(domain_name.clone())
                .or_insert_with(|| Domain::new(domain_name.to_string()));
            let mut builder = domain.event_attributes_builder();
            if let Some(c) = &data.category {
                builder = builder.category_name(c.to_string());
            }
            if let Some(s) = &data.message {
                builder = builder.message(s.to_string());
            }
            if let Some(c) = data.color {
                builder = builder.color(c);
            }
            if let Some(p) = data.payload {
                builder = builder.payload(p);
            }
            range_id = Some(domain.range_start(builder.build()));
        };
        if let Some(range) = range_id {
            ctx.span(id).unwrap().extensions_mut().insert(NvtxId(range));
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).unwrap();
        let maybe_data = span.extensions_mut().remove::<NvtxData>();
        let domain_name = maybe_data
            .map(|data| data.domain)
            .unwrap()
            .unwrap_or_else(|| DEFAULT_DOMAIN_NAME.to_string());
        let mut lock = self.domains.lock().unwrap();
        let domain = lock
            .entry(domain_name.clone())
            .or_insert_with(|| Domain::new(domain_name));

        let maybe_id = span.extensions_mut().remove::<NvtxId>();
        if let Some(NvtxId(id)) = maybe_id {
            domain.range_end(id)
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
        }
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "payload" {
            self.data.payload = Some(Payload::Uint64(value));
        }
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        if field.name() == "payload" {
            self.data.payload = Some(Payload::Int32(value as i32));
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        let owned = value.to_string();
        match field.name() {
            "color" => {
                if let Ok([r, g, b]) = color_name::Color::val().by_string(owned) {
                    self.data.color = Some(Color::new(r, g, b, 255));
                }
            }
            "message" => {
                self.data.message = Some(owned);
            }
            "domain" => {
                self.data.domain = Some(owned);
            }
            "category" => {
                self.data.category = Some(owned);
            }
            _ => (),
        }
    }
}
