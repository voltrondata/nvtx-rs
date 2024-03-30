use crate::{Str, TypeValueEncodable};
use std::{
    marker::PhantomData,
    sync::atomic::{AtomicU32, Ordering},
};

mod category;
pub use category::Category;

mod event_argument;
pub use event_argument::EventArgument;

mod event_attributes;
pub use event_attributes::{EventAttributes, EventAttributesBuilder};

mod identifier;
#[cfg(feature = "cuda")]
pub use self::identifier::CudaIdentifier;
#[cfg(feature = "cuda_runtime")]
pub use self::identifier::CudaRuntimeIdentifier;
#[cfg(target_family = "unix")]
pub use self::identifier::PThreadIdentifier;
pub use identifier::{GenericIdentifier, Identifier};

mod local_range;
pub use local_range::LocalRange;

mod message;
pub use message::Message;

mod range;
pub use range::Range;

mod registered_string;
pub use registered_string::RegisteredString;

mod resource;
pub use resource::Resource;

/// user-defined synchronization objects
pub mod sync;

/// Represents a domain for high-level grouping within NSight profilers.
#[derive(Debug)]
pub struct Domain {
    handle: nvtx_sys::DomainHandle,
    registered_categories: AtomicU32,
}

impl Domain {
    /// Register a NVTX domain.
    ///
    /// See [`Str`] for valid conversions.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// ```
    pub fn new(name: impl Into<Str>) -> Self {
        Domain {
            handle: match &name.into() {
                Str::Ascii(s) => nvtx_sys::domain_create_ascii(s),
                Str::Unicode(s) => nvtx_sys::domain_create_unicode(s),
            },
            registered_categories: AtomicU32::new(0),
        }
    }

    /// Gets a new builder instance for event attributes in the current domain.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// let builder = domain.event_attributes_builder();
    /// ```
    pub fn event_attributes_builder(&self) -> EventAttributesBuilder<'_> {
        EventAttributesBuilder {
            domain: self,
            category: None,
            color: None,
            payload: None,
            message: None,
        }
    }

    /// Registers an immutable string within the current domain.
    ///
    /// Returns a handle to the immutable string registered to NVTX.
    ///
    /// See [`Str`] for valid conversions.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// let my_str = domain.register_string("My immutable string");
    /// ```
    pub fn register_string(&self, string: impl Into<Str>) -> RegisteredString<'_> {
        let handle = match &string.into() {
            Str::Ascii(s) => nvtx_sys::domain_register_string_ascii(self.handle, s),
            Str::Unicode(s) => nvtx_sys::domain_register_string_unicode(self.handle, s),
        };
        RegisteredString {
            handle,
            domain: self,
        }
    }

    /// Register many immutable strings within the current domain.
    ///
    /// Returns an array of handles to the immutable strings registered to NVTX.
    ///
    /// See [`Str`] for valid conversions.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// let [a, b, c] = domain.register_strings(["A", "B", "C"]);
    /// ```
    pub fn register_strings<const N: usize>(
        &self,
        strings: [impl Into<Str>; N],
    ) -> [RegisteredString<'_>; N] {
        strings.map(|string| self.register_string(string))
    }

    /// Register a new category within the domain.
    ///
    /// Returns a handle to the category registered to NVTX.
    ///
    /// See [`Str`] for valid conversions.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// let cat = domain.register_category("Category");
    /// ```
    pub fn register_category(&self, name: impl Into<Str>) -> Category<'_> {
        let id = 1 + self.registered_categories.fetch_add(1, Ordering::SeqCst);
        match &name.into() {
            Str::Ascii(s) => nvtx_sys::domain_name_category_ascii(self.handle, id, s),
            Str::Unicode(s) => nvtx_sys::domain_name_category_unicode(self.handle, id, s),
        }
        Category { id, domain: self }
    }

    /// Register new categories within the domain.
    ///
    /// Returns an array of handles to the categories registered to NVTX.
    ///
    /// See [`Str`] for valid conversions.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// let [cat_a, cat_b] = domain.register_categories(["CatA", "CatB"]);
    /// ```
    pub fn register_categories<const N: usize>(
        &self,
        names: [impl Into<Str>; N],
    ) -> [Category<'_>; N] {
        names.map(|name| self.register_category(name))
    }

    /// Marks an instantaneous event in the application belonging to a domain.
    ///
    /// A marker can contain a text message or specify information using the event
    /// attributes structure. These attributes include a text message, color, category,
    /// and a payload. Each of the attributes is optional.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// domain.mark("Sample mark");
    ///
    /// domain.mark(c"Another example");
    ///
    /// domain.mark(
    ///   domain.event_attributes_builder()
    ///     .message("Interesting example")
    ///     .color([255, 0, 0])
    ///     .build());
    ///
    /// let reg_str = domain.register_string("Registered String");
    /// domain.mark(&reg_str);
    /// ```
    pub fn mark<'a>(&'a self, arg: impl Into<EventArgument<'a>>) {
        let attribute: EventAttributes<'a> = match arg.into() {
            EventArgument::Attributes(attr) => attr,
            EventArgument::Message(m) => m.into(),
        };
        let encoded = attribute.encode();
        nvtx_sys::domain_mark_ex(self.handle, &encoded)
    }

    /// Create an RAII-friendly, domain-owned range type which (1) cannot be moved across
    /// thread boundaries and (2) automatically ended when dropped.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    ///
    /// // creation from Rust string
    /// let range = domain.local_range("simple name");
    ///
    /// // creation from C string (since 1.77)
    /// let range = domain.local_range(c"simple name");
    ///
    /// // creation from EventAttributes
    /// let attr = domain
    ///     .event_attributes_builder()
    ///     .payload(1)
    ///     .message("complex range")
    ///     .build();
    /// let range = domain.local_range(attr);
    ///
    /// // explicitly end a range
    /// drop(range)
    /// ```
    pub fn local_range<'a>(&'a self, arg: impl Into<EventArgument<'a>>) -> LocalRange<'a> {
        LocalRange::new(arg, self)
    }

    /// Create an RAII-friendly, domain-owned range type which (1) can be moved across
    /// thread boundaries and (2) automatically ended when dropped.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    ///
    /// // creation from a unicode string
    /// let range = domain.range("simple name");
    ///
    /// // creation from a c string (from rust 1.77+)
    /// let range = domain.range(c"simple name");
    ///
    /// // creation from EventAttributes
    /// let attr = domain
    ///     .event_attributes_builder()
    ///     .payload(1)
    ///     .message("complex range")
    ///     .build();
    /// let range = domain.range(attr);
    ///
    /// // explicitly end a range
    /// drop(range)
    /// ```
    pub fn range<'a>(&'a self, arg: impl Into<EventArgument<'a>>) -> Range<'a> {
        Range::new(arg, self)
    }

    /// Internal function for starting a range and returning a raw Range Id
    pub(crate) fn range_start<'a>(&self, arg: impl Into<EventArgument<'a>>) -> u64 {
        let arg = match arg.into() {
            EventArgument::Attributes(attr) => attr,
            EventArgument::Message(m) => m.into(),
        };
        nvtx_sys::domain_range_start_ex(self.handle, &arg.encode())
    }

    /// Internal function for ending a range given a raw Range Id
    pub(crate) fn range_end(&self, range_id: u64) {
        nvtx_sys::domain_range_end(self.handle, range_id);
    }

    /// Name a resource
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// let pthread_id = 13854;
    /// domain.name_resource(
    ///     nvtx::domain::GenericIdentifier::PosixThread(pthread_id),
    ///     "My custom name");
    /// #[cfg(feature = "cuda")]
    /// domain.name_resource(nvtx::domain::CudaIdentifier::Device(0), "My device");
    /// #[cfg(feature = "cuda_runtime")]
    /// domain.name_resource(nvtx::domain::CudaRuntimeIdentifier::Device(1), "My device");
    /// ```
    pub fn name_resource<'a>(
        &'a self,
        identifier: impl Into<Identifier>,
        name: impl Into<Message<'a>>,
    ) -> Resource<'a> {
        let materialized_identifier: Identifier = identifier.into();
        let materialized_name: Message = name.into();
        let (msg_type, msg_value) = materialized_name.encode();
        let (id_type, id_value) = materialized_identifier.encode();
        let attrs = nvtx_sys::ResourceAttributes {
            version: nvtx_sys::NVTX_VERSION as u16,
            size: 32,
            identifierType: id_type as i32,
            identifier: id_value,
            messageType: msg_type as i32,
            message: msg_value,
        };
        Resource {
            handle: nvtx_sys::domain_resource_create(self.handle, attrs),
            _lifetime: PhantomData,
        }
    }

    /// Create a user defined synchronization object This is used to track non-OS
    /// synchronization working with spinlocks and atomics.
    pub fn user_sync<'a>(&'a self, name: impl Into<Message<'a>>) -> sync::UserSync<'a> {
        let message = name.into();
        let (msg_type, msg_value) = message.encode();
        let attrs = nvtx_sys::SyncUserAttributes {
            version: nvtx_sys::NVTX_VERSION as u16,
            size: 16,
            messageType: msg_type as i32,
            message: msg_value,
        };
        let handle = nvtx_sys::domain_syncuser_create(self.handle, attrs);
        sync::UserSync {
            handle,
            _lifetime: PhantomData,
        }
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        nvtx_sys::domain_destroy(self.handle)
    }
}
