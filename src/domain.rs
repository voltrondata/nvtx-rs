use crate::{Str, TypeValueEncodable};
use std::{
    marker::PhantomData,
    sync::atomic::{AtomicU32, Ordering},
};

mod category;
mod event_argument;
mod event_attributes;
mod identifier;
mod local_range;
mod message;
mod range;
mod registered_string;
mod resource;
/// user-defined synchronization objects
pub mod sync;

pub use self::{
    category::Category,
    event_argument::EventArgument,
    event_attributes::{EventAttributes, EventAttributesBuilder},
    identifier::Identifier,
    local_range::LocalRange,
    message::Message,
    range::Range,
    registered_string::RegisteredString,
    resource::Resource,
};

/// Represents a domain for high-level grouping
#[derive(Debug)]
pub struct Domain {
    handle: nvtx_sys::ffi::nvtxDomainHandle_t,
    registered_categories: AtomicU32,
}

impl Domain {
    /// Register a NVTX domain
    ///
    /// See [`Str`] for valid conversions
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// ```
    pub fn new(name: impl Into<Str>) -> Self {
        Domain {
            handle: match name.into() {
                Str::Ascii(s) => unsafe { nvtx_sys::ffi::nvtxDomainCreateA(s.as_ptr()) },
                Str::Unicode(s) => unsafe { nvtx_sys::ffi::nvtxDomainCreateW(s.as_ptr().cast()) },
            },
            registered_categories: AtomicU32::new(0),
        }
    }

    /// Gets a new builder instance for event attribute construction in the current domain
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

    /// Registers an immutable string within the current domain
    ///
    /// Returns a handle to the immutable string registered to nvtx.
    ///
    /// See [`Str`] for valid conversions
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// let my_str = domain.register_string("My immutable string");
    /// ```
    pub fn register_string(&self, string: impl Into<Str>) -> RegisteredString<'_> {
        let handle = match string.into() {
            Str::Ascii(s) => unsafe {
                nvtx_sys::ffi::nvtxDomainRegisterStringA(self.handle, s.as_ptr())
            },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxDomainRegisterStringW(self.handle, s.as_ptr().cast())
            },
        };
        RegisteredString {
            handle,
            domain: self,
        }
    }

    /// Register many immutable strings within the current domain
    ///
    /// Returns an array of handles to the immutable strings registered to nvtx.
    ///
    /// See [`Str`] for valid conversions
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

    /// Register a new category within the domain. Categories are used to group sets of events.
    ///
    /// Returns a handle to the category registered to nvtx.
    ///
    /// See [`Str`] for valid conversions
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// let cat = domain.register_category("Category");
    /// ```
    pub fn register_category(&self, name: impl Into<Str>) -> Category<'_> {
        let id = 1 + self.registered_categories.fetch_add(1, Ordering::SeqCst);
        match name.into() {
            Str::Ascii(s) => unsafe {
                nvtx_sys::ffi::nvtxDomainNameCategoryA(self.handle, id, s.as_ptr())
            },
            Str::Unicode(s) => unsafe {
                nvtx_sys::ffi::nvtxDomainNameCategoryW(self.handle, id, s.as_ptr().cast())
            },
        }
        Category { id, domain: self }
    }

    /// Register new categories within the domain. Categories are used to group sets of events.
    ///
    /// Returns an array of handles to the categories registered to nvtx.
    ///
    /// See [`Str`] for valid conversions
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
    /// A marker can contain a text message or specify additional information using the event attributes structure. These attributes include a text message, color, category, and a payload. Each of the attributes is optional.
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// // ...
    /// domain.mark("Sample mark");
    ///
    /// domain.mark(c"Another example");
    ///
    /// domain.mark(domain.event_attributes_builder().message("Interesting example").color([255, 0, 0]).build());
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
        unsafe { nvtx_sys::ffi::nvtxDomainMarkEx(self.handle, &encoded) }
    }

    /// Create an RAII-friendly, domain-owned range type which (1) cannot be moved across thread boundaries and (2) automatically ended when dropped. Panics on drop() if the opening level doesn't match the closing level (since it must model a perfect stack).
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
    /// let attr = domain.event_attributes_builder().payload(1).message("complex range").build();
    /// let range = domain.local_range(attr);
    ///
    /// // explicitly end a range
    /// drop(range)
    /// ```
    pub fn local_range<'a>(&'a self, arg: impl Into<EventArgument<'a>>) -> LocalRange<'a> {
        LocalRange::new(arg, self)
    }

    /// Create an RAII-friendly, domain-owned range type which (1) can be moved across thread boundaries and (2) automatically ended when dropped
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
    /// let attr = domain.event_attributes_builder().payload(1).message("complex range").build();
    /// let range = domain.range(attr);
    ///
    /// // explicitly end a range
    /// drop(range)
    /// ```
    pub fn range<'a>(&'a self, arg: impl Into<EventArgument<'a>>) -> Range<'a> {
        Range::new(arg, self)
    }

    /// Name a resource
    ///
    /// ```
    /// let domain = nvtx::Domain::new("Domain");
    /// let pthread_id = 13854;
    /// domain.name_resource(nvtx::GenericIdentifier::PosixThread(pthread_id), "My custom name");
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
        let mut attrs = nvtx_sys::ffi::nvtxResourceAttributes_t {
            version: nvtx_sys::ffi::NVTX_VERSION as u16,
            size: 32,
            identifierType: id_type as i32,
            identifier: id_value,
            messageType: msg_type as i32,
            message: msg_value,
        };
        let ptr: *mut nvtx_sys::ffi::nvtxResourceAttributes_v0 = &mut attrs;
        let handle = unsafe { nvtx_sys::ffi::nvtxDomainResourceCreate(self.handle, ptr) };
        Resource {
            handle,
            _lifetime: PhantomData,
        }
    }

    /// Create a user defined synchronization object This is used to track non-OS synchronization working with spinlocks and atomics.
    pub fn user_sync<'a>(&'a self, name: impl Into<Message<'a>>) -> sync::UserSync<'a> {
        let message = name.into();
        let (msg_type, msg_value) = message.encode();
        let attrs = nvtx_sys::ffi::nvtxSyncUserAttributes_t {
            version: nvtx_sys::ffi::NVTX_VERSION as u16,
            size: 16,
            messageType: msg_type as i32,
            message: msg_value,
        };
        let handle = unsafe { nvtx_sys::ffi::nvtxDomainSyncUserCreate(self.handle, &attrs) };
        sync::UserSync {
            handle,
            _lifetime: PhantomData,
        }
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        unsafe { nvtx_sys::ffi::nvtxDomainDestroy(self.handle) }
    }
}

unsafe impl Send for Domain {}

unsafe impl Sync for Domain {}
