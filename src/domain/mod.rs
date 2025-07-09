use crate::{common::event_attributes::GenericEventAttributesBuilder, Str, TypeValueEncodable};
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{
        atomic::{AtomicU32, Ordering},
        Mutex,
    },
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
    registered_strings: AtomicU32,
    registered_categories: AtomicU32,
    strings: Mutex<HashMap<String, (nvtx_sys::StringHandle, u32)>>,
    categories: Mutex<HashMap<String, u32>>,
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
            handle: match name.into() {
                Str::Ascii(s) => nvtx_sys::domain_create_ascii(&s),
                Str::Unicode(s) => nvtx_sys::domain_create_unicode(&s),
            },
            registered_strings: AtomicU32::new(0),
            registered_categories: AtomicU32::new(0),
            strings: Mutex::new(HashMap::default()),
            categories: Mutex::new(HashMap::default()),
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
            inner: GenericEventAttributesBuilder::default(),
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
        let into_string: Str = string.into();
        let owned_string = match &into_string {
            Str::Ascii(s) => s.to_str().unwrap().to_string(),
            Str::Unicode(s) => s.to_string().unwrap(),
        };
        let (handle, uid) = *self
            .strings
            .lock()
            .unwrap()
            .entry(owned_string)
            .or_insert_with(|| {
                let id = 1 + self.registered_strings.fetch_add(1, Ordering::SeqCst);
                let handle = match into_string {
                    Str::Ascii(s) => nvtx_sys::domain_register_string_ascii(self.handle, &s),
                    Str::Unicode(s) => nvtx_sys::domain_register_string_unicode(self.handle, &s),
                };
                (handle, id)
            });
        RegisteredString::new(handle, uid, self)
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
        let into_name: Str = name.into();
        let owned_name = match &into_name {
            Str::Ascii(s) => s.to_str().unwrap().to_string(),
            Str::Unicode(s) => s.to_string().unwrap(),
        };

        let id = *self
            .categories
            .lock()
            .unwrap()
            .entry(owned_name)
            .or_insert_with(|| {
                let id = 1 + self.registered_categories.fetch_add(1, Ordering::SeqCst);
                match into_name {
                    Str::Ascii(s) => nvtx_sys::domain_name_category_ascii(self.handle, id, &s),
                    Str::Unicode(s) => nvtx_sys::domain_name_category_unicode(self.handle, id, &s),
                }
                id
            });
        Category::new(id, self)
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
    /// domain.mark(reg_str);
    /// ```
    pub fn mark<'a>(&'a self, arg: impl Into<EventArgument<'a>>) {
        match arg.into() {
            EventArgument::Attributes(attr) => {
                if let Some(category) = &attr.category {
                    assert!(
                        std::ptr::eq(category.domain(), self),
                        "EventAttributes' Domain does not match current Domain"
                    );
                }
                if let Some(Message::Registered(reg_str)) = &attr.message {
                    assert!(
                        std::ptr::eq(reg_str.domain(), self),
                        "EventAttributes' Domain does not match current Domain"
                    );
                }
                let encoded = attr.encode();
                nvtx_sys::domain_mark_ex(self.handle, &encoded)
            }
            EventArgument::Message(m) => {
                if let Message::Registered(reg_str) = m {
                    assert!(
                        std::ptr::eq(reg_str.domain(), self),
                        "EventAttributes' Domain does not match current Domain"
                    );
                }
                let attr: EventAttributes = m.into();
                let encoded = attr.encode();
                nvtx_sys::domain_mark_ex(self.handle, &encoded)
            }
        }
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
        let event_arg = arg.into();
        match &event_arg {
            EventArgument::Attributes(attr) => {
                // Validate that all categories and messages belong to this domain
                if let Some(category) = &attr.category {
                    assert!(
                        std::ptr::eq(category.domain(), self),
                        "EventAttributes' Domain does not match current Domain"
                    );
                }
                if let Some(Message::Registered(reg_str)) = &attr.message {
                    assert!(
                        std::ptr::eq(reg_str.domain(), self),
                        "EventAttributes' Domain does not match current Domain"
                    );
                }
            }
            EventArgument::Message(m) => {
                // NEW: Add validation here
                if let Message::Registered(reg_str) = m {
                    assert!(
                        std::ptr::eq(reg_str.domain(), self),
                        "EventAttributes' Domain does not match current Domain"
                    );
                }
            }
        }
        Range::new(event_arg, self)
    }

    /// Internal function for starting a range and returning a raw Range Id
    pub(super) fn range_start<'a>(&self, arg: impl Into<EventArgument<'a>>) -> u64 {
        let arg = match arg.into() {
            EventArgument::Attributes(attr) => attr,
            EventArgument::Message(m) => m.into(),
        };
        nvtx_sys::domain_range_start_ex(self.handle, &arg.encode())
    }

    /// Internal function for ending a range given a raw Range Id
    pub(super) fn range_end(&self, range_id: u64) {
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

    /// Create a user defined synchronization object.
    ///
    /// This is used to track non-OS synchronization working with spinlocks and atomics.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::TestUtils;

    #[test]
    fn test_register_string() {
        let d = Domain::new("d");
        TestUtils::assert_domain_string_registration(&d, "hello");

        // Test different strings are different
        let s1 = d.register_string("hello");
        let s2 = d.register_string("hello2");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_register_strings() {
        let d = Domain::new("d");
        let [s1, s2, s3] = d.register_strings(["hello", "hello2", "hello"]);
        assert_eq!(s1, s3);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_register_category() {
        let d = Domain::new("d");
        TestUtils::assert_domain_category_registration(&d, "hello");

        // Test different categories are different
        let c1 = d.register_category("hello");
        let c2 = d.register_category("hello2");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_register_categories() {
        let d = Domain::new("d");
        let [c1, c2, c3] = d.register_categories(["hello", "hello2", "hello"]);
        assert_eq!(c1, c3);
        assert_ne!(c1, c2);
    }
}
