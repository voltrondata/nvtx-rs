use crate::Domain;

/// Represents a domain-owned category for use with mark and range grouping.
///
/// See [`Domain::register_category`], [`Domain::register_categories`]
#[derive(Debug, Clone, Copy)]
pub struct Category<'a> {
    id: u32,
    domain: &'a Domain,
}

impl<'a> PartialEq for Category<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && std::ptr::eq(self.domain, other.domain)
    }
}

impl<'a> Eq for Category<'a> {}

impl<'a> Category<'a> {
    pub(super) fn new(id: u32, domain: &'a Domain) -> Category<'a> {
        Category { id, domain }
    }

    pub(super) fn id(&self) -> u32 {
        self.id
    }

    pub(super) fn domain(&self) -> &Domain {
        self.domain
    }
}
