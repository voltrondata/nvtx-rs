use crate::Domain;

/// Represents a domain-owned category for use with event and range grouping.
///
/// See [`Domain::register_category`], [`Domain::register_categories`]
#[derive(Debug, Clone, Copy)]
pub struct Category<'a> {
    pub(crate) id: u32,
    pub(crate) domain: &'a Domain,
}
