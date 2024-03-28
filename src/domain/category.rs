use crate::Domain;

/// Represents a category for use with event and range grouping. See [`Domain::register_category`], [`Domain::register_categories`]
#[derive(Debug, Clone, Copy)]
pub struct Category<'a> {
    pub(super) id: u32,
    pub(super) domain: &'a Domain,
}
