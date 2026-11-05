//! The macros that write the components' pyo3 classes, which pyo3 can't take from a trait.

mod component_kind;
mod component_module;
pub(crate) mod fields;
mod sequence_class;

pub(crate) use component_kind::component_kind;
pub(crate) use component_module::component_module;
pub(crate) use sequence_class::sequence_class;
