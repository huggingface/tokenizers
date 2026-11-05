//! Macros to write the Python classes of the components.
//!
//! pyo3 needs each Python class written explicitly with a `#[pyclass]` struct
//! and `#[pymethods]` methods.
//! For example, the `Strip` normalizer:
//!
//! ```rust,ignore
//! #[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
//! pub struct Strip;
//!
//! #[pymethods]
//! impl Strip {
//!     #[new]
//!     #[pyo3(signature = (strip_left = true, strip_right = true))]
//!     fn new(strip_left: bool, strip_right: bool) -> PyResult<PyClassInitializer<Self>> { ... }
//!
//!     #[getter]
//!     fn strip_left(slf: &Bound<'_, Self>) -> PyResult<bool> { ... }
//!
//!     #[setter]
//!     fn set_strip_left(slf: &Bound<'_, Self>, strip_left: bool) -> PyResult<()> { ... }
//!
//!     // The same getter and setter for `strip_right`.
//! }
//! ```
//!
//! Most components need similar code, each with their own name and fields.
//! A pyo3 `#[pyclass]` can't be generic, and its Python methods can't come from a trait.
//!
//! For these reasons, we use rust macros to generate most of the pyo3 code.
//! The macros hold as little logic as possible.
//! Most of the logic is extracted to traits:  [`ComponentKind`], [`ComponentSequence`] and [`JsonFieldsExt`].
//!
//! # Three macros
//!
//! This module defines three macros to write the pyo3 code:
//!
//! - [`component_kind!`](macro@component_kind) writes the base class of a kind, such as `Normalizer`.
//! - [`sequence_class!`](macro@sequence_class) writes the `Sequence` class of a kind.
//! - [`component_module!`](macro@component_module) writes a whole Python module, such as
//!   `tokenizers.normalizers`, from a list in `tk::schemas`.
//!   It calls the two macros above, then writes one class per definition in the list.
//!   [`field_types`] holds the Rust types of each field type.
//!
//! # Where they are called
//!
//! ```text
//! normalizers.rs, pre_tokenizers.rs, decoders.rs:
//!
//!     tk::normalizer_schemas!(component_module! { ... })
//!         └─▶ component_module!
//!               ├─▶ component_kind!   the kind's class: Normalizer
//!               ├─▶ sequence_class!   its Sequence class
//!               └─▶ one class per definition: Strip, Prepend, ...
//!
//! models.rs, processors.rs:
//!
//!     component_kind!   the kind's class: Model, PostProcessor
//! ```
//!
//! Models and post-processors have no list in `tk::schemas`.
//!
//! [`ComponentKind`]: crate::components::ComponentKind
//! [`ComponentSequence`]: crate::components::component_sequence::ComponentSequence
//! [`JsonFieldsExt`]: crate::components::JsonFieldsExt

mod component_kind;
mod component_module;
pub(crate) mod field_types;
mod sequence_class;

pub(crate) use component_kind::component_kind;
pub(crate) use component_module::component_module;
pub(crate) use sequence_class::sequence_class;
