//! Macro to generate the `Sequence` class pyo3 definitions
//!
//! Normalizers, pre-tokenizers and decoders each have a `Sequence` class, such as `normalizers.Sequence`.
//! pyo3 can't make a generic class, so each kind needs its own definition of the `Sequence` class.
//!
//! The macro writes only the pyo3 code: the class and its Python methods.
//! Each method defers logic to the [`ComponentSequence`] trait.
//!
//! For the `Normalizer` kind, the macro generates this pyo3 class:
//!
//! ```rust,ignore
//! /// Applies each `Normalizer` in turn.
//! #[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
//! #[derive(Default)]
//! pub struct Sequence;
//!
//! impl ComponentSequence<Normalizer> for Sequence {}
//!
//! #[pymethods]
//! impl Sequence {
//!     #[new]
//!     #[pyo3(signature = (items, /))]
//!     fn new(items: Vec<Bound<'_, Normalizer>>) -> PyClassInitializer<Self> {
//!         Self::initializer(&items)
//!     }
//!
//!     fn __len__(slf: &Bound<'_, Self>) -> usize {
//!         Self::spec(slf).len()
//!     }
//!
//!     fn __getitem__(slf: &Bound<'_, Self>, index: isize) -> PyResult<Py<Normalizer>> {
//!         Self::item(slf, index)
//!     }
//!
//!     fn __setitem__(slf: &Bound<'_, Self>, index: isize, item: Bound<'_, Normalizer>) -> PyResult<()> {
//!         Self::set_item(slf, index, &item)
//!     }
//! }
//! ```
//!
//! [`ComponentSequence`]: crate::components::component_sequence::ComponentSequence

/// Writes the `Sequence` class of a component kind, such as `normalizers.Sequence`.
///
/// ```rust,ignore
/// sequence_class! {
///     kind: Normalizer,
///     python_module: "tokenizers.normalizers",
/// }
/// ```
///
/// Shall be called next to the kind class, here `Normalizer`. It writes:
///
/// - `Sequence`, a Python class that extends `Normalizer`, in `tokenizers.normalizers`.
/// - Its constructor, `Sequence(items)`, which takes a list of `Normalizer`s.
/// - methods supporting `len(seq)`, `seq[i]` and `seq[i] = item`.
///
/// Its methods call [`ComponentSequence`], which needs the kind to have an `items_key`.
///
/// [`ComponentSequence`]: crate::components::component_sequence::ComponentSequence
macro_rules! sequence_class {
    (kind: $kind:ident, python_module: $python_module:literal $(,)?) => {
        // pyo3 writes the `__setitem__` slot as an item next to the impl, calling an unsafe fn
        // through `Sequence`, whose span is this macro's. The allow has to sit on a module.
        #[allow(unsafe_op_in_unsafe_fn)]
        mod sequence {
            use pyo3::prelude::*;

            use super::$kind;
            use $crate::components::component_sequence::ComponentSequence;

            #[doc = concat!("Applies each `", stringify!($kind), "` in turn.")]
            #[pyclass(frozen, extends = $kind, module = $python_module)]
            #[derive(Default)]
            pub struct Sequence;

            impl ComponentSequence<$kind> for Sequence {}

            #[pymethods]
            impl Sequence {
                #[new]
                #[pyo3(signature = (items, /))]
                fn new(items: Vec<Bound<'_, $kind>>) -> PyClassInitializer<Self> {
                    Self::initializer(&items)
                }

                fn __len__(slf: &Bound<'_, Self>) -> usize {
                    Self::spec(slf).len()
                }

                fn __getitem__(slf: &Bound<'_, Self>, index: isize) -> PyResult<Py<$kind>> {
                    Self::item(slf, index)
                }

                fn __setitem__(
                    slf: &Bound<'_, Self>,
                    index: isize,
                    item: Bound<'_, $kind>,
                ) -> PyResult<()> {
                    Self::set_item(slf, index, &item)
                }
            }
        }
        pub use sequence::Sequence;
    };
}

pub(crate) use sequence_class;
