//! The `Sequence` class of a component kind, written by [`sequence_class!`].

/// Writes the `Sequence` class of the kind `kind`, in the Python module `python_module`.
/// The caller imports [`ComponentSequence`](crate::components::component_sequence::ComponentSequence).
macro_rules! sequence_class {
    (kind: $kind:ident, python_module: $python_module:literal $(,)?) => {
        // pyo3 writes the `__setitem__` slot as an item next to the impl, calling an unsafe fn
        // through `Sequence`, whose span is this macro's. The allow has to sit on a module.
        #[allow(unsafe_op_in_unsafe_fn)]
        mod sequence {
            use super::*;

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
