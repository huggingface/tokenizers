//! A kind's `Sequence`, which runs several components of that kind in turn.
//!
//! Every kind with a `Sequence` needs the same Python class: a constructor taking the items,
//! `len`, indexing and item assignment. [`sequence_component!`] writes it.

/// Generates a frozen pyclass that represents a sequence of components, such as `normalizers.Sequence`.
///
/// The generated class has four Python methods:
///
/// - The constructor `Sequence(items)` (`__new__`) takes a list of components of the kind.
/// - `len(seq)` returns the number of items.
/// - `seq[i]` returns the item at `i`. Negative indices count from the end. The item is an
///   editable live view of the underlying component.
/// - `seq[i] = item` replaces the item at `i`. It raises an error when `item` is a `Sequence`.
///
/// The macro's inputs:
///
/// - `extends` names the kind, and `module` the Python module that exports the class.
/// - `items_key` is the JSON key that lists the items, such as `"normalizers"`.
/// - The doc comment becomes the Python docstring.
///
/// # Example
///
/// ```ignore
/// sequence_component! {
///     /// Applies each normalizer in turn.
///     #[pyclass(extends = Normalizer, module = "tokenizers.normalizers")]
///     pub struct Sequence;
///     items_key = SEQUENCE_ITEMS_KEY;
/// }
/// ```
///
/// [`ComponentKind::from_spec`]: crate::components::ComponentKind::from_spec
macro_rules! sequence_component {
    (
        $(#[doc = $doc:literal])*
        #[pyclass(extends = $kind:ident, module = $module:literal)]
        pub struct $class:ident;
        items_key = $items_key:expr;
    ) => {
        $(#[doc = $doc])*
        #[::pyo3::pyclass(frozen, extends = $kind, module = $module)]
        pub struct $class;

        #[::pyo3::pymethods]
        impl $class {
            #[new]
            #[pyo3(signature = (items, /))]
            fn new(items: Vec<::pyo3::Bound<'_, $kind>>) -> ::pyo3::PyClassInitializer<Self> {
                use $crate::components::ComponentKind;
                use $crate::components::component_spec::{ComponentSpec, SequenceSpec};

                let mut flat = Vec::with_capacity(items.len());
                for item in &items {
                    match <$kind as ComponentKind>::spec(item) {
                        ComponentSpec::Single(single) => flat.push(single.clone()),
                        ComponentSpec::Sequence(sequence) => flat.extend(sequence.items()),
                    }
                }
                let sequence = SequenceSpec::new($items_key, flat);
                let spec = ComponentSpec::Sequence(::std::sync::Arc::new(sequence));
                <$kind as ComponentKind>::initializer(spec, Self)
            }

            fn __len__(slf: &::pyo3::Bound<'_, Self>) -> usize {
                use $crate::components::ComponentKind;
                <$kind as ComponentKind>::sequence_spec(slf).len()
            }

            fn __getitem__(
                slf: &::pyo3::Bound<'_, Self>,
                index: isize,
            ) -> ::pyo3::PyResult<::pyo3::Py<$kind>> {
                use $crate::components::ComponentKind;
                let spec = <$kind as ComponentKind>::sequence_spec(slf);
                let item = spec.index(index).and_then(|index| spec.get(index));
                let item = item.ok_or_else(|| {
                    ::pyo3::exceptions::PyIndexError::new_err("Sequence index out of range")
                })?;
                <$kind as ComponentKind>::from_spec(slf.py(), item)
            }

            fn __setitem__(
                slf: &::pyo3::Bound<'_, Self>,
                index: isize,
                item: ::pyo3::Bound<'_, $kind>,
            ) -> ::pyo3::PyResult<()> {
                use $crate::components::ComponentKind;
                let spec = <$kind as ComponentKind>::sequence_spec(slf);
                let Some(item) = <$kind as ComponentKind>::spec(&item).as_single() else {
                    return Err($crate::error::err("a Sequence item can't be a Sequence"));
                };
                spec.index(index)
                    .and_then(|index| spec.set(index, item.clone()))
                    .ok_or_else(|| {
                        ::pyo3::exceptions::PyIndexError::new_err(
                            "Sequence assignment index out of range",
                        )
                    })
            }
        }
    };
}

pub(crate) use sequence_component;
