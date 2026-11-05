//! The class of a component kind, such as `normalizers.Normalizer`, written by
//! [`component_kind!`].
//!
//! A kind needs its own class, its own `_unpickle` (pyo3 can't take `#[pymethods]` from a trait),
//! and a [`ComponentKind`](crate::components::ComponentKind) impl. The kinds differ only in the
//! arguments of the macro.

/// Writes the class `kind` and its [`ComponentKind`](crate::components::ComponentKind) impl.
/// The kind's spec is the `tokenizer_field` field of
/// [`TokenizerSpec`](crate::tokenizer::TokenizerSpec), and a `Sequence` of the kind lists its
/// items under `sequence_item_key`. The kind's
/// [`ConcreteClasses`](crate::components::ConcreteClasses) impl is written with its classes.
macro_rules! component_kind {
    (
        kind: $kind:ident,
        python_module: $python_module:literal,
        tokenizer_field: $tokenizer_field:ident,
        sequence_item_key: $sequence_item_key:expr,
    ) => {
        #[doc = concat!("Base class of the classes in `", $python_module, "`.")]
        #[pyclass(frozen, subclass, extends = Component, module = $python_module)]
        #[derive(Default)]
        pub struct $kind;

        #[pymethods]
        impl $kind {
            /// Rebuilds a component from its `__getstate__`
            #[staticmethod]
            fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
                Self::from_state(py, state)
            }
        }

        impl ComponentKind for $kind {
            const SEQUENCE_ITEM_KEY: Option<&'static str> = $sequence_item_key;

            fn tokenizer_field_mut(spec: &mut TokenizerSpec) -> &mut Option<ComponentSpec> {
                &mut spec.$tokenizer_field
            }
        }
    };
}

pub(crate) use component_kind;
