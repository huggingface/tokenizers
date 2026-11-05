//! The class of a component kind, such as `normalizers.Normalizer`, written by
//! [`component_kind!`].
//!
//! A kind needs its own class, its own `_unpickle` (pyo3 can't take `#[pymethods]` from a trait),
//! and a [`ComponentKind`](crate::components::ComponentKind) impl. The kinds differ only in the
//! arguments of the macro.

/// Writes the class `kind` and its [`ComponentKind`](crate::components::ComponentKind) impl.
/// The kind's spec is the `tokenizer_field` field of
/// [`TokenizerSpec`](crate::tokenizer::TokenizerSpec), and a `Sequence` of the kind lists its
/// items under `items_key`.
/// `classes` maps each `"type"` to its concrete class. A `"type"` that isn't listed gets the
/// plain `kind` class.
// rustfmt indents the multi-line `pyclass` attribute deeper on each run.
#[rustfmt::skip]
macro_rules! component_kind {
    (
        kind: $kind:ident,
        python_module: $python_module:literal,
        tokenizer_field: $tokenizer_field:ident,
        items_key: $items_key:expr,
        classes: { $($json_type:pat => $class:ident),* $(,)? } $(,)?
    ) => {
        #[doc = concat!("Base class of the classes in `", $python_module, "`.")]
        #[::pyo3::pyclass(
            frozen,
            subclass,
            extends = $crate::components::Component,
            module = $python_module
        )]
        #[derive(Default)]
        pub struct $kind;

        #[::pyo3::pymethods]
        impl $kind {
            /// Rebuilds a component from its `__getstate__`
            #[staticmethod]
            fn _unpickle(
                py: ::pyo3::Python<'_>,
                state: &[u8],
            ) -> ::pyo3::PyResult<::pyo3::Py<Self>> {
                <Self as $crate::components::ComponentKind>::from_state(py, state)
            }
        }

        impl $crate::components::ComponentKind for $kind {
            const ITEMS_KEY: Option<&'static str> = $items_key;

            fn tokenizer_field_mut(
                spec: &mut $crate::tokenizer::TokenizerSpec,
            ) -> &mut Option<$crate::components::component_spec::ComponentSpec> {
                &mut spec.$tokenizer_field
            }

            fn from_spec(
                py: ::pyo3::Python<'_>,
                spec: $crate::components::component_spec::ComponentSpec,
            ) -> ::pyo3::PyResult<::pyo3::Py<Self>> {
                use $crate::components::component_spec::ComponentSpec;
                let json_type = match &spec {
                    ComponentSpec::Single(single) => single.json_type(),
                    ComponentSpec::Sequence(_) => Some("Sequence".to_owned()),
                };
                match json_type.as_deref() {
                    $(Some($json_type) => Self::concrete_component(py, spec, $class),)*
                    _ => Self::default_component(py, spec),
                }
            }
        }
    };
}

pub(crate) use component_kind;
