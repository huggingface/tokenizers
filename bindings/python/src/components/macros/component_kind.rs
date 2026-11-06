//! Macro to generate the pyo3 definitions of a component kind base class, such as
//! `normalizers.Normalizer`.
//!
//! There are five kinds of components: models, normalizers, pre-tokenizers, post-processors and
//! decoders.
//! Each kind has a base Python class, such as `Normalizer`.
//! Every normalizer class, such as `Strip`, extends `Normalizer`.
//! So `tokenizer.normalizer = x` accepts any normalizer.
//!
//! pyo3 can't make a generic class. It also can't take `#[pymethods]` from a trait.
//! So each kind needs its own copy of the same code, and this macro writes it.
//!
//! The macro writes as little as it can: the pyo3 class and the [`ComponentKind`] impl.
//! The default methods of the [`ComponentKind`] trait hold the shared logic.
//!
//! For the `Normalizer` kind, the macro generates this pyo3 class:
//!
//! ```rust,ignore
//! /// Base class of the classes in `tokenizers.normalizers`.
//! #[pyclass(frozen, subclass, extends = Component, module = "tokenizers.normalizers")]
//! #[derive(Default)]
//! pub struct Normalizer;
//!
//! #[pymethods]
//! impl Normalizer {
//!     /// Rebuilds a component from its `__getstate__`
//!     #[staticmethod]
//!     fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
//!         <Self as ComponentKind>::from_state(py, state)
//!     }
//! }
//!
//! impl ComponentKind for Normalizer {
//!     const ITEMS_KEY: Option<&'static str> = Some("normalizers");
//!
//!     fn tokenizer_field_mut(spec: &mut TokenizerSpec) -> &mut Option<ComponentSpec> {
//!         &mut spec.normalizer
//!     }
//!
//!     fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
//!         let json_type = match &spec {
//!             ComponentSpec::Single(single) => single.json_type(),
//!             ComponentSpec::Sequence(_) => Some("Sequence".to_owned()),
//!         };
//!         match json_type.as_deref() {
//!             Some("Strip") => Self::concrete_component(py, spec, Strip),
//!             Some("Sequence") => Self::concrete_component(py, spec, Sequence),
//!             // One arm per class in `classes`.
//!             _ => Self::default_component(py, spec),
//!         }
//!     }
//! }
//! ```
//!
//! [`ComponentKind`]: crate::components::ComponentKind

/// Writes the base class of a component kind, such as `normalizers.Normalizer`.
///
/// ```rust,ignore
/// component_kind! {
///     kind: Normalizer,
///     python_module: "tokenizers.normalizers",
///     tokenizer_field: normalizer,
///     items_key: Some("normalizers"),
///     classes: { "Strip" => Strip, "Sequence" => Sequence },
/// }
/// ```
///
/// It writes:
///
/// - `Normalizer`, a Python class that extends `Component`, in `tokenizers.normalizers`.
/// - Its `_unpickle` static method, which pickle calls to rebuild a component.
/// - The [`ComponentKind`] impl of `Normalizer`. The other arguments fill it in:
///   - `tokenizer_field` names the field of [`TokenizerSpec`] that stores the normalizer.
///   - `items_key` is the JSON key that holds the items of a normalizer `Sequence`.
///     Use `None` if the kind has no `Sequence`.
///   - `classes` maps each JSON `"type"` to its Python class.
///     A `"type"` missing from `classes` gets the plain `Normalizer` class.
///
/// [`ComponentKind`]: crate::components::ComponentKind
/// [`TokenizerSpec`]: crate::tokenizer::TokenizerSpec
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
