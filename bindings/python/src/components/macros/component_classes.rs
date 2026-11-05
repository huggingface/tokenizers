//! The classes of the simple components, written by [`component_classes!`] from the core's lists.
//!
//! `tk::normalizer_schemas!(component_classes! { ... })` hands each normalizer's entry (its doc,
//! its JSON `"type"` and its fields) to [`component_classes!`], which writes, for the kind:
//!
//! 1. one class per component, with a constructor, and a getter and a setter per field,
//! 2. the kind's [`ConcreteClasses`](crate::components::ConcreteClasses) impl, which picks the
//!    class of a component from its `"type"`,
//! 3. the kind's `#[pymodule]`.
//!
//! A field's shape, such as `Bool`, names its types in [`fields`](super::fields).

/// Writes a kind's component classes from its `tk::*_schemas!` entries.
macro_rules! component_classes {
    ({ kind: $kind:ident, module: $module:ident, python_module: $python_module:literal $(,)? } $(
        $(#[doc = $doc:literal])+
        $class:ident $({
            // A default is a `tt`. pyo3 writes an `expr` default as `...` in the stubs.
            $($key:ident: $field_type:ident $(($choices:expr))? $(= $default:tt)?),* $(,)?
        })?
    ),* $(,)?) => {
        use $crate::components::JsonFieldsExt;
        use $crate::components::macros::fields::{arg, choice_json, choice_name, value};

        $(
            $(#[doc = $doc])+
            #[allow(clippy::upper_case_acronyms)]
            #[pyclass(frozen, extends = $kind, module = $python_module)]
            pub struct $class;

            // pyo3 reads the methods before any macro inside them expands, so `paste!` names the
            // setters first.
            pastey::paste! {
                #[pymethods]
                impl $class {
                    #[new]
                    #[pyo3(signature = ($($($key $(= $default)?),*)?))]
                    fn new($($($key: arg::$field_type),*)?) -> PyResult<PyClassInitializer<Self>> {
                        $($($(let $key = choice_json(stringify!($key), $key, $choices)?;)?)*)?
                        let spec = ComponentSpec::from_literal(serde_json::json!({
                            "type": stringify!($class),
                            $($(stringify!($key): $key,)*)?
                        }));
                        Ok($kind::initializer(spec, Self))
                    }

                    $($(
                        #[getter]
                        fn $key(slf: &Bound<'_, Self>) -> PyResult<value::$field_type> {
                            let value = slf.field(stringify!($key))?;
                            $(let value = choice_name(stringify!($key), value, $choices)?;)?
                            Ok(value)
                        }

                        #[setter]
                        fn [<set_ $key>](slf: &Bound<'_, Self>, $key: arg::$field_type) -> PyResult<()> {
                            $(let $key = choice_json(stringify!($key), $key, $choices)?;)?
                            slf.set_field(stringify!($key), $key)
                        }
                    )*)?
                }
            }
        )*

        impl ConcreteClasses for $kind {
            fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<$kind>> {
                match &spec {
                    ComponentSpec::Sequence(_) => $kind::concrete_component(py, spec, Sequence),
                    ComponentSpec::Single(single) => match single.json_type().as_deref() {
                        $(Some(stringify!($class)) => $kind::concrete_component(py, spec, $class),)*
                        _ => $kind::default_component(py, spec),
                    },
                }
            }
        }

        #[pymodule]
        pub mod $module {
            #[pymodule_export]
            use super::{$kind, Sequence, $($class),*};
        }
    };
}

pub(crate) use component_classes;
