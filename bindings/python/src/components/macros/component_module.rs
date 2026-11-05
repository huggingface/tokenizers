//! The Python module of a component kind with a schema list, such as `tokenizers.normalizers`,
//! written by [`component_module!`] from the core's list.
//!
//! `tk::normalizer_schemas!(component_module! { ... })` hands each normalizer's entry (its doc,
//! its JSON `"type"` and its fields) to [`component_module!`], which writes:
//!
//! 1. the kind's class, with [`component_kind!`](macro@super::component_kind), and its
//!    `Sequence`, with [`sequence_class!`](macro@super::sequence_class),
//! 2. one class per component, with a constructor, and a getter and a setter per field,
//! 3. the kind's `#[pymodule]`.
//!
//! A field's shape, such as `Bool`, names its types in [`fields`](super::fields).

/// Writes a kind's Python module from its `tk::*_schemas!` entries.
macro_rules! component_module {
    ({
        kind: $kind:ident,
        module: $module:ident,
        python_module: $python_module:literal,
        tokenizer_field: $tokenizer_field:ident,
        items_key: $items_key:literal $(,)?
    } $(
        $(#[doc = $doc:literal])+
        $class:ident $({
            // A default is a `tt`. pyo3 writes an `expr` default as `...` in the stubs.
            $($key:ident: $field_type:ident $(($choices:expr))? $(= $default:tt)?),* $(,)?
        })?
    ),* $(,)?) => {
        mod classes {
            use pyo3::prelude::*;

            use $crate::components::component_spec::ComponentSpec;
            use $crate::components::macros::fields::{arg, choice_json, choice_name, value};
            use $crate::components::{ComponentKind, JsonFieldsExt};

            $crate::components::macros::component_kind! {
                kind: $kind,
                python_module: $python_module,
                tokenizer_field: $tokenizer_field,
                items_key: Some($items_key),
                classes: { "Sequence" => Sequence, $(stringify!($class) => $class),* },
            }

            $crate::components::macros::sequence_class! {
                kind: $kind,
                python_module: $python_module,
            }

            $(
                $(#[doc = $doc])+
                #[allow(clippy::upper_case_acronyms)]
                #[pyclass(frozen, extends = $kind, module = $python_module)]
                pub struct $class;

                // The getter is `fn $key`, so the setter needs another Rust name, `set_$key`.
                // `paste!` wraps the whole impl because pyo3 reads method names before inner macros
                // expand.
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
                                // The core reads a missing field as its default, and so does
                                // the getter.
                                $(if !slf.has_field(stringify!($key)) {
                                    let default: arg::$field_type = $default;
                                    return Ok(default.into());
                                })?
                                let value = slf.field(stringify!($key))?;
                                $(let value = choice_name(stringify!($key), value, $choices)?;)?
                                Ok(value)
                            }

                            #[setter]
                            fn [<set_ $key>](slf: &Bound<'_, Self>, $key: arg::$field_type) -> PyResult<()> {
                                $(let $key = choice_json(stringify!($key), $key, $choices)?;)?
                                slf.set_field(stringify!($key), $key);
                                Ok(())
                            }
                        )*)?
                    }
                }
            )*
        }

        pub use classes::{$kind, Sequence, $($class),*};

        #[::pyo3::pymodule]
        pub mod $module {
            #[pymodule_export]
            use super::{$kind, Sequence, $($class),*};
        }
    };
}

pub(crate) use component_module;
