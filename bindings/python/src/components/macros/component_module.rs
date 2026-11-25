//! Macro to generate the pyo3 definitions of a whole Python module of components, such as
//! `tokenizers.normalizers`.
//!
//! Normalizers, pre-tokenizers and decoders python objects are plain data views:
//! Each of their python classes only reads and writes the keys of a small JSON object.
//! `tk::schemas` lists these components, with their JSON fields.
//! This macro turns such a list into Python classes.
//!
//! The macro calls [`component_kind!`](macro@super::component_kind) to write the base class, and
//! [`sequence_class!`](macro@super::sequence_class) to write the `Sequence` class.
//! Then it writes one class per definition in the list, and the `#[pymodule]` that exports them.
//! `also_exports` adds other classes to the module, such as `DecodeStream` to
//! `tokenizers.decoders`.
//! [`field_types`](super::field_types) gives the Rust type of each field type, such as `Bool`.
//!
//! For example, `tk::schemas` has this definition:
//!
//! ```text
//! /// Removes whitespace from the start and the end of the text.
//! Strip { strip_left: Bool = true, strip_right: Bool = true },
//! ```
//!
//! From it, the macro generates this pyo3 class, shown here with the field types resolved:
//!
//! ```rust,ignore
//! /// Removes whitespace from the start and the end of the text.
//! #[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
//! pub struct Strip;
//!
//! #[pymethods]
//! impl Strip {
//!     #[new]
//!     #[pyo3(signature = (strip_left = true, strip_right = true))]
//!     fn new(strip_left: bool, strip_right: bool) -> PyResult<PyClassInitializer<Self>> {
//!         let spec = ComponentSpec::from_literal(serde_json::json!({
//!             "type": "Strip",
//!             "strip_left": strip_left,
//!             "strip_right": strip_right,
//!         }));
//!         Ok(Normalizer::initializer(spec, Self))
//!     }
//!
//!     #[getter]
//!     fn strip_left(slf: &Bound<'_, Self>) -> PyResult<bool> {
//!         if !slf.has_field("strip_left") {
//!             return Ok(true);
//!         }
//!         slf.field("strip_left")
//!     }
//!
//!     #[setter]
//!     fn set_strip_left(slf: &Bound<'_, Self>, strip_left: bool) -> PyResult<()> {
//!         slf.set_field("strip_left", strip_left);
//!         Ok(())
//!     }
//!
//!     // The same getter and setter for `strip_right`.
//! }
//! ```
//!

/// Writes a whole Python module of components, such as `tokenizers.normalizers`, from a list in
/// `tk::schemas`.
///
/// The list calls this macro with its definitions:
///
/// ```rust,ignore
/// tk::normalizer_schemas!(component_module! {
///     kind: Normalizer,
///     module: normalizers,
///     python_module: "tokenizers.normalizers",
///     tokenizer_field: normalizer,
///     items_key: "normalizers",
/// });
/// ```
///
/// It writes:
///
/// - `Normalizer`, the base class, with [`component_kind!`](macro@super::component_kind).
/// - `Sequence`, with [`sequence_class!`](macro@super::sequence_class).
/// - One class per definition, such as `Strip`.
///   Each class has a constructor, and a getter and a setter per field.
///   A getter returns the field's default when the JSON leaves the field out.
/// - `normalizers`, the `#[pymodule]` that exports all these classes.
///
/// The arguments:
///
/// - `kind` names the base class.
/// - `module` names the `#[pymodule]`, and `python_module` is its path in Python.
/// - `tokenizer_field` names the field of `TokenizerSpec` that stores the normalizer.
/// - `items_key` is the JSON key that holds the items of a normalizer `Sequence`.
///
/// The classes are then in scope where the macro is called.
macro_rules! component_module {
    ({
        kind: $kind:ident,
        module: $module:ident,
        python_module: $python_module:literal,
        tokenizer_field: $tokenizer_field:ident,
        items_key: $items_key:literal
        $(, also_exports: [$($export:path),* $(,)?])? $(,)?
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
            use $crate::components::macros::field_types::{arg, choice_json, choice_name, value};
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
            $($(
                #[pymodule_export]
                use $export;
            )*)?
        }
    };
}

pub(crate) use component_module;
