//! Components with no settings, such as `NFC`: their JSON is `{"type": ...}` alone.
//!
//! Each needs a Python class all the same, so that `isinstance` and pickling see its type.
//! [`stateless_component!`] writes that class: a constructor and nothing else.

/// Defines a frozen pyclass with no arguments. Its `name` is also its JSON `"type"`.
///
/// The kind's `from_spec` still lists the `name`, so that a loaded component gets this class.
macro_rules! stateless_component {
    (
        $(#[doc = $doc:literal])*
        #[pyclass(extends = $kind:ident, name = $json_type:literal, module = $module:literal)]
        pub struct $class:ident;
    ) => {
        $(#[doc = $doc])*
        #[::pyo3::pyclass(frozen, extends = $kind, name = $json_type, module = $module)]
        pub struct $class;

        #[::pyo3::pymethods]
        impl $class {
            #[new]
            fn new() -> ::pyo3::PyClassInitializer<Self> {
                let spec = $crate::components::component_spec::ComponentSpec::from_literal(
                    ::serde_json::json!({"type": $json_type}),
                );
                <$kind as $crate::components::ComponentKind>::initializer(spec, Self)
            }
        }
    };
}

pub(crate) use stateless_component;
