#[macro_export]
macro_rules! forward_cxx_enum {
    ($e:expr, $enum_name:ident, $($enum_value:ident),+) => {
        match $e {
            $(ffi::$enum_name::$enum_value => tk::$enum_name::$enum_value,)+
            x => panic!("Illegal {} value {}", stringify!($enum_name), x.repr)
        }
    }
}

#[macro_export]
macro_rules! wrap_option {
    ($e:expr, $struct_name:ident, $default:expr) => {{
        let x = $e;
        $struct_name {
            has_value: x.is_some(),
            value: x.unwrap_or($default),
        }
    }}
}

// TODO why is this necessary?
#[macro_export]
macro_rules! impl_extern_type {
    ($name:ident, $ffi_name:literal) => {
        unsafe impl cxx::ExternType for $name {
            type Id = cxx::type_id!($ffi_name);

            type Kind = cxx::kind::Opaque;
        }
    };
}
