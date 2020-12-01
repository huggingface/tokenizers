#[macro_export]
macro_rules! forward_cxx_enum {
    ($e:expr, $enum_name:ident, $($enum_value:ident),+) => {
        match $e {
            $(ffi::$enum_name::$enum_value => tk::$enum_name::$enum_value,)+
            x => panic!("Illegal {} value {}", stringify!($enum_name), x.repr)
        }
    }
}
