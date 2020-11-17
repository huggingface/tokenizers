#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    extern "C++" {
        include!("tests.h");

        // returns null pointer or empty vector on success,
        // vector of error messages otherwise
        pub fn run_tests() -> Result<UniquePtr<CxxVector<CxxString>>>;
    }
}

#[test]
fn run_cpp_test_suite() {
    match ffi::run_tests() {
        Ok(error_messages) =>
            if !(error_messages.is_null() || error_messages.is_empty()) {
                panic!("C++ test suite reported errors: {:#?}", error_messages.as_slice());
            }
        Err(exception) => panic!("C++ test suite threw exception with message: {}", exception.what())
    }
}
