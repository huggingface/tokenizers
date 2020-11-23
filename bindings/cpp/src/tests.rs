#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    extern "C++" {
        include!("tests.h");

        // returns true on success
        pub fn run_tests() -> bool;
    }
}

#[test]
fn run_cpp_test_suite() {
    if !ffi::run_tests() {
        panic!("C++ test suite reported errors");
    }
}
