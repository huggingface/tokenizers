#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    unsafe extern "C++" {
        include!("tokenizers-cpp/tests.h");

        // returns true on success
        pub fn run_tests(data_dir: &str) -> bool;
    }
}

#[test]
fn run_cpp_test_suite() {
    let mut data_dir = std::env::current_dir().expect("Don't have a working directory?");
    data_dir.push("../../tokenizers/data");
    if !data_dir.is_dir() {
        panic!("{} should be the directory containing data files, please run `make test` in tokenizers", data_dir.to_string_lossy());
    }
    if !ffi::run_tests(
        data_dir
            .to_str()
            .expect("Working directory is not valid UTF-8"),
    ) {
        panic!("C++ test suite reported errors");
    }
}
