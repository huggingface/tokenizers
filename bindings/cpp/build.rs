fn main() {
    #[allow(unused_mut)]
    let mut modules = vec!["normalizers", "pre_tokenizers"];

    // can't do just cfg!(test), see https://github.com/rust-lang/cargo/issues/2549
    if cfg!(feature = "test") {
        modules.push("tests");
    }

    let rust_sources: Vec<_> = modules
        .iter()
        .map(|&name| format!("src/{}.rs", name))
        .collect();
    let mut cpp_headers: Vec<_> = modules
        .iter()
        .map(|&name| format!("src/{}.h", name))
        .collect();
    cpp_headers.push("src/tokenizers_util.h".to_string());

    let standard = "c++14";

    cxx_build::bridges(&rust_sources)
        .includes(&["src", "thirdparty"])
        .flag_if_supported(format!("-std={}", &standard).as_str())
        .flag_if_supported(format!("/std:{}", &standard).as_str())
        // enable exception handling for MSVC
        .flag_if_supported("/EHsc")
        .compile("tokenizers-cpp");

    for file in rust_sources.iter().chain(cpp_headers.iter()) {
        println!("cargo:rerun-if-changed={}", file);
    }

    if cfg!(feature = "test") {
        cc::Build::new()
            .includes(&["src", "target/cxxbridge", "thirdparty"])
            .file("src/redefine_result_tests.cpp")
            .flag_if_supported("/EHsc")
            .compile("redefine_result_tests");
    }
}
