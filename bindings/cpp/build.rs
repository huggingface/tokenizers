use std::env;

use cc::Build;

fn main() {
    #[allow(unused_mut)]
    let mut modules = vec![
        "normalizers",
        "pre_tokenizers",
        "models",
        "processors",
        "decoders",
        "tokenizer",
        "tokens",
    ];

    // can't do just cfg!(test), see https://github.com/rust-lang/cargo/issues/2549
    if cfg!(feature = "test") {
        modules.push("tests");
    }

    let root_dir = env::current_dir()
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();

    let rust_sources: Vec<_> = modules
        .iter()
        .map(|&name| format!("tokenizers-cpp/{}.rs", name))
        .collect();
    let mut cpp_headers: Vec<_> = modules
        .iter()
        .map(|&name| format!("tokenizers-cpp/{}.h", name))
        .collect();
    cpp_headers.extend_from_slice(&[
        "tokenizers-cpp/common.h".to_string(),
        "tokenizers-cpp/input_sequence.h".to_string(),
    ]);

    let standard = "c++14";
    let out_dir = env::var("OUT_DIR").unwrap();
    let generated_include_dir = format!("{}/cxxbridge/include/tokenizers-cpp", out_dir);
    let thirdparty_dir = format!("{}/thirdparty", root_dir);
    let include_dirs = &[root_dir, generated_include_dir, thirdparty_dir];

    let compile = |build: &mut Build, output: &str| {
        build
            .includes(include_dirs)
            .flag_if_supported(format!("-std={}", &standard).as_str())
            .flag_if_supported(format!("/std:{}", &standard).as_str())
            // enable exception handling for MSVC
            .flag_if_supported("/EHsc")
            .compile(output);
    };

    compile(&mut cxx_build::bridges(&rust_sources), "tokenizers-cpp");

    for file in rust_sources.iter().chain(cpp_headers.iter()) {
        println!("cargo:rerun-if-changed={}", file);
    }

    if cfg!(feature = "test") {
        compile(
            cc::Build::new()
                .file("tokenizers-cpp/redefine_result_tests.cpp")
                .include(format!("{}/cxxbridge/include", out_dir)),
            "redefine_result_tests",
        );
    }
}
