fn main() {
    let modules = ["normalizers", "pre_tokenizers"];

    let rust_sources: Vec<_> = modules
        .iter()
        .map(|&name| format!("src/{}.rs", name))
        .collect();
    let cpp_headers: Vec<_> = modules
        .iter()
        .map(|&name| format!("src/{}.h", name))
        .collect();

    let standard = "c++14";

    cxx_build::bridges(&rust_sources)
        .include("src")
        .flag_if_supported(format!("-std={}", &standard).as_str())
        .flag_if_supported(format!("/std:{}", &standard).as_str())
        .compile("tokenizers-cpp");

    for file in rust_sources.iter().chain(cpp_headers.iter()) {
        println!("cargo:rerun-if-changed={}", file);
    }
}
