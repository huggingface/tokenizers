fn main() {
    // On macOS an extension module leaves the interpreter's symbols unresolved until Python
    // loads it. maturin passes the linker flag for that itself; this covers a plain
    // `cargo build`, which stub-gen relies on.
    pyo3_build_config::add_extension_module_link_args();
}
