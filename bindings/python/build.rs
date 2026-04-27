// Re-emit pyo3's build cfgs (notably `Py_GIL_DISABLED` on free-threaded Python)
// for this crate so we can `#[cfg(Py_GIL_DISABLED)]` in our own source. Cargo's
// `rustc-cfg` directives don't propagate to dependents by default.
fn main() {
    pyo3_build_config::use_pyo3_cfgs();
}
