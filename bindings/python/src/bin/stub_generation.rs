use pyo3::prelude::*;

#[cfg(feature = "stub-gen")]
fn main() {
    use std::path::{Path, PathBuf};
    // Python::attach(|py| {
    //     let sysconfig = PyModule::import(py, "sysconfig").unwrap();
    //     let python_version = sysconfig.call_method0("get_python_version").unwrap();
    //     println!("Using python version: {}", python_version);
    //     let python_lib = sysconfig
    //         .call_method("get_config_var", ("LIBDEST",), None)
    //         .unwrap();
    //     println!("Using python lib: {}", python_lib);
    //     let python_site_packages = sysconfig
    //         .call_method("get_path", ("purelib",), None)
    //         .unwrap();
    //     println!("Using python site-packages: {}", python_site_packages);
    // });

    env_logger::init();
    println!("Generating stub files");
    let lib_name = String::from("/home/arthur/Work/tokenizers/bindings/python/target/release/libtokenizers.so");
    let path = Path::new(&lib_name);
    assert!(path.is_file(), "Failed to locate cdylib at {}", lib_name);
    println!("Found cdylib at {}", lib_name);

    let main_module_name = "tokenizers";
    let python_module = pyo3_introspection::introspect_cdylib(path, main_module_name)
        .expect(format!("Failed introspection of {}", main_module_name).as_str());
    let type_stubs = pyo3_introspection::module_stub_files(&python_module);

    let stubst_string = type_stubs
        .get(&PathBuf::from("__init__.pyi"))
        .expect("Failed to get __init__.pyi");
    std::fs::write("tokenizers.pyi", stubst_string).expect("Failed to write stubs file");
    println!("Generated stubs: {}", "tokenizers.pyi")
} 

#[cfg(not(feature = "stub-gen"))]
fn main() {
    panic!(
        "The `stub_generation` binary requires the `stub-gen` feature.\n\
         Run with: cargo run --bin stub_generation --features stub-gen"
    );
}
