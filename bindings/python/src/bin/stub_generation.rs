use pyo3::prelude::*;
use pyo3::types::PyList;
#[cfg(feature = "stub-gen")]
fn main() {
    use std::path::{Path, PathBuf};
    let lib_name = "/home/arthur/Work/tokenizers/bindings/python/tokenizers.abi3.so";
    let path = Path::new(lib_name);
    let so_dir = path.parent().unwrap();
    Python::initialize();
    println!("Gathering Python environment information...");
    Python::attach(|py| {
        let sys = py.import("sys").unwrap();
        println!("sys.version = {}", sys.getattr("version").unwrap());
        println!("sys.executable = {}", sys.getattr("executable").unwrap());
        println!("sys.prefix = {}", sys.getattr("prefix").unwrap());
        println!("sys.base_prefix = {}", sys.getattr("base_prefix").unwrap());

        let bindings = sys.getattr("path").unwrap();
        let sys_path = bindings.cast::<PyList>().unwrap();
        sys_path.insert(0, so_dir.to_str().unwrap()).unwrap();
        use std::path::Path;

        let lib_path = Path::new("/home/arthur/Work/tokenizers/bindings/python/tokenizers.abi3.so");
        let so_dir = lib_path.parent().unwrap();

        let old = std::env::var_os("PYTHONPATH");
        let mut new = std::ffi::OsString::new();
        new.push(so_dir);
        if let Some(old) = old {
            new.push(":");
            new.push(old);
        }
        std::env::set_var("PYTHONPATH", new);
        let sysconfig = PyModule::import(py, "sysconfig").unwrap();
        let python_version = sysconfig.call_method0("get_python_version").unwrap();
        println!("Using python version: {}", python_version);
        let python_lib = sysconfig
            .call_method("get_config_var", ("LIBDEST",), None)
            .unwrap();
        println!("Using python lib: {}", python_lib);
        let python_site_packages = sysconfig
            .call_method("get_path", ("purelib",), None)
            .unwrap();
        println!("Using python site-packages: {}", python_site_packages);
        py.run(c"import tokenizers; import sys; print('import ok:', tokenizers.__file__); print('sys.path[0]=', sys.path[0])",
           None, None).unwrap();
        env_logger::init();
        println!("Generating stub files");
        let lib_name =
            String::from("/home/arthur/Work/tokenizers/bindings/python/tokenizers.abi3.so");
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
    });
}

#[cfg(not(feature = "stub-gen"))]
fn main() {
    panic!(
        "The `stub_generation` binary requires the `stub-gen` feature.\n\
         Run with: cargo run --bin stub_generation --features stub-gen"
    );
}
