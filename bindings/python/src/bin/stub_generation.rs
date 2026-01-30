#[cfg(feature = "stub-gen")]
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3::types::PyList;
#[cfg(feature = "stub-gen")]
fn main() {
    use std::path::Path;
    let lib_name = format!("{}/tokenizers.abi3.so", env!("CARGO_MANIFEST_DIR"));
    let path = Path::new(&lib_name);
    let so_dir = path.parent().unwrap();
    println!("Initializing python");
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

        let old = std::env::var_os("PYTHONPATH");
        let mut new = std::ffi::OsString::new();
        new.push(so_dir);
        if let Some(old) = old {
            new.push(":");
            new.push(old);
        }
        std::env::set_var("PYTHONPATH", new);
        println!("New PYTHONPATH={:?}", std::env::var_os("PYTHONPATH"));
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
            None, None).unwrap_or_else(|e| panic!("Failed to import tokenizers: {:?}", e));

        env_logger::init();
        println!("Generating stub files");
        let path = Path::new(&lib_name);
        assert!(path.is_file(), "Failed to locate cdylib at {}", lib_name);
        println!("Found cdylib at {}", lib_name);

        let main_module_name = "tokenizers";
        let python_module = pyo3_introspection::introspect_cdylib(path, main_module_name)
            .unwrap_or_else(|_| panic!("Failed introspection of {}", main_module_name));
        let type_stubs = pyo3_introspection::module_stub_files(&python_module);
        let out_dir = Path::new("py_src/tokenizers");

        for (rel_path, contents) in type_stubs {
            let out_path = out_dir.join(&rel_path);
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)
                    .unwrap_or_else(|_| panic!("Failed introspection of {}", main_module_name))
            }
            std::fs::write(&out_path, contents).expect("Failed to write stubs file");
            println!("Generated stub: {}", out_path.display());
        }
    });
}

#[cfg(not(feature = "stub-gen"))]
fn main() {
    panic!(
        "The `stub_generation` binary requires the `stub-gen` feature.\n\
         Run with: cargo run --bin stub_generation --features stub-gen"
    );
}
