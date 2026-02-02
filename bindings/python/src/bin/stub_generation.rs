#[cfg(feature = "stub-gen")]
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3::types::PyList;
#[cfg(feature = "stub-gen")]
use std::ffi::OsString;
#[cfg(feature = "stub-gen")]
use std::path::{Path, PathBuf};
#[cfg(feature = "stub-gen")]
use std::process::Command;

#[cfg(feature = "stub-gen")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::try_init().ok();

    let cdylib = default_cdylib_path();
    let out_dir = default_out_dir();

    build_extension()?;
    refresh_cdylib(&cdylib)?;
    generate_stubs(&cdylib, &out_dir)?;
    Ok(())
}

#[cfg(feature = "stub-gen")]
fn generate_stubs(cdylib: &Path, out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !cdylib.is_file() {
        return Err(format!("Failed to locate cdylib at {}", cdylib.display()).into());
    }

    println!("Initializing python");
    Python::initialize();
    let cdylib = cdylib.to_path_buf();
    let out_dir = out_dir.to_path_buf();

    Python::attach(|py| -> PyResult<()> {
        println!("Gathering Python environment information...");
        let sys = py.import("sys")?;
        println!("sys.version = {}", sys.getattr("version")?);
        println!("sys.executable = {}", sys.getattr("executable")?);
        println!("sys.prefix = {}", sys.getattr("prefix")?);
        println!("sys.base_prefix = {}", sys.getattr("base_prefix")?);

        let so_dir = cdylib
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        let bindings = sys.getattr("path")?;
        let sys_path = bindings.cast::<PyList>()?;
        sys_path.insert(0, so_dir.to_str().unwrap())?;

        let old = std::env::var_os("PYTHONPATH");
        let mut new = OsString::new();
        new.push(&so_dir);
        if let Some(old) = old {
            new.push(":");
            new.push(old);
        }
        std::env::set_var("PYTHONPATH", &new);
        println!("New PYTHONPATH={:?}", std::env::var_os("PYTHONPATH"));
        let sysconfig = PyModule::import(py, "sysconfig")?;
        let python_version = sysconfig.call_method0("get_python_version")?;
        println!("Using python version: {}", python_version);
        let python_lib = sysconfig.call_method("get_config_var", ("LIBDEST",), None)?;
        println!("Using python lib: {}", python_lib);
        let python_site_packages = sysconfig.call_method("get_path", ("purelib",), None)?;
        println!("Using python site-packages: {}", python_site_packages);
        py.run(
            c"import tokenizers; import sys; print('import ok:', tokenizers.__file__); print('sys.path[0]=', sys.path[0])",
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("Failed to import tokenizers: {:?}", e));

        println!("Generating stub files");
        assert!(
            cdylib.is_file(),
            "Failed to locate cdylib at {}",
            cdylib.display()
        );
        println!("Found cdylib at {}", cdylib.display());

        let main_module_name = "tokenizers";
        let python_module = pyo3_introspection::introspect_cdylib(&cdylib, main_module_name)
            .unwrap_or_else(|_| panic!("Failed introspection of {}", main_module_name));
        let type_stubs = pyo3_introspection::module_stub_files(&python_module);

        for (rel_path, contents) in type_stubs {
            let out_path = out_dir.join(&rel_path);
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)
                    .unwrap_or_else(|_| panic!("Failed introspection of {}", main_module_name))
            }
            std::fs::write(&out_path, contents).expect("Failed to write stubs file");
            println!("Generated stub: {}", out_path.display());
        }

        Ok(())
    })?;

    Ok(())
}

#[cfg(feature = "stub-gen")]
fn build_extension() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building and installing extension (release with stub-gen enabled)...");
    let status = Command::new("maturin")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["develop", "--release", "--features", "stub-gen"])
        .status()?;

    if !status.success() {
        return Err("`maturin develop` failed".into());
    }

    Ok(())
}

#[cfg(feature = "stub-gen")]
fn refresh_cdylib(cdylib: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let built_cdylib = built_cdylib_path();
    if !built_cdylib.is_file() {
        return Err(format!(
            "Could not find built cdylib at {}. Pass --build or provide --cdylib.",
            built_cdylib.display()
        )
        .into());
    }

    println!(
        "Refreshing cdylib used for introspection: {}",
        cdylib.display()
    );
    std::fs::copy(&built_cdylib, cdylib)?;
    Ok(())
}

#[cfg(feature = "stub-gen")]
fn built_cdylib_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "target/release/{}tokenizers.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    ))
}

#[cfg(feature = "stub-gen")]
fn default_cdylib_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tokenizers.abi3.so")
}

#[cfg(feature = "stub-gen")]
fn default_out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("py_src/tokenizers")
}

#[cfg(not(feature = "stub-gen"))]
fn main() {
    panic!(
        "The `stub_generation` binary requires the `stub-gen` feature.\n\
         Run with: cargo run --bin stub_generation --features stub-gen"
    );
}
