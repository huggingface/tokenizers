use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_introspection::model::{Class, Module};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::try_init().ok();

    let manifest_dir = find_manifest_dir()?;
    let cdylib = manifest_dir.join("tokenizers.abi3.so");
    let out_dir = manifest_dir.join("py_src/tokenizers");
    println!("Using manifest directory: {}", manifest_dir.display());
    println!("Using cdylib: {}", cdylib.display());
    println!("Using output directory: {}", out_dir.display());
    build_extension(&manifest_dir)?;
    refresh_cdylib(&manifest_dir, &cdylib)?;
    setup_python_env()?;
    generate_stubs(&cdylib, &out_dir)?;
    Ok(())
}

/// Set up PYTHONHOME environment variable if not already set.
/// This is needed for PyO3 embedded Python to find the standard library,
/// especially when using virtual environments created by uv.
fn setup_python_env() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("PYTHONHOME").is_some() {
        return Ok(());
    }

    // Query Python for its base_prefix (the actual Python installation, not venv)
    let output = Command::new("python3")
        .args(["-c", "import sys; print(sys.base_prefix, end='')"])
        .output()?;

    if !output.status.success() {
        return Err("Failed to query Python base_prefix".into());
    }

    let base_prefix = String::from_utf8(output.stdout)?;
    if !base_prefix.is_empty() {
        println!("Setting PYTHONHOME={}", base_prefix);
        std::env::set_var("PYTHONHOME", &base_prefix);
    }

    Ok(())
}

fn find_manifest_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Look for the bindings/python directory relative to current working directory
    // or from the tool's location
    let cwd = std::env::current_dir()?;

    // Check if we're already in bindings/python
    if cwd.join("pyproject.toml").exists() && cwd.join("py_src").exists() {
        return Ok(cwd);
    }

    // Check if bindings/python exists relative to cwd
    let bindings_python = cwd.join("bindings/python");
    if bindings_python.join("pyproject.toml").exists() {
        return Ok(bindings_python);
    }

    // Try to find it from the executable location
    if let Ok(exe) = std::env::current_exe() {
        // Go up from tools/stub-gen/target/... to bindings/python
        let mut path = exe.as_path();
        for _ in 0..10 {
            if let Some(parent) = path.parent() {
                if parent.join("pyproject.toml").exists() && parent.join("py_src").exists() {
                    return Ok(parent.to_path_buf());
                }
                path = parent;
            }
        }
    }

    Err("Could not find bindings/python directory. Run from the tokenizers root or bindings/python directory.".into())
}

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

        // Sanity check: if docstrings are missing the patched pyo3 in
        // .cargo/config.toml didn't actually apply to the cdylib build (most
        // common cause: the rev pinned in tools/stub-gen/Cargo.toml and the
        // version requirement in bindings/python/Cargo.toml have drifted, so
        // cargo silently ignores `[patch.crates-io]`). Check that at least
        // one well-known class still carries its docstring before writing
        // out otherwise-empty stubs.
        assert_introspection_has_docstrings(&python_module);

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

/// Walk the introspected module tree and count classes, functions, and
/// attributes that carry a docstring. Returns `(with_docstring, total)`.
fn count_docstrings(module: &Module) -> (usize, usize) {
    let (mut with_doc, mut total) = (0, 0);
    for f in &module.functions {
        total += 1;
        if f.docstring.is_some() {
            with_doc += 1;
        }
    }
    for a in &module.attributes {
        total += 1;
        if a.docstring.is_some() {
            with_doc += 1;
        }
    }
    fn walk_class(c: &Class, with_doc: &mut usize, total: &mut usize) {
        *total += 1;
        if c.docstring.is_some() {
            *with_doc += 1;
        }
        for m in &c.methods {
            *total += 1;
            if m.docstring.is_some() {
                *with_doc += 1;
            }
        }
        for inner in &c.inner_classes {
            walk_class(inner, with_doc, total);
        }
    }
    for c in &module.classes {
        walk_class(c, &mut with_doc, &mut total);
    }
    for sub in &module.modules {
        let (sw, st) = count_docstrings(sub);
        with_doc += sw;
        total += st;
    }
    (with_doc, total)
}

/// Abort if the introspected module tree is missing docstrings — usually
/// caused by `[patch.crates-io]` in `.cargo/config.toml` not actually
/// applying to the cdylib build. Better to fail loudly here than silently
/// strip every docstring from the published stubs.
fn assert_introspection_has_docstrings(module: &Module) {
    let (with_doc, total) = count_docstrings(module);
    println!(
        "Docstring coverage: {}/{} items carry a docstring",
        with_doc, total
    );
    assert!(
        with_doc > 0,
        "stub-gen produced 0/{} docstrings — pyo3-introspection is reading \
         the cdylib but every docstring slot is empty. Most likely the \
         `[patch.crates-io]` in `.cargo/config.toml` (injected by `make \
         style`) did not actually apply: the rev pinned in \
         tools/stub-gen/Cargo.toml must match the version requirement in \
         bindings/python/Cargo.toml. Check `cargo build` output for \
         `warning: patch \\`pyo3 vX.Y.Z\\` was not used in the crate graph` \
         and align the versions before re-running.",
        total,
    );
}

fn build_extension(manifest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("Building and installing extension (release)...");
    match Command::new("maturin").current_dir(manifest_dir).args(["develop", "--release"]).status() {
        Ok(_) => {}
        Err(e) => { eprintln!("Hint: Failed to run `maturin develop`: {:?}. Is maturin even installed? ;)", e) }
    } ;

    Ok(())
}

fn refresh_cdylib(manifest_dir: &Path, cdylib: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let built_cdylib = manifest_dir.join(format!(
        "target/release/{}tokenizers.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    ));

    if !built_cdylib.is_file() {
        return Err(format!(
            "Could not find built cdylib at {}.",
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
