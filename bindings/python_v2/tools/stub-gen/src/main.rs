//! Regenerates `python/tokenizers/*.pyi` from the built extension module.
//!
//! `pyo3-introspection` reads a compiled cdylib and turns its `#[pyclass]`/`#[pymethods]`
//! signatures and docstrings into `.pyi` stubs, so the type hints an editor or type checker
//! sees can never drift from what the extension actually exports. Ported from
//! `bindings/python/tools/stub-gen`.

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_introspection::model::{Class, Module};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::try_init().ok();

    let manifest_dir = find_manifest_dir()?;
    // Under `target/`, not the crate root: `python -c`/`python -m pytest` insert cwd (or "") at
    // the front of `sys.path`, so a copy sitting next to `pyproject.toml` would shadow the real,
    // properly-versioned extension under `python/tokenizers/` for anyone running Python from
    // here -- it did, repeatedly, before this moved.
    let cdylib = manifest_dir.join("target/introspection/tokenizers.abi3.so");
    let out_dir = manifest_dir.join("python/tokenizers");
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
        // SAFETY: main is still single-threaded at this point, so no other
        // threads can race on the environment, and the embedded Python
        // interpreter (which reads PYTHONHOME) hasn't been initialized yet.
        // FIXME: doesn't look great
        unsafe { std::env::set_var("PYTHONHOME", &base_prefix) };
    }

    Ok(())
}

fn find_manifest_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Look for the bindings/python_v2 directory relative to current working directory
    // or from the tool's location
    let cwd = std::env::current_dir()?;

    // Check if we're already in bindings/python_v2
    if cwd.join("pyproject.toml").exists() && cwd.join("python").exists() {
        return Ok(cwd);
    }

    // Check if bindings/python_v2 exists relative to cwd
    let bindings_python_v2 = cwd.join("bindings/python_v2");
    if bindings_python_v2.join("pyproject.toml").exists() {
        return Ok(bindings_python_v2);
    }

    // Try to find it from the executable location
    if let Ok(exe) = std::env::current_exe() {
        // Go up from tools/stub-gen/target/... to bindings/python_v2
        let mut path = exe.as_path();
        for _ in 0..10 {
            if let Some(parent) = path.parent() {
                if parent.join("pyproject.toml").exists() && parent.join("python").exists() {
                    return Ok(parent.to_path_buf());
                }
                path = parent;
            }
        }
    }

    Err("Could not find bindings/python_v2 directory. Run from the tokenizers root or bindings/python_v2 directory.".into())
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

        // Sanity check: if docstrings are missing, introspection ran against a cdylib built
        // without pyo3's `experimental-inspect` feature. Fail loudly here rather than silently
        // write out stubs with every docstring stripped.
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

        generate_root_reexport_shim(&python_module, &out_dir)
            .expect("Failed to write python/tokenizers/tokenizers.pyi");

        Ok(())
    })?;

    Ok(())
}

/// `python/tokenizers/tokenizers.pyi`: a hand-typed shim exists only because `ty` can't resolve
/// `tokenizers.tokenizers` (the compiled submodule maturin's `module-name` puts it under) on its
/// own. Generating it from the same introspection data as everything else means it can't drift
/// from what the extension actually exports, the same reason `__init__.pyi` is generated instead
/// of hand-typed. Mirrors the released `tokenizers` package's own `tokenizers.pyi` exactly, down
/// to the `X as X` explicit-reexport style.
fn generate_root_reexport_shim(
    module: &pyo3_introspection::model::Module,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut names: Vec<&str> = module.classes.iter().map(|c| c.name.as_str()).collect();
    names.extend(module.functions.iter().map(|f| f.name.as_str()));
    names.extend(module.attributes.iter().map(|a| a.name.as_str()));
    names.sort_unstable();

    let mut contents = String::from("# Generated content DO NOT EDIT\nfrom tokenizers import (\n");
    for name in names {
        contents.push_str(&format!("    {name} as {name},\n"));
    }
    contents.push_str(")\n");

    let out_path = out_dir.join("tokenizers.pyi");
    std::fs::write(&out_path, contents)?;
    println!("Generated: {}", out_path.display());
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

/// Abort if the introspected module tree is missing docstrings. Usually caused by the cdylib
/// having been built without pyo3's `experimental-inspect` feature (check that it's listed in
/// `bindings/python_v2/Cargo.toml`'s `pyo3` dependency).
fn assert_introspection_has_docstrings(module: &Module) {
    let (with_doc, total) = count_docstrings(module);
    println!(
        "Docstring coverage: {}/{} items carry a docstring",
        with_doc, total
    );
    assert!(
        with_doc > 0,
        "stub-gen produced 0/{} docstrings -- pyo3-introspection is reading the cdylib but \
         every docstring slot is empty. Check that `experimental-inspect` is enabled on the \
         `pyo3` dependency in bindings/python_v2/Cargo.toml.",
        total,
    );
}

fn build_extension(manifest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("Building and installing extension (release)...");
    match Command::new("maturin")
        .current_dir(manifest_dir)
        .args(["develop", "--release"])
        .status()
    {
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "Hint: Failed to run `maturin develop`: {:?}. Is maturin even installed? ;)",
                e
            )
        }
    };

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
    if let Some(parent) = cdylib.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(&built_cdylib, cdylib)?;
    Ok(())
}
