//! Regenerates `python/tokenizers/tokenizers.pyi` from the built extension module.
//!
//! pyo3, built with its `experimental-inspect` feature, embeds every `#[pyclass]` and
//! `#[pymethods]` signature and docstring in the cdylib. `pyo3-introspection` reads them back
//! out of the binary, so the stub a type checker sees cannot drift from what the extension
//! exports.

use std::error::Error;
use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    // tools/stub-gen/ sits two levels below the extension crate. `cargo run` sets the variable
    // at run time; a compile-time `env!` would keep pointing at wherever the checkout used to be.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let crate_dir = Path::new(&manifest_dir).ancestors().nth(2).unwrap();

    let built = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(crate_dir)
        .status()?;
    if !built.success() {
        return Err("cargo build --release failed".into());
    }

    let cdylib = crate_dir.join(format!(
        "target/release/{}tokenizers.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    ));
    let module = pyo3_introspection::introspect_cdylib(&cdylib, "tokenizers")?;
    if module.classes.iter().all(|c| c.docstring.is_none()) {
        return Err(format!(
            "no class in {} carries a docstring: was it built with pyo3's `experimental-inspect` feature?",
            cdylib.display()
        )
        .into());
    }

    let mut stubs = pyo3_introspection::module_stub_files(&module);
    let stub = stubs
        .remove(Path::new("__init__.pyi"))
        .ok_or("pyo3-introspection wrote no root stub")?;
    if !stubs.is_empty() {
        return Err("the extension grew submodules; decide where their stubs go".into());
    }
    let out = crate_dir.join("python/tokenizers/tokenizers.pyi");
    std::fs::write(&out, stub)?;
    println!("wrote {}", out.display());
    Ok(())
}
