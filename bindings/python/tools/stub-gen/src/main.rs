//! Generates the `.pyi` stubs under `py_src/tokenizers/` from the
//! introspection metadata pyo3 embeds in the built extension (the
//! `experimental-inspect` feature). Run after `maturin develop --release`:
//!
//! ```sh
//! cargo run --manifest-path tools/stub-gen/Cargo.toml
//! ```
//!
//! Return types beyond introspection's reach come from the
//! `#[pyo3(signature = (...) -> "Type")]` annotations in the sources; numpy
//! imports for those annotations are injected here.

use std::path::{Path, PathBuf};

const MODULE: &str = "tokenizers";
/// The `#[pymodule]` name inside the cdylib.
const NATIVE_MODULE: &str = "_native";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("tools/stub-gen sits two levels under the crate root")
        .to_path_buf();
    let cdylib = crate_dir.join(format!(
        "target/release/{}{NATIVE_MODULE}.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    ));
    let out_dir = crate_dir.join("py_src").join(MODULE);

    if !cdylib.is_file() {
        return Err(format!(
            "no cdylib at {} — run `maturin develop --release` first",
            cdylib.display()
        )
        .into());
    }

    let module = pyo3_introspection::introspect_cdylib(&cdylib, NATIVE_MODULE)?;
    assert_has_docstrings(&module);

    for (rel_path, contents) in pyo3_introspection::module_stub_files(&module) {
        let out_path = out_dir.join(place(&rel_path));
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut contents = postprocess(&contents);
        if rel_path == Path::new("__init__.pyi") {
            // `create_exception!` types carry no introspection metadata.
            contents.push_str("\nclass TokenizersError(Exception): ...\n");
        }
        std::fs::write(&out_path, &contents)?;
        println!("generated {}", out_path.display());
    }
    Ok(())
}

/// Map the introspected layout onto the package layout: the root module stub
/// becomes `__init__.pyi`, and each submodule stub lands inside its runtime
/// shim package (`models.pyi` -> `models/__init__.pyi`) so it shadows the
/// `.py` re-exports for type checkers.
fn place(rel_path: &Path) -> PathBuf {
    let name = rel_path
        .file_name()
        .and_then(|n| n.to_str())
        .expect("stub paths are utf-8 files");
    match name.strip_suffix(".pyi") {
        Some("__init__") | None => rel_path.to_path_buf(),
        Some(module) => rel_path.with_file_name(module).join("__init__.pyi"),
    }
}

fn postprocess(contents: &str) -> String {
    // Cross-submodule references come out relative to the extension root;
    // absolutize them to the package.
    let mut contents = contents
        .replace("from . import", &format!("from {MODULE} import"))
        .replace("from .", &format!("from {MODULE}."));
    // Annotated numpy return types need their imports.
    if contents.contains("npt.") || contents.contains("np.") {
        contents = format!(
            "import numpy as np\nimport numpy.typing as npt\n\n{contents}"
        );
    }
    contents
}

/// Fail loudly if introspection came back without docstrings — that means the
/// cdylib was built without `experimental-inspect` (or the feature broke) and
/// the stubs would silently lose all documentation.
fn assert_has_docstrings(module: &pyo3_introspection::model::Module) {
    fn count(module: &pyo3_introspection::model::Module) -> (usize, usize) {
        let mut with_doc = 0;
        let mut total = 0;
        for f in &module.functions {
            total += 1;
            with_doc += f.docstring.is_some() as usize;
        }
        for c in &module.classes {
            total += 1;
            with_doc += c.docstring.is_some() as usize;
            for m in &c.methods {
                total += 1;
                with_doc += m.docstring.is_some() as usize;
            }
        }
        for sub in &module.modules {
            let (w, t) = count(sub);
            with_doc += w;
            total += t;
        }
        (with_doc, total)
    }
    let (with_doc, total) = count(module);
    println!("docstring coverage: {with_doc}/{total}");
    assert!(
        with_doc > 0,
        "introspection returned 0/{total} docstrings — was the cdylib built \
         with the `experimental-inspect` pyo3 feature?"
    );
}
