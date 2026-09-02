//! Generates `include/tokenizers/tokenizers.h` from this crate's `#[unsafe(no_mangle)]` exports with cbindgen.
//! Before generating, we check that every exported fn is guarded against panics unwinding into C, which is undefined behavior;
//! see `check_every_exported_fn_catches_panic`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    check_every_exported_fn_catches_panic(&Path::new(&crate_dir).join("src"));

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cbindgen::Config::from_root_or_default(&crate_dir))
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(Path::new(&crate_dir).join("include/tokenizers/tokenizers.h"));
}

// Allowlist of exported functions that don't require to catch panic unwinding
const PANIC_CATCH_EXEMPT: &[&str] = &["tk_error_message"];

/// Checks every exported functions properly handle panics and returns a *mut Handle<Error>
fn check_every_exported_fn_catches_panic(src_dir: &Path) {
    let unguarded: Vec<String> = rust_files(src_dir)
        .iter()
        .flat_map(|path| unguarded_fns_in(path))
        .collect();

    if !unguarded.is_empty() {
        panic!(
            "{} exported fn(s) (#[unsafe(no_mangle)]) aren't declared `-> Handle<Error>`:\n\
             {}\n\n\
             A panic in one of these would unwind into C code, which is undefined behavior.\
             Every fn reachable from C must be declared `-> Handle<Error>` and call `catch_panic` (`src/panic.rs`) \
             as its body's tail expression, delegate its whole body to `free_handle` like the other `*_free` fns do, \
             or be added to PANIC_CATCH_EXEMPT in build.rs.",
            unguarded.len(),
            unguarded
                .iter()
                .map(|f| format!("  - {f}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
}

fn unguarded_fns_in(path: &Path) -> Vec<String> {
    let src = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("ffi-guard check: couldn't read {}: {e}", path.display()));
    let file = syn::parse_file(&src)
        .unwrap_or_else(|e| panic!("ffi-guard check: couldn't parse {}: {e}", path.display()));

    file.items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(f) => Some(f),
            _ => None,
        })
        .filter(|f| f.attrs.iter().any(is_no_mangle))
        .filter(|f| !PANIC_CATCH_EXEMPT.contains(&f.sig.ident.to_string().as_str()))
        .filter(|f| !is_free_helper(f))
        .filter(|f| !is_guarded(f))
        .map(|f| format!("{} ({})", f.sig.ident, path.display()))
        .collect()
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("ffi-guard check: couldn't read {}: {e}", dir.display()))
        .flat_map(|entry| {
            let path = entry
                .unwrap_or_else(|e| panic!("ffi-guard check: couldn't read a dir entry: {e}"))
                .path();
            if path.is_dir() {
                rust_files(&path)
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                vec![path]
            } else {
                vec![]
            }
        })
        .collect()
}

fn is_no_mangle(attr: &syn::Attribute) -> bool {
    if attr.path().is_ident("no_mangle") {
        return true;
    }
    // `#[unsafe(no_mangle)]`: `unsafe` is the attribute's path, `no_mangle` is its argument.
    attr.path().is_ident("unsafe")
        && attr
            .parse_args::<syn::Path>()
            .is_ok_and(|inner| inner.is_ident("no_mangle"))
}

/// Checks whether `f`'s entire body is `unsafe { free_handle(..) }`
fn is_free_helper(f: &syn::ItemFn) -> bool {
    let [syn::Stmt::Expr(syn::Expr::Unsafe(unsafe_block), _)] = f.block.stmts.as_slice() else {
        return false;
    };
    let [syn::Stmt::Expr(syn::Expr::Call(call), _)] = unsafe_block.block.stmts.as_slice() else {
        return false;
    };
    matches!(call.func.as_ref(), syn::Expr::Path(p) if p.path.is_ident("free_handle"))
}

/// Checks whether `f` returns `Handle<Error>` and catches panic unwinding
fn is_guarded(f: &syn::ItemFn) -> bool {
    let syn::ReturnType::Type(_, ty) = &f.sig.output else {
        return false;
    };
    let syn::Type::Path(path) = ty.as_ref() else {
        return false;
    };
    let Some(segment) = path.path.segments.last() else {
        return false;
    };
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return false;
    };
    segment.ident == "Handle"
        && args.args.len() == 1
        && matches!(&args.args[0], syn::GenericArgument::Type(t) if path_ends_in(t, "Error"))
}

fn path_ends_in(ty: &syn::Type, name: &str) -> bool {
    let syn::Type::Path(path) = ty else {
        return false;
    };
    path.path.segments.last().is_some_and(|s| s.ident == name)
}
