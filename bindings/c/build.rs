//! Generates `include/tokenizers/tokenizers.h` from this crate's `#[unsafe(no_mangle)]` exports with cbindgen.
//! Before generating, we check that every exported fn runs under a panic guard. A panic reaching an
//! `extern "C"` boundary aborts the whole host process; see `check_every_exported_fn_catches_panic`.

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

// Exported fns that run no code that could panic, and so are allowed to skip the guard.
const PANIC_CATCH_EXEMPT: &[&str] = &[
    "tk_error_message",
    "tk_string_cstr",
    "tk_string_len",
    "tk_encoding_ids",
    "tk_encoding_type_ids",
];

// The helpers that run a body under `catch_unwind` and report the outcome as a `*mut Error`.
const GUARDS: &[&str] = &["wrap_in_ptr", "catch_panic"];

/// Fails the build unless every exported fn either runs its whole body under one of the [`GUARDS`]
/// and returns the resulting `*mut Error`, is a `*_free` fn whose whole body is `free_ptr(..)`, or is
/// listed in [`PANIC_CATCH_EXEMPT`].
fn check_every_exported_fn_catches_panic(src_dir: &Path) {
    let unguarded: Vec<String> = rust_files(src_dir)
        .iter()
        .flat_map(|path| unguarded_fns_in(path))
        .collect();

    if !unguarded.is_empty() {
        panic!(
            "{} exported fn(s) (#[unsafe(no_mangle)]) don't catch panics:\n\
             {}\n\n\
             A panic in one of these would reach the `extern \"C\"` boundary and abort the host process. \
             Every fn reachable from C must return `*mut Error` and call to one of {GUARDS:?} (see `src/utils.rs` and `src/error.rs`); \
             or be a `*_free` fn whose whole body is `free_ptr(..)`;
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

/// Whether `f`'s entire body is `free_ptr(..)`, optionally inside `unsafe { }`.
fn is_free_helper(f: &syn::ItemFn) -> bool {
    let [stmt] = f.block.stmts.as_slice() else {
        return false;
    };
    called_fn(stmt).is_some_and(|name| name == "free_ptr")
}

/// Whether `f` returns `*mut Error` and runs everything it does under one of the [`GUARDS`]: its
/// body is closure `let`s (which run nothing) followed by the guard call as the tail expression.
fn is_guarded(f: &syn::ItemFn) -> bool {
    let Some((tail, before)) = f.block.stmts.split_last() else {
        return false;
    };
    returns_error_ptr(f)
        && before.iter().all(is_closure_binding)
        && called_fn(tail).is_some_and(|name| GUARDS.contains(&name.as_str()))
}

fn returns_error_ptr(f: &syn::ItemFn) -> bool {
    let syn::ReturnType::Type(_, ty) = &f.sig.output else {
        return false;
    };
    let syn::Type::Ptr(ptr) = ty.as_ref() else {
        return false;
    };
    ptr.mutability.is_some() && path_ends_in(&ptr.elem, "Error")
}

fn is_closure_binding(stmt: &syn::Stmt) -> bool {
    let syn::Stmt::Local(local) = stmt else {
        return false;
    };
    local
        .init
        .as_ref()
        .is_some_and(|init| matches!(init.expr.as_ref(), syn::Expr::Closure(_)))
}

/// The name of the fn `stmt` calls when it is `f(..)` or `unsafe { f(..) }`.
fn called_fn(stmt: &syn::Stmt) -> Option<String> {
    let syn::Stmt::Expr(expr, _) = stmt else {
        return None;
    };
    let expr = match expr {
        syn::Expr::Unsafe(unsafe_block) => {
            let [syn::Stmt::Expr(inner, _)] = unsafe_block.block.stmts.as_slice() else {
                return None;
            };
            inner
        }
        other => other,
    };
    let syn::Expr::Call(call) = expr else {
        return None;
    };
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    path.path.get_ident().map(ToString::to_string)
}

fn path_ends_in(ty: &syn::Type, name: &str) -> bool {
    let syn::Type::Path(path) = ty else {
        return false;
    };
    path.path.segments.last().is_some_and(|s| s.ident == name)
}
