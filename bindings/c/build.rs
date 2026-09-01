use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    check_every_exported_fn_is_guarded(&Path::new(&crate_dir).join("src"));

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cbindgen::Config::from_root_or_default(&crate_dir))
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(Path::new(&crate_dir).join("include/tokenizers/tokenizers.h"));
}

/// Exported fns that don't need to route through `catch_panic`, with why:
/// - `tk_error_message`, `tk_error_free` (`src/panic.rs`): they're the error-reporting path
///   itself, so there's nowhere left to report a panic in here to; their bodies can't
///   realistically panic either.
const FFI_EXEMPT: &[&str] = &["tk_error_message", "tk_error_free"];

/// Nothing stops writing a `#[unsafe(no_mangle)] pub extern "C" fn` that isn't declared to
/// return `TkHandle<TkError>` -- and every such fn is expected to call `catch_panic`
/// (`src/panic.rs`) as its body's tail expression, which only type-checks if the fn returns
/// exactly that. This parses every `.rs` file under `src/` the same way cbindgen does
/// (statically, via `syn`) and fails the build if any `no_mangle` fn's return type doesn't
/// match, unless the fn is named in `FFI_EXEMPT` above with a comment explaining why. Walking
/// the whole `src/` tree rather than just `lib.rs` matters now that the panic-handling code has
/// its own module: nothing else would notice a new file added later without also being wired
/// into this check.
fn check_every_exported_fn_is_guarded(src_dir: &Path) {
    let unguarded: Vec<String> = rust_files(src_dir)
        .iter()
        .flat_map(|path| unguarded_fns_in(path))
        .collect();

    if !unguarded.is_empty() {
        panic!(
            "{} exported fn(s) (#[unsafe(no_mangle)]) aren't declared `-> TkHandle<TkError>`:\n\
             {}\n\n\
             A panic in one of these would unwind straight into C, which is undefined \
             behavior. Every fn reachable from C must be declared `-> TkHandle<TkError>` and \
             call `catch_panic` (`src/panic.rs`) as its body's tail expression, or be added to \
             FFI_EXEMPT in build.rs with a comment explaining why it doesn't need one.",
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
        .filter(|f| !FFI_EXEMPT.contains(&f.sig.ident.to_string().as_str()))
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

/// Does `f` return `TkHandle<TkError>` (however `TkError` itself is spelled: `TkError`,
/// `panic::TkError`, `crate::panic::TkError`, ...)? Matched structurally rather than by exact
/// path, since the fn's own `use` imports determine which spelling is in scope. Can't tell a
/// `TkHandle<TkError>` from any other `TkHandle<T>` by type alone (aliases aren't resolved
/// here), so this only checks the syntax.
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
    segment.ident == "TkHandle"
        && args.args.len() == 1
        && matches!(&args.args[0], syn::GenericArgument::Type(t) if path_ends_in(t, "TkError"))
}

fn path_ends_in(ty: &syn::Type, name: &str) -> bool {
    let syn::Type::Path(path) = ty else {
        return false;
    };
    path.path.segments.last().is_some_and(|s| s.ident == name)
}
