//! Bakes the shared classify tables at compile time via the `bitmap_gen` crate (build-dep, uses
//! `unicode_categories`; not a runtime dep). Writes `$OUT_DIR/atom_tables.rs`, which
//! `src/atom_tables.rs` `include!`s. `bitmap_gen` self-validates the tables — a mismatch fails the build.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let out = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out}/atom_tables.rs"), bitmap_gen::generate_atom_tables()).unwrap();
}
