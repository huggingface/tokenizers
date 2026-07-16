//! Bench-only, opt-in build step. Without `--features xxutf` this is a no-op — atomsplit still builds
//! with no C toolchain and no network (the whole crate is otherwise build-script-free). With the feature
//! (only the `normalize` bench needs it), it downloads the pinned MIT xxUTF amalgamation and compiles it,
//! so the bench can call xxUTF's C SIMD normalizer for comparison.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "xxutf")]
    xxutf::build();
}

#[cfg(feature = "xxutf")]
mod xxutf {
    use std::path::Path;
    use std::process::Command;

    const VERSION: &str = "0.2.0";
    const URL: &str =
        "https://github.com/dzfrias/xxUTF/releases/download/v0.2.0/xxutf-amalgamation.zip";

    pub fn build() {
        let out = std::env::var("OUT_DIR").expect("OUT_DIR");
        let c_file = format!("{out}/xxutf-amalgamation-{VERSION}/xxutf.c");
        if !Path::new(&c_file).exists() {
            let zip = format!("{out}/xxutf.zip");
            run(
                Command::new("curl").args(["-fsSL", URL, "-o", &zip]),
                "download xxUTF amalgamation",
            );
            run(
                Command::new("unzip").args(["-oq", &zip, "-d", &out]),
                "unzip xxUTF amalgamation",
            );
        }
        assert!(
            Path::new(&c_file).exists(),
            "xxUTF amalgamation missing after download: {c_file}"
        );
        // -O3, warnings off (it's generated), no libc needed per upstream.
        cc::Build::new()
            .file(&c_file)
            .opt_level(3)
            .warnings(false)
            .compile("xxutf");
    }

    fn run(cmd: &mut Command, what: &str) {
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to {what} (is the tool installed?): {e}"));
        assert!(status.success(), "failed to {what} (exit {status})");
    }
}
