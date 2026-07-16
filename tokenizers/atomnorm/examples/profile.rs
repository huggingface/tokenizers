//! Tight loop for sampling profilers: cargo run -p atomnorm --release --example profile <lang> <form>
fn main() {
    let lang = std::env::args().nth(1).unwrap_or("he".into());
    let form = std::env::args().nth(2).unwrap_or("nfd".into());
    let path = format!(
        "{}/../atomsplit/benches/data/{lang}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let s = std::fs::read_to_string(&path).unwrap();
    let mut c = s.len().min(180_000);
    while c > 0 && !s.is_char_boundary(c) {
        c -= 1;
    }
    let text = &s[..c];
    let f: fn(&str) -> std::borrow::Cow<'_, str> = match form.as_str() {
        "nfc" => atomnorm::nfc,
        "nfkc" => atomnorm::nfkc,
        "nfkd" => atomnorm::nfkd,
        _ => atomnorm::nfd,
    };
    loop {
        std::hint::black_box(f(std::hint::black_box(text)));
    }
}
