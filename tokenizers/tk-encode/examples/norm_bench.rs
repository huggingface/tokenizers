//! End-to-end BertNormalizer bench across scripts, three ways:
//!   current  = pipeline `Normalizer::normalize` (the ASCII-lane path on the branch)
//!   tag+scal = classify_scalar::<NormClass> + normalize_from_tags
//!   tag+simd = classify::<NormClass>        + normalize_from_tags
//! The two tag ways differ ONLY in how the classification is produced; both include classify + normalize.
//!   cargo run --release --example norm_bench
use atomsplit::classify::{classify, classify_scalar};
use atomsplit::norm_classify::NormClass;
use std::time::Instant;
use tk_encode::normalizers::BertNormalizer;
use tk_encode::pipeline::Normalizer as _;

fn samples() -> Vec<(&'static str, String)> {
    let base: &[(&str, &str)] = &[
        ("eng_Latn", "The quick brown fox Jumps over the LAZY dog. "),
        (
            "rus_Cyrl",
            "Съешь же ещё этих мягких французских булок да выпей чаю. ",
        ),
        (
            "ell_Grek",
            "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία και ζω χαρούμενος. ",
        ),
        ("arb_Arab", "هذا نصٌّ عربيٌّ للاختبار مع بعض الحركات والتشكيل. "),
        ("heb_Hebr", "זֶהוּ טֶקְסְט עִבְרִי לִבְדִיקָה עִם נִקּוּד מָלֵא. "),
        ("hin_Deva", "यह एक हिन्दी परीक्षण वाक्य है जिसमें कुछ शब्द हैं। "),
        ("ben_Beng", "এটি একটি বাংলা পরীক্ষার বাক্য যাতে কিছু শব্দ আছে। "),
        ("tam_Taml", "இது சில சொற்களைக் கொண்ட ஒரு தமிழ் சோதனை வாக்கியம். "),
        ("tha_Thai", "นี่คือประโยคทดสอบภาษาไทยที่มีคำหลายคำอยู่ในนั้น "),
        ("amh_Ethi", "ይህ አንዳንድ ቃላትን የያዘ የአማርኛ የሙከራ ዓረፍተ ነገር ነው። "),
        (
            "kat_Geor",
            "ეს არის ქართული სატესტო წინადადება რამდენიმე სიტყვით. ",
        ),
        ("cmn_Hani", "这是一个包含若干汉字的中文测试句子。 "),
        (
            "jpn_Jpan",
            "これは日本語のテスト文で、いくつかの単語を含みます。 ",
        ),
        (
            "kor_Hang",
            "이것은 몇 개의 단어를 포함하는 한국어 테스트 문장입니다. ",
        ),
    ];
    base.iter().map(|(n, s)| (*n, s.repeat(5000))).collect()
}

fn time<F: FnMut()>(bytes: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let iters = 30;
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_nanos() as f64 / (iters as f64 * bytes as f64)
}

fn main() {
    let n = BertNormalizer::new(true, true, None, true); // bert-base-uncased config
    println!(
        "{:>9}  {:>8}  {:>8}  {:>8}   speedup(simd vs current)",
        "script", "current", "tag+scal", "tag+simd"
    );
    let mut tags = Vec::new();
    let mut sink = 0usize;
    for (name, text) in samples() {
        let b = text.len();

        let cur = time(b, || {
            sink = sink.wrapping_add(n.normalize(&text).unwrap().len())
        });

        let scal = time(b, || {
            tags.clear();
            tags.resize(text.len(), 0);
            classify_scalar::<NormClass>(text.as_bytes(), &mut tags);
            sink = sink.wrapping_add(n.normalize_from_tags(&text, &tags).len());
        });

        let simd = time(b, || {
            tags.clear();
            tags.resize(text.len(), 0);
            classify::<NormClass>(text.as_bytes(), &mut tags);
            sink = sink.wrapping_add(n.normalize_from_tags(&text, &tags).len());
        });

        println!(
            "{name:>9}  {cur:7.2}   {scal:7.2}   {simd:7.2}    {:.2}x",
            cur / simd
        );
    }
    eprintln!("(sink={sink})");
}
