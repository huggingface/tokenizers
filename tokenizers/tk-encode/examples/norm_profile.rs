//! One consolidated table per script (ns/byte): SIMD classify, scalar classify, then the normalization
//! broken down per stage (copy / nfd / lower / nfd+strip / full = nfd+strip+lower, the bert transform).
//!   cargo run --release --example norm_profile
use atomsplit::classify::{classify, classify_scalar};
use atomsplit::norm_classify::NormClass;
use std::time::Instant;
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

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

fn time<F: FnMut() -> usize>(bytes: usize, mut f: F) -> f64 {
    let mut s = 0usize;
    for _ in 0..3 {
        s = s.wrapping_add(f());
    }
    let it = 30;
    let t = Instant::now();
    for _ in 0..it {
        s = s.wrapping_add(f());
    }
    std::hint::black_box(s);
    t.elapsed().as_nanos() as f64 / (it as f64 * bytes as f64)
}

fn main() {
    let mut tags = Vec::new();
    println!("ns/byte per script — classify (simd/scalar), then normalization per stage");
    println!(
        "{:>9} │ {:>8} {:>8} │ {:>5} {:>6} {:>6} {:>9} {:>6}",
        "script", "cls_simd", "cls_scal", "copy", "nfd", "lower", "nfd+strip", "full"
    );
    for (name, t) in samples() {
        let b = t.len();
        tags.resize(b, 0);
        let cls_simd = time(b, || {
            classify::<NormClass>(t.as_bytes(), &mut tags);
            tags[0] as usize
        });
        let cls_scal = time(b, || {
            classify_scalar::<NormClass>(t.as_bytes(), &mut tags);
            tags[0] as usize
        });
        let copy = time(b, || t.chars().collect::<String>().len());
        let nfd = time(b, || t.nfd().collect::<String>().len());
        let lower = time(b, || {
            t.chars()
                .flat_map(char::to_lowercase)
                .collect::<String>()
                .len()
        });
        let strip = time(b, || {
            t.nfd()
                .filter(|c| !c.is_mark_nonspacing())
                .collect::<String>()
                .len()
        });
        let full = time(b, || {
            t.nfd()
                .filter(|c| !c.is_mark_nonspacing())
                .flat_map(char::to_lowercase)
                .collect::<String>()
                .len()
        });
        println!(
            "{name:>9} │ {cls_simd:8.2} {cls_scal:8.2} │ {copy:5.2} {nfd:6.2} {lower:6.2} {strip:9.2} {full:6.2}"
        );
    }
}
