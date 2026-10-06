//! Ported from `serde_json` 1.0, `src/de.rs`, its default (non-`float_roundtrip`) path.
//!
//! ```text
//! Copyright (c) Erick Tryzelaar and David Tolnay
//! Licensed under either of Apache License, Version 2.0 or MIT license, at your option.
//! https://github.com/serde-rs/json
//! ```
//!
//! Bug-compatible on purpose, double rounding included: Unigram scores feed a Viterbi lattice, so
//! being *more* accurate here moves ids. `1e400` is the one divergence left (infinity, not a range
//! error). Do not "fix" the arithmetic.

#[rustfmt::skip]
static POW10: [f64; 309] = [
    1e000, 1e001, 1e002, 1e003, 1e004, 1e005, 1e006, 1e007,
    1e008, 1e009, 1e010, 1e011, 1e012, 1e013, 1e014, 1e015,
    1e016, 1e017, 1e018, 1e019, 1e020, 1e021, 1e022, 1e023,
    1e024, 1e025, 1e026, 1e027, 1e028, 1e029, 1e030, 1e031,
    1e032, 1e033, 1e034, 1e035, 1e036, 1e037, 1e038, 1e039,
    1e040, 1e041, 1e042, 1e043, 1e044, 1e045, 1e046, 1e047,
    1e048, 1e049, 1e050, 1e051, 1e052, 1e053, 1e054, 1e055,
    1e056, 1e057, 1e058, 1e059, 1e060, 1e061, 1e062, 1e063,
    1e064, 1e065, 1e066, 1e067, 1e068, 1e069, 1e070, 1e071,
    1e072, 1e073, 1e074, 1e075, 1e076, 1e077, 1e078, 1e079,
    1e080, 1e081, 1e082, 1e083, 1e084, 1e085, 1e086, 1e087,
    1e088, 1e089, 1e090, 1e091, 1e092, 1e093, 1e094, 1e095,
    1e096, 1e097, 1e098, 1e099, 1e100, 1e101, 1e102, 1e103,
    1e104, 1e105, 1e106, 1e107, 1e108, 1e109, 1e110, 1e111,
    1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
    1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127,
    1e128, 1e129, 1e130, 1e131, 1e132, 1e133, 1e134, 1e135,
    1e136, 1e137, 1e138, 1e139, 1e140, 1e141, 1e142, 1e143,
    1e144, 1e145, 1e146, 1e147, 1e148, 1e149, 1e150, 1e151,
    1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
    1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167,
    1e168, 1e169, 1e170, 1e171, 1e172, 1e173, 1e174, 1e175,
    1e176, 1e177, 1e178, 1e179, 1e180, 1e181, 1e182, 1e183,
    1e184, 1e185, 1e186, 1e187, 1e188, 1e189, 1e190, 1e191,
    1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199,
    1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207,
    1e208, 1e209, 1e210, 1e211, 1e212, 1e213, 1e214, 1e215,
    1e216, 1e217, 1e218, 1e219, 1e220, 1e221, 1e222, 1e223,
    1e224, 1e225, 1e226, 1e227, 1e228, 1e229, 1e230, 1e231,
    1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239,
    1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247,
    1e248, 1e249, 1e250, 1e251, 1e252, 1e253, 1e254, 1e255,
    1e256, 1e257, 1e258, 1e259, 1e260, 1e261, 1e262, 1e263,
    1e264, 1e265, 1e266, 1e267, 1e268, 1e269, 1e270, 1e271,
    1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279,
    1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287,
    1e288, 1e289, 1e290, 1e291, 1e292, 1e293, 1e294, 1e295,
    1e296, 1e297, 1e298, 1e299, 1e300, 1e301, 1e302, 1e303,
    1e304, 1e305, 1e306, 1e307, 1e308,
];

#[cfg(any(all(feature = "serialize", feature = "unigram"), test))]
pub(crate) fn f64_from_literal(digits: &str) -> f64 {
    let (positive, significand, exponent) = number(digits);
    f64_from_parts(positive, significand, exponent)
}

pub(crate) fn f64_from_parts(positive: bool, significand: u64, mut exponent: i32) -> f64 {
    let mut f = significand as f64;
    loop {
        match POW10.get(exponent.unsigned_abs() as usize) {
            Some(&pow10) => {
                if exponent >= 0 {
                    f *= pow10;
                    if f.is_infinite() {
                        return if positive {
                            f64::INFINITY
                        } else {
                            f64::NEG_INFINITY
                        };
                    }
                } else {
                    f /= pow10;
                }
                break;
            }
            None => {
                if f == 0.0 {
                    break;
                }
                if exponent >= 0 {
                    return if positive {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                }
                f /= 1e308;
                exponent += 308;
            }
        }
    }
    if positive { f } else { -f }
}

pub(crate) fn number(digits: &str) -> (bool, u64, i32) {
    let s = digits.as_bytes();
    let mut i = 0;
    let positive = if s.first() == Some(&b'-') {
        i = 1;
        false
    } else {
        true
    };

    let mut significand: u64 = 0;
    let mut exponent: i32 = 0;
    let mut saturated = false;
    while let Some(c) = s.get(i).filter(|c| c.is_ascii_digit()) {
        let digit = u64::from(c - b'0');
        if saturated {
            exponent += 1;
        } else {
            match significand
                .checked_mul(10)
                .and_then(|v| v.checked_add(digit))
            {
                Some(v) => significand = v,
                None => {
                    saturated = true;
                    exponent += 1;
                }
            }
        }
        i += 1;
    }

    if s.get(i) == Some(&b'.') {
        i += 1;
        while let Some(c) = s.get(i).filter(|c| c.is_ascii_digit()) {
            let digit = u64::from(c - b'0');
            if !saturated {
                match significand
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(digit))
                {
                    Some(v) => {
                        significand = v;
                        exponent -= 1;
                    }
                    None => saturated = true,
                }
            }
            i += 1;
        }
    }

    if matches!(s.get(i), Some(b'e' | b'E')) {
        i += 1;
        let negate = match s.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let mut exp: i32 = 0;
        while let Some(c) = s.get(i).filter(|c| c.is_ascii_digit()) {
            exp = exp.saturating_mul(10).saturating_add(i32::from(c - b'0'));
            i += 1;
        }
        exponent = exponent.saturating_add(if negate { -exp } else { exp });
    }

    (positive, significand, exponent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::Json;

    fn p(s: &str) -> Json<'_> {
        Json::parse(s).expect("should parse")
    }

    #[test]
    fn matches_serde_not_from_str_on_a_real_unigram_score() {
        let lit = "-3.8403830528259277";
        let ours = Json::parse(lit).unwrap().as_f64().unwrap();
        let serde: f64 = serde_json::from_str(lit).unwrap();
        let correctly_rounded: f64 = lit.parse().unwrap();

        assert_eq!(
            ours.to_bits(),
            serde.to_bits(),
            "we must reproduce serde_json bit-for-bit"
        );
        assert_ne!(
            serde.to_bits(),
            correctly_rounded.to_bits(),
            "if serde ever becomes correctly rounded by default, this emulation should be deleted"
        );
    }

    #[test]
    fn numbers_are_bit_identical_to_serde_json() {
        for lit in [
            "0",
            "-0",
            "1",
            "-1",
            "100",
            "50256",
            "4294967295",
            "100277",
            "0.0",
            "-0.0",
            "1.5",
            "0.1",
            "0.3",
            "1e10",
            "1e-10",
            "1E+2",
            "-2.5e-3",
            "-3.8403830528259277",
            "-3.1902313232421875",
            "-10.522398948669434",
            "-0.0001",
            "-1e-300",
            "1e308",
            "1e-320",
            "123456789012345678901234567890",
            "1.2345678901234567890123456789",
            "0.000000000000000000000000001",
            "18446744073709551615",
            "18446744073709551616",
        ] {
            let ours = Json::parse(lit).unwrap().as_f64().unwrap();
            let serde: f64 = serde_json::from_str(lit).unwrap();
            assert_eq!(
                ours.to_bits(),
                serde.to_bits(),
                "{lit}: ours={ours} ({:016x}) serde={serde} ({:016x})",
                ours.to_bits(),
                serde.to_bits()
            );
        }

        for prefix in [
            "1844674407370955161",
            "1844674407370955160",
            "9223372036854775807",
            "1234567890123456789",
        ] {
            for suffix in 0..1000u32 {
                let lit = format!("0.{prefix}{suffix:03}");
                let ours = Json::parse(&lit).unwrap().as_f64().unwrap();
                let serde: f64 = serde_json::from_str(&lit).unwrap();
                assert_eq!(
                    ours.to_bits(),
                    serde.to_bits(),
                    "{lit}: ours={ours:?} serde={serde:?}"
                );
            }
        }
    }

    #[test]
    fn f64_from_literal_agrees_with_the_accessor() {
        for literal in [
            "0",
            "-0",
            "1.5",
            "-13.5321998596191",
            "-3.8403830528259277",
            "1e-9",
            "1.2345678901234567890123456789",
        ] {
            assert_eq!(
                f64_from_literal(literal).to_bits(),
                p(literal)
                    .as_f64()
                    .expect("a number reads as an f64")
                    .to_bits(),
                "{literal}"
            );
        }
    }

    #[test]
    fn an_overflowing_exponent_is_the_last_divergence_from_serde() {
        assert_eq!(p("1e400").as_f64(), Some(f64::INFINITY));
        assert_eq!(p("-1e400").as_f64(), Some(f64::NEG_INFINITY));
        assert!(serde_json::from_str::<f64>("1e400").is_err());
    }
}
