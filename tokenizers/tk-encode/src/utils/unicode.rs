use crate::vocab::bucket_vocab_store::mix;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};

// The same type as the vocabulary's and the BPE pair map's, so the binary carries one copy of
// PtrHash's code.
type Mphf = FastPtrHash<NoHash, u64>;

const _: () = assert!(
    matches!(unicode_normalization::UNICODE_VERSION, (17, 0, 0)),
    concat!(
        "`unicode-normalization` updated its Unicode version, ",
        "combining mark exceptions need to be updated too in tk-encode/src/utils/unicode.rs"
    )
);

/// A perfect hash lookup for chars that are combining marks in Unicode 17 but not in Unicode 9.
/// We need to conform to unicode 9 to preserve backwards compatibility for legacy models:
/// The normalization could differ and the model emit invalid ids (although it's very unlikely)
#[derive(Clone)]
pub(crate) struct Unicode9IgnoredCombiningMarks {
    mphf: Mphf,
    slots: Box<[u32]>,
}

impl Unicode9IgnoredCombiningMarks {
    pub(crate) fn new() -> Self {
        // `NoHash` takes keys as already hashed, so the code points go through `mix` first.
        // `mix` is a bijection, so the slot can still store and compare the char itself.
        let keys: Vec<u64> = MARKS_ONLY_IN_UNICODE_17
            .iter()
            .map(|&c| mix(c as u64))
            .collect();
        let mphf = Mphf::new(&keys, PtrHashParams::default_fast());
        let mut slots = vec![u32::MAX; mphf.max_index()].into_boxed_slice();
        for &c in &MARKS_ONLY_IN_UNICODE_17 {
            slots[mphf.index(&mix(c as u64))] = c as u32;
        }
        Self { mphf, slots }
    }

    pub(crate) fn contains(&self, c: char) -> bool {
        self.slots[self.mphf.index(&mix(c as u64))] == c as u32
    }
}

/// Combining marks in Unicode 17 that were not combining marks in Unicode 9, in code point order.
#[rustfmt::skip]
pub(crate) static MARKS_ONLY_IN_UNICODE_17: [char; 448] = [
    '\u{07FD}', '\u{0897}', '\u{0898}', '\u{0899}', '\u{089A}', '\u{089B}',
    '\u{089C}', '\u{089D}', '\u{089E}', '\u{089F}', '\u{08CA}', '\u{08CB}',
    '\u{08CC}', '\u{08CD}', '\u{08CE}', '\u{08CF}', '\u{08D0}', '\u{08D1}',
    '\u{08D2}', '\u{08D3}', '\u{09FE}', '\u{0AFA}', '\u{0AFB}', '\u{0AFC}',
    '\u{0AFD}', '\u{0AFE}', '\u{0AFF}', '\u{0B55}', '\u{0C04}', '\u{0C3C}',
    '\u{0CF3}', '\u{0D00}', '\u{0D3B}', '\u{0D3C}', '\u{0D81}', '\u{0EBA}',
    '\u{0ECE}', '\u{1715}', '\u{180F}', '\u{1ABF}', '\u{1AC0}', '\u{1AC1}',
    '\u{1AC2}', '\u{1AC3}', '\u{1AC4}', '\u{1AC5}', '\u{1AC6}', '\u{1AC7}',
    '\u{1AC8}', '\u{1AC9}', '\u{1ACA}', '\u{1ACB}', '\u{1ACC}', '\u{1ACD}',
    '\u{1ACE}', '\u{1ACF}', '\u{1AD0}', '\u{1AD1}', '\u{1AD2}', '\u{1AD3}',
    '\u{1AD4}', '\u{1AD5}', '\u{1AD6}', '\u{1AD7}', '\u{1AD8}', '\u{1AD9}',
    '\u{1ADA}', '\u{1ADB}', '\u{1ADC}', '\u{1ADD}', '\u{1AE0}', '\u{1AE1}',
    '\u{1AE2}', '\u{1AE3}', '\u{1AE4}', '\u{1AE5}', '\u{1AE6}', '\u{1AE7}',
    '\u{1AE8}', '\u{1AE9}', '\u{1AEA}', '\u{1AEB}', '\u{1CF7}', '\u{1DF6}',
    '\u{1DF7}', '\u{1DF8}', '\u{1DF9}', '\u{1DFA}', '\u{A82C}', '\u{A8FF}',
    '\u{10D24}', '\u{10D25}', '\u{10D26}', '\u{10D27}', '\u{10D69}', '\u{10D6A}',
    '\u{10D6B}', '\u{10D6C}', '\u{10D6D}', '\u{10EAB}', '\u{10EAC}', '\u{10EFA}',
    '\u{10EFB}', '\u{10EFC}', '\u{10EFD}', '\u{10EFE}', '\u{10EFF}', '\u{10F46}',
    '\u{10F47}', '\u{10F48}', '\u{10F49}', '\u{10F4A}', '\u{10F4B}', '\u{10F4C}',
    '\u{10F4D}', '\u{10F4E}', '\u{10F4F}', '\u{10F50}', '\u{10F82}', '\u{10F83}',
    '\u{10F84}', '\u{10F85}', '\u{11070}', '\u{11073}', '\u{11074}', '\u{110C2}',
    '\u{11145}', '\u{11146}', '\u{111C9}', '\u{111CE}', '\u{111CF}', '\u{11241}',
    '\u{1133B}', '\u{113B8}', '\u{113B9}', '\u{113BA}', '\u{113BB}', '\u{113BC}',
    '\u{113BD}', '\u{113BE}', '\u{113BF}', '\u{113C0}', '\u{113C2}', '\u{113C5}',
    '\u{113C7}', '\u{113C8}', '\u{113C9}', '\u{113CA}', '\u{113CC}', '\u{113CD}',
    '\u{113CE}', '\u{113CF}', '\u{113D0}', '\u{113D2}', '\u{113E1}', '\u{113E2}',
    '\u{1145E}', '\u{1182C}', '\u{1182D}', '\u{1182E}', '\u{1182F}', '\u{11830}',
    '\u{11831}', '\u{11832}', '\u{11833}', '\u{11834}', '\u{11835}', '\u{11836}',
    '\u{11837}', '\u{11838}', '\u{11839}', '\u{1183A}', '\u{11930}', '\u{11931}',
    '\u{11932}', '\u{11933}', '\u{11934}', '\u{11935}', '\u{11937}', '\u{11938}',
    '\u{1193B}', '\u{1193C}', '\u{1193D}', '\u{1193E}', '\u{11940}', '\u{11942}',
    '\u{11943}', '\u{119D1}', '\u{119D2}', '\u{119D3}', '\u{119D4}', '\u{119D5}',
    '\u{119D6}', '\u{119D7}', '\u{119DA}', '\u{119DB}', '\u{119DC}', '\u{119DD}',
    '\u{119DE}', '\u{119DF}', '\u{119E0}', '\u{119E4}', '\u{11A01}', '\u{11A02}',
    '\u{11A03}', '\u{11A04}', '\u{11A05}', '\u{11A06}', '\u{11A07}', '\u{11A08}',
    '\u{11A09}', '\u{11A0A}', '\u{11A33}', '\u{11A34}', '\u{11A35}', '\u{11A36}',
    '\u{11A37}', '\u{11A38}', '\u{11A39}', '\u{11A3B}', '\u{11A3C}', '\u{11A3D}',
    '\u{11A3E}', '\u{11A47}', '\u{11A51}', '\u{11A52}', '\u{11A53}', '\u{11A54}',
    '\u{11A55}', '\u{11A56}', '\u{11A57}', '\u{11A58}', '\u{11A59}', '\u{11A5A}',
    '\u{11A5B}', '\u{11A8A}', '\u{11A8B}', '\u{11A8C}', '\u{11A8D}', '\u{11A8E}',
    '\u{11A8F}', '\u{11A90}', '\u{11A91}', '\u{11A92}', '\u{11A93}', '\u{11A94}',
    '\u{11A95}', '\u{11A96}', '\u{11A97}', '\u{11A98}', '\u{11A99}', '\u{11B60}',
    '\u{11B61}', '\u{11B62}', '\u{11B63}', '\u{11B64}', '\u{11B65}', '\u{11B66}',
    '\u{11B67}', '\u{11D31}', '\u{11D32}', '\u{11D33}', '\u{11D34}', '\u{11D35}',
    '\u{11D36}', '\u{11D3A}', '\u{11D3C}', '\u{11D3D}', '\u{11D3F}', '\u{11D40}',
    '\u{11D41}', '\u{11D42}', '\u{11D43}', '\u{11D44}', '\u{11D45}', '\u{11D47}',
    '\u{11D8A}', '\u{11D8B}', '\u{11D8C}', '\u{11D8D}', '\u{11D8E}', '\u{11D90}',
    '\u{11D91}', '\u{11D93}', '\u{11D94}', '\u{11D95}', '\u{11D96}', '\u{11D97}',
    '\u{11EF3}', '\u{11EF4}', '\u{11EF5}', '\u{11EF6}', '\u{11F00}', '\u{11F01}',
    '\u{11F03}', '\u{11F34}', '\u{11F35}', '\u{11F36}', '\u{11F37}', '\u{11F38}',
    '\u{11F39}', '\u{11F3A}', '\u{11F3E}', '\u{11F3F}', '\u{11F40}', '\u{11F41}',
    '\u{11F42}', '\u{11F5A}', '\u{13440}', '\u{13447}', '\u{13448}', '\u{13449}',
    '\u{1344A}', '\u{1344B}', '\u{1344C}', '\u{1344D}', '\u{1344E}', '\u{1344F}',
    '\u{13450}', '\u{13451}', '\u{13452}', '\u{13453}', '\u{13454}', '\u{13455}',
    '\u{1611E}', '\u{1611F}', '\u{16120}', '\u{16121}', '\u{16122}', '\u{16123}',
    '\u{16124}', '\u{16125}', '\u{16126}', '\u{16127}', '\u{16128}', '\u{16129}',
    '\u{1612A}', '\u{1612B}', '\u{1612C}', '\u{1612D}', '\u{1612E}', '\u{1612F}',
    '\u{16F4F}', '\u{16F7F}', '\u{16F80}', '\u{16F81}', '\u{16F82}', '\u{16F83}',
    '\u{16F84}', '\u{16F85}', '\u{16F86}', '\u{16F87}', '\u{16FE4}', '\u{16FF0}',
    '\u{16FF1}', '\u{1CF00}', '\u{1CF01}', '\u{1CF02}', '\u{1CF03}', '\u{1CF04}',
    '\u{1CF05}', '\u{1CF06}', '\u{1CF07}', '\u{1CF08}', '\u{1CF09}', '\u{1CF0A}',
    '\u{1CF0B}', '\u{1CF0C}', '\u{1CF0D}', '\u{1CF0E}', '\u{1CF0F}', '\u{1CF10}',
    '\u{1CF11}', '\u{1CF12}', '\u{1CF13}', '\u{1CF14}', '\u{1CF15}', '\u{1CF16}',
    '\u{1CF17}', '\u{1CF18}', '\u{1CF19}', '\u{1CF1A}', '\u{1CF1B}', '\u{1CF1C}',
    '\u{1CF1D}', '\u{1CF1E}', '\u{1CF1F}', '\u{1CF20}', '\u{1CF21}', '\u{1CF22}',
    '\u{1CF23}', '\u{1CF24}', '\u{1CF25}', '\u{1CF26}', '\u{1CF27}', '\u{1CF28}',
    '\u{1CF29}', '\u{1CF2A}', '\u{1CF2B}', '\u{1CF2C}', '\u{1CF2D}', '\u{1CF30}',
    '\u{1CF31}', '\u{1CF32}', '\u{1CF33}', '\u{1CF34}', '\u{1CF35}', '\u{1CF36}',
    '\u{1CF37}', '\u{1CF38}', '\u{1CF39}', '\u{1CF3A}', '\u{1CF3B}', '\u{1CF3C}',
    '\u{1CF3D}', '\u{1CF3E}', '\u{1CF3F}', '\u{1CF40}', '\u{1CF41}', '\u{1CF42}',
    '\u{1CF43}', '\u{1CF44}', '\u{1CF45}', '\u{1CF46}', '\u{1E08F}', '\u{1E130}',
    '\u{1E131}', '\u{1E132}', '\u{1E133}', '\u{1E134}', '\u{1E135}', '\u{1E136}',
    '\u{1E2AE}', '\u{1E2EC}', '\u{1E2ED}', '\u{1E2EE}', '\u{1E2EF}', '\u{1E4EC}',
    '\u{1E4ED}', '\u{1E4EE}', '\u{1E4EF}', '\u{1E5EE}', '\u{1E5EF}', '\u{1E6E3}',
    '\u{1E6E6}', '\u{1E6EE}', '\u{1E6EF}', '\u{1E6F5}',
];

/// Combining marks in Unicode 9 that are no longer combining marks in Unicode 17.
#[rustfmt::skip]
pub(crate) static MARKS_ONLY_IN_UNICODE_9: [char; 2] = [
    '\u{1CF2}', '\u{1CF3}',
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_exactly_the_marks_only_in_unicode_17() {
        assert!(MARKS_ONLY_IN_UNICODE_17.is_sorted());
        let set = Unicode9IgnoredCombiningMarks::new();
        for c in (0..=char::MAX as u32).filter_map(char::from_u32) {
            assert_eq!(
                set.contains(c),
                MARKS_ONLY_IN_UNICODE_17.binary_search(&c).is_ok(),
                "U+{:04X}",
                c as u32
            );
        }
    }
}
