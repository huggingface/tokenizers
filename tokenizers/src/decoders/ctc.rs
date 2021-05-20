use crate::decoders::wordpiece;
use crate::tokenizer::{Decoder, Result};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// The CTC (Connectionist Temporal Classification) decoder takes care
/// of sanitizing a list of inputs token.
/// Due to some alignement problem the output of some models can come
/// with duplicated token.
#[serde(tag = "type")]
#[non_exhaustive]
pub struct CTC {
    /// The pad token used by CTC to delimit a new token.
    pub pad_token: String,
    /// The word delimiter token. It will be replaced by a <space>
    pub word_delimiter_token: String,
    /// Whether to cleanup some tokenization artifacts.
    /// Mainly spaces before punctuation, and some abbreviated english forms.
    pub cleanup: bool,
}

impl CTC {
    pub fn new(pad_token: String, word_delimiter_token: String, cleanup: bool) -> Self {
        Self {
            pad_token,
            word_delimiter_token,
            cleanup,
        }
    }
}

impl Default for CTC {
    fn default() -> Self {
        Self {
            pad_token: "<pad>".to_string(),
            word_delimiter_token: "|".to_string(),
            cleanup: true,
        }
    }
}

impl Decoder for CTC {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let mut output = tokens
            .into_iter()
            .dedup()
            .join("")
            .replace(&self.pad_token, "");
        if self.cleanup {
            output = wordpiece::cleanup(output).replace(&self.word_delimiter_token, " ");
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handmade_sample() {
        let ctc_decoder = CTC::default();
        let id_to_string_result = "<pad> <pad> h e e l l <pad> l o o o <pad>"
            .split(' ')
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            ctc_decoder.decode(id_to_string_result).unwrap(),
            "hello".to_string()
        );
    }
    #[test]
    fn handmade_with_delimiter_sample() {
        let ctc_decoder = CTC::default();
        let id_to_string_result = "<pad> <pad> h e e l l <pad> l o o o <pad> <pad> | <pad> w o o o r <pad> <pad> l l d <pad> <pad> <pad> <pad>"
            .split(' ')
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            ctc_decoder.decode(id_to_string_result).unwrap(),
            "hello world".to_string()
        );
    }
    #[test]
    fn librispeech_sample() {
        let ctc_decoder = CTC::default();
        let id_to_string_result = "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> A | | <pad> M <pad> <pad> <pad> <pad> A <pad> <pad> N <pad> <pad> <pad> | | | <pad> <pad> <pad> <pad> S <pad> <pad> <pad> A I <pad> D D | | T T <pad> O <pad> | | T H E E | | | <pad> U U <pad> N N <pad> I <pad> <pad> V <pad> <pad> <pad> E R R <pad> <pad> <pad> S E E | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> S S <pad> <pad> <pad> <pad> I <pad> R R <pad> <pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> I <pad> <pad> <pad> | <pad> <pad> <pad> E X <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> I <pad> S <pad> <pad> T <pad> <pad> | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>".split(' ').map(|s| s.to_string()).collect();
        assert_eq!(
            ctc_decoder.decode(id_to_string_result).unwrap(),
            "A MAN SAID TO THE UNIVERSE SIR I EXIST ".to_string()
        );
    }
    #[test]
    fn another_librispeech_sample() {
        let ctc_decoder = CTC::default();
        let id_to_string_result = "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> H <pad> I <pad> S S | | <pad> <pad> <pad> I N <pad> <pad> S <pad> T T <pad> <pad> A N C C T <pad> | | | | | <pad> <pad> <pad> <pad> P <pad> <pad> <pad> <pad> A <pad> <pad> N N N <pad> <pad> I <pad> C <pad> <pad> | | <pad> W <pad> <pad> A S <pad> | | <pad> <pad> <pad> F <pad> <pad> O L <pad> <pad> L L O O W E E D | | <pad> B <pad> <pad> <pad> Y <pad> | | | A | | <pad> S S S <pad> M M <pad> <pad> <pad> A L L <pad> <pad> <pad> <pad> L <pad> | | | <pad> <pad> <pad> <pad> S H H <pad> <pad> <pad> <pad> A R R <pad> <pad> P <pad> <pad> | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> B <pad> <pad> L L <pad> <pad> <pad> <pad> <pad> O W W <pad> <pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> H <pad> <pad> <pad> <pad> <pad> <pad> <pad> I G H H | | <pad> <pad> O N <pad> | | H <pad> I S S | | <pad> <pad> C H H <pad> <pad> <pad> E <pad> S S <pad> T T <pad> <pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>".split(' ').map(|s| s.to_string()).collect();
        assert_eq!(
            ctc_decoder.decode(id_to_string_result).unwrap(),
            "HIS INSTANCT PANIC WAS FOLLOWED BY A SMALL SHARP BLOW HIGH ON HIS CHEST ".to_string()
        );
    }
}
