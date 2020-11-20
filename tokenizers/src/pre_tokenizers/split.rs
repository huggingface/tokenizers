use regex::Regex;
// use serde::{Serialize};

use crate::tokenizer::{
    pattern::Invert, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};

// #[derive(Clone, Debug, Serialize)]
#[derive(Clone, Debug)]
// #[serde(tag = "type")]
pub struct Split {
    re: Regex,
    behavior: SplitDelimiterBehavior,
    invert: bool
}

impl PreTokenizer for Split {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        // if self.invert {
        //     let proc_re = Invert(&self.re);
        // } else {
        //     let proc_re = &self.re;
        // }
        pretokenized.split(|_, normalized| {
            normalized.split(&self.re, &self.behavior)
        })
    }
}
