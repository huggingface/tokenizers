mod error;

use error::Result;

pub struct PreTokenSplits<'a> {
    string: &'a str,
    splits: Vec<(usize, usize)>,
}

impl<'a> PreTokenSplits<'a> {
	pub fn from(string: &'a str, splits: Vec<(usize, usize)>) -> Self {
		Self { string, splits }
	}
}

pub trait PreTokenizer: Send + Sync {
    fn pre_tokenize<'a>(&mut self, string: &'a str) -> Result<PreTokenSplits<'a>>;
}

pub enum PreTokenizerPlan {
    SplitWhitespace(SplitWhitespacePreTokenizer),
}

impl PreTokenizer for PreTokenizerPlan {
	fn pre_tokenize<'a>(&mut self, string: &'a str) -> Result<PreTokenSplits<'a>> {
		match self {
			PreTokenizerPlan::SplitWhitespace(pre_tokenizer) => pre_tokenizer.pre_tokenize(string)
		}
	}
}


#[derive(Debug)]
pub struct SplitWhitespacePreTokenizer;

impl PreTokenizer for SplitWhitespacePreTokenizer {
    fn pre_tokenize<'a>(&mut self, string: &'a str) -> Result<PreTokenSplits<'a>> {
        let mut splits: Vec<(usize, usize)> = vec![];
        for split in string.split_whitespace() {
            let start = split.as_ptr() as usize - string.as_ptr() as usize;
			splits.push((start, start + split.len()));
        }
		Ok(PreTokenSplits::from(
			string,
			splits
		))
    }
}
