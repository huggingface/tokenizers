mod error;

use error::Result;

pub use error::PreTokenizerError;

pub struct PreTokenSplits<'a> {
    string: &'a str,
    splits: Vec<(usize, usize)>,
}

impl<'a> PreTokenSplits<'a> {
	pub fn from(string: &'a str, splits: Vec<(usize, usize)>) -> Self {
		Self { string, splits }
	}

	pub fn get_bytes(&self, index: usize) -> Option<&[u8]> {
		self.splits.get(index).map(|(start, end)| &self.string.as_bytes()[*start..*end])
	}

	pub fn iter(&'a self) -> PreTokenIterator<'a> {
		PreTokenIterator::new(self)
	}

	pub fn len(&self) -> usize {
		self.splits.len()
	}
}

pub struct PreTokenIterator<'a> {
	splits: &'a PreTokenSplits<'a>,
	index: usize
}

impl<'a> PreTokenIterator<'a> {
	fn new(splits: &'a PreTokenSplits) -> Self {
		Self {
			splits,
			index: 0
		}
	}
}

impl<'a> Iterator for PreTokenIterator<'a> {
	type Item = &'a [u8];

	fn next(&mut self) -> Option<Self::Item> {
		let next_item = self.splits.get_bytes(self.index);
		self.index += 1;
		next_item
	}
}

pub trait PreTokenizer: Send + Sync {
    fn pre_tokenize<'a>(&mut self, string: &'a str) -> Result<PreTokenSplits<'a>>;
}

pub enum PreTokenizePlan {
    SplitWhitespace(SplitWhitespacePreTokenizer),
}

impl PreTokenizer for PreTokenizePlan {
	fn pre_tokenize<'a>(&mut self, string: &'a str) -> Result<PreTokenSplits<'a>> {
		match self {
			PreTokenizePlan::SplitWhitespace(pre_tokenizer) => pre_tokenizer.pre_tokenize(string)
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
