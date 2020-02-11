pub mod bert;
pub mod roberta;
use crate::tokenizer::{Encoding, PostProcessor, Result};
use regex::Regex;

#[derive(Debug)]
pub enum Error {
    NoPairTemplate
}

impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::NoPairTemplate => {
                write!(fmt, "Unable to process pair input. No pair template was provided.")
            }
        }
    }
}
impl std::error::Error for Error {}


struct Template<'a> {
    template: &'a str,
    required_keys: Vec<&'a str>,
}

pub struct TemplateBasedProcessor<'a>{
    template: Template<'a>,
    pair_template: Option<Template<'a>>,
    keys: Hashmap<&'a str, u8>
}

impl<'a> Template<'a> {
    pub fn from_string(template: &'a str) -> Self {
        lazy_static!{
            static ref RE: Regex = Regex::new("(\\[(.+?)\\])").unwrap();
        }

        let required_keys: Vec<&str> = RE.captures_iter(template)
            .map(|capture| capture.get(2).unwrap().as_str())
            .collect();

        return Template { template, required_keys}
    }
}

impl<'a> TemplateBasedProcessor<'a>{
    pub fn new(template: &'a str, pair_template: &'a Option<&'a str>, keys: Hashmap<&'a str, u8>) -> Self {
        if let Some(pair_template) = *pair_template {
            TemplateBasedProcessor {
                template: Template::from_string(template),
                pair_template: Some(Template::from_string(pair_template)),
                keys
            }
        }else {
            TemplateBasedProcessor {
                template: Template::from_string(template),
                pair_template: None,
                keys
            }
        }
    }
}

impl<'a> PostProcessor for TemplateBasedProcessor<'a>{
    fn added_tokens(&self, encoding: &Encoding, pair_encoding: &Option<Encoding>) -> Result<usize> {
        match pair_encoding {
            Some(pair_encoding) => {
                match self.pair_template {
                    Some(pair) => {
                        Ok(pair.required_keys.len())
                    },
                    _ => Err(Error::NoPairTemplateProvided)
                }
            }
            None => {
                self.template.required_keys.len()
            }
        }
    }

    fn process(&self, encoding: Encoding, pair_encoding: Option<Encoding>) -> Result<Encoding> {
        match pair_encoding{
            Some(pair) => {

            },
            None = {

            }
        }
    }
}