use std::fmt;

use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};

// use crate::pattern::Pattern;
use crate::tokenizer::{
    pattern::{Invert, Pattern}, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};
use serde::de::{Error, Visitor};

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub struct Split {
    #[serde(default = "default_regex", skip)]
    re: Regex,
    #[serde(default = "default_behavior", skip)]
    behavior: SplitDelimiterBehavior,
    invert: bool
}

fn default_regex() -> Regex {
    Regex::new(r"\w+|[^\w\s]+").unwrap()
}

fn default_behavior() -> SplitDelimiterBehavior {
    SplitDelimiterBehavior::Removed
}

impl Default for Split {
    fn default() -> Self {
        Self {
            re: default_regex(),
            behavior: default_behavior(),
            invert: false,
        }
    }
}

impl Split {
    pub fn new(re: Regex, behavior: SplitDelimiterBehavior, invert: bool) -> Self {
        Split {
            re,
            behavior,
            invert
        }
    }
}

impl PreTokenizer for Split {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        use SplitDelimiterBehavior::*;
        match self.behavior {
            Removed => {
                if self.invert { 
                    pretokenized.split(|_, normalized| {
                        normalized.split(Invert(&self.re),Removed)
                    })
                } else {
                    pretokenized.split(|_, normalized| {
                        normalized.split(&self.re, Removed)
                    })
                }
            },
            Isolated => {
                if self.invert { 
                    pretokenized.split(|_, normalized| {
                        normalized.split(Invert(&self.re), Isolated)
                    })
                } else {
                    pretokenized.split(|_, normalized| {
                        normalized.split(&self.re, Isolated)
                    })
                }  
            },
            MergedWithPrevious => {
                if self.invert { 
                    pretokenized.split(|_, normalized| {
                        normalized.split(Invert(&self.re), MergedWithPrevious)
                    })
                } else {
                    pretokenized.split(|_, normalized| {
                        normalized.split(&self.re, MergedWithPrevious)
                    })
                }  
            },
            MergedWithNext => {
                if self.invert { 
                    pretokenized.split(|_, normalized| {
                        normalized.split(Invert(&self.re), MergedWithNext)
                    })
                } else {
                    pretokenized.split(|_, normalized| {
                        normalized.split(&self.re, MergedWithNext)
                    })
                }  
            },
            Contiguous => {
                if self.invert { 
                    pretokenized.split(|_, normalized| {
                        normalized.split(Invert(&self.re), Contiguous)
                    })
                } else {
                    pretokenized.split(|_, normalized| {
                        normalized.split(&self.re, Contiguous)
                    })
                }  
            },
        }
    }
}


// manually implement deserialize because Split is not a unit-struct but is
// serialized like one.
impl<'de> Deserialize<'de> for Split {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(SplitVisitor)
    }
}

struct SplitVisitor;
impl<'de> Visitor<'de> for SplitVisitor {
    type Value = Split;
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Split")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let maybe_type = map.next_entry::<String, String>()?;
        let maybe_type_str = maybe_type.as_ref().map(|(k, v)| (k.as_str(), v.as_str()));
        match maybe_type_str {
            Some(("type", "Split")) => Ok(Split::default()),
            Some((_, ty)) => Err(Error::custom(&format!("Expected Split, got {}", ty))),
            None => Err(Error::custom("Expected type: Split")),
        }
    }
}

