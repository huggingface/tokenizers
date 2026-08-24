/// Wraps the first sequence in `cls` ... `sep`, and appends a second `sep` to the pair sequence.
///
/// The tag is *optional* on the way in, and that is a documented requirement rather than an
/// accident: `PostProcessorWrapper` is untagged in both directions, and
/// `post_processor_deserialization_no_type` asserts that `{"sep":["[SEP]",102],"cls":["[CLS]",101]}`
/// loads as a `Bert`. What discriminates the variants is the set of *required fields*, which is also
/// why `Roberta` has to stay ahead of `Bert` in the enum: a Roberta object satisfies Bert's shape
/// but not the other way round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BertProcessing {
    pub sep: (String, u32),
    pub cls: (String, u32),
}

impl Default for BertProcessing {
    fn default() -> Self {
        Self {
            sep: ("[SEP]".into(), 102),
            cls: ("[CLS]".into(), 101),
        }
    }
}

impl BertProcessing {
    pub fn new(sep: (String, u32), cls: (String, u32)) -> Self {
        Self { sep, cls }
    }

    pub fn get_sep_copy(&self) -> (String, u32) {
        (self.sep.0.clone(), self.sep.1)
    }

    pub fn get_cls_copy(&self) -> (String, u32) {
        (self.cls.0.clone(), self.cls.1)
    }
}
