/// Like [`crate::processors::bert::BertProcessing`], except every encoding keeps `type_id == 0` and
/// the offsets can first be trimmed of the `ByteLevel` space marker.
///
/// Note there is no `#[serde(default)]` on `trim_offsets` or `add_prefix_space`: all four fields are
/// required, and that is the whole reason `Roberta` can be told apart from `Bert` by an untagged
/// enum. The tag itself is optional here too -- see [`super::bert::BertProcessing`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobertaProcessing {
    pub sep: (String, u32),
    pub cls: (String, u32),
    pub trim_offsets: bool,
    pub add_prefix_space: bool,
}

impl Default for RobertaProcessing {
    fn default() -> Self {
        Self {
            sep: ("</s>".into(), 2),
            cls: ("<s>".into(), 0),
            trim_offsets: true,
            add_prefix_space: true,
        }
    }
}

impl RobertaProcessing {
    pub fn new(sep: (String, u32), cls: (String, u32)) -> Self {
        Self {
            sep,
            cls,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    pub fn get_sep_copy(&self) -> (String, u32) {
        (self.sep.0.clone(), self.sep.1)
    }

    pub fn get_cls_copy(&self) -> (String, u32) {
        (self.cls.0.clone(), self.cls.1)
    }
}
