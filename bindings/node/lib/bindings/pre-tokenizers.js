const native = require("./native");

module.exports = {
  byteLevelPreTokenizer: native.pre_tokenizers_ByteLevel,
  byteLevelAlphabet: native.pre_tokenizers_ByteLevel_Alphabet,
  whitespacePreTokenizer: native.pre_tokenizers_Whitespace,
  whitespaceSplitPreTokenizer: native.pre_tokenizers_WhitespaceSplit,
  bertPreTokenizer: native.pre_tokenizers_BertPreTokenizer,
  metaspacePreTokenizer: native.pre_tokenizers_Metaspace,
  charDelimiterSplitPreTokenizer: native.pre_tokenizers_CharDelimiterSplit,
  punctuationPreTokenizer: native.pre_tokenizers_Punctuation,
  sequencePreTokenizer: native.pre_tokenizers_Sequence,
  digitsPreTokenizer: native.pre_tokenizers_Digits,
  splitPreTokenizer: native.pre_tokenizers_Split,
};
