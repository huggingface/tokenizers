var addon = require('../bin-package');

module.exports = {
  byteLevelPreTokenizer:  addon.pre_tokenizers_ByteLevel,
  byteLevelAlphabet:      addon.pre_tokenizers_ByteLevel_Alphabet,
  whitespacePreTokenizer: addon.pre_tokenizers_Whitespace,
  bertPreTokenizer:       addon.pre_tokenizers_BertPreTokenizer,
  metaspacePreTokenizer:  addon.pre_tokenizers_Metaspace
};
