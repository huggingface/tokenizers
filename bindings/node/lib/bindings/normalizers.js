const native = require("./native");

module.exports = {
  bertNormalizer: native.normalizers_BertNormalizer,
  lowercaseNormalizer: native.normalizers_Lowercase,
  nfcNormalizer: native.normalizers_NFC,
  nfdNormalizer: native.normalizers_NFD,
  nfkcNormalizer: native.normalizers_NFKC,
  nfkdNormalizer: native.normalizers_NFKD,
  sequenceNormalizer: native.normalizers_Sequence,
  stripNormalizer: native.normalizers_Strip,
  stripAccentsNormalizer: native.normalizers_StripAccents,
};
