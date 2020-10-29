const native = require("./native");

module.exports = {
  bertNormalizer: native.normalizers_BertNormalizer,
  nfcNormalizer: native.normalizers_NFC,
  nfdNormalizer: native.normalizers_NFD,
  nfkcNormalizer: native.normalizers_NFKC,
  nfkdNormalizer: native.normalizers_NFKD,
  sequenceNormalizer: native.normalizers_Sequence,
  lowercaseNormalizer: native.normalizers_Lowercase,
  stripNormalizer: native.normalizers_Strip,
  stripAccentsNormalizer: native.normalizers_StripAccents,
  nmtNormalizer: native.normalizers_Nmt,
  precompiledNormalizer: native.normalizers_Precompiled,
  replaceNormalizer: native.normalizers_Replace,
};
