const native = require("./native");

const normOptions = {
  SEPARATE_INTEGERS: 1 << 1,
  SEPARATE_SYMBOLS: 1 << 2,
  SIMPL_TO_TRAD: 1 << 3,
  TRAD_TO_SIMPL: 1 << 4,
  ZH_NORM_MAPPING: 1 << 5,
};

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
  openccEnabled:
    native.normalizers_OpenccEnabled ||
    function () {
      return false;
    },
  normOptions: normOptions,
};
