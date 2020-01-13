var native = require('./native');

module.exports = {
  bertNormalizer:      native.normalizers_BertNormalizer,
  nfdNormalizer:       native.normalizers_NFD,
  nfkdNormalizer:      native.normalizers_NFKD,
  nfcNormalizer:       native.normalizers_NFC,
  nfkcNormalizer:      native.normalizers_NFKC,
  sequenceNormalizer:  native.normalizers_Sequence,
  lowercaseNormalizer: native.normalizers_Lowercase
};
