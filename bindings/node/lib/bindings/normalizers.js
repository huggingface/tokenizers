var addon = require('../bin-package');

module.exports = {
  bertNormalizer:      addon.normalizers_BertNormalizer,
  nfdNormalizer:       addon.normalizers_NFD,
  nfkdNormalizer:      addon.normalizers_NFKD,
  nfcNormalizer:       addon.normalizers_NFC,
  nfkcNormalizer:      addon.normalizers_NFKC,
  sequenceNormalizer:  addon.normalizers_Sequence,
  lowercaseNormalizer: addon.normalizers_Lowercase
};
