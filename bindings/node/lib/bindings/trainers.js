var addon = require('../bin-package');

module.exports = {
  bpeTrainer:       addon.trainers_BPETrainer,
  wordPieceTrainer: addon.trainers_WordPieceTrainer
};
