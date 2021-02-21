const native = require("./native");

module.exports = {
  bpeTrainer: native.trainers_BPETrainer,
  wordPieceTrainer: native.trainers_WordPieceTrainer,
  wordLevelTrainer: native.trainers_WordLevelTrainer,
  unigramTrainer: native.trainers_UnigramTrainer,
};
