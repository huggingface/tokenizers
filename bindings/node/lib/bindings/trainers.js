const native = require("./native");

module.exports = {
  bpeTrainer: native.trainers_BPETrainer,
  wordPieceTrainer: native.trainers_WordPieceTrainer,
  unigramTrainer: native.trainers_UnigramTrainer,
};
