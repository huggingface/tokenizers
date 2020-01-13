var native = require('./native');

module.exports = {
  bpe: {
    fromFiles: native.models_BPE_from_files,
    empty:     native.models_BPE_empty,
  },
  wordPiece: {
    fromFiles: native.models_WordPiece_from_files,
    empty:     native.models_WordPiece_empty,
  }
}
