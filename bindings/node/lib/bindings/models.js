const native = require("./native");

module.exports = {
  BPE: {
    fromFiles: native.models_BPE_from_files,
    empty: native.models_BPE_empty,
  },
  WordPiece: {
    fromFiles: native.models_WordPiece_from_files,
    empty: native.models_WordPiece_empty,
  },
};
