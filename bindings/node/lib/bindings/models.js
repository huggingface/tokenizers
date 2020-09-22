const native = require("./native");

module.exports = {
  BPE: {
    init: native.models_BPE_init,
    fromFiles: native.models_BPE_from_files,
    empty: native.models_BPE_empty,
  },
  WordPiece: {
    init: native.models_WordPiece_init,
    fromFiles: native.models_WordPiece_from_files,
    empty: native.models_WordPiece_empty,
  },
};
