const native = require("./native");

module.exports = {
  BPE: {
    init: native.models_BPE_init,
    fromFile: native.models_BPE_from_file,
    empty: native.models_BPE_empty,
  },
  WordPiece: {
    init: native.models_WordPiece_init,
    fromFile: native.models_WordPiece_from_file,
    empty: native.models_WordPiece_empty,
  },
};
