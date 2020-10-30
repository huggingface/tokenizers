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
  WordLevel: {
    init: native.models_WordLevel_init,
    fromFile: native.models_WordLevel_from_file,
    empty: native.models_WordLevel_empty,
  },
  Unigram: {
    init: native.models_Unigram_init,
    empty: native.models_Unigram_empty,
  },
};
