var addon = require('../bin-package');

module.exports = {
  byteLevelDecoder: addon.decoders_ByteLevel,
  wordPieceDecoder: addon.decoders_WordPiece,
  metaspaceDecoder: addon.decoders_Metaspace,
  bpeDecoder:       addon.decoders_BPEDecoder
};
