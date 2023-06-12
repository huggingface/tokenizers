const native = require("./native");

module.exports = {
  byteLevelDecoder: native.decoders_ByteLevel,
  replaceDecoder: native.decoders_Replace,
  wordPieceDecoder: native.decoders_WordPiece,
  byteFallbackDecoder: native.decoders_ByteFallback,
  fuseDecoder: native.decoders_Fuse,
  stripDecoder: native.decoders_Strip,
  metaspaceDecoder: native.decoders_Metaspace,
  bpeDecoder: native.decoders_BPEDecoder,
  ctcDecoder: native.decoders_CTC,
  sequenceDecoder: native.decoders_Sequence,
};
