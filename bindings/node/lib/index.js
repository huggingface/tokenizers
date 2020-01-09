var addon = require('../native');

exports.Tokenizer = addon.tokenizer_Tokenizer;
exports.models = {
	BPE: {
		fromFiles: addon.models_BPE_from_files,
		empty: addon.models_BPE_empty,
	},
	WordPiece: addon.models_WordPiece,
}
exports.decoders = {
	ByteLevel: addon.decoders_ByteLevel,
	WordPiece: addon.decoders_WordPiece,
	Metaspace: addon.decoders_Metaspace,
	BPEDecoder: addon.decoders_BPEDecoder,
}
exports.post_processors = {
	BertProcessing: addon.processors_BertProcessing,
}
