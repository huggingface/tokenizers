var addon = require('../native');

exports.Tokenizer = addon.tokenizer_Tokenizer;
exports.models = {
	BPE: {
		from_files: addon.models_create_BPE_from_files,
		empty: addon.models_create_BPE_empty,
	},
	WordPiece: addon.models_WordPiece,
}
