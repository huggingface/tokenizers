var addon = require('../../native');

module.exports = {
	bpe: {
		fromFiles: addon.models_BPE_from_files,
		empty: addon.models_BPE_empty,
	},
	wordPiece: {
		fromFiles: addon.models_WordPiece_from_files,
		empty: addon.models_WordPiece_empty,
	}
}
