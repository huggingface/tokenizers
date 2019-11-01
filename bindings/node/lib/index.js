var addon = require('../native');

let s = "Hey man!";
if (typeof process.argv[2] == 'string') {
	s = process.argv[2];
}

console.log(addon.WhitespaceTokenizer.tokenize(s));
