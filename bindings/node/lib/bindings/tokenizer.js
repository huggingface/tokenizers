const native = require("./native");

class Tokenizer extends native.tokenizer_Tokenizer {
  static fromString = native.tokenizer_Tokenizer_from_string;
  static fromFile = native.tokenizer_Tokenizer_from_file;
}

module.exports = {
  AddedToken: native.tokenizer_AddedToken,
  Tokenizer,
};
