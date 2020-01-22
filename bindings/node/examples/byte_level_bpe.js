const tokenizers = require("..");

let bpe = tokenizers.models.BPE.fromFiles(
  "./data/gpt2-vocab.json",
  "./data/gpt2-merges.txt"
);
let tokenizer = new tokenizers.Tokenizer(bpe);
console.log(bpe, tokenizer);
