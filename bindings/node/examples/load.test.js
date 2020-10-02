/*eslint-disable no-undef*/
const tokenizers = require("..");
const { promisify } = require("util");

describe("loadExample", () => {
  beforeAll(async () => {});
  it("", async () => {
    const example = "This is an example";
    const ids = [713, 16, 41, 1246];
    const tokens = ["This", "Ġis", "Ġan", "Ġexample"];

    // START load_tokenizer
    const tokenizer = tokenizers.Tokenizer.fromFile("data/roberta.json");
    // END load_tokenizer

    // You could also use regular callbacks
    const encode = promisify(tokenizer.encode.bind(tokenizer));
    const decode = promisify(tokenizer.decode.bind(tokenizer));

    const encoded = await encode(example);
    expect(encoded.getIds()).toEqual(ids);
    expect(encoded.getTokens()).toEqual(tokens);

    const decoded = await decode(ids);
    expect(decoded).toEqual(example);
  });
});
