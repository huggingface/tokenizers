/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-empty-function */

const tokenizers = await import("tokenizers");

describe("loadExample", () => {
  it("", () => {
    const example = "This is an example";
    const ids = [713, 16, 41, 1246];
    const tokens = ["This", "Ġis", "Ġan", "Ġexample"];

    const tokenizer = tokenizers.Tokenizer.fromFile("data/roberta.json");
    const encoded = tokenizer.encode(example);

    expect(encoded.ids).toBe(ids);
    expect(encoded.tokens).toBe(tokens);

    expect(tokenizer.decode(ids)).toBe(example);
  });
});
