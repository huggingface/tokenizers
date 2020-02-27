import { byteLevelPreTokenizer, metaspacePreTokenizer } from "./pre-tokenizers";

describe("byteLevelPreTokenizer", () => {
  it("accepts `undefined` as parameter", () => {
    expect(byteLevelPreTokenizer(undefined)).toBeDefined();
  });
});

describe("metaspacePreTokenizer", () => {
  it("accepts `undefined` as first parameter", () => {
    expect(metaspacePreTokenizer(undefined)).toBeDefined();
  });

  it("accepts `undefined` as second parameter", () => {
    expect(metaspacePreTokenizer("test", undefined)).toBeDefined();
  });
});
