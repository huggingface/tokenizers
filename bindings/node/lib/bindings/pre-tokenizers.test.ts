import {
  byteLevelPreTokenizer,
  metaspacePreTokenizer,
  punctuationPreTokenizer,
  sequencePreTokenizer,
  splitPreTokenizer,
  whitespaceSplitPreTokenizer,
} from "./pre-tokenizers";

describe("byteLevelPreTokenizer", () => {
  it("instantiates correctly", () => {
    const processor = byteLevelPreTokenizer();
    expect(processor.constructor.name).toEqual("PreTokenizer");
  });
});

describe("metaspacePreTokenizer", () => {
  it("instantiates correctly without any parameter", () => {
    const processor = metaspacePreTokenizer();
    expect(processor.constructor.name).toEqual("PreTokenizer");
  });

  it("accepts `undefined` as first parameter", () => {
    expect(metaspacePreTokenizer(undefined)).toBeDefined();
  });

  it("accepts `undefined` as second parameter", () => {
    expect(metaspacePreTokenizer("t", undefined)).toBeDefined();
  });

  it("can pre-tokenize strings", () => {
    const pretok = metaspacePreTokenizer();
    expect(pretok.preTokenizeString("Hello there friend")).toEqual([
      ["▁Hello", [0, 5]],
      ["▁there", [5, 11]],
      ["▁friend", [11, 18]],
    ]);
  });
});

describe("punctuationPreTokenizer", () => {
  it("instantiates correctly without any parameter", () => {
    const processor = punctuationPreTokenizer();
    expect(processor.constructor.name).toEqual("PreTokenizer");
  });
});

describe("splitPreTokenizer", () => {
  it("instantiates correctly with invert parameter", () => {
    const processor = splitPreTokenizer(" ", "mergedWithPrevious", false);
    expect(processor.constructor.name).toEqual("PreTokenizer");
  });
});

describe("sequencePreTokenizer", () => {
  it("instantiates correctly", () => {
    const punctuation = punctuationPreTokenizer();
    const whitespace = whitespaceSplitPreTokenizer();
    const sequence2 = sequencePreTokenizer([]);
    expect(sequence2.constructor.name).toEqual("PreTokenizer");
    const sequence3 = sequencePreTokenizer([punctuation, whitespace]);
    expect(sequence3.constructor.name).toEqual("PreTokenizer");
  });
});
