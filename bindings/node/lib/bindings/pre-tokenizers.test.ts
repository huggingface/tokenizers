import {
  byteLevelPreTokenizer,
  metaspacePreTokenizer,
  punctuationPreTokenizer,
  sequencePreTokenizer,
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
});

describe("punctuationPreTokenizer", () => {
  it("instantiates correctly without any parameter", () => {
    const processor = punctuationPreTokenizer();
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
