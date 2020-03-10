import { byteLevelPreTokenizer, metaspacePreTokenizer } from "./pre-tokenizers";

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
    expect(metaspacePreTokenizer("test", undefined)).toBeDefined();
  });
});
