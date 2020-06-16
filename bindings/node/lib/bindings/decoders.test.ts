import { bpeDecoder, metaspaceDecoder, wordPieceDecoder } from "./decoders";

describe("wordPieceDecoder", () => {
  it("accepts `undefined` as first parameter", () => {
    expect(wordPieceDecoder(undefined)).toBeDefined();
  });

  it("accepts `undefined` as second parameter", () => {
    expect(wordPieceDecoder("test", undefined)).toBeDefined();
  });
});

describe("metaspaceDecoder", () => {
  it("accepts `undefined` as first parameter", () => {
    expect(metaspaceDecoder(undefined)).toBeDefined();
  });

  it("accepts `undefined` as second parameter", () => {
    expect(metaspaceDecoder("t", undefined)).toBeDefined();
  });
});

describe("bpeDecoder", () => {
  it("accepts `undefined` as parameter", () => {
    expect(bpeDecoder(undefined)).toBeDefined();
  });
});
