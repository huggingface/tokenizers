import { bpeDecoder, ctcDecoder, metaspaceDecoder, wordPieceDecoder } from "./decoders";

describe("wordPieceDecoder", () => {
  it("accepts `undefined` as first parameter", () => {
    expect(wordPieceDecoder(undefined)).toBeDefined();
  });

  it("accepts `undefined` as second parameter", () => {
    expect(wordPieceDecoder("test", undefined)).toBeDefined();
  });

  it("can decode arrays of strings", () => {
    expect(
      wordPieceDecoder().decode(["Hel", "##lo", "there", "my", "fr", "##iend"])
    ).toEqual(["Hel", "lo", " there", " my", " fr", "iend"]);
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

describe("ctcDecoder", () => {
  it("accepts `undefined` as parameter", () => {
    expect(ctcDecoder(undefined)).toBeDefined();
  });
  it("encodes correctly", () => {
    expect(
      ctcDecoder().decode(["<pad>", "h", "h", "e", "e", "l", "l", "<pad>", "l", "l", "o"])
    ).toEqual(["h", "e", "l", "l", "o"]);
  });
});
