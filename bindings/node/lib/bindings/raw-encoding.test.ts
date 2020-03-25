import { promisify } from "util";

import { Model, WordPiece, WordPieceOptions } from "./models";
import { whitespacePreTokenizer } from "./pre-tokenizers";
import { RawEncoding } from "./raw-encoding";
import { Tokenizer } from "./tokenizer";

const MOCKS_DIR = __dirname + "/__mocks__";

describe("RawEncoding", () => {
  const originalString = "my name is john";
  let encoding: RawEncoding;

  beforeEach(async () => {
    const model = await promisify<string, WordPieceOptions, Model>(WordPiece.fromFiles)(
      `${MOCKS_DIR}/vocab.txt`,
      {
        continuingSubwordPrefix: "##"
      }
    );

    const tokenizer = new Tokenizer(model);
    tokenizer.setPreTokenizer(whitespacePreTokenizer());

    const encode = promisify(tokenizer.encode.bind(tokenizer));
    encoding = await encode(originalString, null, false);
  });

  it("has a list of defined methods", async () => {
    expect(typeof encoding.charToToken).toBe("function");
    expect(typeof encoding.charToTokenOffsets).toBe("function");
    expect(typeof encoding.charToWordOffsets).toBe("function");
    expect(typeof encoding.getAttentionMask).toBe("function");
    expect(typeof encoding.getIds).toBe("function");
    expect(typeof encoding.getLength).toBe("function");
    expect(typeof encoding.getOffsets).toBe("function");
    expect(typeof encoding.getOverflowing).toBe("function");
    expect(typeof encoding.getSpecialTokensMask).toBe("function");
    expect(typeof encoding.getTokens).toBe("function");
    expect(typeof encoding.getTypeIds).toBe("function");
    expect(typeof encoding.getWords).toBe("function");
    expect(typeof encoding.pad).toBe("function");
    expect(typeof encoding.tokenToWordOffsets).toBe("function");
    expect(typeof encoding.truncate).toBe("function");
  });

  describe("truncate", () => {
    it("accepts `undefined` as second parameter", () => {
      expect(encoding.truncate(10, undefined)).toBeUndefined();
    });
  });

  describe("getWords", () => {
    it("returns the correct list of indexes", () => {
      const indexes = encoding.getWords();
      expect(indexes).toEqual([0, 1, 2, 3, 3]);
    });
  });

  describe("charToToken", () => {
    it("returns the correct index", () => {
      const index = encoding.charToToken(3);
      expect(index).toEqual(1);
    });

    it("returns undefined when out of range char", () => {
      const index = encoding.charToToken(100);
      expect(index).toBeUndefined();
    });
  });

  describe("charToTokenOffsets", () => {
    it("returns the correct offset", () => {
      const offset = encoding.charToTokenOffsets(11);
      expect(offset).toEqual([11, 13]);
    });

    it("returns undefined when out of range char", () => {
      const offset = encoding.charToTokenOffsets(100);
      expect(offset).toBeUndefined();
    });
  });

  describe("charToWordOffsets", () => {
    it("returns the correct offset", () => {
      const offset = encoding.charToWordOffsets(11);
      expect(offset).toEqual([11, 15]);
    });

    it("returns undefined when out of range char", () => {
      const offset = encoding.charToWordOffsets(100);
      expect(offset).toBeUndefined();
    });
  });

  describe("tokenToWordOffsets", () => {
    it("returns the correct offset", () => {
      const offset = encoding.tokenToWordOffsets(3);
      expect(offset).toEqual([11, 15]);
    });

    it("returns undefined when out of range char", () => {
      const offset = encoding.tokenToWordOffsets(100);
      expect(offset).toBeUndefined();
    });
  });
});
