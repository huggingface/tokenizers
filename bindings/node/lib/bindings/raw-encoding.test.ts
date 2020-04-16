import { promisify } from "util";

import { Model, WordPiece, WordPieceOptions } from "./models";
import { whitespacePreTokenizer } from "./pre-tokenizers";
import { RawEncoding } from "./raw-encoding";
import { Tokenizer } from "./tokenizer";

const MOCKS_DIR = __dirname + "/__mocks__";

describe("RawEncoding", () => {
  const originalString = "my name is john";
  let encoding: RawEncoding;
  let encode: (
    sequence: string,
    pair: string | null,
    addSpecialTokens: boolean
  ) => Promise<RawEncoding>;

  beforeAll(async () => {
    const model = await promisify<string, WordPieceOptions, Model>(WordPiece.fromFiles)(
      `${MOCKS_DIR}/vocab.txt`,
      {
        continuingSubwordPrefix: "##"
      }
    );

    const tokenizer = new Tokenizer(model);
    tokenizer.setPreTokenizer(whitespacePreTokenizer());
    encode = promisify(tokenizer.encode.bind(tokenizer));
  });

  beforeEach(async () => {
    encoding = await encode(originalString, null, false);
  });

  it("has a list of defined methods", async () => {
    expect(typeof encoding.wordToTokens).toBe("function");
    expect(typeof encoding.wordToChars).toBe("function");
    expect(typeof encoding.tokenToChars).toBe("function");
    expect(typeof encoding.tokenToWord).toBe("function");
    expect(typeof encoding.charToToken).toBe("function");
    expect(typeof encoding.charToWord).toBe("function");
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

  describe("wordToTokens", () => {
    it("returns the correct indexes", () => {
      const indexes = encoding.wordToTokens(3);
      expect(indexes).toEqual([3, 5]);
    });

    it("returns undefined when out of range word", () => {
      const index = encoding.wordToTokens(100);
      expect(index).toBeUndefined();
    });
  });

  describe("wordToChars", () => {
    it("returns the correct offsets", () => {
      const offsets = encoding.wordToChars(3);
      expect(offsets).toEqual([11, 15]);
    });

    it("returns undefined when out of range word", () => {
      const offsets = encoding.wordToChars(100);
      expect(offsets).toBeUndefined();
    });
  });

  describe("tokenToChars", () => {
    it("returns the correct offsets", () => {
      const offsets = encoding.tokenToChars(3);
      expect(offsets).toEqual([11, 13]);
    });

    it("returns undefined when out of range token", () => {
      const offsets = encoding.tokenToChars(100);
      expect(offsets).toBeUndefined();
    });
  });

  describe("tokenToWord", () => {
    it("returns the correct index", () => {
      const index = encoding.tokenToWord(3);
      expect(index).toEqual(3);
    });

    it("returns undefined when out of range token", () => {
      const index = encoding.tokenToWord(100);
      expect(index).toBeUndefined();
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

  describe("charToWord", () => {
    it("returns the correct index", () => {
      const index = encoding.charToWord(3);
      expect(index).toEqual(1);
    });

    it("returns undefined when out of range char", () => {
      const index = encoding.charToWord(100);
      expect(index).toBeUndefined();
    });
  });
});
