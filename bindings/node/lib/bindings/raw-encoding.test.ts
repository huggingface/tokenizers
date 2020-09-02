import { promisify } from "util";

import { PaddingDirection } from "./enums";
import { Model, WordPiece, WordPieceOptions } from "./models";
import {
  punctuationPreTokenizer,
  sequencePreTokenizer,
  whitespacePreTokenizer,
} from "./pre-tokenizers";
import { RawEncoding } from "./raw-encoding";
import { EncodeOptions, InputSequence, Tokenizer } from "./tokenizer";

const MOCKS_DIR = __dirname + "/__mocks__";

describe("Can modify pretokenizers on the fly", () => {
  let encoding: RawEncoding;
  let encode: (
    sequence: InputSequence,
    pair?: InputSequence | null,
    options?: EncodeOptions | null
  ) => Promise<RawEncoding>;
  let tokenizer: Tokenizer;

  beforeAll(async () => {
    const model = await promisify<string, WordPieceOptions, Model>(WordPiece.fromFiles)(
      `${MOCKS_DIR}/vocab.txt`,
      {
        continuingSubwordPrefix: "##",
      }
    );

    tokenizer = new Tokenizer(model);
    encode = promisify(tokenizer.encode.bind(tokenizer));
  });

  it("Can change pre tokenizer", async () => {
    const input = "my  name is john.!?";
    tokenizer.setPreTokenizer(sequencePreTokenizer([whitespacePreTokenizer()]));

    encoding = await encode(input, null);
    expect(encoding.getIds()).toEqual([0, 1, 2, 3, 4, 6]);

    // Change pre tokenizer
    tokenizer.setPreTokenizer(
      sequencePreTokenizer([whitespacePreTokenizer(), punctuationPreTokenizer()])
    );

    encoding = await encode(input, null);
    expect(encoding.getIds()).toEqual([0, 1, 2, 3, 4, 6, 6, 6]);
  });
});

describe("RawEncoding", () => {
  const originalString = "my name is john";
  let encoding: RawEncoding;
  let encode: (
    sequence: InputSequence,
    pair?: InputSequence | null,
    options?: EncodeOptions | null
  ) => Promise<RawEncoding>;

  beforeAll(async () => {
    const model = await promisify<string, WordPieceOptions, Model>(WordPiece.fromFiles)(
      `${MOCKS_DIR}/vocab.txt`,
      {
        continuingSubwordPrefix: "##",
      }
    );

    const tokenizer = new Tokenizer(model);
    tokenizer.setPreTokenizer(whitespacePreTokenizer());
    encode = promisify(tokenizer.encode.bind(tokenizer));
  });

  beforeEach(async () => {
    encoding = await encode(originalString, null);
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

  describe("pad", () => {
    it("works correctly with only one parameter", () => {
      encoding.pad(10);
      expect(encoding.getTokens()).toHaveLength(10);
    });

    it("accepts `undefined` as second parameter", () => {
      encoding.pad(10, undefined);
      expect(encoding.getTokens()).toHaveLength(10);
    });

    it("accepts options as second parameter", () => {
      encoding.pad(10, {
        direction: PaddingDirection.Left,
        padToken: "[PA]",
        padTypeId: 10,
        padId: 400,
      });

      const tokens = encoding.getTokens();
      expect(tokens).toHaveLength(10);
      expect(tokens[0]).toBe("[PA]");
      expect(encoding.getTypeIds()[0]).toBe(10);
      expect(encoding.getIds()[0]).toBe(400);
    });
  });
});
