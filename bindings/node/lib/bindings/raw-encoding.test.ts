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
    const model = await promisify<string, WordPieceOptions, Model>(WordPiece.fromFile)(
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
    expect(encoding.getIds()).toEqual([0, 1, 2, 3, 4, 8]);

    // Change pre tokenizer
    tokenizer.setPreTokenizer(
      sequencePreTokenizer([whitespacePreTokenizer(), punctuationPreTokenizer()])
    );

    encoding = await encode(input, null);
    expect(encoding.getIds()).toEqual([0, 1, 2, 3, 4, 8, 8, 8]);
  });
});

describe("RawEncoding", () => {
  const originalString = "my name is john";
  const originalPairString = "what is yours?";
  let encoding: RawEncoding;
  let encodingDual: RawEncoding;
  let encode: (
    sequence: InputSequence,
    pair?: InputSequence | null,
    options?: EncodeOptions | null
  ) => Promise<RawEncoding>;

  beforeAll(async () => {
    const model = await promisify<string, WordPieceOptions, Model>(WordPiece.fromFile)(
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
    encodingDual = await encode(originalString, originalPairString);
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
    expect(typeof encoding.getWordIds).toBe("function");
    expect(typeof encoding.getSequenceIds).toBe("function");
    expect(typeof encoding.pad).toBe("function");
    expect(typeof encoding.truncate).toBe("function");
  });

  describe("truncate", () => {
    it("accepts `undefined` as second parameter", () => {
      expect(encoding.truncate(10, undefined)).toBeUndefined();
    });
  });

  describe("getWordIds", () => {
    it("returns the correct list of indexes", () => {
      const indexes = encoding.getWordIds();
      expect(indexes).toEqual([0, 1, 2, 3, 3]);
    });
  });

  describe("getSequenceIds", () => {
    it("returns the correct list of indexes", () => {
      expect(encoding.getSequenceIds()).toEqual([0, 0, 0, 0, 0]);
      expect(encodingDual.getSequenceIds()).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1]);
    });
  });

  describe("wordToTokens", () => {
    it("returns the correct indexes", () => {
      const indexes = encoding.wordToTokens(3);
      expect(indexes).toEqual([3, 5]);
    });

    it("returns the corrent indexes with pair sequences", () => {
      expect(encodingDual.wordToTokens(3, 0)).toEqual([3, 5]);
      expect(encodingDual.wordToTokens(3, 1)).toEqual([8, 9]);
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

    it("returns the correct offsets with pair sequences", () => {
      expect(encodingDual.wordToChars(3, 0)).toEqual([11, 15]);
      expect(encodingDual.wordToChars(3, 1)).toEqual([13, 14]);
    });

    it("returns undefined when out of range word", () => {
      const offsets = encoding.wordToChars(100);
      expect(offsets).toBeUndefined();
    });
  });

  describe("tokenToSequence", () => {
    it("returns the correct value", () => {
      expect(encodingDual.tokenToSequence(4)).toEqual(0);
      expect(encodingDual.tokenToSequence(6)).toEqual(1);
    });
  });

  describe("tokenToChars", () => {
    it("returns the correct offsets", () => {
      const offsets = encoding.tokenToChars(3);
      expect(offsets).toEqual([11, 13]);
    });

    it("returns the correct offsets with pair sequences", () => {
      expect(encodingDual.tokenToChars(3)).toEqual([11, 13]);
      expect(encodingDual.tokenToChars(7)).toEqual([8, 13]);
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

    it("returns the correct index with pair sequences", () => {
      expect(encodingDual.tokenToWord(3)).toEqual(3);
      expect(encodingDual.tokenToWord(7)).toEqual(2);
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

    it("returns the correct index with pair sequences", () => {
      expect(encodingDual.charToToken(3, 0)).toEqual(1);
      expect(encodingDual.charToToken(3, 1)).toEqual(5);
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

    it("returns the correct index with pair sequences", () => {
      expect(encodingDual.charToWord(3, 0)).toEqual(1);
      expect(encodingDual.charToWord(3, 1)).toEqual(0);
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
