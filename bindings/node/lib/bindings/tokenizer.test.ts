/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-empty-function */

import { promisify } from "util";

import { PaddingDirection, TruncationStrategy } from "./enums";
import { BPE } from "./models";
import { lowercaseNormalizer } from "./normalizers";
import { RawEncoding } from "./raw-encoding";
import {
  AddedToken,
  PaddingConfiguration,
  Tokenizer,
  TruncationConfiguration
} from "./tokenizer";

// jest.mock('../bindings/tokenizer');
// jest.mock('../bindings/models', () => ({
//   __esModule: true,
//   Model: jest.fn()
// }));

// Or:
// jest.mock('../bindings/models', () => {
//   return require('../bindings/__mocks__/models');
// });

// const TokenizerMock = mocked(Tokenizer);

describe("AddedToken", () => {
  it("instantiates with only content", () => {
    const addToken = new AddedToken("test");
    expect(addToken.constructor.name).toEqual("AddedToken");
  });

  it("instantiates with empty options", () => {
    const addToken = new AddedToken("test", {});
    expect(addToken.constructor.name).toEqual("AddedToken");
  });

  it("instantiates with options", () => {
    const addToken = new AddedToken("test", {
      leftStrip: true,
      rightStrip: true,
      singleWord: true
    });
    expect(addToken.constructor.name).toEqual("AddedToken");
  });
});

describe("Tokenizer", () => {
  it("has expected methods", () => {
    const model = BPE.empty();
    const tokenizer = new Tokenizer(model);

    expect(typeof tokenizer.addSpecialTokens).toBe("function");
    expect(typeof tokenizer.addTokens).toBe("function");
    expect(typeof tokenizer.decode).toBe("function");
    expect(typeof tokenizer.decodeBatch).toBe("function");
    expect(typeof tokenizer.disablePadding).toBe("function");
    expect(typeof tokenizer.disableTruncation).toBe("function");
    expect(typeof tokenizer.encode).toBe("function");
    expect(typeof tokenizer.encodeBatch).toBe("function");
    expect(typeof tokenizer.getDecoder).toBe("function");
    expect(typeof tokenizer.getNormalizer).toBe("function");
    expect(typeof tokenizer.getPostProcessor).toBe("function");
    expect(typeof tokenizer.getPreTokenizer).toBe("function");
    expect(typeof tokenizer.getVocabSize).toBe("function");
    expect(typeof tokenizer.idToToken).toBe("function");
    expect(typeof tokenizer.runningTasks).toBe("function");
    expect(typeof tokenizer.setDecoder).toBe("function");
    expect(typeof tokenizer.setModel).toBe("function");
    expect(typeof tokenizer.setNormalizer).toBe("function");
    expect(typeof tokenizer.setPadding).toBe("function");
    expect(typeof tokenizer.setPostProcessor).toBe("function");
    expect(typeof tokenizer.setPreTokenizer).toBe("function");
    expect(typeof tokenizer.setTruncation).toBe("function");
    expect(typeof tokenizer.tokenToId).toBe("function");
    expect(typeof tokenizer.train).toBe("function");
  });

  describe("addTokens", () => {
    it("accepts a list of string as new tokens when initial model is empty", () => {
      const model = BPE.empty();
      const tokenizer = new Tokenizer(model);

      const nbAdd = tokenizer.addTokens(["my", "name", "is", "john", "pair"]);
      expect(nbAdd).toBe(5);
    });

    it("accepts a list of AddedToken as new tokens when initial model is empty", () => {
      const model = BPE.empty();
      const tokenizer = new Tokenizer(model);
      const addedToken = new AddedToken("test");

      const nbAdd = tokenizer.addTokens([addedToken]);
      expect(nbAdd).toBe(1);
    });
  });

  describe("encode", () => {
    let tokenizer: Tokenizer;
    let encode: (
      sequence: string,
      pair: string | null,
      addSpecialTokens: boolean
    ) => Promise<RawEncoding>;

    beforeEach(() => {
      // Clear all instances and calls to constructor and all methods:
      // TokenizerMock.mockClear();

      const model = BPE.empty();
      tokenizer = new Tokenizer(model);
      tokenizer.addTokens(["my", "name", "is", "john", "pair"]);

      // const my = new AddedToken("my");
      // const name = new AddedToken("name");
      // const is = new AddedToken("is");
      // const john = new AddedToken("john");
      // const pair = new AddedToken("pair");
      // tokenizer.addTokens([my, name, is, john, pair]);

      encode = promisify(tokenizer.encode.bind(tokenizer));
    });

    it("accepts a pair of strings as parameters", async () => {
      const encoding = await encode("my name is john", "pair", false);
      expect(encoding).toBeDefined();
    });

    it("accepts a string with a null pair", async () => {
      const encoding = await encode("my name is john", null, false);
      expect(encoding).toBeDefined();
    });

    it("throws if called with only two arguments", async () => {
      await expect((encode as any)("my name is john", null)).rejects.toThrow(
        "failed downcast to boolean"
      );
    });

    it("returns an Encoding", async () => {
      const encoding = await encode("my name is john", "pair", false);

      expect(encoding.getAttentionMask()).toEqual([1, 1, 1, 1, 1]);

      const ids = encoding.getIds();
      expect(Array.isArray(ids)).toBe(true);
      expect(ids).toHaveLength(5);
      for (const id of ids) {
        expect(typeof id).toBe("number");
      }

      expect(encoding.getOffsets()).toEqual([
        [0, 2],
        [2, 6],
        [6, 8],
        [8, 12],
        [0, 4]
      ]);
      expect(encoding.getOverflowing()).toEqual([]);
      expect(encoding.getSpecialTokensMask()).toEqual([0, 0, 0, 0, 0]);
      expect(encoding.getTokens()).toEqual(["my", "name", "is", "john", "pair"]);
      expect(encoding.getTypeIds()).toEqual([0, 0, 0, 0, 1]);
    });

    describe("when truncation is enabled", () => {
      it("truncates with default if no truncation options provided", async () => {
        tokenizer.setTruncation(2);

        const singleEncoding = await encode("my name is john", null, false);
        expect(singleEncoding.getTokens()).toEqual(["my", "name"]);

        const pairEncoding = await encode("my name is john", "pair", false);
        expect(pairEncoding.getTokens()).toEqual(["my", "pair"]);
      });

      it("throws an error with strategy `only_second` and no pair is encoded", async () => {
        tokenizer.setTruncation(2, { strategy: TruncationStrategy.OnlySecond });
        await expect(encode("my name is john", null, false)).rejects.toThrow();
      });
    });

    describe("when padding is enabled", () => {
      it("does not pad anything with default options", async () => {
        tokenizer.setPadding();

        const singleEncoding = await encode("my name", null, false);
        expect(singleEncoding.getTokens()).toEqual(["my", "name"]);

        const pairEncoding = await encode("my name", "pair", false);
        expect(pairEncoding.getTokens()).toEqual(["my", "name", "pair"]);
      });

      it("pads to the right by default", async () => {
        tokenizer.setPadding({ maxLength: 5 });

        const singleEncoding = await encode("my name", null, false);
        expect(singleEncoding.getTokens()).toEqual([
          "my",
          "name",
          "[PAD]",
          "[PAD]",
          "[PAD]"
        ]);

        const pairEncoding = await encode("my name", "pair", false);
        expect(pairEncoding.getTokens()).toEqual([
          "my",
          "name",
          "pair",
          "[PAD]",
          "[PAD]"
        ]);
      });
    });
  });

  describe("decode", () => {
    let tokenizer: Tokenizer;

    beforeEach(() => {
      const model = BPE.empty();
      tokenizer = new Tokenizer(model);
      tokenizer.addTokens(["my", "name", "is", "john", "pair"]);
    });

    it("returns `undefined`", () => {
      expect(tokenizer.decode([0, 1, 2, 3], true, () => {})).toBeUndefined();
    });

    it("has its callback called with the decoded string", async () => {
      const decode = promisify(tokenizer.decode.bind(tokenizer));
      await expect(decode([0, 1, 2, 3], true)).resolves.toEqual("my name is john");
    });
  });

  describe("decodeBatch", () => {
    let tokenizer: Tokenizer;

    beforeEach(() => {
      const model = BPE.empty();
      tokenizer = new Tokenizer(model);
      tokenizer.addTokens(["my", "name", "is", "john", "pair"]);
    });

    it("returns `undefined`", () => {
      expect(tokenizer.decodeBatch([[0, 1, 2, 3], [4]], true, () => {})).toBeUndefined();
    });

    it("has its callback called with the decoded string", async () => {
      const decodeBatch = promisify(tokenizer.decodeBatch.bind(tokenizer));
      await expect(decodeBatch([[0, 1, 2, 3], [4]], true)).resolves.toEqual([
        "my name is john",
        "pair"
      ]);
    });
  });

  describe("getVocabSize", () => {
    it("accepts `undefined` as parameter", () => {
      const model = BPE.empty();
      const tokenizer = new Tokenizer(model);

      expect(tokenizer.getVocabSize(undefined)).toBeDefined();
    });
  });

  describe("setTruncation", () => {
    it("returns the full truncation configuration", () => {
      const model = BPE.empty();
      const tokenizer = new Tokenizer(model);

      const truncation = tokenizer.setTruncation(2);
      const expectedConfig: TruncationConfiguration = {
        maxLength: 2,
        strategy: TruncationStrategy.LongestFirst,
        stride: 0
      };
      expect(truncation).toEqual(expectedConfig);
    });
  });

  describe("setPadding", () => {
    it("returns the full padding params", () => {
      const model = BPE.empty();
      const tokenizer = new Tokenizer(model);

      const padding = tokenizer.setPadding();
      const expectedConfig: PaddingConfiguration = {
        direction: PaddingDirection.Right,
        padId: 0,
        padToken: "[PAD]",
        padTypeId: 0
      };
      expect(padding).toEqual(expectedConfig);
    });
  });

  describe("normalize", () => {
    it("normalizes a string correctly", () => {
      const model = BPE.empty();
      const tokenizer = new Tokenizer(model);
      tokenizer.setNormalizer(lowercaseNormalizer());

      expect(tokenizer.normalize("MY NAME IS JOHN")).toEqual("my name is john");
    });
  });
});
