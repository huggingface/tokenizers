import { promisify } from "util";

import { BPE } from "./models";
import { Tokenizer } from "./tokenizer";

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

describe("Tokenizer", () => {
  describe("addTokens", () => {
    it("accepts a list of string as new tokens when initial model is empty", () => {
      const model = BPE.empty();
      const tokenizer = new Tokenizer(model);

      const nbAdd = tokenizer.addTokens(["my", "name", "is", "john", "pair"]);
      expect(nbAdd).toBe(5);
    });
  });

  describe("encode", () => {
    let tokenizer: Tokenizer;

    beforeEach(() => {
      // Clear all instances and calls to constructor and all methods:
      // TokenizerMock.mockClear();

      const model = BPE.empty();
      tokenizer = new Tokenizer(model);
      tokenizer.addTokens(["my", "name", "is", "john", "pair"]);
    });

    it("is a function w/ parameters", async () => {
      expect(typeof tokenizer.encode).toBe("function");
    });

    it("accepts a pair of strings as parameters", async () => {
      const encode = promisify(tokenizer.encode.bind(tokenizer));
      const encoding = await encode("my name is john", "pair");
      expect(encoding).toBeDefined();
    });

    it("accepts a string with a null pair", async () => {
      const encode = promisify(tokenizer.encode.bind(tokenizer));
      const encoding = await encode("my name is john", null);
      expect(encoding).toBeDefined();
    });

    it("returns an Encoding", async () => {
      const encode = promisify(tokenizer.encode.bind(tokenizer));
      const encoding = await encode("my name is john", "pair");

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
        [12, 16]
      ]);
      expect(encoding.getOverflowing()).toBeUndefined();
      expect(encoding.getSpecialTokensMask()).toEqual([0, 0, 0, 0, 0]);
      expect(encoding.getTokens()).toEqual(["my", "name", "is", "john", "pair"]);
      expect(encoding.getTypeIds()).toEqual([0, 0, 0, 0, 1]);
    });
  });
});
