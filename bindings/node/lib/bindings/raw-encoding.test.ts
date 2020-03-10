import { promisify } from "util";

import { BPE } from "./models";
import { RawEncoding } from "./raw-encoding";
import { Tokenizer } from "./tokenizer";

describe("RawEncoding", () => {
  const originalString = "my name is john";
  let encoding: RawEncoding;

  beforeEach(async () => {
    const model = BPE.empty();
    const tokenizer = new Tokenizer(model);
    tokenizer.addTokens(["my", "name", "is", "john", "pair"]);

    const encode = promisify(tokenizer.encode.bind(tokenizer));
    encoding = await encode(originalString, null, false);
  });

  it("has a list of defined methods", async () => {
    expect(typeof encoding.getAttentionMask).toBe("function");
    expect(typeof encoding.getIds).toBe("function");
    expect(typeof encoding.getLength).toBe("function");
    expect(typeof encoding.getOffsets).toBe("function");
    expect(typeof encoding.getOriginalString).toBe("function");
    expect(typeof encoding.getOverflowing).toBe("function");
    expect(typeof encoding.getSpecialTokensMask).toBe("function");
    expect(typeof encoding.getTokens).toBe("function");
    expect(typeof encoding.getTypeIds).toBe("function");
    expect(typeof encoding.pad).toBe("function");
    expect(typeof encoding.truncate).toBe("function");
  });

  describe("getOriginalString", () => {
    it("returns the full original string when no params", () => {
      const original = encoding.getOriginalString();
      expect(original).toEqual(originalString);
    });

    it("accepts `undefined` as first parameter", () => {
      const original = encoding.getOriginalString(undefined);
      expect(original).toEqual(originalString);
    });

    it("accepts `undefined` as second parameter", () => {
      const original = encoding.getOriginalString(0, undefined);
      expect(original).toEqual(originalString);
    });

    it("throws an error when `begin` is out of range", () => {
      expect(() => encoding.getOriginalString(1000)).toThrow();
    });

    it("returns the original string starting at the specified index", () => {
      const original = encoding.getOriginalString(3);
      expect(original).toEqual("name is john");
    });

    it("throws an error when `end` is out of range", () => {
      expect(() => encoding.getOriginalString(0, 1000)).toThrow();
    });

    it("returns the original string between the two specified indexes", () => {
      const original = encoding.getOriginalString(3, 7);
      expect(original).toEqual("name");
    });

    describe("with only a negative `begin`", () => {
      it("returns the original string counting from the end when in the range", () => {
        const original = encoding.getOriginalString(-4);
        expect(original).toEqual("john");
      });

      it("throws an error when out of range", () => {
        expect(() => encoding.getOriginalString(-1000)).toThrow();
      });
    });

    describe("with a positive `begin` and a negative `end`", () => {
      it("returns the original string when resulting range is valid", () => {
        const original = encoding.getOriginalString(3, -5);
        expect(original).toEqual("name is");
      });

      it("throws an error when resulting `end` index is lower than `begin`", () => {
        expect(() => encoding.getOriginalString(7, -10)).toThrow();
      });

      it("throws an error when `begin` is out of range", () => {
        expect(() => encoding.getOriginalString(1000, -10)).toThrow();
      });

      it("throws an error when resulting `end` index is out of range", () => {
        expect(() => encoding.getOriginalString(7, -1000)).toThrow();
      });
    });

    describe("with a negative `begin` and a positive `end`", () => {
      it("returns the original string when resulting range is valid", () => {
        const original = encoding.getOriginalString(-7, 10);
        expect(original).toEqual("is");
      });

      it("throws an error when resulting `begin` index is upper than `end`", () => {
        expect(() => encoding.getOriginalString(-3, 5)).toThrow();
      });

      it("throws an error when `end` is out of range", () => {
        expect(() => encoding.getOriginalString(-5, 1000)).toThrow();
      });

      it("throws an error when resulting `begin` index is out of range", () => {
        expect(() => encoding.getOriginalString(-1000, 10)).toThrow();
      });
    });

    describe("with negatives `begin` and `end`", () => {
      it("returns the original string when resulting range is valid", () => {
        const original = encoding.getOriginalString(-7, -5);
        expect(original).toEqual("is");
      });

      it("throws an error when resulting `end` index is lower than `begin`", () => {
        expect(() => encoding.getOriginalString(-5, -10)).toThrow();
      });

      it("throws an error when resulting `begin` index is out of range", () => {
        expect(() => encoding.getOriginalString(-1000, -10)).toThrow();
      });

      it("throws an error when resulting `end` index is out of range", () => {
        expect(() => encoding.getOriginalString(-10, -1000)).toThrow();
      });
    });
  });

  describe("truncate", () => {
    it("accepts `undefined` as second parameter", () => {
      expect(encoding.truncate(10, undefined)).toBeUndefined();
    });
  });
});
