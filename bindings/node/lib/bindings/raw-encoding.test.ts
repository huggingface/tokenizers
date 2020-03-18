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
    expect(typeof encoding.getOverflowing).toBe("function");
    expect(typeof encoding.getSpecialTokensMask).toBe("function");
    expect(typeof encoding.getTokens).toBe("function");
    expect(typeof encoding.getTypeIds).toBe("function");
    expect(typeof encoding.pad).toBe("function");
    expect(typeof encoding.truncate).toBe("function");
  });

  describe("truncate", () => {
    it("accepts `undefined` as second parameter", () => {
      expect(encoding.truncate(10, undefined)).toBeUndefined();
    });
  });
});
