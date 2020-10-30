import { stripAccentsNormalizer, stripNormalizer } from "./normalizers";

describe("stripNormalizer", () => {
  it("instantiates with no parameters", () => {
    const normalizer = stripNormalizer();
    expect(normalizer.constructor.name).toEqual("Normalizer");
  });

  it("accepts `undefined` as first parameter", () => {
    expect(stripNormalizer(undefined)).toBeDefined();
  });

  it("accepts `undefined` as second parameter", () => {
    expect(stripNormalizer(false, undefined)).toBeDefined();
  });

  it("instantiates with one parameter", () => {
    const normalizer = stripNormalizer(false);
    expect(normalizer.constructor.name).toEqual("Normalizer");
  });

  it("instantiates with two parameters", () => {
    const normalizer = stripNormalizer(false, true);
    expect(normalizer.constructor.name).toEqual("Normalizer");
  });

  it("can normalize strings", () => {
    const normalizer = stripNormalizer();
    expect(normalizer.normalizeString("     Hello there   ")).toEqual("Hello there");
  });
});

describe("stripAccentsNormalizer", () => {
  it("initialize", () => {
    const normalizer = stripAccentsNormalizer();
    expect(normalizer.constructor.name).toEqual("Normalizer");
  });
});
