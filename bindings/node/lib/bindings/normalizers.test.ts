import { stripNormalizer } from "./normalizers";

describe("stripNormalizer", () => {
  it("instantiates with no parameters", () => {
    const normalizer = stripNormalizer();
    expect(normalizer.constructor.name).toEqual("Normalizer");
  });

  it("instantiates with one parameter", () => {
    const normalizer = stripNormalizer(false);
    expect(normalizer.constructor.name).toEqual("Normalizer");
  });

  it("instantiates with two parameter", () => {
    const normalizer = stripNormalizer(false, true);
    expect(normalizer.constructor.name).toEqual("Normalizer");
  });
});
