import { slice } from "./utils";

describe("slice", () => {
  const text = "My name is John 👋";
  const sliceText = slice.bind({}, text);

  it("returns the full text when no params", () => {
    const sliced = sliceText();
    expect(sliced).toEqual(text);
  });

  it("accepts `undefined` as second parameter", () => {
    const original = sliceText(undefined);
    expect(original).toEqual(text);
  });

  it("accepts `undefined` as third parameter", () => {
    const original = sliceText(0, undefined);
    expect(original).toEqual(text);
  });

  it("throws an error when `begin` is out of range", () => {
    expect(() => sliceText(1000)).toThrow();
  });

  it("returns slice starting at the specified index", () => {
    const original = sliceText(3);
    expect(original).toEqual("name is John 👋");
  });

  it("throws an error when `end` is out of range", () => {
    expect(() => sliceText(0, 1000)).toThrow();
  });

  it("returns the text between the two specified indexes", () => {
    const original = sliceText(3, 7);
    expect(original).toEqual("name");
  });

  describe("with only a negative `begin`", () => {
    it("returns the original string counting from the end when in the range", () => {
      const original = sliceText(-1);
      expect(original).toEqual("👋");
    });

    it("throws an error when out of range", () => {
      expect(() => sliceText(-1000)).toThrow();
    });
  });

  describe("with a positive `begin` and a negative `end`", () => {
    it("returns correct slice when resulting range is valid", () => {
      const original = sliceText(3, -7);
      expect(original).toEqual("name is");
    });

    it("throws an error when resulting `end` index is lower than `begin`", () => {
      expect(() => sliceText(7, -12)).toThrow();
    });

    it("throws an error when `begin` is out of range", () => {
      expect(() => sliceText(1000, -12)).toThrow();
    });

    it("throws an error when resulting `end` index is out of range", () => {
      expect(() => sliceText(7, -1000)).toThrow();
    });
  });

  describe("with a negative `begin` and a positive `end`", () => {
    it("returns correct slice when resulting range is valid", () => {
      const original = sliceText(-9, 10);
      expect(original).toEqual("is");
    });

    it("throws an error when resulting `begin` index is upper than `end`", () => {
      expect(() => sliceText(-3, 5)).toThrow();
    });

    it("throws an error when `end` is out of range", () => {
      expect(() => sliceText(-5, 1000)).toThrow();
    });

    it("throws an error when resulting `begin` index is out of range", () => {
      expect(() => sliceText(-1000, 10)).toThrow();
    });
  });

  describe("with negatives `begin` and `end`", () => {
    it("returns correct slice when resulting range is valid", () => {
      const original = sliceText(-9, -7);
      expect(original).toEqual("is");
    });

    it("throws an error when resulting `end` index is lower than `begin`", () => {
      expect(() => sliceText(-5, -10)).toThrow();
    });

    it("throws an error when resulting `begin` index is out of range", () => {
      expect(() => sliceText(-1000, -10)).toThrow();
    });

    it("throws an error when resulting `end` index is out of range", () => {
      expect(() => sliceText(-10, -1000)).toThrow();
    });
  });
});
