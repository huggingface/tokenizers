import { byteLevelProcessing, robertaProcessing } from "./post-processors";

describe("byteLevelProcessing", () => {
  it("instantiates correctly without any parameter", () => {
    const processor = byteLevelProcessing();
    expect(processor.constructor.name).toEqual("Processor");
  });

  it("accepts `undefined` as first parameter", () => {
    expect(byteLevelProcessing(undefined)).toBeDefined();
  });

  it("accepts `boolean` as first parameter", () => {
    expect(byteLevelProcessing(true)).toBeDefined();
  });
});

describe("robertaProcessing", () => {
  it("instantiates correctly with only two parameters", () => {
    const processor = robertaProcessing(["sep", 1], ["cls", 2]);
    expect(processor.constructor.name).toEqual("Processor");
  });

  it("accepts `undefined` as third and fourth parameters", () => {
    expect(robertaProcessing(["sep", 1], ["cls", 2], undefined, undefined)).toBeDefined();
  });

  it("accepts `boolean` as third and fourth parameter", () => {
    expect(robertaProcessing(["sep", 1], ["cls", 2], true, true)).toBeDefined();
  });
});
