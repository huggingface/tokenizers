import { byteLevelProcessing } from "./post-processors";

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
