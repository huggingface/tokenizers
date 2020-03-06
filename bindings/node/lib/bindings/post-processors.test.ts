import { byteLevelProcessing } from "./post-processors";

describe("byteLevelProcessing", () => {
  it("instantiates correctly", () => {
    const processor = byteLevelProcessing();
    expect(processor.constructor.name).toEqual("Processor");
  });
});
