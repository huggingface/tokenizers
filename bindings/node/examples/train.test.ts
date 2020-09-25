/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-empty-function */

const {
  Tokenizer,
  models,
  normalizers,
  pre_tokenizers,
  post_processors,
  decoders,
  trainers,
  AddedToken,
} = await import("tokenizers");

describe("trainExample", () => {
  it("", () => {
    const vocab_size = 100;

    const tokenizer = new Tokenizer(models.BPE.empty());
    tokenizer.normalizer = normalizers.sequenceNormalizer([
      normalizers.stripNormalizer(),
      normalizers.nfcNormalizer(),
    ]);
    tokenizer.pre_tokenizer = pre_tokenizers.byteLevelPreTokenizer();
    tokenizer.post_processor = post_processors.byteLevelProcessing();
    tokenizer.decoder = decoders.byteLevelDecoder();

    const trainer = trainers.bpeTrainer({
      vocab_size,
      min_frequency: 0,
      special_tokens: [
        new AddedToken("<s>", true),
        new AddedToken("<pad>", true),
        new AddedToken("</s>", true),
        new AddedToken("<unk>", true),
        new AddedToken("<mask>", true),
      ],
      show_progress: true,
    });

    tokenizer.train(trainer, ["data/small.txt"]);
    tokenizer.save("data/tokenizer.json");

    expect(1).toBe(1);
  });
});
