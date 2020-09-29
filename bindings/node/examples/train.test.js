/*eslint-disable no-undef*/

const {
  Tokenizer,
  models,
  normalizers,
  pre_tokenizers,
  post_processors,
  decoders,
  trainers,
  AddedToken,
} = require("..");

describe("trainExample", () => {
  it("", () => {
    const vocabSize = 100;

    const tokenizer = new Tokenizer(models.BPE.empty());
    tokenizer.normalizer = normalizers.sequenceNormalizer([
      normalizers.stripNormalizer(),
      normalizers.nfcNormalizer(),
    ]);
    tokenizer.pre_tokenizer = pre_tokenizers.byteLevelPreTokenizer();
    tokenizer.post_processor = post_processors.byteLevelProcessing();
    tokenizer.decoder = decoders.byteLevelDecoder();

    const trainer = trainers.bpeTrainer({
      vocabSize,
      minFrequency: 0,
      specialTokens: [
        new AddedToken("<s>", true),
        new AddedToken("<pad>", true),
        new AddedToken("</s>", true),
        new AddedToken("<unk>", true),
        new AddedToken("<mask>", true),
      ],
      showProgress: false,
    });

    tokenizer.train(trainer, ["data/small.txt"]);
    tokenizer.save("data/tokenizer.json");

    expect(1).toBe(1);
  });
});
