import { byteLevelDecoder } from "../bindings/decoders";
import { BPE, Model } from "../bindings/models";
import { nfkcNormalizer } from "../bindings/normalizers";
import { byteLevelAlphabet, byteLevelPreTokenizer } from "../bindings/pre-tokenizers";
import { Tokenizer } from "../bindings/tokenizer";
import { bpeTrainer } from "../bindings/trainers";
import { BaseTokenizer } from "./base.tokenizer";

export interface ByteLevelBPETokenizerOptions {
  /**
   * @default false
   */
  addPrefixSpace?: boolean;
  mergesFile?: string;
  vocabFile?: string;
}

export interface ByteLevelBPETrainOptions {
  /**
   * @default 2
   */
  minFrequency?: number;
  /**
   * @default true
   */
  showProgress?: boolean;
  /**
   * @default []
   */
  specialTokens?: string[];
  /**
   * @default 30000
   */
  vocabSize?: number;
}

/**
 * Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
 */
export class ByteLevelBPETokenizer extends BaseTokenizer {
  private static readonly defaultOptions: ByteLevelBPETokenizerOptions &
    Required<Pick<ByteLevelBPETokenizerOptions, "addPrefixSpace">> = {
    addPrefixSpace: false
  };

  private readonly defaultTrainOptions: Required<ByteLevelBPETrainOptions> = {
    minFrequency: 2,
    showProgress: true,
    specialTokens: ["<unk>"],
    vocabSize: 30000
  };

  private constructor(tokenizer: Tokenizer) {
    super(tokenizer);
  }

  static async fromOptions(
    options?: ByteLevelBPETokenizerOptions
  ): Promise<ByteLevelBPETokenizer> {
    const opts = { ...this.defaultOptions, ...options };

    let model: Model;
    if (opts.vocabFile && opts.mergesFile) {
      // const fromFiles = promisify(BPE.fromFiles);
      model = BPE.fromFiles(opts.vocabFile, opts.mergesFile);
      // model = await fromFiles(mergedOptions.vocabFile, mergedOptions.mergesFile, null);
    } else {
      model = BPE.empty();
    }

    const tokenizer = new Tokenizer(model);
    tokenizer.setNormalizer(nfkcNormalizer());

    const preTokenizer = byteLevelPreTokenizer(opts.addPrefixSpace);
    tokenizer.setPreTokenizer(preTokenizer);
    tokenizer.setDecoder(byteLevelDecoder());

    return new ByteLevelBPETokenizer(tokenizer);
  }

  /**
   * Train the model using the given files
   *
   * @param files Files to use for training
   * @param [options] Training options
   */
  async train(files: string[], options?: ByteLevelBPETrainOptions): Promise<void> {
    const mergedOptions = { ...this.defaultTrainOptions, ...options };
    const trainer = bpeTrainer({
      ...mergedOptions,
      initialAlphabet: byteLevelAlphabet()
    });

    this.tokenizer.train(trainer, files);
  }
}
