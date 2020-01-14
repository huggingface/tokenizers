import { promisify } from "util";
import { BaseTokenizer } from "./base.tokenizer";
import { Model, bpe } from "../bindings/models";
import { Tokenizer } from "../bindings/tokenizer";
import { sequenceNormalizer, nfkcNormalizer, lowercaseNormalizer } from "../bindings/normalizers";
import { whitespacePreTokenizer } from "../bindings/pre-tokenizers";
import { bpeDecoder } from "../bindings/decoders";
import { bpeTrainer } from "../bindings/trainers";

export interface BPETokenizerOptions {
  dropout?:    number;
  mergesFile?: string;
  /**
   * @default "</w>"
   */
  suffix?:     string;
  /**
   * @default "<unk>"
   */
  unkToken?:   string;
  vocabFile?:  string;
}

export interface BPETokenizerTrainOptions {
  /**
   * @default []
   */
  initialAlphabet?: string[];
  /**
   * @default 1000
   */
  limitAlphabet?:   number;
  /**
   * @default 2
   */
  minFrequency?:    number;
  /**
   * @default true
   */
  showProgress?:    boolean;
  /**
   * @default ["<unk>"]
   */
  specialTokens?:   string[];
  /**
   * @default "</w>"
   */
  suffix?:          string;
  /**
   * @default 30000
   */
  vocabSize?:       number;
}

/**
 * Original BPE Tokenizer.
 * Represents the BPE algorithm, as introduced by Rico Sennrich (https://arxiv.org/abs/1508.07909)
 */
export class BPETokenizer extends BaseTokenizer {
  private static readonly defaultBPEOptions:
    BPETokenizerOptions & Required<Pick<BPETokenizerOptions, "unkToken" | "suffix">> = {
    suffix:   "</w>",
    unkToken: "<unk>"
  };

  private readonly defaultTrainOptions: Required<BPETokenizerTrainOptions> = {
    initialAlphabet: [],
    limitAlphabet:   1000,
    minFrequency:    2,
    showProgress:    true,
    specialTokens:   ["<unk>"],
    suffix:          "</w>",
    vocabSize:       30000
  };

  private constructor(tokenizer: Tokenizer) {
    super(tokenizer);
  }

  /**
   * Instantiate and returns a new BPE tokenizer
   * @param [options] Optional tokenizer options
   */
  static async fromOptions(options?: BPETokenizerOptions): Promise<BPETokenizer> {
    const mergedOptions = { ...this.defaultBPEOptions, ...options };

    let model: Model;
    if (mergedOptions.vocabFile && mergedOptions.mergesFile) {
      // const fromFiles = promisify(BPE.fromFiles);
      const modelOptions: bpe.BPEModelOptions = {
        dropout:         mergedOptions.dropout,
        endOfWordSuffix: mergedOptions.suffix,
        unkToken:        mergedOptions.unkToken
      };

      model = bpe.fromFiles(mergedOptions.vocabFile, mergedOptions.mergesFile, modelOptions);
      // model = await fromFiles(mergedOptions.vocabFile, mergedOptions.mergesFile, modelOptions);
    } else {
      model = bpe.empty();
    }
  
    const tokenizer = new Tokenizer(model);

    const normalizer = sequenceNormalizer([nfkcNormalizer(), lowercaseNormalizer()]);
    tokenizer.setNormalizer(normalizer);
    tokenizer.setPreTokenizer(whitespacePreTokenizer());

    const decoder = bpeDecoder(mergedOptions.suffix);
    tokenizer.setDecoder(decoder);

    return new BPETokenizer(tokenizer);
  }

  /**
   * Train the model using the given files
   *
   * @param files Files to use for training
   * @param [options] Training options
   */
  async train(files: string[], options?: BPETokenizerTrainOptions): Promise<void> {
    const mergedOptions = { ...this.defaultTrainOptions, ...options };
    const trainer = bpeTrainer(mergedOptions);

    this.tokenizer.train(trainer, files);
  }
}
