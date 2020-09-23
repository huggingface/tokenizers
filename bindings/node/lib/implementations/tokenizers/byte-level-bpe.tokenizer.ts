import { promisify } from "util";

import { byteLevelDecoder } from "../../bindings/decoders";
import { BPE, BPEOptions, Model } from "../../bindings/models";
import {
  lowercaseNormalizer,
  nfkcNormalizer,
  sequenceNormalizer,
} from "../../bindings/normalizers";
import { byteLevelProcessing } from "../../bindings/post-processors";
import { byteLevelAlphabet, byteLevelPreTokenizer } from "../../bindings/pre-tokenizers";
import { Tokenizer } from "../../bindings/tokenizer";
import { bpeTrainer } from "../../bindings/trainers";
import { BaseTokenizer, Token } from "./base.tokenizer";

export interface ByteLevelBPETokenizerOptions {
  /**
   * @default false
   */
  addPrefixSpace?: boolean;
  /**
   * The prefix to attach to subword units that don't represent a beginning of word
   */
  continuingSubwordPrefix?: string;
  /**
   * @default false
   */
  lowercase?: boolean;
  /**
   * The BPE dropout to use. Must be an float between 0 and 1
   */
  dropout?: number;
  /**
   * The suffix to attach to subword units that represent an end of word
   */
  endOfWordSuffix?: string;
  mergesFile?: string;
  unicodeNormalizer?: string;
  /**
   * Whether to trim the whitespaces from the produced offsets
   * @default false
   */
  trimOffsets?: boolean;
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
  specialTokens?: Token[];
  /**
   * @default 30000
   */
  vocabSize?: number;
}

type ByteLevelBPETokenizerConfig = ByteLevelBPETokenizerOptions &
  Required<Pick<ByteLevelBPETokenizerOptions, "addPrefixSpace">>;

/**
 * Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
 */
export class ByteLevelBPETokenizer extends BaseTokenizer<ByteLevelBPETokenizerConfig> {
  private static readonly defaultOptions: ByteLevelBPETokenizerConfig = {
    addPrefixSpace: false,
    trimOffsets: false,
  };

  private readonly defaultTrainOptions: Required<ByteLevelBPETrainOptions> = {
    minFrequency: 2,
    showProgress: true,
    specialTokens: ["<unk>"],
    vocabSize: 30000,
  };

  private constructor(tokenizer: Tokenizer, configuration: ByteLevelBPETokenizerConfig) {
    super(tokenizer, configuration);
  }

  static async fromOptions(
    options?: ByteLevelBPETokenizerOptions
  ): Promise<ByteLevelBPETokenizer> {
    const opts = { ...this.defaultOptions, ...options };

    let model: Model;
    if (opts.vocabFile && opts.mergesFile) {
      const fromFile = promisify<string, string, BPEOptions, Model>(BPE.fromFile);
      model = await fromFile(opts.vocabFile, opts.mergesFile, opts);
    } else {
      model = BPE.empty();
    }

    const tokenizer = new Tokenizer(model);

    if (opts.lowercase) {
      tokenizer.setNormalizer(
        sequenceNormalizer([nfkcNormalizer(), lowercaseNormalizer()])
      );
    } else {
      tokenizer.setNormalizer(nfkcNormalizer());
    }

    const preTokenizer = byteLevelPreTokenizer(opts.addPrefixSpace);
    tokenizer.setPreTokenizer(preTokenizer);
    tokenizer.setDecoder(byteLevelDecoder());
    tokenizer.setPostProcessor(byteLevelProcessing(opts.trimOffsets));

    return new ByteLevelBPETokenizer(tokenizer, opts);
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
      initialAlphabet: byteLevelAlphabet(),
    });

    this.tokenizer.train(trainer, files);
  }
}
