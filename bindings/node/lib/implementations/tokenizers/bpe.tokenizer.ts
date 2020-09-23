import { promisify } from "util";

import { bpeDecoder } from "../../bindings/decoders";
import { BPE, BPEOptions, Model } from "../../bindings/models";
import {
  lowercaseNormalizer,
  nfkcNormalizer,
  sequenceNormalizer,
} from "../../bindings/normalizers";
import { whitespaceSplitPreTokenizer } from "../../bindings/pre-tokenizers";
import { Tokenizer } from "../../bindings/tokenizer";
import { bpeTrainer } from "../../bindings/trainers";
import { BaseTokenizer, getTokenContent, Token } from "./base.tokenizer";

export interface BPETokenizerOptions {
  /**
   * The BPE dropout to use. Must be an float between 0 and 1
   */
  dropout?: number;
  /**
   * @default false
   */
  lowercase?: boolean;
  mergesFile?: string;
  /**
   * @default "</w>"
   */
  suffix?: string;
  /**
   * The unknown token to be used by the model
   * @default "<unk>"
   */
  unkToken?: Token;
  vocabFile?: string;
}

export interface BPETokenizerTrainOptions {
  /**
   * @default []
   */
  initialAlphabet?: string[];
  /**
   * @default 1000
   */
  limitAlphabet?: number;
  /**
   * @default 2
   */
  minFrequency?: number;
  /**
   * @default true
   */
  showProgress?: boolean;
  /**
   * @default ["<unk>"]
   */
  specialTokens?: Token[];
  /**
   * @default "</w>"
   */
  suffix?: string;
  /**
   * @default 30000
   */
  vocabSize?: number;
}

type BPETokenizerConfig = BPETokenizerOptions &
  Required<Pick<BPETokenizerOptions, "unkToken" | "suffix">>;

/**
 * Original BPE Tokenizer.
 * Represents the BPE algorithm, as introduced by Rico Sennrich (https://arxiv.org/abs/1508.07909)
 */
export class BPETokenizer extends BaseTokenizer<BPETokenizerConfig> {
  private static readonly defaultBPEOptions: BPETokenizerConfig = {
    suffix: "</w>",
    unkToken: "<unk>",
  };

  private readonly defaultTrainOptions: Required<BPETokenizerTrainOptions> = {
    initialAlphabet: [],
    limitAlphabet: 1000,
    minFrequency: 2,
    showProgress: true,
    specialTokens: ["<unk>"],
    suffix: "</w>",
    vocabSize: 30000,
  };

  private constructor(tokenizer: Tokenizer, configuration: BPETokenizerConfig) {
    super(tokenizer, configuration);
  }

  /**
   * Instantiate and returns a new BPE tokenizer
   * @param [options] Optional tokenizer options
   */
  static async fromOptions(options?: BPETokenizerOptions): Promise<BPETokenizer> {
    const opts = { ...this.defaultBPEOptions, ...options };
    const unkToken = getTokenContent(opts.unkToken);

    let model: Model;
    if (opts.vocabFile && opts.mergesFile) {
      const modelOptions: BPEOptions = {
        dropout: opts.dropout,
        endOfWordSuffix: opts.suffix,
        unkToken: unkToken,
      };

      const fromFile = promisify<string, string, BPEOptions, Model>(BPE.fromFile);
      model = await fromFile(opts.vocabFile, opts.mergesFile, modelOptions);
    } else {
      model = BPE.empty();
    }

    const tokenizer = new Tokenizer(model);
    if (tokenizer.tokenToId(unkToken) !== undefined) {
      tokenizer.addSpecialTokens([opts.unkToken]);
    }

    if (opts.lowercase) {
      tokenizer.setNormalizer(
        sequenceNormalizer([nfkcNormalizer(), lowercaseNormalizer()])
      );
    } else {
      tokenizer.setNormalizer(nfkcNormalizer());
    }

    tokenizer.setPreTokenizer(whitespaceSplitPreTokenizer());

    const decoder = bpeDecoder(opts.suffix);
    tokenizer.setDecoder(decoder);

    return new BPETokenizer(tokenizer, opts);
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
