import { BaseTokenizer } from "./base.tokenizer";
import { Tokenizer } from "../bindings/tokenizer";
import { Model, bpe } from "../bindings/models";
import { nfkcNormalizer } from "../bindings/normalizers";
import { metaspacePreTokenizer } from "../bindings/pre-tokenizers";
import { metaspaceDecoder } from "../bindings/decoders";
import { bpeTrainer } from "../bindings/trainers";

export interface SentencePieceBPETokenizerOptions extends OptionsWithDefaults {
  dropout?:    number;
  mergesFile?: string;
  vocabFile?:  string;
}

interface OptionsWithDefaults {
  /**
   * @default true
   */
  addPrefixSpace?: boolean;
  /**
   * @default "▁"
   */
  replacement?:    string;
  /**
   * @default "<unk>"
   */
  unkToken?:       string;
}

export interface SentencePieceBPETrainOptions {
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
   * @default 30000
   */
  vocabSize?:       number;
}

/**
 * Represents the BPE algorithm, with the pretokenization used by SentencePiece
 */
export class SentencePieceBPETokenizer extends BaseTokenizer {
  private static readonly defaultOptions: SentencePieceBPETokenizerOptions & Required<OptionsWithDefaults> = {
    addPrefixSpace: true,
    replacement:    '▁',
    unkToken:       '<unk>'
  };

  private readonly defaultTrainOptions: Required<SentencePieceBPETrainOptions> = {
    initialAlphabet: [],
    limitAlphabet:   1000,
    minFrequency:    2,
    showProgress:    true,
    specialTokens:   ['<unk>'],
    vocabSize:       30000
  };

  private constructor(tokenizer: Tokenizer) {
    super(tokenizer);
  }

  static async fromOptions(options?: SentencePieceBPETokenizerOptions): Promise<SentencePieceBPETokenizer> {
    const mergedOptions = { ...this.defaultOptions, ...options };

    let model: Model;
    if (mergedOptions.vocabFile && mergedOptions.mergesFile) {
      // const fromFiles = promisify(BPE.fromFiles);
      const modelOptions: bpe.BPEModelOptions = {
        dropout:  mergedOptions.dropout,
        unkToken: mergedOptions.unkToken
      };

      model = bpe.fromFiles(mergedOptions.vocabFile, mergedOptions.mergesFile, modelOptions);
      // model = await fromFiles(mergedOptions.vocabFile, mergedOptions.mergesFile, null);
    } else {
      model = bpe.empty();
    }
    
    const tokenizer = new Tokenizer(model);
    tokenizer.setNormalizer(nfkcNormalizer());

    const preTokenizer = metaspacePreTokenizer(mergedOptions.replacement, mergedOptions.addPrefixSpace);
    tokenizer.setPreTokenizer(preTokenizer);

    const decoder = metaspaceDecoder(mergedOptions.replacement, mergedOptions.addPrefixSpace);
    tokenizer.setDecoder(decoder);

    return new SentencePieceBPETokenizer(tokenizer);
  }

  /**
   * Train the model using the given files
   *
   * @param files Files to use for training
   * @param [options] Training options
   */
  async train(files: string[], options?: SentencePieceBPETrainOptions): Promise<void> {
    const mergedOptions = { ...this.defaultTrainOptions, ...options };
    const trainer = bpeTrainer(mergedOptions);

    this.tokenizer.train(trainer, files);
  }
}
