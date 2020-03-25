import { promisify } from "util";

import { wordPieceDecoder } from "../../bindings/decoders";
import { Model, WordPiece, WordPieceOptions } from "../../bindings/models";
import { bertNormalizer } from "../../bindings/normalizers";
import { bertProcessing } from "../../bindings/post-processors";
import { bertPreTokenizer } from "../../bindings/pre-tokenizers";
import { AddedToken, Tokenizer } from "../../bindings/tokenizer";
import { wordPieceTrainer } from "../../bindings/trainers";
import { BaseTokenizer } from "./base.tokenizer";

export interface BertWordPieceOptions {
  /**
   * @default true
   */
  addSpecialTokens?: boolean;
  /**
   * @default true
   */
  cleanText?: boolean;
  /**
   * @default "[CLS]"
   */
  clsToken?: string | AddedToken;
  /**
   * @default true
   */
  handleChineseChars?: boolean;
  /**
   * @default true
   */
  lowercase?: boolean;
  /**
   * @default "[MASK]"
   */
  maskToken?: string | AddedToken;
  /**
   * @default "[PAD]"
   */
  padToken?: string | AddedToken;
  /**
   * @default "[SEP]"
   */
  sepToken?: string | AddedToken;
  /**
   * @default true
   */
  stripAccents?: boolean;
  /**
   * @default "[UNK]"
   */
  unkToken?: string | AddedToken;
  vocabFile?: string;
  /**
   * The prefix to attach to subword units that don't represent a beginning of word
   * @default "##"
   */
  wordpiecesPrefix?: string;
}

export interface BertWordPieceTrainOptions {
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
   * @default ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
   */
  specialTokens?: (string | AddedToken)[];
  /**
   * @default 30000
   */
  vocabSize?: number;
  /**
   * The prefix to attach to subword units that don't represent a beginning of word
   * @default "##"
   */
  wordpiecesPrefix?: string;
}

type BertTokenizerConfig = Required<Omit<BertWordPieceOptions, "vocabFile">> & {
  vocabFile?: string;
};

/**
 * Bert WordPiece Tokenizer
 */
export class BertWordPieceTokenizer extends BaseTokenizer<BertTokenizerConfig> {
  private static readonly defaultBertOptions: BertTokenizerConfig = {
    addSpecialTokens: true,
    cleanText: true,
    clsToken: "[CLS]",
    handleChineseChars: true,
    lowercase: true,
    maskToken: "[MASK]",
    padToken: "[PAD]",
    sepToken: "[SEP]",
    stripAccents: true,
    unkToken: "[UNK]",
    wordpiecesPrefix: "##"
  };

  private readonly defaultTrainOptions: Required<BertWordPieceTrainOptions> = {
    initialAlphabet: [],
    limitAlphabet: 1000,
    minFrequency: 2,
    showProgress: true,
    specialTokens: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    vocabSize: 30000,
    wordpiecesPrefix: "##"
  };

  private constructor(tokenizer: Tokenizer, configuration: BertTokenizerConfig) {
    super(tokenizer, configuration);
  }

  /**
   * Instantiate and returns a new Bert WordPiece tokenizer
   * @param [options] Optional tokenizer options
   */
  static async fromOptions(
    options?: BertWordPieceOptions
  ): Promise<BertWordPieceTokenizer> {
    const opts = { ...this.defaultBertOptions, ...options };

    let model: Model;
    if (opts.vocabFile) {
      const fromFiles = promisify<string, WordPieceOptions, Model>(WordPiece.fromFiles);
      model = await fromFiles(opts.vocabFile, {
        unkToken: opts.unkToken,
        continuingSubwordPrefix: opts.wordpiecesPrefix
      });
    } else {
      model = WordPiece.empty();
    }

    const tokenizer = new Tokenizer(model);

    for (const token of [
      opts.clsToken,
      opts.sepToken,
      opts.unkToken,
      opts.padToken,
      opts.maskToken
    ]) {
      if (tokenizer.tokenToId(token) !== undefined) {
        tokenizer.addSpecialTokens([token]);
      }
    }

    const normalizer = bertNormalizer(opts);
    tokenizer.setNormalizer(normalizer);
    tokenizer.setPreTokenizer(bertPreTokenizer());

    if (opts.vocabFile && opts.addSpecialTokens) {
      const sepTokenId = tokenizer.tokenToId(opts.sepToken);
      if (sepTokenId === undefined) {
        throw new Error("sepToken not found in the vocabulary");
      }

      const clsTokenId = tokenizer.tokenToId(opts.clsToken);
      if (clsTokenId === undefined) {
        throw new Error("clsToken not found in the vocabulary");
      }

      const processor = bertProcessing(
        [opts.sepToken, sepTokenId],
        [opts.clsToken, clsTokenId]
      );
      tokenizer.setPostProcessor(processor);
    }

    const decoder = wordPieceDecoder(opts.wordpiecesPrefix);
    tokenizer.setDecoder(decoder);

    return new BertWordPieceTokenizer(tokenizer, opts);
  }

  /**
   * Train the model using the given files
   *
   * @param files Files to use for training
   * @param [options] Training options
   */
  async train(files: string[], options?: BertWordPieceTrainOptions): Promise<void> {
    const mergedOptions = { ...this.defaultTrainOptions, ...options };
    const trainer = wordPieceTrainer(mergedOptions);

    this.tokenizer.train(trainer, files);
  }
}
