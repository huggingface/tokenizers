import { Decoder } from "./decoders";
import { Encoding } from "./encoding";
import { Model } from "./models";
import { Normalizer } from "./normalizers";
import { PostProcessor } from "./post-processors";
import { PreTokenizer } from "./pre-tokenizers";
import { Trainer } from "./trainers";

export interface TruncationOptions {
  /**
   * The length of the previous sequence to be included in the overflowing sequence
   * @default 0
   */
  stride?: number;
  /**
   * Strategy to use:
   * - `longest_first` Iteratively reduce the inputs sequence until the input is under max_length
   * starting from the longest one at each token (when there is a pair of input sequences).
   * - `only_first` Only truncate the first sequence.
   * - `only_second` Only truncate the second sequence.
   * @default "longest_first"
   */
  strategy?: "longest_first" | "only_first" | "only_second";
}

export interface PaddingOptions {
  /**
   * @default "right"
   */
  direction?: "left" | "right";
  /**
   * Padding length. If not provided:
   * - Will default to the longest sequence when encoding in batch.
   * - No padding will be applied when single encoding
   */
  maxLength?: number;
  /**
   * The index to be used when padding
   * @default 0
   */
  padId?: number;
  /**
   * The type index to be used when padding
   * @default 0
   */
  padTypeId?: number;
  /**
   * The pad token to be used when padding
   * @default "[PAD]"
   */
  padToken?: string;
}

/**
 * A Tokenizer works as a pipeline, it processes some raw text as input and outputs
 * an `Encoding`.
 * The various steps of the pipeline are:
 * 1. The `Normalizer`: in charge of normalizing the text. Common examples of
 *    normalization are the unicode normalization standards, such as NFD or NFKC.
 * 2. The `PreTokenizer`: in charge of creating initial words splits in the text.
 *    The most common way of splitting text is simply on whitespace.
 * 3. The `Model`: in charge of doing the actual tokenization. An example of a
 *    `Model` would be `BPE` or `WordPiece`.
 * 4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything
 *    relevant that, for example, a language model would need, such as special tokens.
 */
export class Tokenizer {
  /**
   * Instantiate a new Tokenizer using the given Model
   */
  constructor(model: Model);

  /**
   * Add the given tokens to the vocabulary
   *
   * @param tokens A list of tokens to add to the vocabulary.
   * Each token can either be a string, or a tuple with a string representing the token,
   * and a boolean option representing whether to match on single words only.
   * If the boolean is not included, it defaults to False
   * @returns The number of tokens that were added to the vocabulary
   */
  addTokens(tokens: (string | [string, boolean])[]): number;

  /**
   * Add the given special tokens to the vocabulary, and treat them as special tokens.
   * The special tokens will never be processed by the model, and will be removed while decoding.
   *
   * @param tokens The list of special tokens to add
   * @returns The number of tokens that were added to the vocabulary
   */
  addSpecialTokens(tokens: string[]): number;

  /**
   * Encode the given sequence
   *
   * @param sequence The sequence to encode
   * @param pair The optional pair sequence
   * @param __callback Callback called when encoding is complete
   */
  encode(
    sequence: string,
    pair: string | null,
    __callback: (err: any, encoding: Encoding) => void
  ): void;

  /**
   * Encode the given sequences or pair of sequences
   *
   * @param sequences A list of sequences or pair of sequences. The list can contain both at the same time.
   * @param __callback Callback called when encoding is complete
   */
  encodeBatch(
    sequences: (string | [string, string])[],
    __callback: (err: any, encodings: Encoding[]) => void
  ): void;

  /**
   * Decode the given list of ids to a string sequence
   *
   * @param ids A list of ids to be decoded
   * @param [skipSpecialTokens=true] Whether to remove all the special tokens from the output string
   * @returns The decoded string
   */
  decode(ids: number[], skipSpecialTokens?: boolean): string;

  /**
   * Decode the list of sequences to a list of string sequences
   *
   * @param sequences A list of sequence of ids to be decoded
   * @param [skipSpecialTokens] Whether to remove all the special tokens from the output strings
   * @returns A list of decoded strings
   */
  decodeBatch(sequences: number[][], skipSpecialTokens?: boolean): string[];

  /**
   * Convert the given token id to its corresponding string
   *
   * @param id The token id to convert
   * @returns The corresponding string if it exists
   */
  idToToken(id: number): string | undefined;

  /**
   * Convert the given token to its corresponding id
   *
   * @param token The token to convert
   * @returns The corresponding id if it exists
   */
  tokenToId(token: string): number | undefined;

  /**
   * Enable/change padding with specified options
   * @param [options] Padding options
   */
  setPadding(options?: PaddingOptions): void;

  /**
   * Disable padding
   */
  disablePadding(): void;

  /**
   * Enable/change truncation with specified options
   *
   * @param maxLength The maximum length at which to truncate
   * @param [options] Additional truncation options
   */
  setTruncation(maxLength: number, options?: TruncationOptions): void;

  /**
   * Disable truncation
   */
  disableTruncation(): void;

  /**
   * Train the model using the given files
   *
   * @param trainer Trainer to use
   * @param files List of files to use
   */
  train(trainer: Trainer, files: string[]): void;

  /**
   * Returns the size of the vocabulary
   *
   * @param [withAddedTokens=true] Whether to include the added tokens in the vocabulary's size
   */
  getVocabSize(withAddedTokens?: boolean): number;

  /**
   * Returns the number of encoding tasks running currently
   */
  runningTasks(): number;

  /**
   * Returns the model in use
   */
  getModel(): Model;

  /**
   * Change the model to use with this Tokenizer
   * @param model New model to use
   * @throws Will throw an error if any task is running
   * @throws Will throw an error if the model is already used in another Tokenizer
   */
  setModel(model: Model): void;

  /**
   * Returns the normalizer in use
   */
  getNormalizer(): Normalizer | undefined;

  /**
   * Change the normalizer to use with this Tokenizer
   * @param normalizer New normalizer to use
   * @throws Will throw an error if any task is running
   * @throws Will throw an error if the normalizer is already used in another Tokenizer
   */
  setNormalizer(normalizer: Normalizer): void;

  /**
   * Returns the pre-tokenizer in use
   */
  getPreTokenizer(): PreTokenizer | undefined;

  /**
   * Change the pre-tokenizer to use with this Tokenizer
   * @param preTokenizer New pre-tokenizer to use
   * @throws Will throw an error if any task is running
   * @throws Will throw an error if the pre-tokenizer is already used in another Tokenizer
   */
  setPreTokenizer(preTokenizer: PreTokenizer): void;

  /**
   * Returns the post-processor in use
   */
  getPostProcessor(): PostProcessor | undefined;

  /**
   * Change the post-processor to use with this Tokenizer
   * @param postProcessor New post-processor to use
   * @throws Will throw an error if any task is running
   * @throws Will throw an error if the post-processor is already used in another Tokenizer
   */
  setPostProcessor(processor: PostProcessor): void;

  /**
   * Returns the decoder in use
   */
  getDecoder(): Decoder | undefined;

  /**
   * Change the decoder to use with this Tokenizer
   * @param decoder New decoder to use
   * @throws Will throw an error if any task is running
   * @throws Will throw an error if the decoder is already used in another Tokenizer
   */
  setDecoder(decoder: Decoder): void;
}
