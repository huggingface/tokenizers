import { Decoder } from "./decoders";
import { PaddingDirection, TruncationStrategy } from "./enums";
import { Model } from "./models";
import { Normalizer } from "./normalizers";
import { PostProcessor } from "./post-processors";
import { PreTokenizer } from "./pre-tokenizers";
import { RawEncoding } from "./raw-encoding";
import { Trainer } from "./trainers";

export interface TruncationOptions {
  /**
   * The length of the previous sequence to be included in the overflowing sequence
   * @default 0
   */
  stride?: number;
  /**
   * Strategy to use:
   * - `TruncationStrategy.LongestFirst` Iteratively reduce the inputs sequence until the input is under max_length
   * starting from the longest one at each token (when there is a pair of input sequences).
   * - `TruncationStrategy.OnlyFirst` Only truncate the first sequence.
   * - `TruncationStrategy.OnlySecond` Only truncate the second sequence.
   * @default TruncationStrategy.LongestFirst
   */
  strategy?: TruncationStrategy;
}

export interface TruncationConfiguration extends Required<TruncationOptions> {
  /**
   * The maximum length at which to truncate
   */
  maxLength: number;
}

export type PaddingConfiguration = Required<
  Omit<PaddingOptions, "maxLength" | "padToMultipleOf">
> &
  Pick<PaddingOptions, "maxLength" | "padToMultipleOf">;

export interface PaddingOptions {
  /**
   * @default PaddingDirection.Right
   */
  direction?: PaddingDirection;
  /**
   * Padding length. If not provided:
   * - Will default to the longest sequence when encoding in batch.
   * - No padding will be applied when single encoding
   */
  maxLength?: number;
  /**
   * If specified, the padding will snap to a multiple of the given value.
   * @default undefined
   */
  padToMultipleOf?: number;
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

export type TextInputSequence = string;
export type PreTokenizedInputSequence = string[];
export type InputSequence = TextInputSequence | PreTokenizedInputSequence;

export type TextEncodeInput = TextInputSequence | [TextInputSequence, TextInputSequence];
export type PreTokenizedEncodeInput =
  | PreTokenizedInputSequence
  | [PreTokenizedInputSequence, PreTokenizedInputSequence];
export type EncodeInput = TextEncodeInput | PreTokenizedEncodeInput;

export interface EncodeOptions {
  /**
   * Whether the given sequence is pre-tokenized
   * @default false
   */
  isPretokenized?: boolean;
  /**
   * Whether we should add special tokens
   * @default true
   */
  addSpecialTokens?: boolean;
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
   * Instantiate a new Tokenizer from the given file
   * @param path Path to a file containing a Tokenizer
   */
  static fromFile(path: string): Tokenizer;

  /**
   * Instantiate a new Tokenizer from the given JSON string
   * @param s A JSON string representation of the Tokenizer
   */
  static fromString(s: string): Tokenizer;

  /**
   * Add the given tokens to the vocabulary
   *
   * @param tokens A list of tokens to add to the vocabulary.
   * Each token can either be a string, or an instance of {@link AddedToken}.
   * @returns The number of tokens that were added to the vocabulary
   */
  addTokens(tokens: (string | AddedToken)[]): number;

  /**
   * Add the given special tokens to the vocabulary, and treat them as special tokens.
   * The special tokens will never be processed by the model, and will be removed while decoding.
   *
   * @param tokens The list of special tokens to add.
   * Each token can either be a string or an instance of {@link AddedToken}.
   * @returns The number of tokens that were added to the vocabulary
   */
  addSpecialTokens(tokens: (string | AddedToken)[]): number;

  /**
   * Encode the given sequence
   *
   * @param sequence The sequence to encode
   * @param pair The optional pair sequence
   * @param addSpecialTokens Whether to add the special tokens while encoding
   * @param __callback Callback called when encoding is complete
   */
  encode(
    sequence: InputSequence,
    pair?: InputSequence | null,
    options?: EncodeOptions | null, // |(err: Error, encoding: RawEncoding) => void,
    __callback?: (err: Error, encoding: RawEncoding) => void
  ): void;

  /**
   * Encode the given sequences or pair of sequences
   *
   * @param sequences A list of sequences or pair of sequences. The list can contain both at the same time.
   * @param addSpecialTokens Whether to add the special tokens while encoding
   * @param __callback Callback called when encoding is complete
   */
  encodeBatch(
    inputs: EncodeInput[],
    options?: EncodeOptions | null, // (err: Error, encodings: RawEncoding[]) => void,
    __callback?: (err: Error, encodings: RawEncoding[]) => void
  ): void;

  /**
   * Decode the given list of ids to a string sequence
   *
   * @param ids A list of ids to be decoded
   * @param skipSpecialTokens Whether to remove all the special tokens from the output string
   * @param __callback Callback called with decoded string
   */
  decode(
    ids: number[],
    skipSpecialTokens: boolean,
    __callback: (err: Error, encodings: string) => void
  ): void;

  /**
   * Decode the list of sequences to a list of string sequences
   *
   * @param sequences A list of sequence of ids to be decoded
   * @param skipSpecialTokens Whether to remove all the special tokens from the output strings
   * @param __callback Callback called with decoded strings
   */
  decodeBatch(
    sequences: number[][],
    skipSpecialTokens: boolean,
    __callback: (err: Error, encodings: string[]) => void
  ): void[];

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
  setPadding(options?: PaddingOptions): PaddingConfiguration;

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
  setTruncation(maxLength: number, options?: TruncationOptions): TruncationConfiguration;

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
   * Returns the vocabulary
   *
   * @param [withAddedTokens=true] Whether to include the added tokens in the vocabulary
   */
  getVocab(withAddedTokens?: boolean): { [token: string]: number };

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

  /**
   * Apply all the post-processing steps to the given encodings.
   * The various steps are:
   * 1. Truncate according to global params (@see setTruncation)
   * 2. Apply the PostProcessor
   * 3. Pad according to global params (@see setPadding)
   * @param encoding The main Encoding to post process
   * @param [pair] An optional pair Encoding
   * @param [addSpecialTokens=true] Whether to add special tokens. Default to `true`.
   * @since 0.6.0
   */
  postProcess(
    encoding: RawEncoding,
    pair?: RawEncoding,
    addSpecialTokens?: boolean
  ): RawEncoding;

  /**
   * Save the Tokenizer as JSON to the given path
   * @param path Path to the JSON file to write
   * @param [pretty=false] Whether the JSON string should be prettified
   */
  save(path: string, pretty?: boolean): void;

  /**
   * Get a serialized JSON version of the Tokenizer as a string
   * @param [pretty=false] Whether the JSON string should be prettified
   */
  toString(pretty?: boolean): string;
}

/**
 * Options used to construct an AddedToken
 * @since 0.6.0
 */
export interface AddedTokenOptions {
  /**
   * Whether this token should strip all potential whitespaces on the left side.
   * If True, this token will greedily match any whitespace on the left and then strip
   * them out.
   * @default False
   */
  leftStrip?: boolean;
  /**
   * Whether this token should strip all potential whitespaces on the right side.
   * If True, this token will greedily match any whitespace on the right and then strip
   * them out.
   * @default False
   */
  rightStrip?: boolean;
  /**
   * Whether this token should only match against single word.
   * If True, this token will never match inside of a word.
   * @default False
   */
  singleWord?: boolean;
  /**
   * Whether this token should match on the normalized version of the text. For example
   * with the added token `yesterday` and a normalizer in charge of lowercasing the text,
   * the input `I saw a lion Yesterday` would match the token.
   * This is False for special tokens by default, true otherwise
   * @default True
   */
  normalized?: boolean;
}

/**
 * AddedToken represents a token to be added to a Tokenizer.
 * An AddedToken can have special options defining the way it should behave.
 *
 * @since 0.6.0
 */
export class AddedToken {
  /**
   * Instantiate a new AddedToken
   * @param content The content of the token
   * @param special Whether this is a special token
   * @param [options] Options for the token
   */
  constructor(content: string, special: boolean, options?: AddedTokenOptions);

  /**
   * Get the content of the AddedToken
   */
  getContent(): string;
}
