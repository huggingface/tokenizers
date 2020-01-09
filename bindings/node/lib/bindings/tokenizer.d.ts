import { Model } from "./models";

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
   * Encode the given sequence
   *
   * @param {string} sequence The sequence to encode
   * @param {(string | null)} pair The optional pair sequence
   * @param {(err: any, encoding: Encoding) => void} __callback Callback called when encoding is complete
   */
  encode(sequence: string, pair: string | null, __callback: (err: any, encoding: Encoding) => void): void;

  /**
   * Encode the given sequences or pair of sequences
   *
   * @param {((string | [string, string])[])} sequences A list of sequences or pair of sequences.
   * The list can contain both at the same time.
   * @param {(err: any, encodings: Encoding[]) => void} __callback Callback called when encoding is complete
   */
  encodeBatch(sequences: (string | [string, string])[], __callback: (err: any, encodings: Encoding[]) => void): void;

  /**
   * Returns the size of the vocabulary
   *
   * @param {boolean} [withAddedTokens=true] Whether to include the added tokens in the vocabulary's size
   */
  getVocabSize(withAddedTokens?: boolean): number;

  /**
   * Returns the number of encoding tasks running currently
   */
  runningTasks(): number;

  /**
   * Change the model to use with this Tokenizer
   * @param model New model to use
   * @throws Will throw an error if any task is running
   */
  setModel(model: Model): void;
}

/**
 * An Encoding as returned by the Tokenizer
 */
declare class Encoding {
  /**
   * Returns the attention mask
   */
  getAttentionMask(): number[];

  /**
   * Returns the tokenized ids
   */
  getIds(): number[];

  /**
   * Returns the offsets
   */
  getOffsets(): [number, number][];

  /**
   * Returns the overflowing encoding, after truncation
   */
  getOverflowing(): Encoding | undefined;

  /**
   * Returns the special tokens mask
   */
  getSpecialTokensMask(): number;

  /**
   * Returns the tokenized string
   */
  getTokens(): string[];

  /**
   * Returns the type ids
   */
  getTypeIds(): number[];

  /**
   * Pad the current Encoding at the given length
   *
   * @param {number} length The length at which to pad
   * @param {PaddingOptions} [options] Padding options
   */
  pad(length: number, options?: PaddingOptions): void;

  /**
   * Truncate the current Encoding at the given max_length
   *
   * @param {number} length The maximum length to be kept
   * @param {number} [stride=0] The length of the previous first sequence
   * to be includedin the overflowing sequence
   */
  truncate(length: number, stride?: number): void;
}

interface PaddingOptions {
  /**
   * @default "right"
   */
  direction?: 'left' | 'right';
  /**
   * The indice to be used when padding
   * @default 0
   */
  padId?: number;
  /**
   * The type indice to be used when padding
   * @default 0
   */
  padTypeId?: number;
  /**
   * The pad token to be used when padding
   * @default "[PAD]"
   */
  padToken?: string;
}
