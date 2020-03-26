import { PaddingDirection } from "./enums";

/**
 * An Encoding as returned by the Tokenizer
 */
export interface RawEncoding {
  /**
   * Find the index of the token at the position of the given char
   * @param pos The position of a char in the input string
   * @since 0.6.0
   */
  charToToken(pos: number): number | undefined;

  /**
   * Find the offsets of the token that contains the character at the specified position
   * @param pos The position of a char in the input string
   * @since 0.6.0
   */
  charToTokenOffsets(pos: number): [number, number] | undefined;

  /**
   * Find the offsets of the word that contains the character at the specified position
   * @param pos The position of a char in the input string
   * @since 0.6.0
   */
  charToWordOffsets(pos: number): [number, number] | undefined;

  /**
   * Returns the attention mask
   */
  getAttentionMask(): number[];

  /**
   * Returns the tokenized ids
   */
  getIds(): number[];

  /**
   * Returns the number of tokens
   */
  getLength(): number;

  /**
   * Returns the offsets
   */
  getOffsets(): [number, number][];

  /**
   * Returns the overflowing encodings, after truncation
   */
  getOverflowing(): RawEncoding[];

  /**
   * Returns the special tokens mask
   */
  getSpecialTokensMask(): number[];

  /**
   * Returns the tokenized string
   */
  getTokens(): string[];

  /**
   * Returns the type ids
   */
  getTypeIds(): number[];

  /**
   * The tokenized words indexes
   * @since 0.6.0
   */
  getWords(): number[];

  /**
   * Pad the current Encoding at the given length
   *
   * @param length The length at which to pad
   * @param [options] Padding options
   */
  pad(length: number, options?: PaddingOptions): void;

  /**
   * Find the offsets of the word that contains the token at the given index
   * @param index The index of a token
   * @since 0.6.0
   */
  tokenToWordOffsets(index: number): [number, number] | undefined;

  /**
   * Truncate the current Encoding at the given max_length
   *
   * @param length The maximum length to be kept
   * @param [stride=0] The length of the previous first sequence
   * to be included in the overflowing sequence
   */
  truncate(length: number, stride?: number): void;
}

interface PaddingOptions {
  /**
   * @default "right"
   */
  direction?: PaddingDirection;
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
