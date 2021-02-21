import { PaddingDirection } from "./enums";

/**
 * An Encoding as returned by the Tokenizer
 */
export interface RawEncoding {
  /**
   * Get the encoded tokens corresponding to the word at the given index in one of the input
   * sequences, with the form [startToken, endToken+1]
   * @param word The position of a word in one of the input sequences
   * @param seqId The index of the input sequence that contains said word
   * @since 0.7.0
   */
  wordToTokens(word: number, seqId?: number): [number, number] | undefined;

  /**
   * Get the offsets of the word at the given index in the input sequence
   * @param word The index of the word in the input sequence
   * @param seqId The index of the input sequence that contains said word
   * @since 0.7.0
   */
  wordToChars(word: number, seqId?: number): [number, number] | undefined;

  /**
   * Get the index of the sequence that contains the given token
   * @param token The index of the token in the encoded sequence
   */
  tokenToSequence(token: number): number | undefined;

  /**
   * Get the offsets of the token at the given index
   *
   * The returned offsets are related to the input sequence that contains the
   * token.  In order to determine in which input sequence it belongs, you
   * must call `tokenToSequence`.
   *
   * @param token The index of the token in the encoded sequence
   * @since 0.7.0
   */
  tokenToChars(token: number): [number, number] | undefined;

  /**
   * Get the word that contains the token at the given index
   *
   * The returned index is related to the input sequence that contains the
   * token.  In order to determine in which input sequence it belongs, you
   * must call `tokenToSequence`.
   *
   * @param token The index of the token  in the encoded sequence
   * @since 0.7.0
   */
  tokenToWord(token: number): number | undefined;

  /**
   * Find the index of the token at the position of the given char
   * @param pos The position of a char in one of the input strings
   * @param seqId The index of the input sequence that contains said char
   * @since 0.6.0
   */
  charToToken(pos: number, seqId?: number): number | undefined;

  /**
   * Get the word that contains the given char
   * @param pos The position of a char in the input string
   * @param seqId The index of the input sequence that contains said char
   * @since 0.7.0
   */
  charToWord(pos: number, seqId?: number): number | undefined;

  /**
   * Returns the attention mask
   */
  getAttentionMask(): number[];

  /**
   * Returns the number of sequences
   */
  getNSequences(): number;

  /**
   * Set the sequence id for this encoding
   */
  setSequenceId(seqId: number): undefined;

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
  getWordIds(): (number | undefined)[];

  /**
   * The sequences indices
   */
  getSequenceIds(): (number | undefined)[];

  /**
   * Pad the current Encoding at the given length
   *
   * @param length The length at which to pad
   * @param [options] Padding options
   */
  pad(length: number, options?: PaddingOptions): void;

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
