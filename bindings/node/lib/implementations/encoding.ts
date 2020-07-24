import { PaddingOptions, RawEncoding } from "../bindings/raw-encoding";
import { mergeEncodings } from "../bindings/utils";

export class Encoding {
  private _attentionMask?: number[];
  private _ids?: number[];
  private _length?: number;
  private _offsets?: [number, number][];
  private _overflowing?: Encoding[];
  private _specialTokensMask?: number[];
  private _tokens?: string[];
  private _typeIds?: number[];
  private _wordIndexes?: (number | undefined)[];

  constructor(private _rawEncoding: RawEncoding) {}

  /**
   * Merge a list of Encoding into one final Encoding
   * @param encodings The list of encodings to merge
   * @param [growingOffsets=false] Whether the offsets should accumulate while merging
   */
  static merge(encodings: Encoding[], growingOffsets?: boolean): Encoding {
    const mergedRaw = mergeEncodings(
      encodings.map((e) => e.rawEncoding),
      growingOffsets
    );

    return new Encoding(mergedRaw);
  }

  /**
   * Attention mask
   */
  get attentionMask(): number[] {
    if (this._attentionMask) {
      return this._attentionMask;
    }

    return (this._attentionMask = this._rawEncoding.getAttentionMask());
  }

  /**
   * Tokenized ids
   */
  get ids(): number[] {
    if (this._ids) {
      return this._ids;
    }

    return (this._ids = this._rawEncoding.getIds());
  }

  /**
   * Number of tokens
   */
  get length(): number {
    if (this._length !== undefined) {
      return this._length;
    }

    return (this._length = this._rawEncoding.getLength());
  }

  /**
   * Offsets
   */
  get offsets(): [number, number][] {
    if (this._offsets) {
      return this._offsets;
    }

    return (this._offsets = this._rawEncoding.getOffsets());
  }

  /**
   * Overflowing encodings, after truncation
   */
  get overflowing(): Encoding[] {
    if (this._overflowing) {
      return this._overflowing;
    }

    return (this._overflowing = this._rawEncoding
      .getOverflowing()
      .map((e) => new Encoding(e)));
  }

  /**
   * __⚠️ DANGER ZONE: do not touch unless you know what you're doing ⚠️__
   * Access to the `rawEncoding` returned by the internal Rust code.
   * @private
   * @ignore
   * @since 0.6.0
   */
  get rawEncoding(): Readonly<RawEncoding> {
    return this._rawEncoding;
  }

  /**
   * Special tokens mask
   */
  get specialTokensMask(): number[] {
    if (this._specialTokensMask) {
      return this._specialTokensMask;
    }

    return (this._specialTokensMask = this._rawEncoding.getSpecialTokensMask());
  }

  /**
   * Tokenized string
   */
  get tokens(): string[] {
    if (this._tokens) {
      return this._tokens;
    }

    return (this._tokens = this._rawEncoding.getTokens());
  }

  /**
   * Type ids
   */
  get typeIds(): number[] {
    if (this._typeIds) {
      return this._typeIds;
    }

    return (this._typeIds = this._rawEncoding.getTypeIds());
  }

  /**
   * The tokenized words indexes
   */
  get wordIndexes(): (number | undefined)[] {
    if (this._wordIndexes) {
      return this._wordIndexes;
    }

    return (this._wordIndexes = this._rawEncoding.getWords());
  }

  /**
   * Get the encoded tokens corresponding to the word at the given index in the input
   * sequence, with the form [startToken, endToken+1]
   * @param word The position of a word in the input sequence
   * @since 0.7.0
   */
  wordToTokens(word: number): [number, number] | undefined {
    return this._rawEncoding.wordToTokens(word);
  }

  /**
   * Get the offsets of the word at the given index in the input sequence
   * @param word The index of the word in the input sequence
   * @since 0.7.0
   */
  wordToChars(word: number): [number, number] | undefined {
    return this._rawEncoding.wordToChars(word);
  }

  /**
   * Get the offsets of the token at the given index
   * @param token The index of the token in the encoded sequence
   * @since 0.7.0
   */
  tokenToChars(token: number): [number, number] | undefined {
    return this._rawEncoding.tokenToChars(token);
  }

  /**
   * Get the word that contains the token at the given index
   * @param token The index of the token  in the encoded sequence
   * @since 0.7.0
   */
  tokenToWord(token: number): number | undefined {
    return this._rawEncoding.tokenToWord(token);
  }

  /**
   * Find the index of the token at the position of the given char
   * @param pos The position of a char in the input string
   */
  charToToken(pos: number): number | undefined {
    return this._rawEncoding.charToToken(pos);
  }

  /**
   * Pad the current Encoding at the given length
   *
   * @param length The length at which to pad
   * @param [options] Padding options
   */
  pad(length: number, options?: PaddingOptions): void {
    this._rawEncoding.pad(length, options);
    this.resetInternalProperties();
  }

  /**
   * Truncate the current Encoding at the given max length
   *
   * @param length The maximum length to be kept
   * @param [stride=0] The length of the previous first sequence
   * to be included in the overflowing sequence
   */
  truncate(length: number, stride?: number): void {
    this._rawEncoding.truncate(length, stride);
    this.resetInternalProperties();
  }

  private resetInternalProperties(): void {
    for (const prop of [
      "_attentionMask",
      "_ids",
      "_length",
      "_offsets",
      "_overflowing",
      "_specialTokensMask",
      "_tokens",
      "_typeIds",
      "_wordIndexes",
    ]) {
      delete this[prop as keyof this];
    }
  }
}
