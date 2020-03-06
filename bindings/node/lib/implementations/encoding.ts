import { PaddingOptions, RawEncoding } from "../bindings/raw-encoding";

export class Encoding {
  private _attentionMask?: number[];
  private _ids?: number[];
  private _length?: number;
  private _offsets?: [number, number][];
  private _originalString?: string;
  private _overflowing?: Encoding[];
  private _specialTokensMask?: number[];
  private _tokens?: string[];
  private _typeIds?: number[];

  constructor(private rawEncoding: RawEncoding) {}

  /**
   * Attention mask
   */
  get attentionMask(): number[] {
    if (this._attentionMask) {
      return this._attentionMask;
    }

    return (this._attentionMask = this.rawEncoding.getAttentionMask());
  }

  /**
   * Tokenized ids
   */
  get ids(): number[] {
    if (this._ids) {
      return this._ids;
    }

    return (this._ids = this.rawEncoding.getIds());
  }

  /**
   * Number of tokens
   */
  get length(): number {
    if (this._length !== undefined) {
      return this._length;
    }

    return (this._length = this.rawEncoding.getLength());
  }

  /**
   * Offsets
   */
  get offsets(): [number, number][] {
    if (this._offsets) {
      return this._offsets;
    }

    return (this._offsets = this.rawEncoding.getOffsets());
  }

  /**
   * Overflowing encodings, after truncation
   */
  get overflowing(): Encoding[] {
    if (this._overflowing) {
      return this._overflowing;
    }

    return (this._overflowing = this.rawEncoding
      .getOverflowing()
      .map(e => new Encoding(e)));
  }

  /**
   * Special tokens mask
   */
  get specialTokensMask(): number[] {
    if (this._specialTokensMask) {
      return this._specialTokensMask;
    }

    return (this._specialTokensMask = this.rawEncoding.getSpecialTokensMask());
  }

  /**
   * Tokenized string
   */
  get tokens(): string[] {
    if (this._tokens) {
      return this._tokens;
    }

    return (this._tokens = this.rawEncoding.getTokens());
  }

  /**
   * Type ids
   */
  get typeIds(): number[] {
    if (this._typeIds) {
      return this._typeIds;
    }

    return (this._typeIds = this.rawEncoding.getTypeIds());
  }

  /**
   * Returns the original string
   *
   * @param [begin] The index from which to start (can be negative).
   * @param [end] The index (excluded) to which to stop (can be negative).
   * Stopping at the end of the string if not provided.
   * @returns The full original string if no parameter is provided,
   * otherwise the original string between `begin` and `end`
   */
  getOriginalString(begin?: number, end?: number): string {
    if (begin === undefined && end === undefined) {
      if (this._originalString !== undefined) {
        return this._originalString;
      } else {
        return (this._originalString = this.rawEncoding.getOriginalString());
      }
    }

    return this.rawEncoding.getOriginalString(begin, end);
  }

  /**
   * Pad the current Encoding at the given length
   *
   * @param length The length at which to pad
   * @param [options] Padding options
   */
  pad(length: number, options?: PaddingOptions): void {
    this.rawEncoding.pad(length, options);
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
    this.rawEncoding.truncate(length, stride);
    this.resetInternalProperties();
  }

  private resetInternalProperties(): void {
    for (const prop of [
      "_attentionMask",
      "_ids",
      "_length",
      "_offsets",
      "_originalString",
      "_overflowing",
      "_specialTokensMask",
      "_tokens",
      "_typeIds"
    ]) {
      delete this[prop as keyof this];
    }
  }
}
