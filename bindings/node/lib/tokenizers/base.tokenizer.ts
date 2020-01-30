import { promisify } from "util";

import { Encoding } from "../bindings/encoding";
import { PaddingOptions, Tokenizer, TruncationOptions } from "../bindings/tokenizer";

export { Encoding, TruncationOptions };

export class BaseTokenizer {
  constructor(protected tokenizer: Tokenizer) {}

  /**
   * Encode the given sequence
   *
   * @param sequence The sequence to encode
   * @param pair The optional pair sequence
   */
  async encode(sequence: string, pair?: string): Promise<Encoding> {
    const encode = promisify(this.tokenizer.encode.bind(this.tokenizer));
    return encode(sequence, pair ?? null);
  }

  /**
   * Encode the given sequences or pair of sequences
   *
   * @param sequences A list of sequences or pair of sequences.
   * The list can contain both at the same time.
   */
  async encodeBatch(sequences: (string | [string, string])[]): Promise<Encoding[]> {
    const encodeBatch = promisify(this.tokenizer.encodeBatch.bind(this.tokenizer));
    return encodeBatch(sequences);
  }

  /**
   * Enable/change truncation with specified options
   *
   * @param maxLength The maximum length at which to truncate
   * @param options Additional truncation options
   */
  setTruncation(maxLength: number, options?: TruncationOptions): void {
    return this.tokenizer.setTruncation(maxLength, options);
  }

  /**
   * Disable truncation
   */
  disableTruncation(): void {
    return this.tokenizer.disableTruncation();
  }

  /**
   * Enable/change padding with specified options
   * @param [options] Padding options
   */
  setPadding(options?: PaddingOptions): void {
    return this.tokenizer.setPadding(options);
  }

  /**
   * Disable padding
   */
  disablePadding(): void {
    return this.tokenizer.disablePadding();
  }
}
