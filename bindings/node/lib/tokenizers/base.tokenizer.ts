import { promisify } from "util";

import { Encoding } from "../bindings/encoding";
import { PaddingOptions, Tokenizer, TruncationOptions } from "../bindings/tokenizer";


export class BaseTokenizer {
  constructor(protected tokenizer: Tokenizer) {}

  /**
   * Add the given tokens to the vocabulary
   *
   * @param tokens A list of tokens to add to the vocabulary.
   * Each token can either be a string, or a tuple with a string representing the token,
   * and a boolean option representing whether to match on single words only.
   * If the boolean is not included, it defaults to False
   * @returns The number of tokens that were added to the vocabulary
   */
  addTokens(tokens: (string | [string, boolean])[]): number {
    return this.tokenizer.addTokens(tokens);
  }

  /**
   * Add the given special tokens to the vocabulary, and treat them as special tokens.
   * The special tokens will never be processed by the model, and will be removed while decoding.
   *
   * @param tokens The list of special tokens to add
   * @returns The number of tokens that were added to the vocabulary
   */
  addSpecialTokens(tokens: string[]): number {
    return this.tokenizer.addSpecialTokens(tokens);
  }

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

  /**
   * Convert the given token id to its corresponding string
   *
   * @param id The token id to convert
   * @returns The corresponding string if it exists
   */
  idToToken(id: number): string | undefined {
    return this.tokenizer.idToToken(id);
  }

  /**
   * Convert the given token to its corresponding id
   *
   * @param token The token to convert
   * @returns The corresponding id if it exists
   */
  tokenToId(token: string): number | undefined {
    return this.tokenizer.tokenToId(token);
  }
}
