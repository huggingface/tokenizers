import { promisify } from "util";
import { Encoding, Tokenizer } from "../bindings/tokenizer";

export class BaseTokenizer {
  constructor(protected tokenizer: Tokenizer) {}

  /**
   * Encode the given sequence
   *
   * @param {string} sequence The sequence to encode
   * @param {(string | null)} pair The optional pair sequence
   */
  async encode(sequence: string, pair?: string): Promise<Encoding> {
    const encode = promisify(this.tokenizer.encode.bind(this.tokenizer));
    return encode(sequence, pair ?? null);
  }

  /**
   * Encode the given sequences or pair of sequences
   *
   * @param {((string | [string, string])[])} sequences A list of sequences or pair of sequences.
   * The list can contain both at the same time.
   */
  async encodeBatch(sequences: (string | [string, string])[]): Promise<Encoding[]> {
    const encodeBatch = promisify(this.tokenizer.encodeBatch.bind(this.tokenizer));
    return encodeBatch(sequences);
  }
}
