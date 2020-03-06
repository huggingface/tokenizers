/**
 * This class is not supposed to be instantiated directly. Instead, any implementation of
 * a Model will return a instance of this class when instantiated.
 */
interface Model {
  /**
   * Save the current model in the given folder, using the given name
   * for the various files that will get created.
   * Any file with the same name that already exist in this folder will be overwritten.
   *
   * @param folder Name of the destination folder
   * @param name Prefix to use in the name of created files
   */
  save(folder: string, name?: string): string[];
}

export interface BPEOptions {
  /**
   * The number of words that the BPE cache can contain. The cache allows
   * to speed-up the process by keeping the result of the merge operations
   * for a number of words.
   * @default 10_000
   */
  cacheCapacity?: number;
  /**
   * The BPE dropout to use. Must be an float between 0 and 1
   */
  dropout?: number;
  /**
   * The unknown token to be used by the model
   */
  unkToken?: string;
  /**
   * The prefix to attach to subword units that don't represent a beginning of word
   */
  continuingSubwordPrefix?: string;
  /**
   * The suffix to attach to subword units that represent an end of word
   */
  endOfWordSuffix?: string;
}

export namespace BPE {
  /**
   * Instantiate a BPE model from the given vocab and merges files
   *
   * @param vocab Path to a vocabulary JSON file
   * @param merges Path to a merge file
   * @param options BPE model options
   * @param __callback Callback called when model is loaded
   */
  export function fromFiles(
    vocab: string,
    merges: string,
    options: BPEOptions,
    __callback: (err: Error, encoding: Model) => void
  ): void;
  /**
   * Instantiate a BPE model from the given vocab and merges files
   *
   * @param vocab Path to a vocabulary JSON file
   * @param merges Path to a merge file
   * @param __callback Callback called when model is loaded
   */
  export function fromFiles(
    vocab: string,
    merges: string,
    __callback: (err: Error, encoding: Model) => void
  ): void;

  /**
   * Instantiate an empty BPE Model
   */
  export function empty(): Model;
}

export interface WordPieceOptions {
  /**
   * The prefix to attach to subword units that don't represent a beginning of word
   * @default "##"
   */
  continuingSubwordPrefix?: string;
  /**
   * The maximum number of characters to authorize in a single word.
   * @default 100
   */
  maxInputCharsPerWord?: number;
  /**
   * The unknown token to be used by the model.
   * @default "[UNK]"
   */
  unkToken?: string;
}

export namespace WordPiece {
  /**
   * Instantiate a WordPiece model from the given vocab file
   *
   * @param vocab Path to a vocabulary file
   * @param options WordPiece model options
   * @param __callback Callback called when model is loaded
   */
  export function fromFiles(
    vocab: string,
    options: WordPieceOptions,
    __callback: (err: Error, encoding: Model) => void
  ): void;
  /**
   * Instantiate a WordPiece model from the given vocab file
   *
   * @param vocab Path to a vocabulary file
   * @param __callback Callback called when model is loaded
   */
  export function fromFiles(
    vocab: string,
    __callback: (err: Error, encoding: Model) => void
  ): void;

  /**
   * Instantiate an empty WordPiece model
   */
  export function empty(): Model;
}
