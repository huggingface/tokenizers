/**
 * Base class for all models
 * This class is not supposed to be instantiated directly. Instead, any implementation of
 * a Model will return a instance of this class when instantiated.
 */
declare class Model {

}

export namespace models {
  export namespace BPE {
    export interface BPEOptions {
      /**
       * The number of words that the BPE cache can contain. The cache allows
       * to speed-up the process by keeping the result of the merge operations
       * for a number of words.
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

    /**
     * Instantiate a BPE model from the given vocab and merges files
     *
     * @param {string} vocab Path to a vocabulary JSON file
     * @param {string} merges Path to a merge file
     * @param {BPEOptions} [options] BPE model options
     * @param {(err: any, model: Model) => void} __callback Callback called when model is loaded
     */
    export function fromFiles(
      vocab: string,
      merges: string,
      options: BPEOptions | null,
      __callback: (err: any, model: Model) => void
    ): void;

    /**
     * Instantiate an empty BPE Model
     */
    export function empty(): Model;
  }

  export namespace WordPiece {
    /**
     * Instantiate a WordPiece model from the given vocab file
     *
     * @param {string} vocab Path to a vocabulary file
     * @param {string} [unkToken] The unknown token to be used by the model
     * @param {number} [maxInputCharsPerWord] The maximum number of characters to authorize in a single word
     * @param {(err: any, model: Model) => void} __callback Callback called when model is loaded
     */
    export function fromFiles(
      vocab: string,
      unkToken: string,
      maxInputCharsPerWord: number | null,
      __callback: (err: any, model: Model) => void
    ): void;

    /**
     * Instantiate an empty WordPiece model
     */
    export function empty(): Model;
  }
}
