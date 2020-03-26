import { RawEncoding } from "./raw-encoding";

/**
 * Returns a subpart of a string according to specified indexes, and respecting unicode characters
 *
 * @param text The text for which to return a subpart
 * @param [begin] The index from which to start (can be negative).
 * @param [end] The index (excluded) to which to stop (can be negative).
 * Stopping at the end of the string if not provided.
 * @returns The full string if no start/end indexes are provided,
 * otherwise the original string between `begin` (included) and `end` (excluded)
 * @since 0.6.0
 */
export function slice(text: string, start?: number, end?: number): string;

/**
 * Merge the list of RawEncoding into one final RawEncoding
 * @param encodings The list of encodings to merge
 * @param [growingOffsets=false] Whether the offsets should accumulate while merging
 */
export function mergeEncodings(
  encodings: RawEncoding[],
  growingOffsets?: boolean
): RawEncoding;
