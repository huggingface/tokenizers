/* eslint-disable @typescript-eslint/no-explicit-any */

import { RawEncoding } from "../bindings/raw-encoding";
import { Encoding } from "./encoding";

describe("Encoding", () => {
  let encoding: Encoding;
  const rawEncodingMock = jest.fn<Partial<RawEncoding>, any>();

  describe("ids", () => {
    const getIdsMock = jest.fn(() => [3]);
    const m = rawEncodingMock.mockImplementation(() => ({
      getIds: getIdsMock,
    }));

    encoding = new Encoding(m() as RawEncoding);

    it("returns the ids from the raw encoding when not called before", () => {
      const ids = encoding.ids;

      expect(getIdsMock).toHaveBeenCalledTimes(1);
      expect(ids).toEqual([3]);
    });

    it("returns the ids without using the raw encoding when already called before", () => {
      getIdsMock.mockReset();
      const ids = encoding.ids;

      expect(getIdsMock).toHaveBeenCalledTimes(0);
      expect(ids).toEqual([3]);
    });
  });

  describe("pad", () => {
    it('reset internal "cache" properties', () => {
      const getIdsMock = jest.fn(() => [4]);
      const m = rawEncodingMock.mockImplementation(() => ({
        getIds: getIdsMock,
        pad: jest.fn(),
      }));

      encoding = new Encoding(m() as RawEncoding);
      encoding["_ids"] = [3];

      encoding.pad(10);
      const ids = encoding.ids;

      expect(getIdsMock).toHaveBeenCalledTimes(1);
      expect(ids).toEqual([4]);
    });
  });

  describe("truncate", () => {
    it('reset internal "cache" properties', () => {
      const getIdsMock = jest.fn(() => [4]);
      const m = rawEncodingMock.mockImplementation(() => ({
        getIds: getIdsMock,
        truncate: jest.fn(),
      }));

      encoding = new Encoding(m() as RawEncoding);
      encoding["_ids"] = [3];

      encoding.truncate(10);
      const ids = encoding.ids;

      expect(getIdsMock).toHaveBeenCalledTimes(1);
      expect(ids).toEqual([4]);
    });
  });
});
