import io
import json
import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from run import DecodeError, SecondWriter, Timings, batches, decoded_frames, process_shard


def encoded_clip(rotation=0, first_pts=10):
    target = io.BytesIO()
    with av.open(target, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.set_display_rotation(rotation)
        for tick in (0, 1, 4, 12):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:24, :32, 2] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            frame.pts = tick + first_pts
            frame.time_base = Fraction(1, 4)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return target.getvalue()


class FixedDetector:
    batch_size = 3

    def predict(self, images):
        for _ in images:
            yield [1, 0, 0, 0, 0]


class IndexTests(unittest.TestCase):
    def test_ties_rounded_median_and_empty_second(self):
        output = io.StringIO()
        writer = SecondWriter("a.parquet:0", output)
        writer.add(0, [0, 3, 0, 0, 0])
        writer.add(0, [3, 0, 0, 0, 0])
        writer.add(2, [1, 0, 0, 2, 0])
        writer.finish("decode failed")
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        expected = [
            {"second": 0, "objects": ["person", "car"], "people": 1},
            {"second": 1, "objects": [], "people": 0},
            {"second": 2, "objects": ["person", "laptop"], "people": 1},
            {"error": "decode failed"},
        ]
        self.assertEqual(rows, [{"video": "a.parquet:0", **row} for row in expected])

    def test_variable_timestamps_and_display_rotation(self):
        frames = list(decoded_frames(encoded_clip(90)))
        self.assertEqual([second for second, _ in frames], [0, 0, 1, 3])
        self.assertTrue(all(image.shape == (64, 48, 3) for _, image in frames))
        self.assertGreater(frames[0][1][48, 12, 2], 200)
        self.assertLess(frames[0][1][16, 12, 2], 20)

    def test_partial_batch_before_decode_failure(self):
        def broken(_):
            yield 0, np.zeros((8, 8, 3), np.uint8)
            raise DecodeError("truncated video")

        with patch("run.decoded_frames", broken):
            [(seconds, images, error)] = list(batches(b"test", 16, Timings()))
        self.assertEqual((seconds, len(images), error), ([0], 1, "truncated video"))

    def test_bad_row_between_good_rows(self):
        with TemporaryDirectory() as folder:
            shard = Path(folder) / "test.parquet"
            media = encoded_clip()
            pq.write_table(pa.table({"mp4": [media, b"broken", media]}), shard)
            output = io.StringIO()
            timing = process_shard(shard, output, FixedDetector(), baseline=False)

        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual((timing.videos, timing.frames, timing.errors), (3, 8, 1))
        self.assertEqual([r["video"] for r in rows],
                         ["test.parquet:0"] * 4 + ["test.parquet:1"] + ["test.parquet:2"] * 4)
        self.assertEqual([r.get("people") for r in rows], [1, 1, 0, 1, None, 1, 1, 0, 1])
        self.assertIn("error", rows[4])


if __name__ == "__main__":
    unittest.main()
