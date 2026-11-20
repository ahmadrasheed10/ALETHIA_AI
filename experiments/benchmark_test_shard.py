"""Time model inference on frames from the first, middle, and last videos."""

import argparse
import json
from contextlib import closing
from io import BytesIO
from itertools import islice
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import av
import pyarrow.parquet as pq

from run import DecodeError, Detector, decoded_frames


def load_samples(path):
    samples = []
    frames_estimate = 0
    estimated_video_count = 0
    with pq.ParquetFile(path) as shard:
        video_count = shard.metadata.num_rows
        selected = {0, video_count // 2, video_count - 1}
        for index, record in enumerate(shard.iter_batches(batch_size=1, columns=["mp4"])):
            payload = record.column(0)[0].as_py()
            try:
                with av.open(BytesIO(payload)) as video:
                    stream = video.streams.video[0]
                    if stream.frames:
                        frames_estimate += stream.frames
                        estimated_video_count += 1
                    elif video.duration and stream.average_rate:
                        frames_estimate += round(video.duration * float(stream.average_rate) / av.time_base)
                        estimated_video_count += 1
                if index in selected:
                    with closing(decoded_frames(payload)) as frames:
                        sample = [image for _, image in islice(frames, 32)]
                    if sample:
                        samples.append(sample)
            except (av.FFmpegError, DecodeError, IndexError, TypeError):
                continue
    if not samples:
        raise RuntimeError("No frames decoded for the benchmark")
    return samples, frames_estimate, estimated_video_count, video_count


def predict(detector, samples):
    next(detector.predict(samples[0][:1]))
    total = 0.0
    results = []
    for frames in samples:
        begin = perf_counter()
        for start in range(0, len(frames), detector.batch_size):
            batch = frames[start:start + detector.batch_size]
            if any(image.shape != batch[0].shape for image in batch):
                for image in batch:
                    results.extend(detector.predict([image]))
            else:
                results.extend(detector.predict(batch))
        total += perf_counter() - begin
    return results, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--adapter", type=int)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    samples, frame_estimate, estimates, video_count = load_samples(args.shard)
    cpu = Detector(SimpleNamespace(device="cpu", baseline=False, fp32=True, batch_size=2))
    reference, seconds = predict(cpu, samples)
    print(json.dumps({"backend": "cpu", "sample_videos": len(samples), "sample_frames": len(reference),
                      "sample_seconds": seconds, "frames_per_second": len(reference) / seconds,
                      "shard_frames_estimate": frame_estimate, "videos_with_estimate": estimates,
                      "shard_videos": video_count}), flush=True)
    if args.adapter is not None:
        from .one_shard_dml import DirectMLDetector

        detector = DirectMLDetector(args.adapter, args.batch_size, args.static, args.half)
        accelerated, elapsed = predict(detector, samples)
        mismatches = sum(a != b for a, b in zip(accelerated, reference, strict=True))
        print(json.dumps({"backend": "directml", "sample_frames": len(accelerated),
                          "sample_seconds": elapsed, "frames_per_second": len(accelerated) / elapsed,
                          "frames_with_different_counts": mismatches}), flush=True)


if __name__ == "__main__":
    main()
