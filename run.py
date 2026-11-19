"""Index FineVideo Parquet shards with the official YOLO11l detector."""

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from time import perf_counter

import av
import numpy as np
import pyarrow.parquet as pq


NAMES = ("person", "car", "dog", "laptop", "cell phone")
WEIGHTS = Path(__file__).resolve().with_name("yolo11l.pt")
MAX_BATCH_BYTES = 128 * 1024**2


class DecodeError(Exception):
    pass


@dataclass
class Timings:
    frames: int = 0
    videos: int = 0
    errors: int = 0
    decode_seconds: float = 0.0
    inference_seconds: float = 0.0


class SecondWriter:
    def __init__(self, video, output):
        self.video = video
        self.output = output
        self.second = 0
        self.votes = [0] * len(NAMES)
        self.people = []

    def write(self, record):
        self.output.write(json.dumps({"video": self.video, **record},
                                     separators=(",", ":")) + "\n")

    def finish_second(self):
        count = len(self.people)
        objects = [name for name, votes in zip(NAMES, self.votes)
                   if count and 2 * votes >= count]
        self.people.sort()
        people = 0
        if count:
            lower = self.people[(count - 1) // 2]
            upper = self.people[count // 2]
            people = (lower + upper) // 2
        self.write({"second": self.second, "objects": objects, "people": people})
        self.second += 1
        self.votes = [0] * len(NAMES)
        self.people.clear()

    def add(self, second, counts):
        while self.second < second:
            self.finish_second()
        for index, count in enumerate(counts):
            self.votes[index] += count > 0
        self.people.append(counts[0])

    def finish(self, error):
        if self.people:
            self.finish_second()
        if error:
            self.write({"error": error})


def rotate_bgr(image, degrees):
    degrees %= 360
    if not degrees or not math.isfinite(degrees):
        return image
    if degrees % 90 == 0:
        return np.rot90(image, int(degrees // 90)).copy()

    import cv2

    height, width = image.shape[:2]
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = math.ceil(width * cos + height * sin)
    new_height = math.ceil(width * sin + height * cos)
    matrix[0, 2] += (new_width - width) / 2
    matrix[1, 2] += (new_height - height) / 2
    return cv2.warpAffine(image, matrix, (new_width, new_height))


def decoded_frames(payload):
    if not payload:
        raise DecodeError("empty video")
    with av.open(io.BytesIO(payload)) as container:
        if not container.streams.video:
            raise DecodeError("no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        origin = previous = None
        rotation = None
        for frame in container.decode(stream):
            if frame.is_corrupt:
                raise DecodeError("corrupt decoded frame")
            if frame.pts is None or frame.time_base is None:
                raise DecodeError("frame has no presentation timestamp")
            timestamp = frame.pts * frame.time_base
            if origin is None:
                origin = timestamp
            if previous is not None and timestamp < previous:
                raise DecodeError("non-monotonic presentation timestamps")
            previous = timestamp
            elapsed = timestamp - origin

            # The display matrix can arrive on the first frame only.
            has_display_matrix = "DISPLAYMATRIX" in frame.side_data
            if rotation is None or has_display_matrix:
                rotation = frame.rotation
                if not has_display_matrix and not rotation:
                    try:
                        rotation = -float(stream.metadata.get("rotate", 0))
                    except ValueError:
                        rotation = 0
            yield elapsed // 1, rotate_bgr(frame.to_ndarray(format="bgr24"), rotation)

        if origin is None:
            raise DecodeError("no decoded frames")


def batches(payload, batch_size, timings):
    seconds, images = [], []
    size = 0
    error = None
    with closing(decoded_frames(payload)) as frames:
        while True:
            start = perf_counter()
            try:
                second, image = next(frames)
            except StopIteration:
                break
            except (av.FFmpegError, DecodeError) as exc:
                error = (str(exc).splitlines() or ["could not decode video"])[0][:160]
                break
            finally:
                timings.decode_seconds += perf_counter() - start

            if images and (image.shape != images[0].shape or
                           size + image.nbytes > MAX_BATCH_BYTES):
                yield seconds, images, None
                seconds, images, size = [], [], 0
            seconds.append(second)
            images.append(image)
            size += image.nbytes
            if len(images) == batch_size:
                yield seconds, images, None
                seconds, images, size = [], [], 0
    if images or error:
        yield seconds, images, error


def prefetch(source):
    """Keep one waiting batch and surface producer failures in the caller."""
    pending = Queue(maxsize=1)
    stop = Event()
    end = object()

    def put(item):
        while not stop.is_set():
            try:
                pending.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def produce():
        try:
            for item in source:
                if not put(item):
                    return
        except BaseException as exc:
            put(exc)
        finally:
            source.close()
            put(end)

    thread = Thread(target=produce, name="video-decode", daemon=True)
    thread.start()
    try:
        while (item := pending.get()) is not end:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join()


class Detector:
    def __init__(self, args):
        import cv2
        import torch
        from ultralytics import YOLO

        cv2.setNumThreads(1)
        self.torch = torch
        device = args.device or ("0" if torch.cuda.is_available() else "cpu")
        gpu = device != "cpu"
        self.device = torch.device(f"cuda:{device}" if gpu else "cpu")
        if gpu:
            if not torch.cuda.is_available() or int(device) >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {device} is unavailable")
            vram = torch.cuda.get_device_properties(self.device).total_memory
            default_batch_size = 16 if vram >= 8 * 1024**3 else 4
        else:
            torch.set_num_threads(min(8, os.cpu_count() or 1))
            default_batch_size = 2
        self.half = gpu and not args.fp32 and not args.baseline
        self.batch_size = 1 if args.baseline else args.batch_size or default_batch_size
        self.model = YOLO(str(WEIGHTS))
        by_name = {name: index for index, name in self.model.names.items()}
        self.ids = [by_name[name] for name in NAMES]

    def predict(self, images):
        results = self.model.predict(
            images, imgsz=640, rect=True, iou=0.7, agnostic_nms=False,
            # Ultralytics applies a strict greater-than comparison in NMS.
            conf=float(np.nextafter(np.float32(0.4), np.float32(0))),
            device=self.device, half=self.half, verbose=False,
        )
        torch = self.torch
        boxes = [result.boxes.data[:, 4:6] for result in results]
        all_boxes = torch.cat(boxes).float().cpu().numpy()
        start = 0
        for batch_boxes in boxes:
            end = start + len(batch_boxes)
            detections = all_boxes[start:end]
            classes = detections[detections[:, 0] >= np.float32(0.4), 1].astype(int)
            counts = np.bincount(classes, minlength=max(self.ids) + 1)
            yield [int(counts[index]) for index in self.ids]
            start = end


def process_shard(path, output, detector, baseline):
    timings = Timings()
    row = 0
    with pq.ParquetFile(path) as shard:
        if "mp4" not in shard.schema_arrow.names:
            raise ValueError(f"{path.name} has no mp4 column")
        for batch in shard.iter_batches(batch_size=1, columns=["mp4"], use_threads=False):
            writer = SecondWriter(f"{path.name}:{row}", output)
            row += 1
            timings.videos += 1
            source = batches(batch.column(0)[0].as_py(), detector.batch_size, timings)
            stream = source if baseline else prefetch(source)
            try:
                for seconds, images, error in stream:
                    if images:
                        begin = perf_counter()
                        for second, counts in zip(seconds, detector.predict(images), strict=True):
                            writer.add(second, counts)
                        timings.inference_seconds += perf_counter() - begin
                        timings.frames += len(images)
                    if error:
                        timings.errors += 1
                        print(f"{writer.video}: {error}", file=sys.stderr)
                        writer.finish(error)
            finally:
                stream.close()
            writer.finish(None)
    return timings


def shard_paths(folder):
    return sorted(folder.glob("*.parquet")) if folder.is_dir() else []


def run(args, shards):
    start = perf_counter()
    detector = Detector(args)
    print(f"device={detector.device} batch={detector.batch_size} "
          f"precision={'FP16' if detector.half else 'FP32'}", file=sys.stderr, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.resume:
        checkpoint = args.output.with_name(args.output.name + ".parts")
        checkpoint.mkdir(exist_ok=True)
        fingerprint = hashlib.sha256(Path(__file__).read_bytes())
        with WEIGHTS.open("rb") as weights:
            fingerprint.update(hashlib.file_digest(weights, "sha256").digest())
        fingerprint.update(repr((detector.device, detector.half,
                                 detector.batch_size, args.baseline)).encode())

    totals = Timings()
    reused = 0
    temp = args.output.with_name(args.output.name + ".incomplete")
    with temp.open("w", encoding="utf-8", newline="\n") as final:
        for index, shard in enumerate(shards, 1):
            print(f"[{index}/{len(shards)}] {shard.name}", file=sys.stderr, flush=True)
            if checkpoint is None:
                timings = process_shard(shard, final, detector, args.baseline)
            else:
                stat = shard.stat()
                key = repr((str(shard.resolve()), stat.st_size, stat.st_mtime_ns))
                shard_hash = fingerprint.copy()
                shard_hash.update(key.encode())
                part = checkpoint / (shard_hash.hexdigest() + ".jsonl")
                if part.exists():
                    reused += 1
                    timings = Timings()
                else:
                    pending = part.with_suffix(".incomplete")
                    with pending.open("w", encoding="utf-8", newline="\n") as writer:
                        timings = process_shard(shard, writer, detector, args.baseline)
                    pending.replace(part)
                with part.open(encoding="utf-8") as saved:
                    shutil.copyfileobj(saved, final)
            for name, value in vars(timings).items():
                setattr(totals, name, getattr(totals, name) + value)
            print(json.dumps({"shard": shard.name, **vars(timings)}),
                  file=sys.stderr, flush=True)
    temp.replace(args.output)
    print(json.dumps({"wall_seconds": perf_counter() - start, "reused_shards": reused,
                      **vars(totals)}), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="directory of Parquet shards")
    parser.add_argument("--output", type=Path, required=True, help="output JSONL path")
    parser.add_argument("--device", help="cpu or a CUDA device index (default: automatic)")
    parser.add_argument("--batch-size", type=int, help="maximum frames per model call")
    parser.add_argument("--fp32", action="store_true", help="use full precision on CUDA")
    parser.add_argument("--baseline", action="store_true", help="single-frame FP32, no prefetch")
    parser.add_argument("--resume", action="store_true", help="reuse completed shard outputs")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device and args.device != "cpu" and not (
            args.device.isascii() and args.device.isdecimal()):
        parser.error("--device must be 'cpu' or a nonnegative CUDA device index")
    shards = shard_paths(args.input)
    if not shards:
        parser.error("input directory contains no .parquet files")
    if args.output.resolve() in {path.resolve() for path in shards}:
        parser.error("--output must not overwrite an input shard")
    if not WEIGHTS.is_file():
        parser.error("missing yolo11l.pt, run python prepare.py first")
    run(args, shards)


if __name__ == "__main__":
    main()
