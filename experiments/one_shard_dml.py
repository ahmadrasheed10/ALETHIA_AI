"""Local DirectML experiment for the FineVideo test shard."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression

from run import NAMES, WEIGHTS, SecondWriter, Timings, batches, prefetch, process_shard, shard_paths


MODEL = Path(__file__).resolve().parents[1] / ".localdeps" / "yolo11l-dynamic.onnx"


def export_model(model):
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    backbone = model.model.eval().float().fuse(verbose=False)
    head = backbone.model[-1]
    head.export = True
    head.dynamic = True
    head.format = "onnx"
    sample = torch.zeros((1, 3, 640, 640))
    temporary = MODEL.with_suffix(".incomplete")
    with torch.inference_mode():
        backbone(sample)
        torch.onnx.export(
            backbone, sample, str(temporary), dynamo=False, opset_version=17,
            input_names=["images"], output_names=["detections"],
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"},
                          "detections": {0: "batch", 2: "anchors"}},
        )
    temporary.replace(MODEL)


class DirectMLDetector:
    def __init__(self, adapter, batch_size, static=False, half=False):
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML is unavailable")
        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        self.batch_size = batch_size
        model = YOLO(str(WEIGHTS))
        ids = {name: index for index, name in model.names.items()}
        self.ids = [ids[name] for name in NAMES]
        if not MODEL.exists():
            export_model(model)
        self.model_path = MODEL.with_name("yolo11l-dynamic-fp16.onnx") if half else MODEL
        if half and not self.model_path.exists():
            import onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16

            converted = convert_float_to_float16(onnx.load(MODEL), keep_io_types=True)
            temporary = self.model_path.with_suffix(".incomplete")
            onnx.save(converted, temporary)
            temporary.replace(self.model_path)
        self.adapter = adapter
        self.static = static
        self.shape = None
        self.session = None if static else self.load_session()
        self.letterbox = LetterBox((640, 640), auto=True, stride=32)
        self.threshold = float(np.nextafter(np.float32(0.4), np.float32(0)))

    def load_session(self, shape=None):
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        if shape is not None:
            batch, _, height, width = shape
            for dimension, size in (("batch", batch), ("height", height), ("width", width)):
                options.add_free_dimension_override_by_name(dimension, size)
        session = ort.InferenceSession(
            str(self.model_path), sess_options=options,
            providers=[("DmlExecutionProvider", {"device_id": self.adapter}), "CPUExecutionProvider"],
        )
        if session.get_providers()[0] != "DmlExecutionProvider":
            raise RuntimeError("ONNX Runtime failed to initialize DirectML")
        return session

    def predict(self, images):
        count = len(images)
        prepared = [self.letterbox(image=image) for image in images]
        if self.static:
            prepared.extend([prepared[-1]] * (self.batch_size - count))
        padded = np.stack(prepared)
        nchw = np.ascontiguousarray(padded[..., ::-1].transpose(0, 3, 1, 2), dtype=np.float32)
        nchw /= 255
        if self.static and self.shape != nchw.shape:
            self.session = None
            self.session = self.load_session(nchw.shape)
            self.shape = nchw.shape
        [raw] = self.session.run(["detections"], {"images": nchw})
        for boxes in non_max_suppression(torch.from_numpy(raw[:count]), self.threshold, 0.7):
            detections = boxes.numpy()
            classes = detections[detections[:, 4] >= np.float32(0.4), 5].astype(int)
            counts = np.bincount(classes, minlength=max(self.ids) + 1)
            yield [int(counts[index]) for index in self.ids]


def index_with_checkpoints(shard_path, output_path, detector):
    stat = shard_path.stat()
    configuration = (str(shard_path.resolve()), stat.st_size, stat.st_mtime_ns,
                     detector.adapter, detector.batch_size, detector.static, detector.model_path.name)
    fingerprint = hashlib.sha256(repr(configuration).encode())
    fingerprint.update(Path(__file__).read_bytes())
    with detector.model_path.open("rb") as model:
        fingerprint.update(hashlib.file_digest(model, "sha256").digest())
    parts = output_path.with_name(output_path.name + ".parts") / fingerprint.hexdigest()
    parts.mkdir(parents=True, exist_ok=True)
    frames_processed = 0
    reused_videos = 0
    with pq.ParquetFile(shard_path) as shard:
        count = shard.metadata.num_rows
        for index, row in enumerate(shard.iter_batches(batch_size=1, columns=["mp4"], use_threads=False)):
            part = parts / f"{index:05d}.jsonl"
            if part.is_file():
                reused_videos += 1
                continue
            begin = perf_counter()
            timings = Timings()
            temporary = part.with_suffix(".incomplete")
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                writer = SecondWriter(f"{shard_path.name}:{index}", output)
                error = None
                stream = prefetch(batches(row.column(0)[0].as_py(), detector.batch_size, timings))
                try:
                    for seconds, images, decode_error in stream:
                        if images:
                            for second, counts in zip(seconds, detector.predict(images), strict=True):
                                writer.add(second, counts)
                            timings.frames += len(images)
                        if decode_error:
                            error = decode_error
                finally:
                    stream.close()
                writer.finish(error)
            temporary.replace(part)
            frames_processed += timings.frames
            print(json.dumps({"video": index + 1, "total_videos": count, "frames": timings.frames,
                              "wall_seconds": perf_counter() - begin, "decode_error": error}), flush=True)
    pending = output_path.with_suffix(".incomplete")
    with pending.open("w", encoding="utf-8", newline="\n") as output:
        for index in range(count):
            with (parts / f"{index:05d}.jsonl").open(encoding="utf-8") as part:
                shutil.copyfileobj(part, output)
    pending.replace(output_path)
    return {"videos": count, "frames": frames_processed, "reused_videos": reused_videos}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = shard_paths(args.input)
    if len(paths) != 1:
        parser.error("The test expects exactly one Parquet shard.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    begin = perf_counter()
    detector = DirectMLDetector(args.adapter, args.batch_size, args.static, args.half)
    if args.resume:
        metrics = index_with_checkpoints(paths[0], args.output, detector)
    else:
        pending = args.output.with_suffix(".incomplete")
        with pending.open("w", encoding="utf-8", newline="\n") as output:
            timings = process_shard(paths[0], output, detector, baseline=False)
        pending.replace(args.output)
        metrics = vars(timings)
    print(json.dumps({"wall_seconds": perf_counter() - begin, **metrics}), flush=True)


if __name__ == "__main__":
    main()
