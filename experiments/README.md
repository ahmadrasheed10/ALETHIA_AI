# Windows one-shard experiment

Run these modules from the repository root. The range downloader uses your saved Hugging Face login, verifies the shard's SHA-256, and saves progress by byte range.

```powershell
python -m experiments.fetch_test_shard
```

The experiment used Python 3.11, CPU PyTorch 2.13.0, ONNX 1.19.1, ONNX Runtime DirectML 1.24.4, and ml-dtypes 0.5.3, with the YOLO11l weights and the decoding libraries used by the main pipeline. Install the DirectML runtime in the experiment environment. The adapter index depends on how Windows enumerates the GPUs. Adapter 1 selected the RTX 2050 on the test laptop.

```powershell
python -m experiments.benchmark_test_shard finevideo/one/data/train-00000-of-01357.parquet --adapter 1 --static --batch-size 8 --half
python -m experiments.one_shard_dml --input finevideo/one/data --output .localdeps/one-shard.jsonl --adapter 1 --static --batch-size 8 --half --resume
```

The benchmark compares per-frame counts across 96 frames from three videos. The checkpointed command writes one file per completed video and assembles the JSONL after the whole shard finishes.
