# FineVideo object indexer

Index every decoded frame with the official YOLO11l model and write one JSONL row per video second. Inputs are the `mp4` bytes in each Parquet shard. Results report person, car, dog, laptop, and cell phone.

## Set up on Ubuntu 22.04

Use Python 3.11 and an NVIDIA GPU with a driver compatible with the CUDA 12.6 PyTorch wheels. The PyAV wheels include FFmpeg libraries.

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare.py
```

`prepare.py` downloads and loads `yolo11l.pt` using Ultralytics. Preparation requires internet access. The timed command reads the weights from the repository directory and runs offline. For a CPU smoke test on a machine without an NVIDIA GPU, the pinned CUDA build also supports `--device cpu`.

## Run

Point `--input` at the directory directly containing the Parquet files.

```bash
python run.py --input /path/to/parquet/files --output results.jsonl
```

The default uses same-shape batches, a bounded decode thread, and FP16 inference on CUDA. CPU inference uses FP32. Use `--fp32` to retain full precision while batching, `--batch-size 4` to reduce GPU memory use, or `--baseline` for single-frame FP32 inference with serial decode. Every mode processes all frames. The detector uses rectangular Ultralytics letterboxing at size 640, BGR inputs, a confidence floor of 0.4, and default class-aware NMS at IoU 0.7. The pipeline uses presentation timestamps relative to each video's first decoded frame.

## Run the submission dataset

Accept the dataset terms on Hugging Face and sign in with a read token. Keep the token in the terminal or a credential store. Put the dataset on a drive with at least 50 GB free.

```bash
python -m pip install huggingface_hub==0.36.2
hf auth login
hf download HuggingFaceFV/finevideo --repo-type dataset --include 'data/train-000[0-4]*' --local-dir finevideo
```

Run the two configurations on the same hardware and shards to compare their wall times and accuracy.

```bash
python run.py --input finevideo/data --output baseline.jsonl --baseline --resume 2> baseline.log
python run.py --input finevideo/data --output results.jsonl --resume 2> results.log
python verify.py compare baseline.jsonl results.jsonl
python verify.py package finevideo/data results.jsonl results.jsonl.gz
```

`--resume` stores each finished shard in a directory named after the output with a `.parts` suffix. If interrupted, rerun the same command with the same output path. Use separate output paths for simultaneous jobs. If you change the code, weights, inference settings, or shard files, the checkpoint keys change. The final JSONL is published after all shards finish.

The final log entry reports wall time, frames processed, and aggregate decode and prediction times. Decode and prediction can overlap, so their elapsed times can sum to more than wall time. Prediction timing includes Ultralytics preprocessing, inference, NMS, transfer, and per-second aggregation. A resumed invocation reports wall time and counters for that invocation. Preserve the logs from each attempt to record total elapsed processing time.

The comparison requires identical second coverage and error video IDs, then reports pooled object F1 and both people-count metrics. The packaging command checks the expected 50 shard names, every Parquet row ID, and the output schema before producing the gzip file.

## Run the local checks

```bash
python -m unittest -v
```

These checks exercise half-frame voting, rounded medians, empty seconds, variable timestamps, display rotation, and recovery from damaged video rows.

See the [Windows DirectML experiment](experiments/README.md) and the [completed-video checkpoint](partial_results/README.md) for the local FineVideo test.
