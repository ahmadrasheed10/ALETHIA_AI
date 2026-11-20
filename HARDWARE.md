# Full-run hardware

The full 50-shard run is pending. Record these fields from the machine that completes it.

| Field | Full-run value |
| --- | --- |
| Compute environment | Pending |
| GPU and VRAM | Pending |
| CPU and physical core count | Pending |
| RAM | Pending |
| Disk type (SSD or HDD) | Pending |
| Operating system | Pending |
| Full-run wall time | Pending |

## Local verification workstation

The CPU smoke test ran on a Windows 11 Home laptop with an AMD Ryzen 5 7535HS (6 cores and 12 threads), 15.2 GiB of usable RAM, a KIOXIA NVMe SSD, and an NVIDIA GeForce RTX 2050 with 4 GiB of VRAM. The pinned environment passed the regression tests. Windows Application Control blocked the CUDA-enabled PyTorch import.

The local DirectML experiment used the RTX 2050. Across 96 frames from three FineVideo videos, FP16 with a batch size of 8 matched the CPU counts on every frame. The measured inference rates were 32.0 frames per second on DirectML and 2.58 frames per second on the CPU. The shard run stopped after its first completed video. See [the partial result](partial_results/README.md).
