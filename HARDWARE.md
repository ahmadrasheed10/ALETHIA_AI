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
