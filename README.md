<!-- ============================================================ -->
<!--     RemoteCUDA — Capabilities & Usage Guide                -->
<!--     Professional Bilingual Documentation                    -->
<!-- ============================================================ -->

<div align="center">

# RemoteCUDA

### Remote GPU. Zero Code Changes.
### GPU راه دور. بدون تغییر حتی یک خط کد.

---

[![PyPI](https://img.shields.io/badge/pypi-v2.0.0-blue?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/remotecuda/)
[![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/remotecuda/remotecuda/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/remotecuda?style=flat-square&color=blue)](https://pypi.org/project/remotecuda/)

</div>

---

<br>

# Table of Contents | فهرست مطالب

| # | Section |
|:--|:--|
| 1 | [Distinctive Capabilities](#1-distinctive-capabilities) |
| 2 | [Comparative Advantage](#2-comparative-advantage) |
| 3 | [Architecture at a Glance](#3-architecture-at-a-glance) |
| 4 | [Usage — Step by Step](#4-usage--step-by-step) |
| 5 | [Performance Benchmarks](#5-performance-benchmarks) |
| 6 | [قابلیت‌های متمایز](#6-قابلیتهای-متمایز) |
| 7 | [مزیت رقابتی](#7-مزیت-رقابتی) |
| 8 | [نحوه استفاده — گام به گام](#8-نحوه-استفاده--گام-به-گام) |

---

<br>
<br>

# 1. Distinctive Capabilities

RemoteCUDA is not merely another remote GPU tool — it is a complete distributed execution framework that operates transparently beneath your existing PyTorch codebase.

## Capability Matrix

| Capability | Description | Impact |
|:--|:--|:--|
| **Zero-Change Integration** | No import rewrites, no decorators, no context managers required. Your `model.cuda()` calls work exactly as written. | Eliminates migration cost entirely. |
| **Network Auto-Discovery** | UDP multicast-based server detection. Clients locate GPU servers without IP addresses, configuration files, or environment variables. | Zero-configuration deployment. |
| **Multi-GPU Scheduling** | Four distinct scheduling strategies — Adaptive, Load-Balancing, Memory-Aware, Latency-Optimized — with automatic strategy switching based on workload profiling. | Optimal resource utilization without manual tuning. |
| **Local Dataset Retention** | Datasets remain on the client filesystem. Only the active mini-batch traverses the network. A 512MB LRU/LFU tensor cache eliminates redundant transfers. | 50GB dataset → zero upfront transfer. |
| **Asynchronous Stream Pipelining** | Data transfer and GPU computation overlap via CUDA stream semantics. While batch N computes on the GPU, batch N+1 is already in flight over the network. | Hides network latency behind computation. |
| **Automatic Failover** | GPU health monitored via heartbeat protocol. Failed tasks automatically reassigned to healthy GPUs with configurable retry logic (default: 3 attempts). | Graceful degradation under hardware failure. |
| **Heterogeneous GPU Support** | Mixed GPU architectures (RTX 4090 + A100 + RTX 3090) within a single pool. Scheduler accounts for compute capability, memory capacity, and current utilization. | Leverage all available hardware simultaneously. |
| **No CUDA on Client** | Client requires only PyTorch CPU build. No NVIDIA drivers, no CUDA toolkit, no GPU hardware. | Works on any laptop, including ultrabooks. |

## Scheduling Strategies
┌─────────────────────────────────────────────────────────────────────────────┐
│ GPU Scheduler — Strategy Selection │
├───────────────────┬─────────────────────────────────────────────────────────┤
│ Strategy │ Behavior │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Adaptive │ Profiles each task; routes memory-heavy ops to GPUs │
│ (default) │ with most free VRAM, latency-sensitive ops to GPUs │
│ │ with lowest historical response time. │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Load-Balancing │ Distributes tasks evenly across all available GPUs. │
│ │ Optimal for homogeneous clusters running uniform │
│ │ workloads (e.g., hyperparameter sweeps). │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Memory-Aware │ Routes each task to the GPU with the most free memory. │
│ │ Essential for large models (LLMs, diffusion models) │
│ │ where VRAM is the primary constraint. │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Latency-Optimized │ Routes to GPUs with lowest historical average │
│ │ forward-pass time. Ideal for real-time inference │
│ │ serving where tail latency matters. │
└───────────────────┴─────────────────────────────────────────────────────────┘


---

<br>
<br>

# 2. Comparative Advantage

RemoteCUDA occupies a unique position in the design space. Unlike SSH-based approaches (which require code relocation) and unlike simpler CUDA-forwarding tools (which lack intelligence), RemoteCUDA provides transparent interception combined with sophisticated resource management.

## Feature-by-Feature Comparison

| Feature | RemoteCUDA | SSH + SCP | VSCode Remote | Chidori GPU | Tensorlink |
|:--|:--:|:--:|:--:|:--:|:--:|
| Zero code changes | **Yes** | No | No | Yes | Yes |
| Local dataset access | **Yes** | No | No | Yes | Yes |
| Network auto-discovery | **Yes** | No | No | No | No |
| Single-command server setup | **Yes** | No | No | Yes | No |
| Multi-GPU parallelism | **Yes** | No | No | No | No |
| Intelligent load balancing | **Yes** | No | No | No | No |
| Automatic failover | **Yes** | No | No | No | No |
| Multiple scheduling strategies | **Yes (4)** | No | No | No | No |
| Tensor caching layer | **Yes** | No | No | No | No |
| Async stream pipelining | **Yes** | No | No | No | No |
| Heterogeneous GPU support | **Yes** | No | No | No | No |
| No CUDA on client | **Yes** | Yes | No | Yes | Yes |
| pip installable | **Yes** | N/A | N/A | Yes | No |
| Windows support | **Yes** | No | No | Yes | Yes |
| Open source (MIT) | **Yes** | N/A | No | Yes | Yes |
## Positioning



Intelligence / Automation
▲
│
│ ┌──────────────┐
│ │ RemoteCUDA │ ← Transparent + Smart
│ └──────────────┘
│
│ ┌──────────────┐
│ │ Chidori GPU │ ← Transparent only
│ └──────────────┘
│
│ ┌──────────────┐
│ │ VSCode │ ← Manual + Remote
│ │ Remote │
│ └──────────────┘
│
│ ┌──────────────┐
│ │ SSH + SCP │ ← Manual + Relocation
│ └──────────────┘
│
└──────────────────────────────►




RemoteCUDA is the only solution that combines **zero code impact** (transparent CUDA interception) with **intelligent resource orchestration** (scheduling, caching, failover, pipelining).

---

<br>
<br>

# 3. Architecture at a Glance
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLIENT MACHINE (No GPU) │
│ │
│ Your Code RemoteCUDA Stack │
│ ┌──────────┐ ┌─────────────────────────────────────────────────────┐ │
│ │model.cuda()──→│ AutoCUDAHook → GPUScheduler → GPUPool │ │
│ │tensor.to() │ │ (intercept) (strategy) (connections) │ │
│ │loss.back() │ └─────────────────────────────────────────────────────┘ │
│ └──────────┘ ┌─────────────────────────────────────────────────────┐ │
│ │ TensorCache │ StreamManager │ TensorBridge │ │
│ │ (512MB LRU) │ (async pipes) │ (lifecycle) │ │
│ └─────────────────────────────────────────────────────┘ │
│ │
└────────────────────────────────────┬──────────────────────────────────────────┘
│
│ TCP/IP (1–10 Gbps Ethernet)
│ Protocol: Binary, zlib/lz4 compressed
│ Discovery: UDP Multicast (239.255.100.100)
│
┌──────────────────────────┼──────────────────────────┐
│ │ │
▼ ▼ ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ GPU Server 1 │ │ GPU Server 2 │ │ GPU Server N │
│ │ │ │ │ │
│ ┌────────────┐ │ │ ┌────────────┐ │ │ ┌────────────┐ │
│ │ GPUWorker 0│ │ │ │ GPUWorker 0│ │ │ │ GPUWorker 0│ │
│ │ (RTX 4090) │ │ │ │ (RTX 3090) │ │ │ │ (A100) │ │
│ ├────────────┤ │ │ ├────────────┤ │ │ ├────────────┤ │
│ │ GPUWorker 1│ │ │ │ GPUWorker 1│ │ │ │ GPUWorker 1│ │
│ │ (RTX 4090) │ │ │ │ (RTX 3090) │ │ │ │ (A100) │ │
│ └────────────┘ │ │ └────────────┘ │ │ └────────────┘ │
│ │ │ │ │ │
│ Discovery: ON │ │ Discovery: ON │ │ Discovery: ON │
└──────────────────┘ └──────────────────┘ └──────────────────┘




**Key Design Decisions:**

- **One Worker per GPU:** Each physical GPU gets its own worker thread, memory manager, and CUDA stream. No cross-GPU contention.
- **Binary Protocol:** Custom 16-byte header with CRC32 integrity check. Payload compressed with zlib (level 3) by default; supports lz4 and zstd as optional accelerators.
- **UDP Discovery:** Servers announce presence via multicast; clients listen passively. No central registry. Servers can join or leave dynamically.

---

<br>
<br>

# 4. Usage — Step by Step

## 4.1 Server Setup (GPU Machine)

**Prerequisites:** Python 3.8+, PyTorch with CUDA, NVIDIA drivers.

```bash
# Install
pip install remotecuda

# Start the service
remotecuda start



 RemoteCUDA GPU Service
 GPU:        NVIDIA GeForce RTX 4090 (24.0 GB)
 Port:       55555
 Status:     Ready — Broadcasting on network
 Discovery:  ON (UDP multicast)

 GPU is now available on the network.
