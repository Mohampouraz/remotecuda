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
