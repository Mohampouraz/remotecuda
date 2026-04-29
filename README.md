# RemoteCUDA v3.0

<p align="center">
  <strong>Remote GPU Access — Zero Client Dependencies — Auto CPU Fallback</strong><br>
  <strong>دسترسی به GPU راه دور — صفر وابستگی در کلاینت — بازگشت خودکار به CPU</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/remotecuda/"><img src="https://img.shields.io/badge/pypi-v3.0.0-blue?style=flat-square&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/remotecuda/"><img src="https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/Mohampouraz/remotecuda/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://github.com/Mohampouraz/remotecuda/stargazers"><img src="https://img.shields.io/github/stars/Mohampouraz/remotecuda?style=flat-square&color=yellow" alt="Stars"></a>
</p>

---

## What is RemoteCUDA?

RemoteCUDA lets you **use a GPU from another computer on your network** with **zero dependencies on your machine**. The client uses only Python standard library — no NumPy, no PyTorch, no CUDA, no anything.

The server **auto-detects GPU availability** and falls back to CPU if no GPU is present.

**Same code. Two modes. No configuration.**

---

## The Problem

Every ML developer knows this:

Your Reality | The Pain
:--|:--
You code on a lightweight laptop | No built-in GPU
You have a powerful GPU machine | Sitting somewhere on the network
You want to use that GPU | SSH, SCP, VSCode Remote are cumbersome
Your datasets are local | Copying 50GB over network is painful
You just want to run `python train.py` | Without learning distributed systems

---




## Why RemoteCUDA?

RemoteCUDA is the **only solution** that combines **zero client dependencies** with **intelligent resource management**.

### Dependency Freedom

| Feature | RemoteCUDA v3 | Chidori GPU | Tensorlink | SSH+SCP | VSCode Remote |
|:--|:--:|:--:|:--:|:--:|:--:|
| Zero client dependencies | ✓ | needs PyTorch | needs CUDA | ✓ | ✗ |
| Pure Python client | ✓ | ✗ | ✗ | N/A | N/A |
| No CUDA on client | ✓ | ✓ | ✗ | ✓ | ✗ |
| No PyTorch on client | ✓ | ✗ | ✓ | ✓ | ✗ |
| No NumPy on client | ✓ | ✗ | ✓ | ✓ | ✗ |
| Works on any OS | ✓ | ✓ | ✓ | ✗ | ✗ |

### Server Intelligence

| Feature | RemoteCUDA v3 | Chidori GPU | Tensorlink | SSH+SCP | VSCode Remote |
|:--|:--:|:--:|:--:|:--:|:--:|
| Auto CPU fallback | ✓ | ✗ | ✗ | N/A | N/A |
| Auto-discovery | ✓ | ✗ | ✗ | ✗ | ✗ |
| One-command setup | ✓ | ✓ | ✗ | ✗ | ✗ |

### Advanced Capabilities

| Feature | RemoteCUDA v3 | Chidori GPU | Tensorlink | SSH+SCP | VSCode Remote |
|:--|:--:|:--:|:--:|:--:|:--:|
| Multi-GPU parallelism | ✓ | ✗ | ✗ | ✗ | ✗ |
| Load balancing | ✓ | ✗ | ✗ | ✗ | ✗ |
| Task scheduler | ✓ | ✗ | ✗ | ✗ | ✗ |
| Tensor caching | ✓ | ✗ | ✗ | ✗ | ✗ |
| Async streaming | ✓ | ✗ | ✗ | ✗ | ✗ |
| Auto failover | ✓ | ✗ | ✗ | ✗ | ✗ |

### Deployment & Protocol

| Feature | RemoteCUDA v3 | Chidori GPU | Tensorlink | SSH+SCP | VSCode Remote |
|:--|:--:|:--:|:--:|:--:|:--:|
| Open source | MIT | MIT | MIT | ✓ | ✗ |
| pip install | ✓ | ✓ | ✗ | N/A | N/A |
| JSON protocol (no pickle) | ✓ | uses pickle | N/A | N/A | N/A |

> **Key:** ✓ = Supported | ✗ = Not supported | N/A = Not applicable
