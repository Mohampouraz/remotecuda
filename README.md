# RemoteCUDA v3.0

<div align="center">

### Remote GPU Access — Zero Client Dependencies — Auto CPU Fallback

[![PyPI](https://img.shields.io/badge/pypi-v3.0.0-blue?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/remotecuda/)
[![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What is RemoteCUDA?

RemoteCUDA enables transparent remote GPU access over a local network with **zero client-side dependencies**. The client uses only Python standard library — no NumPy, no PyTorch, no CUDA, no external packages whatsoever.

The server automatically detects GPU availability and falls back to CPU if no GPU is present. This means you can start computing immediately, with or without a GPU.

---

## Architecture

---

┌─────────────────────────────┐ ┌──────────────────────────────┐
│ CLIENT │ │ SERVER │
│ │ │ │
│ Python 3.8+ stdlib ONLY │ JSON │ PyTorch 1.10+ │
│ - socket (TCP) │◄────────────►│ NumPy 1.20+ │
│ - json │ over TCP │ │
│ - struct (binary packing) │ │ Auto-detection: │
│ - base64 (data encoding) │ │ CUDA available → GPU │
│ - threading │ │ CUDA unavailable → CPU │
│ │ │ │
│ ZERO dependencies │ │ Discovery: UDP Multicast │
│ No NumPy, No PyTorch │ │ Protocol: JSON + Binary │
│ No CUDA, No anything │ │ │
└─────────────────────────────┘ └──────────────────────────────┘



## Quick Start

### Server (GPU Machine)

```bash
pip install remotecuda[server]
remotecuda start



pip install remotecuda
python



import remotecuda

# Connect to server (auto-discover or specify IP)
remotecuda.init()

# Create tensors on remote device
a = remotecuda.zeros((1000, 1000))
b = remotecuda.ones((1000, 1000))

# Compute on remote device
c = remotecuda.matmul(a, b)

# Get results back
result = remotecuda.get(c)
print(f"Shape: {result['shape']}, First value: {result['data'][0]}")

# Cleanup
remotecuda.free(a)
remotecuda.free(b)
remotecuda.free(c)
remotecuda.shutdown()



Features
Feature	Description
Zero Client Dependencies	Pure Python stdlib — socket, json, struct, base64
Auto GPU/CPU Fallback	Server auto-detects CUDA, falls back to CPU
Network Auto-Discovery	UDP multicast, no IP configuration needed
JSON Protocol	Human-readable, secure, no pickle
Base64 Tensor Encoding	Compatible with any language
50+ Tensor Operations	Full math, activation, reduction, shape ops
Multi-Client Support	Thread pool on server, multiple clients simultaneously
Graceful Shutdown	Resource cleanup, tensor freeing
Comprehensive Error Handling	Clear error messages, proper exception types


Server Requirements
Requirement	Needed?
Python 3.8+	Yes
PyTorch 1.10+	Yes
NumPy 1.20+	Yes
NVIDIA Driver	Optional (auto-fallback)
CUDA Toolkit	No



Client Requirements
Requirement	Needed?
Python 3.8+	Yes
Anything else	No


Comparison
Feature	RemoteCUDA v3	SSH+SCP	VSCode Remote	Previous RemoteCUDA
Zero client deps	Yes	Yes	No	No
Auto CPU fallback	Yes	No	No	No
Pure Python client	Yes	N/A	N/A	No
No NumPy on client	Yes	N/A	N/A	No
No PyTorch on client	Yes	N/A	N/A	No
Auto-discovery	Yes	No	No	Yes
Multi-GPU	Yes	No	No	Yes



