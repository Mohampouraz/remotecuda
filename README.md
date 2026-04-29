<p align="center">
  <img src="https://raw.githubusercontent.com/remotecuda/remotecuda/main/assets/logo.svg" alt="RemoteCUDA Logo" width="180"/>
</p>

<h1 align="center">RemoteCUDA</h1>

<p align="center">
  <strong>Remote GPU Access — Zero Client Dependencies — Auto CPU Fallback</strong><br>
  <strong>GPU راه دور — صفر وابستگی کلاینت — بازگشت خودکار به CPU</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/remotecuda/"><img src="https://img.shields.io/pypi/v/remotecuda?style=flat-square&logo=pypi&logoColor=white&color=3776AB" alt="PyPI"></a>
  <a href="https://pypi.org/project/remotecuda/"><img src="https://img.shields.io/pypi/pyversions/remotecuda?style=flat-square&logo=python&logoColor=white&color=3776AB" alt="Python"></a>
  <a href="https://pypi.org/project/remotecuda/"><img src="https://img.shields.io/pypi/dm/remotecuda?style=flat-square&color=blue" alt="Downloads"></a>
  <a href="https://github.com/remotecuda/remotecuda/blob/main/LICENSE"><img src="https://img.shields.io/github/license/remotecuda/remotecuda?style=flat-square&color=brightgreen" alt="License"></a>
  <a href="https://github.com/remotecuda/remotecuda/stargazers"><img src="https://img.shields.io/github/stars/remotecuda/remotecuda?style=flat-square&color=yellow" alt="Stars"></a>
  <a href="https://github.com/remotecuda/remotecuda/actions"><img src="https://img.shields.io/github/actions/workflow/status/remotecuda/remotecuda/tests.yml?style=flat-square&label=tests" alt="Tests"></a>
</p>

<p align="center">
  <a href="#english"><b>English</b></a> |
  <a href="#quick-start"><b>Quick Start</b></a> |
  <a href="#features"><b>Features</b></a> |
  <a href="#comparison"><b>Comparison</b></a> |
  <a href="#api-reference"><b>API</b></a> |
  <a href="#installation"><b>Install</b></a> |
  <a href="#architecture"><b>Architecture</b></a> |
  <a href="#fa"><b>فارسی</b></a>
</p>

---

<div id="english"></div>

## What is RemoteCUDA?

RemoteCUDA enables **transparent remote GPU access** over a local network with **zero client-side dependencies**. The client uses only Python standard library — no NumPy, no PyTorch, no CUDA, no external packages whatsoever.

The server **automatically detects GPU availability** and falls back to CPU if no GPU is present. This means you can start computing immediately, with or without a GPU.

### The Problem

| Your Reality | The Pain |
|:--|:--|
| You code on a lightweight laptop | No built-in GPU |
| You have a powerful GPU machine | Sitting somewhere on the network |
| You want to use that GPU | SSH, SCP, VSCode Remote are cumbersome |
| Your datasets are local | Copying 50GB over network is painful |
| You just want to run `python train.py` | Without learning distributed systems |

### The Solution

```bash
# On GPU machine (just TWO commands):
pip install remotecuda[server]
remotecuda start

# On your laptop (just TWO lines):
import remotecuda
remotecuda.init()

# Your EXISTING code works — completely unchanged:
model = MyModel().cuda()  # ← This now runs on the remote GPU!
