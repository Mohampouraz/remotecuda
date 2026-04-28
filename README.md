<!-- ============================================================ -->
<!--     RemoteCUDA - The Ultimate Remote GPU Solution           -->
<!--     README v2.0 - Professional Bilingual Documentation      -->
<!-- ============================================================ -->

<div align="center">

<img src="https://raw.githubusercontent.com/yourusername/remotecuda/main/assets/logo.svg" alt="RemoteCUDA Logo" width="180"/>

# 🚀 RemoteCUDA

### GPU راه دور • بدون تغییر کد • کشف خودکار
### Remote GPU • Zero Code Changes • Auto-Discovery

---

[![PyPI version](https://img.shields.io/pypi/v/remotecuda?style=for-the-badge&logo=pypi&logoColor=white&color=3776AB)](https://pypi.org/project/remotecuda/)
[![Python](https://img.shields.io/pypi/pyversions/remotecuda?style=for-the-badge&logo=python&logoColor=white&color=3776AB)](https://pypi.org/project/remotecuda/)
[![License](https://img.shields.io/github/license/yourusername/remotecuda?style=for-the-badge&color=brightgreen)](https://github.com/yourusername/remotecuda/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/remotecuda?style=for-the-badge&color=blue)](https://pypi.org/project/remotecuda/)
[![Stars](https://img.shields.io/github/stars/yourusername/remotecuda?style=for-the-badge&logo=github&color=gold)](https://github.com/yourusername/remotecuda/stargazers)

---

<p align="center">
  <b>🇬🇧 English</b> &nbsp;|&nbsp;
  <a href="#-iran">🇮🇷 فارسی</a> &nbsp;|&nbsp;
  <a href="#-quick-start">⚡ Quick Start</a> &nbsp;|&nbsp;
  <a href="#-comparison">📊 Comparison</a>
</p>

</div>

---

## 🌟 What is RemoteCUDA?

**RemoteCUDA** lets you use a GPU from another computer on your local network — **without changing a single line of your PyTorch code.**

> 💻 **Your laptop** (no GPU) → 🔌 connects to → 🖥️ **GPU machine** (RTX 4090)
>
> No SSH. No file copying. No code changes. Just `remotecuda init`.

---

## ❓ The Problem

Every ML developer faces this exact situation:

| Your Reality | The Pain |
|:--|:--|
| 💻 You code on a lightweight laptop | No built-in GPU |
| 🖥️ You have a powerful GPU machine | Sitting somewhere on the network |
| 🔌 You want to use that GPU | But SSH, SCP, VSCode Remote are cumbersome |
| 📁 Your datasets are local | Copying 50GB over network is painful |
| 🐍 You just want to run `python train.py` | Without learning distributed systems |

---

## ✅ The Solution: RemoteCUDA

```bash
# On GPU machine (just TWO commands):
pip install remotecuda
remotecuda start

# On your  (just TWO lines):
import remotecuda
remotecuda.init()

# Your EXISTING code - COMPLETELY UNCHANGED:
model = MyModel().cuda()  # ← This now runs on the remote GPU!
