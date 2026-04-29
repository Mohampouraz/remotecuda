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



<div align="center">

## Comparison with Alternatives

</div>

<table style="border-collapse: collapse; width: 100%; max-width: 100%; font-size: 0.9em; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">
  <thead>
    <tr style="background-color: #161b22; border-bottom: 2px solid #30363d;">
      <th style="padding: 14px 16px; text-align: left; color: #58a6ff; white-space: nowrap;">Feature</th>
      <th style="padding: 14px 12px; text-align: center; color: #3fb950; white-space: nowrap;">
        RemoteCUDA v3<br><span style="font-size: 0.75em; font-weight: normal; color: #8b949e;">(this project)</span>
      </th>
      <th style="padding: 14px 12px; text-align: center; color: #58a6ff; white-space: nowrap;">
        <a href="https://github.com/chidorigi/chidori" style="color: #58a6ff; text-decoration: none;">Chidori GPU</a>
      </th>
      <th style="padding: 14px 12px; text-align: center; color: #58a6ff; white-space: nowrap;">
        <a href="https://github.com/tensorlink/tensorlink" style="color: #58a6ff; text-decoration: none;">Tensorlink</a>
      </th>
      <th style="padding: 14px 12px; text-align: center; color: #8b949e; white-space: nowrap;">SSH+SCP</th>
      <th style="padding: 14px 12px; text-align: center; color: #8b949e; white-space: nowrap;">VSCode Remote</th>
    </tr>
  </thead>
  <tbody>
    <!-- Zero client dependencies -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Zero client dependencies</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">needs PyTorch</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">needs CUDA</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>
    
    <!-- Pure Python client -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; color: #e6edf3;">&nbsp;&nbsp;&#8627; Pure Python client</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
    </tr>

    <!-- No CUDA on client -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; color: #e6edf3;">&nbsp;&nbsp;&#8627; No CUDA on client</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- No PyTorch on client -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; color: #e6edf3;">&nbsp;&nbsp;&#8627; No PyTorch on client</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- No NumPy on client -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; color: #e6edf3;">&nbsp;&nbsp;&#8627; No NumPy on client</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- Separator -->
    <tr style="background-color: #161b22; border-bottom: 2px solid #30363d;">
      <td colspan="6" style="padding: 6px;"></td>
    </tr>

    <!-- Auto CPU fallback -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Auto CPU fallback</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
    </tr>

    <!-- Auto-discovery -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Network auto-discovery</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- One-command server setup -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">One-command server setup</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- Separator -->
    <tr style="background-color: #161b22; border-bottom: 2px solid #30363d;">
      <td colspan="6" style="padding: 6px;"></td>
    </tr>

    <!-- Multi-GPU parallelism -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Multi-GPU parallelism</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- Load balancing -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; color: #e6edf3;">&nbsp;&nbsp;&#8627; Load balancing</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- Task scheduler -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Task scheduler</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- Tensor caching -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Tensor caching</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- Async streaming -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Async streaming</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- Auto failover -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Auto failover</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- Separator -->
    <tr style="background-color: #161b22; border-bottom: 2px solid #30363d;">
      <td colspan="6" style="padding: 6px;"></td>
    </tr>

    <!-- Open source -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Open source</td>
      <td style="padding: 12px 12px; text-align: center; color: #3fb950;">MIT</td>
      <td style="padding: 12px 12px; text-align: center; color: #3fb950;">MIT</td>
      <td style="padding: 12px 12px; text-align: center; color: #3fb950;">MIT</td>
      <td style="padding: 12px 12px; text-align: center; color: #3fb950;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- pip install -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">pip install</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
    </tr>

    <!-- Windows support -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">Windows support</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
      <td style="padding: 12px 12px; text-align: center; color: #f85149; font-size: 1.2em;">&#10007;</td>
    </tr>

    <!-- JSON protocol -->
    <tr style="border-bottom: 1px solid #21262d; background-color: #0d1117;">
      <td style="padding: 12px 16px; font-weight: 600; color: #e6edf3;">JSON protocol (no pickle)</td>
      <td style="padding: 12px 12px; text-align: center; font-weight: 700; color: #3fb950; font-size: 1.2em;">&#10003;</td>
      <td style="padding: 12px 12px; text-align: center; color: #d2991d;">uses pickle</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
      <td style="padding: 12px 12px; text-align: center; color: #8b949e;">N/A</td>
    </tr>

  </tbody>
</table>

<div style="margin-top: 16px; padding: 14px 20px; background-color: #0d1117; border-left: 4px solid #3fb950; border-radius: 0 8px 8px 0; color: #c9d1d9; font-size: 0.9em;">
  <strong style="color: #3fb950;">Key:</strong> 
  <span style="color: #3fb950; font-weight: 600;">&#10003;</span> = Supported &nbsp;|&nbsp; 
  <span style="color: #f85149;">&#10007;</span> = Not supported &nbsp;|&nbsp; 
  <span style="color: #8b949e;">N/A</span> = Not applicable &nbsp;|&nbsp; 
  <span style="color: #d2991d;">text</span> = Partial/with caveats
</div>
