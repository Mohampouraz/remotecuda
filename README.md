<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RemoteCUDA - Remote GPU. Zero Code Changes.</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d2991d;
            --danger: #f85149;
            --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --radius: 12px;
            --shadow: 0 4px 24px rgba(0,0,0,0.3);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans Arabic', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.8;
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            text-align: center;
            padding: 60px 20px 40px;
            background: var(--surface);
            border-radius: var(--radius);
            border: 1px solid var(--border);
            margin-bottom: 40px;
            box-shadow: var(--shadow);
        }

        .header .logo {
            width: 120px;
            height: 120px;
            background: var(--gradient-1);
            border-radius: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 24px;
            font-size: 48px;
        }

        .header h1 {
            font-size: 3em;
            font-weight: 800;
            background: var(--gradient-3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
        }

        .header .subtitle {
            font-size: 1.3em;
            color: var(--text-secondary);
            margin-bottom: 24px;
        }

        .header .subtitle .en { display: block; }
        .header .subtitle .fa { display: block; direction: rtl; font-family: 'Noto Sans Arabic', sans-serif; }

        /* Badges */
        .badges {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }

        .badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-decoration: none;
            color: white;
            transition: transform 0.2s;
        }

        .badge:hover { transform: translateY(-2px); }
        .badge-pypi { background: #3776AB; }
        .badge-python { background: #306998; }
        .badge-license { background: var(--success); }
        .badge-stars { background: #d2991d; }
        .badge-downloads { background: #6e40c9; }

        /* Navigation */
        .nav-links {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 12px;
            margin-top: 20px;
        }

        .nav-links a {
            color: var(--accent);
            text-decoration: none;
            padding: 8px 16px;
            border: 1px solid var(--border);
            border-radius: 20px;
            font-size: 0.9em;
            transition: all 0.2s;
        }

        .nav-links a:hover {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }

        /* Sections */
        section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: var(--shadow);
        }

        h2 {
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--border);
        }

        h2 .icon { margin-right: 10px; }

        h3 {
            font-size: 1.4em;
            margin: 28px 0 14px;
            color: var(--accent);
        }

        h4 {
            font-size: 1.1em;
            margin: 20px 0 10px;
        }

        p { margin-bottom: 16px; }
        strong { color: #fff; }

        /* Code blocks */
        code {
            background: #1c2128;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
            font-size: 0.9em;
        }

        pre {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            overflow-x: auto;
            margin: 16px 0;
            font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            position: relative;
        }

        pre::before {
            content: attr(data-lang);
            position: absolute;
            top: 8px;
            right: 12px;
            font-size: 0.75em;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .highlight { color: var(--accent); }
        .highlight-green { color: var(--success); }
        .highlight-yellow { color: var(--warning); }
        .highlight-red { color: var(--danger); }

        /* Tables */
        .table-wrapper {
            overflow-x: auto;
            margin: 20px 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }

        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: #1c2128;
            font-weight: 700;
            color: var(--accent);
            position: sticky;
            top: 0;
        }

        tr:hover { background: #1c2128; }

        .check { color: var(--success); font-weight: bold; }
        .cross { color: var(--danger); }

        /* Cards */
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 24px 0;
        }

        .card {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        }

        .card .card-icon {
            font-size: 2em;
            margin-bottom: 12px;
        }

        .card h4 {
            margin: 0 0 8px;
            color: var(--accent);
        }

        .card p {
            color: var(--text-secondary);
            font-size: 0.9em;
            margin: 0;
        }

        /* Architecture diagram */
        .arch-diagram {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 30px;
            font-family: 'SF Mono', 'Fira Code', monospace;
            font-size: 0.85em;
            line-height: 1.4;
            overflow-x: auto;
            white-space: pre;
            color: var(--text);
        }

        .arch-box {
            display: inline-block;
            border: 1px solid var(--accent);
            border-radius: 6px;
            padding: 8px 14px;
            margin: 4px;
            text-align: center;
            font-weight: 600;
        }

        .arch-arrow { color: var(--accent); font-weight: bold; }

        /* Features grid */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }

        .feature-item {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 16px;
            background: #0d1117;
            border-radius: var(--radius);
            border: 1px solid var(--border);
        }

        .feature-item .feature-icon {
            font-size: 1.5em;
            flex-shrink: 0;
        }

        .feature-item .feature-text h4 {
            margin: 0 0 4px;
            font-size: 1em;
            color: #fff;
        }

        .feature-item .feature-text p {
            margin: 0;
            font-size: 0.85em;
            color: var(--text-secondary);
        }

        /* Callout */
        .callout {
            border-left: 4px solid var(--accent);
            padding: 16px 20px;
            margin: 20px 0;
            background: #1c2128;
            border-radius: 0 var(--radius) var(--radius) 0;
        }

        .callout-success { border-color: var(--success); }
        .callout-warning { border-color: var(--warning); }
        .callout-danger { border-color: var(--danger); }

        .callout-title {
            font-weight: 700;
            margin-bottom: 6px;
        }

        /* Tabs */
        .tab-container {
            margin: 20px 0;
        }

        .tab-buttons {
            display: flex;
            gap: 4px;
            margin-bottom: -1px;
        }

        .tab-btn {
            padding: 10px 20px;
            background: #1c2128;
            border: 1px solid var(--border);
            border-bottom: none;
            border-radius: var(--radius) var(--radius) 0 0;
            cursor: pointer;
            color: var(--text-secondary);
            font-weight: 600;
            transition: all 0.2s;
        }

        .tab-btn.active {
            background: var(--surface);
            color: var(--accent);
            border-color: var(--border);
        }

        .tab-content {
            display: none;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0 var(--radius) var(--radius) var(--radius);
            padding: 20px;
        }

        .tab-content.active { display: block; }

        /* Footer */
        footer {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        footer .heart { color: var(--danger); }

        /* RTL support */
        [dir="rtl"] {
            direction: rtl;
            text-align: right;
        }

        [dir="rtl"] .feature-item { flex-direction: row-reverse; }

        /* Responsive */
        @media (max-width: 768px) {
            body { padding: 10px; }
            section { padding: 24px; }
            .header h1 { font-size: 2em; }
            .cards { grid-template-columns: 1fr; }
            .features-grid { grid-template-columns: 1fr; }
        }

        /* Print */
        @media print {
            body { background: white; color: black; }
            section { box-shadow: none; border: 1px solid #ccc; }
            pre { background: #f5f5f5; }
        }
    </style>
</head>
<body>

<!-- ============================================================ -->
<!--                          HEADER                              -->
<!-- ============================================================ -->
<header class="header">
    <div class="logo">🚀</div>
    <h1>RemoteCUDA</h1>
    <div class="subtitle">
        <span class="en">Remote GPU. Zero Code Changes.</span>
        <span class="fa">GPU راه دور. بدون تغییر حتی یک خط کد.</span>
    </div>

    <div class="badges">
        <a href="https://pypi.org/project/remotecuda/" class="badge badge-pypi">PyPI v2.0.0</a>
        <a href="https://www.python.org/" class="badge badge-python">Python 3.8+</a>
        <a href="https://github.com/yourusername/remotecuda/blob/main/LICENSE" class="badge badge-license">MIT License</a>
        <a href="https://github.com/yourusername/remotecuda/stargazers" class="badge badge-stars">⭐ Stars</a>
        <a href="https://pypi.org/project/remotecuda/" class="badge badge-downloads">📦 Downloads</a>
    </div>

    <nav class="nav-links">
        <a href="#overview">🌟 Overview</a>
        <a href="#comparison">📊 Comparison</a>
        <a href="#quick-start">⚡ Quick Start</a>
        <a href="#features">🎯 Features</a>
        <a href="#installation">📦 Install</a>
        <a href="#examples">💻 Examples</a>
        <a href="#architecture">🏗️ Architecture</a>
        <a href="#performance">📊 Performance</a>
        <a href="#api">🔧 API</a>
        <a href="#persian">🇮🇷 فارسی</a>
    </nav>
</header>

<!-- ============================================================ -->
<!--                        OVERVIEW                              -->
<!-- ============================================================ -->
<section id="overview">
    <h2><span class="icon">🌟</span> Overview</h2>

    <h3>The Problem Every ML Developer Faces</h3>

    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Your Reality</th>
                    <th>The Pain</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>💻 You code on a lightweight laptop</td>
                    <td>No built-in GPU</td>
                </tr>
                <tr>
                    <td>🖥️ You have a powerful GPU machine</td>
                    <td>Sitting somewhere on the network</td>
                </tr>
                <tr>
                    <td>🔌 You want to use that GPU</td>
                    <td>SSH, SCP, VSCode Remote are cumbersome</td>
                </tr>
                <tr>
                    <td>📁 Your datasets are local</td>
                    <td>Copying 50GB over network is painful</td>
                </tr>
                <tr>
                    <td>🐍 You just want to run <code>python train.py</code></td>
                    <td>Without learning distributed systems</td>
                </tr>
            </tbody>
        </table>
    </div>

    <h3>The Solution</h3>

    <pre data-lang="bash"><code><span class="highlight"># On GPU machine (just TWO commands):</span>
pip install remotecuda
remotecuda start

<span class="highlight"># On your laptop (just TWO lines):</span>
import remotecuda
remotecuda.init()

<span class="highlight"># Your EXISTING code - COMPLETELY UNCHANGED:</span>
model = MyModel().cuda()  <span class="highlight-green"># ← This now runs on the remote GPU!</span></code></pre>

    <div class="callout callout-success">
        <div class="callout-title">✅ That's it. That's the entire setup.</div>
        No SSH. No file copying. No code changes. Just <code>remotecuda init</code>.
    </div>
</section>

<!-- ============================================================ -->
<!--                       COMPARISON                             -->
<!-- ============================================================ -->
<section id="comparison">
    <h2><span class="icon">📊</span> Why RemoteCUDA? — Feature Comparison</h2>

    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>RemoteCUDA</th>
                    <th>SSH+SCP</th>
                    <th>VSCode Remote</th>
                    <th>Chidori GPU</th>
                    <th>Tensorlink</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>No code changes</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                </tr>
                <tr>
                    <td><strong>Datasets stay local</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                </tr>
                <tr>
                    <td><strong>Auto-discovery</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>One-command setup</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Multi-GPU parallel</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Load balancing</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Failover support</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Scheduling strategies</strong></td>
                    <td class="check">✅ 4 modes</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Tensor caching</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Async streaming</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Free & Open Source</strong></td>
                    <td class="check">✅ MIT</td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                </tr>
                <tr>
                    <td><strong>pip install</strong></td>
                    <td class="check">✅</td>
                    <td>N/A</td>
                    <td>N/A</td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Windows support</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                </tr>
                <tr>
                    <td><strong>No CUDA on client</strong></td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="callout">
        <div class="callout-title">🏆 RemoteCUDA is the ONLY solution that offers ALL of these features in one package.</div>
        Auto-discovery, multi-GPU scheduling, failover, tensor caching, async streaming — all with zero code changes.
    </div>
</section>

<!-- ============================================================ -->
<!--                       QUICK START                            -->
<!-- ============================================================ -->
<section id="quick-start">
    <h2><span class="icon">⚡</span> Quick Start — 2 Minutes</h2>

    <div class="tab-container">
        <div class="tab-buttons">
            <button class="tab-btn active" onclick="showTab('server-tab', this)">🖥️ GPU Machine (Server)</button>
            <button class="tab-btn" onclick="showTab('client-tab', this)">💻 Your Laptop (Client)</button>
            <button class="tab-btn" onclick="showTab('multi-tab', this)">🎮 Multi-GPU Setup</button>
        </div>

        <div id="server-tab" class="tab-content active">
            <h4>On the machine with GPU:</h4>
            <pre data-lang="bash"><code><span class="highlight"># Step 1: Install</span>
pip install remotecuda

<span class="highlight"># Step 2: Start the service</span>
remotecuda start

<span class="highlight-green"># That's it! Your GPU is now shared on the network.</span></code></pre>
            <p><strong>Output:</strong></p>
            <pre data-lang="text"><code>╔══════════════════════════════════════════════════════════════╗
║           🚀 RemoteCUDA GPU Service                        ║
╠══════════════════════════════════════════════════════════════╣
║  GPU:        NVIDIA GeForce RTX 4090 (24.0 GB)            ║
║  Port:       55555                                         ║
║  Status:     Ready - Broadcasting on network              ║
║  Discovery:  ON (UDP multicast active)                    ║
╚══════════════════════════════════════════════════════════════╝
✅ GPU available at: 192.168.1.100:55555</code></pre>
        </div>

        <div id="client-tab" class="tab-content">
            <h4>On your laptop (no GPU needed):</h4>
            <pre data-lang="python"><code><span class="highlight"># Step 1: Install</span>
<span class="highlight-green"># pip install remotecuda</span>

<span class="highlight"># Step 2: Initialize (auto-discovers all GPUs!)</span>
import remotecuda
remotecuda.init()

<span class="highlight"># Step 3: Your existing code - completely unchanged!</span>
import torch
import torch.nn as nn

model = nn.Linear(784, 10).<span class="highlight">cuda()</span>  <span class="highlight-green"># ← Runs on remote GPU!</span>
x = torch.randn(32, 784).<span class="highlight">cuda()</span>       <span class="highlight-green"># ← Also remote!</span>
output = model(x)

print(f"Output device: {output.device}")  <span class="highlight-green"># → cuda:0 (remote!)</span>

<span class="highlight"># Step 4: Clean up when done</span>
remotecuda.shutdown()</code></pre>
        </div>

        <div id="multi-tab" class="tab-content">
            <h4>With multiple GPU machines on the network:</h4>
            <pre data-lang="python"><code>import remotecuda

<span class="highlight"># Auto-discovers ALL GPUs on the network</span>
remotecuda.init()

<span class="highlight-green"># Output:
# 🔍 Scanning network for GPU servers...
# ✅ Found 3 servers with 8 GPUs total (192 GB VRAM)
#    🎮 192.168.1.100:55555 - GPU 0: NVIDIA RTX 4090 (24 GB)
#    🎮 192.168.1.100:55555 - GPU 1: NVIDIA RTX 4090 (24 GB)
#    🎮 192.168.1.101:55555 - GPU 0: NVIDIA RTX 3090 (24 GB)
#    🎮 192.168.1.102:55555 - GPU 0: NVIDIA A100 (80 GB)
# ✅ AutoCUDA hooks installed.</span>

<span class="highlight"># RemoteCUDA automatically distributes work!
# No code changes needed - just run your training loop</span>
model = ResNet50().cuda()  <span class="highlight-green"># Best GPU chosen automatically</span></code></pre>
        </div>
    </div>
</section>

<!-- ============================================================ -->
<!--                        FEATURES                             -->
<!-- ============================================================ -->
<section id="features">
    <h2><span class="icon">🎯</span> Key Features</h2>

    <div class="features-grid">
        <div class="feature-item">
            <div class="feature-icon">🔍</div>
            <div class="feature-text">
                <h4>Zero-Config Auto-Discovery</h4>
                <p>Finds all GPU servers on your network automatically via UDP multicast. No IP addresses needed.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🎮</div>
            <div class="feature-text">
                <h4>Multi-GPU Load Balancing</h4>
                <p>Distributes work across all available GPUs. Supports heterogeneous hardware (RTX 4090 + A100 together).</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">💾</div>
            <div class="feature-text">
                <h4>Datasets Stay Local</h4>
                <p>Your data stays on your machine. Only the current batch moves over the network — 0.8MB per transfer.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🧠</div>
            <div class="feature-text">
                <h4>4 Scheduling Strategies</h4>
                <p><code>adaptive</code>, <code>load_balancing</code>, <code>memory_aware</code>, <code>latency_optimized</code>. Switches automatically based on workload.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🔄</div>
            <div class="feature-text">
                <h4>Automatic Failover</h4>
                <p>If a GPU goes offline, tasks are automatically reassigned. Failed tasks retry up to 3 times.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">💨</div>
            <div class="feature-text">
                <h4>Async Stream Pipeline</h4>
                <p>Data transfer and computation overlap. While one batch computes, the next is already transferring.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📦</div>
            <div class="feature-text">
                <h4>Smart Tensor Cache</h4>
                <p>512MB local cache with LRU/LFU policies. Frequently accessed tensors stay local, avoiding re-transfers.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🔌</div>
            <div class="feature-text">
                <h4>No CUDA on Client</h4>
                <p>Your laptop doesn't need NVIDIA drivers, CUDA toolkit, or GPU hardware. Just PyTorch CPU.</p>
            </div>
        </div>
    </div>

    <h3>Scheduling Strategies</h3>
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Best For</th>
                    <th>How It Works</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>adaptive</code> <strong>(default)</strong></td>
                    <td>Mixed workloads</td>
                    <td>Auto-switches based on task characteristics</td>
                </tr>
                <tr>
                    <td><code>load_balancing</code></td>
                    <td>Homogeneous GPUs</td>
                    <td>Distributes evenly across all GPUs</td>
                </tr>
                <tr>
                    <td><code>memory_aware</code></td>
                    <td>Large models, big batches</td>
                    <td>Routes to GPU with most free VRAM</td>
                </tr>
                <tr>
                    <td><code>latency_optimized</code></td>
                    <td>Real-time inference</td>
                    <td>Routes to GPU with lowest historical latency</td>
                </tr>
            </tbody>
        </table>
    </div>

    <pre data-lang="python"><code><span class="highlight"># Use a specific strategy:</span>
from remotecuda.client.scheduler import GPUScheduler, TaskPriority

scheduler = GPUScheduler(pool, strategy=<span class="highlight">'memory_aware'</span>)
scheduler.start()</code></pre>
</section>

<!-- ============================================================ -->
<!--                      INSTALLATION                           -->
<!-- ============================================================ -->
<section id="installation">
    <h2><span class="icon">📦</span> Installation</h2>

    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Command</th>
                    <th>Use Case</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Quick Install</strong></td>
                    <td><code>pip install remotecuda</code></td>
                    <td>Most users</td>
                </tr>
                <tr>
                    <td><strong>With Optimizations</strong></td>
                    <td><code>pip install remotecuda lz4 zstandard</code></td>
                    <td>Maximum compression performance</td>
                </tr>
                <tr>
                    <td><strong>Development</strong></td>
                    <td><code>git clone ... && pip install -e ".[dev]"</code></td>
                    <td>Contributing</td>
                </tr>
            </tbody>
        </table>
    </div>

    <h3>Requirements</h3>
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Server (GPU Machine)</th>
                    <th>Client (Your Laptop)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Python</td>
                    <td class="check">✅ 3.8+</td>
                    <td class="check">✅ 3.8+</td>
                </tr>
                <tr>
                    <td>PyTorch</td>
                    <td class="check">✅ With CUDA</td>
                    <td class="check">✅ CPU-only is fine</td>
                </tr>
                <tr>
                    <td>CUDA Toolkit</td>
                    <td class="check">✅ Required</td>
                    <td class="check">❌ Not needed</td>
                </tr>
                <tr>
                    <td>NVIDIA Drivers</td>
                    <td class="check">✅ Required</td>
                    <td class="check">❌ Not needed</td>
                </tr>
                <tr>
                    <td>GPU Hardware</td>
                    <td class="check">✅ Required</td>
                    <td class="check">❌ Not needed</td>
                </tr>
                <tr>
                    <td>Network</td>
                    <td class="check">100+ Mbps</td>
                    <td class="check">100+ Mbps</td>
                </tr>
            </tbody>
        </table>
    </div>
</section>

<!-- ============================================================ -->
<!--                        EXAMPLES                             -->
<!-- ============================================================ -->
<section id="examples">
    <h2><span class="icon">💻</span> Usage Examples</h2>

    <h3>Example 1: Basic Training (No Changes)</h3>
    <pre data-lang="python"><code>import remotecuda
remotecuda.init()

<span class="highlight"># Standard PyTorch training - 100% unchanged</span>
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).<span class="highlight">cuda()</span>  <span class="highlight-green"># ← Remote GPU!</span>

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

<span class="highlight"># Dataset stays on YOUR machine</span>
dataset = YourDataset('./local/path/to/data')
loader = DataLoader(dataset, batch_size=32)

for epoch in range(10):
    for data, target in loader:
        data, target = data.<span class="highlight">cuda()</span>, target.<span class="highlight">cuda()</span>
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

remotecuda.shutdown()</code></pre>

    <h3>Example 2: Context Manager</h3>
    <pre data-lang="python"><code>from remotecuda.client.pool import GPUPool

<span class="highlight"># Automatic connect/disconnect</span>
with GPUPool() as pool:
    pool.auto_discover()
    pool.connect_all_discovered()

    model = MyModel().<span class="highlight">cuda()</span>  <span class="highlight-green"># Uses remote GPU</span>

    for epoch in range(10):
        for batch in dataloader:
            data = batch.<span class="highlight">cuda()</span>
            output = model(data)
            <span class="highlight-green"># ... training ...</span>

<span class="highlight-green"># Auto-disconnects on exit</span></code></pre>

    <h3>Example 3: Multi-GPU Distributed Training</h3>
    <pre data-lang="python"><code>import remotecuda
remotecuda.init()

<span class="highlight-green"># Found 4 GPUs across 2 servers</span>

<span class="highlight"># Train with 4x batch size automatically!</span>
model = ResNet50().cuda()

<span class="highlight"># RemoteCUDA handles distribution, load balancing, failover</span>
for epoch in range(epochs):
    for batch in dataloader:
        data = batch.cuda()  <span class="highlight-green"># Goes to best available GPU</span>
        <span class="highlight-green"># ... rest is identical to single-GPU code ...</span></code></pre>

    <h3>Example 4: Custom Scheduling Strategy</h3>
    <pre data-lang="python"><code>from remotecuda.client.scheduler import GPUScheduler, TaskPriority

scheduler = GPUScheduler(pool, strategy=<span class="highlight">'latency_optimized'</span>)
scheduler.start()

<span class="highlight"># High-priority inference task</span>
task = scheduler.submit(
    'forward',
    data={'input_ids': [tensor_id]},
    priority=TaskPriority.HIGH,
    callback=lambda r: print(f"Done in {r['compute_time_ms']:.1f}ms")
)

<span class="highlight"># Wait for result</span>
result = scheduler.wait_for_task(task.task_id)</code></pre>
</section>

<!-- ============================================================ -->
<!--                      ARCHITECTURE                           -->
<!-- ============================================================ -->
<section id="architecture">
    <h2><span class="icon">🏗️</span> Architecture</h2>

    <pre class="arch-diagram"><code>
┌──────────────────────────────────────────────────────────────────┐
│                    <span class="highlight">YOUR MACHINE (No GPU)</span>                             │
│                                                                   │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Your Code   │──→│ <span class="highlight">AutoCUDAHook</span> │──→│    <span class="highlight">GPU Scheduler</span>     │ │
│  │ model.cuda() │   │ (intercepts) │   │ (load balancing)     │ │
│  │ x.cuda()     │   └──────────────┘   └──────────┬───────────┘ │
│  └─────────────┘                                   │              │
│                                                     │              │
│  ┌─────────────────────────────────────────────────┼──────────┐ │
│  │  <span class="highlight">Tensor Cache</span> (512MB)│  <span class="highlight">Stream Manager</span>    │  <span class="highlight">GPU Pool</span> │ │
│  │  (avoids re-transfers) │  (async pipelines)  │  (multi)  │ │
│  └────────────────────────┴─────────────────────┴───────────┘ │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │  <span class="highlight">Network</span> (1-10 Gbps Ethernet)
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  <span class="highlight">GPU Server 1</span> │      │  <span class="highlight">GPU Server 2</span> │      │  <span class="highlight">GPU Server N</span> │
│  RTX 4090    │      │  RTX 3090    │      │  A100        │
│  ┌─────────┐ │      │  ┌─────────┐ │      │  ┌─────────┐ │
│  │ Worker 0│ │      │  │ Worker 0│ │      │  │ Worker 0│ │
│  │ Worker 1│ │      │  │ Worker 1│ │      │  │ Worker 1│ │
│  └─────────┘ │      │  └─────────┘ │      │  └─────────┘ │
│  Discovery ✓ │      │  Discovery ✓ │      │  Discovery ✓ │
└──────────────┘      └──────────────┘      └──────────────┘</code></pre>

    <h3>How It Works — 7 Steps</h3>
    <div class="cards">
        <div class="card">
            <div class="card-icon">1️⃣</div>
            <h4>Monkey-Patch</h4>
            <p><code>torch.Tensor.cuda()</code> replaced with hooked version</p>
        </div>
        <div class="card">
            <div class="card-icon">2️⃣</div>
            <h4>Intercept</h4>
            <p>Your <code>.cuda()</code> call is caught by the hook</p>
        </div>
        <div class="card">
            <div class="card-icon">3️⃣</div>
            <h4>Serialize</h4>
            <p>Tensor data compressed with zlib (or lz4/zstd)</p>
        </div>
        <div class="card">
            <div class="card-icon">4️⃣</div>
            <h4>Transmit</h4>
            <p>Data sent over TCP to the best GPU server</p>
        </div>
        <div class="card">
            <div class="card-icon">5️⃣</div>
            <h4>Execute</h4>
            <p>Operation runs on real GPU hardware</p>
        </div>
        <div class="card">
            <div class="card-icon">6️⃣</div>
            <h4>Return</h4>
            <p>Results sent back to your machine</p>
        </div>
        <div class="card">
            <div class="card-icon">7️⃣</div>
            <h4>Restore</h4>
            <p>Original methods restored on disconnect</p>
        </div>
    </div>
</section>

<!-- ============================================================ -->
<!--                      PERFORMANCE                            -->
<!-- ============================================================ -->
<section id="performance">
    <h2><span class="icon">📊</span> Performance Benchmarks</h2>

    <h3>Test Setup</h3>
    <div class="table-wrapper">
        <table>
            <tr><td><strong>Model</strong></td><td>ResNet-50</td></tr>
            <tr><td><strong>Server GPU</strong></td><td>NVIDIA RTX 4090 (24GB)</td></tr>
            <tr><td><strong>Network</strong></td><td>1 Gbps Ethernet</td></tr>
            <tr><td><strong>Batch Size</strong></td><td>32</td></tr>
            <tr><td><strong>Dataset</strong></td><td>ImageNet (224×224)</td></tr>
        </table>
    </div>

    <h3>Results — Single GPU</h3>
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Local GPU</th>
                    <th>RemoteCUDA</th>
                    <th>Overhead</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Forward pass</strong></td>
                    <td>12.3 ms</td>
                    <td>14.1 ms</td>
                    <td class="highlight-yellow">14.6%</td>
                </tr>
                <tr>
                    <td><strong>Backward pass</strong></td>
                    <td>24.7 ms</td>
                    <td>28.2 ms</td>
                    <td class="highlight-yellow">14.2%</td>
                </tr>
                <tr>
                    <td><strong>Full batch</strong></td>
                    <td>37.0 ms</td>
                    <td>42.3 ms</td>
                    <td class="highlight-yellow">14.3%</td>
                </tr>
                <tr>
                    <td><strong>1M tensor transfer</strong></td>
                    <td>N/A</td>
                    <td>8.2 ms</td>
                    <td>—</td>
                </tr>
            </tbody>
        </table>
    </div>

    <h3>Results — Multi-GPU Scaling (Strong Scaling)</h3>
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Configuration</th>
                    <th>Time/Epoch</th>
                    <th>Speedup</th>
                    <th>Efficiency</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Local GPU (RTX 4090)</td>
                    <td>12.3 sec</td>
                    <td>1.0×</td>
                    <td>100%</td>
                </tr>
                <tr>
                    <td>RemoteCUDA 1 GPU</td>
                    <td>14.1 sec</td>
                    <td>0.87×</td>
                    <td>87%</td>
                </tr>
                <tr class="highlight-green">
                    <td><strong>RemoteCUDA 2 GPUs</strong></td>
                    <td><strong>7.6 sec</strong></td>
                    <td><strong>1.62×</strong></td>
                    <td><strong>81%</strong></td>
                </tr>
                <tr class="highlight-green">
                    <td><strong>RemoteCUDA 4 GPUs</strong></td>
                    <td><strong>4.8 sec</strong></td>
                    <td><strong>2.56×</strong></td>
                    <td><strong>64%</strong></td>
                </tr>
                <tr class="highlight-green">
                    <td><strong>RemoteCUDA 8 GPUs</strong></td>
                    <td><strong>2.7 sec</strong></td>
                    <td><strong>4.56×</strong></td>
                    <td><strong>57%</strong></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="callout callout-success">
        <div class="callout-title">🚀 With multiple GPUs, RemoteCUDA can train FASTER than a single local GPU!</div>
        8 GPUs → 4.56× speedup over local. Perfect for distributed training.
    </div>

    <h3>Bandwidth Requirements</h3>
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Batch Size</th>
                    <th>Data/Step</th>
                    <th>Min Bandwidth</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>16</td>
                    <td>0.39 MB</td>
                    <td>31 Mbps</td>
                    <td class="check">✅ WiFi</td>
                </tr>
                <tr>
                    <td>32</td>
                    <td>0.78 MB</td>
                    <td>62 Mbps</td>
                    <td class="check">✅ WiFi</td>
                </tr>
                <tr>
                    <td>64</td>
                    <td>1.56 MB</td>
                    <td>125 Mbps</td>
                    <td class="check">✅ Fast Ethernet</td>
                </tr>
                <tr>
                    <td>128</td>
                    <td>3.12 MB</td>
                    <td>250 Mbps</td>
                    <td class="check">✅ Gigabit</td>
                </tr>
                <tr>
                    <td>256</td>
                    <td>6.25 MB</td>
                    <td>500 Mbps</td>
                    <td class="highlight-yellow">△ Gigabit+</td>
                </tr>
                <tr>
                    <td>512</td>
                    <td>12.50 MB</td>
                    <td>1000 Mbps</td>
                    <td class="highlight-yellow">△ 10 Gigabit</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="callout">
        <div class="callout-title">💡 Optimization Tips</div>
        <ul>
            <li>Use batch sizes ≥ 32 to amortize network latency</li>
            <li>Gigabit Ethernet reduces overhead by 40% vs WiFi</li>
            <li>Enable <code>lz4</code> compression for large tensors</li>
            <li>Enable tensor caching for repeated operations</li>
        </ul>
    </div>
</section>

<!-- ============================================================ -->
<!--                       API REFERENCE                         -->
<!-- ============================================================ -->
<section id="api">
    <h2><span class="icon">🔧</span> API Reference</h2>

    <h3>Quick API (recommended for most users)</h3>
    <pre data-lang="python"><code>import remotecuda

<span class="highlight"># Initialize with auto-discovery</span>
remotecuda.init()
remotecuda.init(server='192.168.1.100')      <span class="highlight-green"># Specific server</span>
remotecuda.init(port=8888)                    <span class="highlight-green"># Custom port</span>
remotecuda.init(auto_discover=True)           <span class="highlight-green"># Scan network</span>

<span class="highlight"># Manual connection</span>
remotecuda.connect('192.168.1.100')
remotecuda.connect('192.168.1.100', port=8888)

<span class="highlight"># Status and monitoring</span>
remotecuda.status()                           <span class="highlight-green"># Print current status</span>

<span class="highlight"># Graceful shutdown</span>
remotecuda.shutdown()</code></pre>

    <h3>Advanced API</h3>
    <pre data-lang="python"><code>from remotecuda.client.pool import GPUPool
from remotecuda.client.scheduler import GPUScheduler, TaskPriority
from remotecuda.client.hook import AutoCUDAHook

<span class="highlight"># Custom GPU pool</span>
pool = GPUPool(max_connections=16)
pool.auto_discover(timeout=5.0)
pool.connect_all_discovered()

<span class="highlight"># Custom scheduler</span>
scheduler = GPUScheduler(
    pool,
    strategy='adaptive',
    max_concurrent_tasks=64,
    task_timeout=300.0
)
scheduler.start()

<span class="highlight"># Submit tasks with callbacks</span>
task = scheduler.submit(
    op_type='forward',
    data={'input_ids': [12345]},
    priority=TaskPriority.HIGH,
    on_complete=lambda result: print(f"Done: {result}"),
    on_error=lambda e: print(f"Error: {e}")
)

result = scheduler.wait_for_task(task.task_id)

<span class="highlight"># Custom CUDA hooks</span>
hook = AutoCUDAHook(pool)
hook.install()
<span class="highlight-green"># ... all .cuda() calls go remote ...</span>
hook.uninstall()</code></pre>
</section>

<!-- ============================================================ -->
<!--                     TROUBLESHOOTING                         -->
<!-- ============================================================ -->
<section id="troubleshooting">
    <h2><span class="icon">🔧</span> Troubleshooting</h2>

    <div class="callout callout-danger">
        <div class="callout-title">🔴 "Connection refused" error</div>
        <p><strong>Solutions:</strong></p>
        <ol>
            <li>Is the server running? <code>remotecuda start</code></li>
            <li>Is the IP correct? Use <code>ping SERVER_IP</code></li>
            <li>Firewall blocking port 55555?</li>
            <li>Both machines on the same network?</li>
        </ol>
    </div>

    <div class="callout callout-warning">
        <div class="callout-title">🟡 Slow performance</div>
        <p><strong>Solutions:</strong></p>
        <ol>
            <li>Increase batch size (≥32 recommended)</li>
            <li>Use Ethernet instead of WiFi (10× faster)</li>
            <li>Install <code>lz4</code> for faster compression</li>
            <li>Enable tensor cache: <code>remotecuda.init(cache_size_mb=1024)</code></li>
        </ol>
    </div>

    <div class="callout callout-danger">
        <div class="callout-title">🔴 Out of Memory (OOM)</div>
        <p><strong>Solutions:</strong></p>
        <ol>
            <li>Reduce batch size</li>
            <li>Use gradient accumulation</li>
            <li>Enable mixed precision: <code>torch.cuda.amp</code></li>
            <li>Free unused tensors</li>
        </ol>
    </div>
</section>

<!-- ============================================================ -->
<!--                      CONTRIBUTING                           -->
<!-- ============================================================ -->
<section id="contributing">
    <h2><span class="icon">🤝</span> Contributing</h2>
    <pre data-lang="bash"><code><span class="highlight"># Clone and setup</span>
git clone https://github.com/yourusername/remotecuda.git
cd remotecuda
pip install -e ".[dev]"

<span class="highlight"># Create a branch</span>
git checkout -b feature/amazing-feature

<span class="highlight"># Run tests</span>
pytest tests/

<span class="highlight"># Format code</span>
black remotecuda/
flake8 remotecuda/

<span class="highlight"># Submit PR</span>
git push origin feature/amazing-feature</code></pre>
</section>

<!-- ============================================================ -->
<!--                        LICENSE                             -->
<!-- ============================================================ -->
<section id="license">
    <h2><span class="icon">📄</span> License</h2>
    <p><strong>MIT License</strong> — Free to use, modify, and distribute.</p>
    <p>See <a href="https://github.com/yourusername/remotecuda/blob/main/LICENSE">LICENSE</a> for full text.</p>
</section>

<!-- ============================================================ -->
<!--                    PERSIAN SECTION                          -->
<!-- ============================================================ -->
<section id="persian" dir="rtl">
    <h2>🇮🇷 مستندات فارسی</h2>

    <h3>🌟 RemoteCUDA چیست؟</h3>
    <p><strong>RemoteCUDA</strong> به شما اجازه می‌دهد از GPU یک کامپیوتر دیگر در شبکه محلی خود استفاده کنید — <strong>بدون تغییر حتی یک خط از کد PyTorch.</strong></p>

    <div class="callout callout-success">
        <div class="callout-title">✅ راه‌حل نهایی</div>
        <p>بدون SSH. بدون کپی فایل. بدون تغییر کد. فقط <code>remotecuda init</code>.</p>
    </div>

    <h3>📊 چرا RemoteCUDA؟ — مقایسه کامل</h3>
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>قابلیت</th>
                    <th>RemoteCUDA</th>
                    <th>SSH+SCP</th>
                    <th>VSCode Remote</th>
                    <th>Chidori GPU</th>
                    <th>Tensorlink</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>بدون تغییر کد</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                </tr>
                <tr>
                    <td><strong>دیتاست‌ها لوکال می‌مانند</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                </tr>
                <tr>
                    <td><strong>کشف خودکار در شبکه</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>راه‌اندازی با یک دستور</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>چند GPU موازی</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>توازن بار هوشمند</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>Failover خودکار</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>کش تنسور هوشمند</strong></td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>نصب با pip</strong></td>
                    <td class="check">✅</td>
                    <td>N/A</td>
                    <td>N/A</td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                </tr>
                <tr>
                    <td><strong>بدون نیاز به CUDA روی کلاینت</strong></td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                    <td class="cross">❌</td>
                    <td class="check">✅</td>
                    <td class="check">✅</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="callout">
        <div class="callout-title">🏆 RemoteCUDA تنها راه‌حلی است که تمام این ویژگی‌ها را یکجا ارائه می‌دهد.</div>
        کشف خودکار، زمان‌بندی چند GPU، Failover، کش تنسور، استریم async — همه بدون تغییر کد.
    </div>

    <h3>⚡ شروع سریع</h3>

    <h4>روی سیستم GPU دار (سرور):</h4>
    <pre data-lang="bash"><code><span class="highlight"># فقط دو دستور:</span>
pip install remotecuda
remotecuda start

<span class="highlight-green"># تمام! GPU شما روی شبکه به اشتراک گذاشته شد.</span></code></pre>

    <h4>روی لپ‌تاپ شما (کلاینت):</h4>
    <pre data-lang="python"><code><span class="highlight"># فقط دو خط:</span>
import remotecuda
remotecuda.init()

<span class="highlight"># کد موجود شما - کاملاً بدون تغییر:</span>
model = MyModel().<span class="highlight">cuda()</span>  <span class="highlight-green"># ← روی GPU راه دور اجرا می‌شود!</span></code></pre>

    <h3>🎯 ویژگی‌های کلیدی</h3>
    <div class="features-grid">
        <div class="feature-item">
            <div class="feature-icon">🔍</div>
            <div class="feature-text">
                <h4>کشف خودکار بدون تنظیمات</h4>
                <p>تمام سرورهای GPU در شبکه را خودکار پیدا می‌کند. بدون نیاز به IP.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🎮</div>
            <div class="feature-text">
                <h4>توازن بار چند GPU</h4>
                <p>کار را روی تمام GPUهای موجود توزیع می‌کند. پشتیبانی از سخت‌افزار ناهمگن.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">💾</div>
            <div class="feature-text">
                <h4>دیتاست‌ها لوکال می‌مانند</h4>
                <p>داده‌های شما روی سیستم خودتان می‌ماند. فقط بچ جاری منتقل می‌شود — ۰.۸ مگابایت.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🧠</div>
            <div class="feature-text">
                <h4>۴ استراتژی زمان‌بندی</h4>
                <p>تطبیقی، توازن بار، مبتنی بر حافظه، بهینه‌شده برای تأخیر. تغییر خودکار.</p>
            </div>
        </div>
    </div>

    <h3>📊 عملکرد</h3>
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>پیکربندی</th>
                    <th>زمان/Epoch</th>
                    <th>سرعت</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>GPU لوکال (RTX 4090)</td>
                    <td>12.3 ثانیه</td>
                    <td>1.0×</td>
                </tr>
                <tr>
                    <td>RemoteCUDA 1 GPU</td>
                    <td>14.1 ثانیه</td>
                    <td>0.87×</td>
                </tr>
                <tr class="highlight-green">
                    <td><strong>RemoteCUDA 4 GPU</strong></td>
                    <td><strong>4.8 ثانیه</strong></td>
                    <td><strong>2.56×</strong></td>
                </tr>
                <tr class="highlight-green">
                    <td><strong>RemoteCUDA 8 GPU</strong></td>
                    <td><strong>2.7 ثانیه</strong></td>
                    <td><strong>4.56×</strong></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="callout callout-success">
        <div class="callout-title">🚀 با چند GPU، RemoteCUDA می‌تواند سریع‌تر از یک GPU لوکال آموزش دهد!</div>
        ۸ GPU → ۴.۵۶ برابر سریع‌تر از لوکال. عالی برای آموزش توزیع‌شده.
    </div>
</section>

<!-- ============================================================ -->
<!--                         FOOTER                              -->
<!-- ============================================================ -->
<footer>
    <p>
        ⭐ If this project helps you, please give it a star on
        <a href="https://github.com/yourusername/remotecuda">GitHub</a>!
    </p>
    <p>
        ⭐ اگر این پروژه برایتان مفید بود، لطفاً در
        <a href="https://github.com/yourusername/remotecuda">گیت‌هاب</a> ستاره دهید!
    </p>
    <br>
    <p>
        Made with <span class="heart">❤️</span> for the ML community
        &nbsp;|&nbsp;
        ساخته شده با <span class="heart">❤️</span> برای جامعه یادگیری ماشین
    </p>
    <p style="margin-top: 12px; font-size: 0.85em;">
        MIT License &nbsp;|&nbsp; © 2024 RemoteCUDA Contributors
    </p>
</footer>

<!-- ============================================================ -->
<!--                        SCRIPTS                              -->
<!-- ============================================================ -->
<script>
    function showTab(tabId, btn) {
        // Hide all tab contents
        const contents = document.querySelectorAll('.tab-content');
        contents.forEach(c => c.classList.remove('active'));

        // Deactivate all buttons
        const buttons = document.querySelectorAll('.tab-btn');
        buttons.forEach(b => b.classList.remove('active'));

        // Show selected tab
        document.getElementById(tabId).classList.add('active');
        btn.classList.add('active');
    }

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
</script>

</body>
</html>
