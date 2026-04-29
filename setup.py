"""
RemoteCUDA v3.0 — Ultimate Zero-Dependency Architecture
========================================================
CLIENT: Pure Python. No NumPy. No PyTorch. No CUDA. Nothing.
SERVER: PyTorch (CUDA preferred, auto-CPU fallback).

Architecture:
    Client (Pure Python)
        │
        │ TCP Socket + JSON Protocol (no pickle, no numpy)
        ▼
    Server (PyTorch)
        │
        ├── GPU available? → CUDA tensors
        └── GPU unavailable? → CPU tensors (auto fallback)
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="remotecuda",
    version="3.1.0",
    author="Abolfazl Mohammadpour",
    author_email="Mohampouraz@Gmail.com",
    description="Remote GPU — pure Python client, zero dependencies, auto CPU fallback",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohampouraz/remotecuda",
    project_urls={
        "Documentation": "https://github.com/remotecuda/remotecuda#readme",
        "Source": "https://github.com/remotecuda/remotecuda",
        "Issues": "https://github.com/remotecuda/remotecuda/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    # NO install_requires — client is pure Python!
    install_requires=[],
    extras_require={
        "server": [
            "torch>=1.10.0",
            "numpy>=1.20.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "build>=0.7",
            "twine>=3.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "remotecuda=remotecuda.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)