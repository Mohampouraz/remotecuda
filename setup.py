"""
RemoteCUDA v2.0 Setup
=====================
Package configuration for pip installation.
"""

from setuptools import setup, find_packages

setup(
    name="remotecuda",
    version="2.0.0",
    author="RemoteCUDA Team",
    description="Transparent remote GPU access - no code changes needed",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/remotecuda",
    
    packages=find_packages(),
    
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
    ],
    
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)