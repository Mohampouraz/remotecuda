"""
Compression Module — Optional Data Compression
===============================================
Provides optional compression for large tensor data transfers.

Supports:
    - zlib (built-in, always available)
    - lz4 (faster, optional)
    - zstandard (best compression, optional)

The compression is transparent — if enabled, data is compressed
before transmission and decompressed on receipt.

Usage:
    from remotecuda.protocol.compression import compress, decompress

    data = b"large binary data..."
    compressed = compress(data)
    original = decompress(compressed)
"""

import zlib
from typing import Optional


# Try to import optional compression libraries
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# Compression level constants
COMPRESSION_NONE = 0
COMPRESSION_ZLIB = 1
COMPRESSION_LZ4 = 2
COMPRESSION_ZSTD = 3

# Default settings
DEFAULT_COMPRESSION = COMPRESSION_ZLIB
COMPRESSION_THRESHOLD = 1024  # bytes — minimum size for compression
ZLIB_LEVEL = 3  # 1=fastest, 9=best compression


def compress(data: bytes, method: int = DEFAULT_COMPRESSION) -> bytes:
    """
    Compress binary data.

    Prefixes the compressed data with a 1-byte method identifier.

    Args:
        data: Raw bytes to compress.
        method: Compression method to use.

    Returns:
        bytes: Compressed data with method prefix byte.

    Example:
        >>> compressed = compress(b"hello world" * 100)
        >>> compressed[0]  # method byte
        1
        >>> len(compressed) < len(b"hello world" * 100)
        True
    """
    # Don't compress small data
    if len(data) < COMPRESSION_THRESHOLD:
        return bytes([COMPRESSION_NONE]) + data

    if method == COMPRESSION_NONE:
        return bytes([COMPRESSION_NONE]) + data

    elif method == COMPRESSION_LZ4 and HAS_LZ4:
        try:
            compressed = lz4.frame.compress(data)
            return bytes([COMPRESSION_LZ4]) + compressed
        except Exception:
            pass

    elif method == COMPRESSION_ZSTD and HAS_ZSTD:
        try:
            compressed = zstandard.compress(data)
            return bytes([COMPRESSION_ZSTD]) + compressed
        except Exception:
            pass

    # Default: zlib (always available)
    compressed = zlib.compress(data, level=ZLIB_LEVEL)
    return bytes([COMPRESSION_ZLIB]) + compressed


def decompress(data: bytes) -> bytes:
    """
    Decompress binary data.

    Reads the 1-byte method identifier prefix to determine
    how to decompress.

    Args:
        data: Compressed data with method prefix byte.

    Returns:
        bytes: Decompressed data.

    Example:
        >>> original = b"hello world" * 100
        >>> compressed = compress(original)
        >>> decompressed = decompress(compressed)
        >>> original == decompressed
        True
    """
    if not data:
        return data

    method = data[0]
    payload = data[1:]

    if method == COMPRESSION_NONE:
        return payload

    elif method == COMPRESSION_LZ4 and HAS_LZ4:
        try:
            return lz4.frame.decompress(payload)
        except Exception:
            pass

    elif method == COMPRESSION_ZSTD and HAS_ZSTD:
        try:
            return zstandard.decompress(payload)
        except Exception:
            pass

    elif method == COMPRESSION_ZLIB:
        try:
            return zlib.decompress(payload)
        except zlib.error:
            pass

    # Fallback: return raw payload
    return payload


def get_compression_info() -> dict:
    """
    Get information about available compression methods.

    Returns:
        dict: Available compression methods and their status.

    Example:
        >>> info = get_compression_info()
        >>> print(info['available'])
        ['zlib']
    """
    available = ['zlib']  # Always available

    if HAS_LZ4:
        available.append('lz4')

    if HAS_ZSTD:
        available.append('zstandard')

    return {
        'available': available,
        'threshold': COMPRESSION_THRESHOLD,
        'zlib_level': ZLIB_LEVEL,
    }


def get_compression_ratio(original: bytes, compressed: bytes) -> float:
    """
    Calculate compression ratio.

    Args:
        original: Original uncompressed data.
        compressed: Compressed data.

    Returns:
        float: Compression ratio (original/compressed).
               > 1.0 means compression helped.

    Example:
        >>> ratio = get_compression_ratio(original, compressed)
        >>> print(f"Compression ratio: {ratio:.2f}x")
    """
    if len(compressed) == 0:
        return 0.0
    return len(original) / len(compressed)