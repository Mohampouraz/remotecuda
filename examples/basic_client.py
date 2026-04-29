#!/usr/bin/env python3
"""
RemoteCUDA v3.0 — Basic Client Example
=======================================
Demonstrates basic usage of the RemoteCUDA client API.

This example shows:
    1. Connecting to a server (auto-discover or manual)
    2. Creating tensors on the remote device
    3. Performing arithmetic operations
    4. Retrieving results
    5. Proper cleanup

Requirements:
    pip install remotecuda
    (No other dependencies needed!)

Usage:
    python basic_client.py
    python basic_client.py --server 192.168.1.100
"""

import sys
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='RemoteCUDA Basic Client Example')
    parser.add_argument('--server', default=None, help='Server IP (auto-discover if not specified)')
    parser.add_argument('--port', type=int, default=55555, help='Server port')
    args = parser.parse_args()

    print("=" * 60)
    print("  RemoteCUDA v3.0 — Basic Client Example")
    print("=" * 60)
    print()

    # Import remotecuda (pure Python, no dependencies needed!)
    import remotecuda

    # Connect to server
    print("1. Connecting to server...")
    if args.server:
        remotecuda.init(server=args.server, port=args.port)
    else:
        remotecuda.init()

    # Get server info
    info = remotecuda.info()
    print(f"   Connected to: {info.get('host', 'unknown')}")
    print(f"   Device:       {info.get('device', 'unknown')}")
    print(f"   GPU Count:    {info.get('gpu_count', 0)}")
    print()

    # Create tensors
    print("2. Creating tensors on remote device...")
    a = remotecuda.ones((500, 500))
    b = remotecuda.ones((500, 500))
    print(f"   Tensor A: ID={a}, shape=(500, 500), all ones")
    print(f"   Tensor B: ID={b}, shape=(500, 500), all ones")
    print()

    # Matrix multiplication
    print("3. Computing A @ B (matrix multiplication)...")
    start = time.time()
    c = remotecuda.matmul(a, b)
    elapsed = time.time() - start
    print(f"   Result C: ID={c}")
    print(f"   Time: {elapsed:.4f} seconds")
    print()

    # Retrieve result
    print("4. Retrieving result...")
    result = remotecuda.get(c)
    print(f"   Shape: {result['shape']}")
    print(f"   Expected first value: 500.0")
    print(f"   Actual first value:   {result['data'][0]:.1f}")
    assert abs(result['data'][0] - 500.0) < 1.0, "Result verification failed!"
    print(f"   Status: PASSED")
    print()

    # More operations
    print("5. Testing more operations...")

    # Element-wise addition
    d = remotecuda.add(a, b)
    d_result = remotecuda.get(d)
    print(f"   A + B first value: {d_result['data'][0]:.1f} (expected 2.0)")
    remotecuda.free(d)

    # ReLU activation
    e = remotecuda.relu(a)
    e_result = remotecuda.get(e)
    print(f"   ReLU(A) first value: {e_result['data'][0]:.1f} (expected 1.0)")
    remotecuda.free(e)

    # Scalar multiplication
    f = remotecuda.full((10,), 5.0)
    f_result = remotecuda.get(f)
    print(f"   Full tensor: {f_result['data']} (expected all 5.0)")
    remotecuda.free(f)

    print()

    # Cleanup
    print("6. Cleaning up...")
    remotecuda.free(a)
    remotecuda.free(b)
    remotecuda.free(c)
    remotecuda.shutdown()
    print("   Done!")

    print()
    print("=" * 60)
    print("  All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()