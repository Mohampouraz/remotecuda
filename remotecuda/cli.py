#!/usr/bin/env python3
"""
RemoteCUDA v3.0 — Command Line Interface
========================================
Pure Python client, zero dependencies.
Server requires PyTorch (auto GPU/CPU fallback).

Server Commands:
    remotecuda start              Start GPU service (auto GPU/CPU)
    remotecuda start --port 8888  Custom port

Client Commands (pure Python, nothing needed):
    remotecuda init               Auto-discover and connect
    remotecuda status             Show server info
    remotecuda shutdown           Disconnect
    remotecuda test               Test server connection
    remotecuda demo               Run demonstration
"""

import argparse
import sys
import json


def cmd_start(args):
    """Start the GPU/CPU computation server."""
    try:
        from remotecuda.server.gpu_service import GPUService
    except ImportError as e:
        print("ERROR: Server dependencies not installed.")
        print("Install with: pip install remotecuda[server]")
        print(f"Details: {e}")
        sys.exit(1)

    service = GPUService(port=args.port, host=args.host)
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        service.stop()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def cmd_init(args):
    """Initialize client connection."""
    import remotecuda

    try:
        remotecuda.init(
            server=args.server if args.server else 'auto',
            port=args.port,
        )
        print("Connected! Use remotecuda.status() for details.")
        print("Press Ctrl+C to shutdown.")

        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        remotecuda.shutdown()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def cmd_status(args):
    """Show server status."""
    import remotecuda
    remotecuda.status()


def cmd_shutdown(args):
    """Disconnect from server."""
    import remotecuda
    remotecuda.shutdown()


def cmd_test(args):
    """Test server connection."""
    import remotecuda

    print(f"Testing connection to {args.server}:{args.port}...")

    try:
        remotecuda.init(server=args.server, port=args.port)
        info = remotecuda.info()
        print(f"  Device: {info.get('device', '?')}")
        print(f"  GPU Count: {info.get('gpu_count', 0)}")

        # Create test tensor
        t1 = remotecuda.ones((100, 100))
        t2 = remotecuda.ones((100, 100))
        t3 = remotecuda.add(t1, t2)
        result = remotecuda.get(t3)

        if result['data'][0] == 2.0:
            print("  Compute test: PASSED")
        else:
            print("  Compute test: FAILED")

        remotecuda.free(t1)
        remotecuda.free(t2)
        remotecuda.free(t3)
        remotecuda.shutdown()

        print("  All tests passed!")

    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


def cmd_demo(args):
    """Run a demonstration of remote computation."""
    import remotecuda

    print("""
╔══════════════════════════════════════════════════════════════╗
║              RemoteCUDA v3.0 — Demo                         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    remotecuda.init(server=args.server if args.server else 'auto', port=args.port)
    info = remotecuda.info()
    print(f"Connected to: {info.get('host')}")
    print(f"Device:       {info.get('device')}")
    print()

    # Matrix multiplication demo
    print("Creating 1000x1000 matrices...")
    import time

    a = remotecuda.ones((1000, 1000))
    b = remotecuda.ones((1000, 1000))

    print("Computing A @ B...")
    start = time.time()
    c = remotecuda.matmul(a, b)
    elapsed = time.time() - start
    print(f"Matrix multiplication: {elapsed:.4f} seconds")

    # Get sample result
    result = remotecuda.get(c)
    print(f"Result shape: {result['shape']}")
    print(f"Sample value: {result['data'][0]:.1f} (should be 1000.0)")
    print()

    # Cleanup
    remotecuda.free(a)
    remotecuda.free(b)
    remotecuda.free(c)
    remotecuda.shutdown()
    print("Demo complete!")


def main():
    parser = argparse.ArgumentParser(
        description='RemoteCUDA v3.0 — Pure Python Client, Zero Dependencies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  remotecuda start                    # Start server (GPU/CPU)
  remotecuda init                     # Connect client (auto-discover)
  remotecuda init --server 10.0.0.5   # Connect to specific server
  remotecuda status                   # Show server info
  remotecuda test --server 10.0.0.5   # Test connection
  remotecuda demo                     # Run demo
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # start
    start_p = subparsers.add_parser('start', help='Start computation server')
    start_p.add_argument('--port', type=int, default=55555)
    start_p.add_argument('--host', default='0.0.0.0')

    # init
    init_p = subparsers.add_parser('init', help='Connect to server')
    init_p.add_argument('--server', default=None)
    init_p.add_argument('--port', type=int, default=55555)

    # status
    subparsers.add_parser('status', help='Show server status')

    # shutdown
    subparsers.add_parser('shutdown', help='Disconnect')

    # test
    test_p = subparsers.add_parser('test', help='Test server connection')
    test_p.add_argument('--server', default='localhost')
    test_p.add_argument('--port', type=int, default=55555)

    # demo
    demo_p = subparsers.add_parser('demo', help='Run demonstration')
    demo_p.add_argument('--server', default=None)
    demo_p.add_argument('--port', type=int, default=55555)

    args = parser.parse_args()

    commands = {
        'start': cmd_start,
        'init': cmd_init,
        'status': cmd_status,
        'shutdown': cmd_shutdown,
        'test': cmd_test,
        'demo': cmd_demo,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()