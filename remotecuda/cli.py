#!/usr/bin/env python3
"""
RemoteCUDA Command Line Interface
=================================
Simple commands for both server and client sides.

Server (GPU machine):
    remotecuda start              # Start GPU service
    remotecuda start --port 8888  # Custom port

Client (your machine):
    remotecuda init               # Auto-discover and connect
    remotecuda connect 192.168.1.100  # Connect to specific server
    remotecuda status             # Show current status
    remotecuda shutdown           # Disconnect
"""

import argparse
import sys


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description='RemoteCUDA - Remote GPU Access Made Simple',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPU Machine (Server):
  remotecuda start
  remotecuda start --port 8888
  
  # Your Machine (Client):
  remotecuda init
  remotecuda init --server 192.168.1.100
  remotecuda connect 192.168.1.100
  remotecuda status
  remotecuda shutdown
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Server commands
    start_parser = subparsers.add_parser('start', help='Start GPU service (run on GPU machine)')
    start_parser.add_argument('--port', type=int, default=55555, help='Port (default: 55555)')
    start_parser.add_argument('--host', default='0.0.0.0', help='Bind address')
    
    # Client commands
    init_parser = subparsers.add_parser('init', help='Initialize client (auto-discover GPUs)')
    init_parser.add_argument('--server', help='Specific server to connect to')
    init_parser.add_argument('--port', type=int, default=55555, help='Server port')
    
    connect_parser = subparsers.add_parser('connect', help='Connect to a GPU server')
    connect_parser.add_argument('server', help='Server IP or hostname')
    connect_parser.add_argument('--port', type=int, default=55555, help='Server port')
    
    subparsers.add_parser('status', help='Show current status')
    subparsers.add_parser('shutdown', help='Disconnect and shut down')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        # Start GPU service on this machine
        from .server.gpu_service import GPUService
        service = GPUService(port=args.port, host=args.host)
        service.start()
    
    elif args.command == 'init':
        # Initialize client
        import remotecuda
        remotecuda.init(
            auto_discover=args.server is None,
            server=args.server,
            port=args.port
        )
        
        # Keep running until interrupted
        try:
            print("\n✅ RemoteCUDA is active. Press Ctrl+C to shut down.\n")
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            remotecuda.shutdown()
    
    elif args.command == 'connect':
        import remotecuda
        remotecuda.connect(args.server, args.port)
    
    elif args.command == 'status':
        import remotecuda
        remotecuda.status()
    
    elif args.command == 'shutdown':
        import remotecuda
        remotecuda.shutdown()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()