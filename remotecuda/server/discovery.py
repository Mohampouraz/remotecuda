"""
Network Discovery Module — Server Side
=======================================
UDP multicast broadcasting for server auto-discovery.

The server broadcasts its presence every 2 seconds on:
    Multicast Group: 239.255.100.100
    Multicast Port:  55556

Message Format (JSON):
{
    "service": "RemoteCUDA",
    "server_id": "abc12345",
    "port": 55555,
    "version": "3.0.0",
    "device": "cuda:0",
    "gpu_count": 1,
    "gpu_name": "NVIDIA RTX 4090",
    "hostname": "gpu-server",
    "active_tensors": 5,
    "memory_used_mb": 128.5,
    "uptime_seconds": 3600,
    "connections": 2,
    "timestamp": 1234567890.123
}

Clients listen for these broadcasts to discover the server.
No central registry or configuration needed.
"""

import socket
import json
import time
import threading
from typing import Dict, Optional


# ============================================================
#  Constants
# ============================================================

MULTICAST_GROUP = '239.255.100.100'
MULTICAST_PORT = 55556
MULTICAST_TTL = 2  # Restrict to local network (2 hops)
SERVICE_NAME = 'RemoteCUDA'
BROADCAST_INTERVAL = 2.0  # seconds between broadcasts


class NetworkDiscovery:
    """
    Zero-configuration network discovery via UDP multicast.

    The server periodically broadcasts its presence and current
    status so that clients can discover it without manual IP setup.

    Features:
        - UDP multicast broadcasting
        - Configurable broadcast interval
        - Automatic TTL management
        - Thread-safe operation
        - Graceful start/stop

    Usage:
        discovery = NetworkDiscovery()
        discovery.start_broadcasting(port=55555, gpu_info={...})
        # ... server runs ...
        discovery.stop()
    """

    def __init__(self, service_name: str = SERVICE_NAME):
        """
        Initialize network discovery.

        Args:
            service_name: Name of the service to broadcast.
                         Clients filter by this name.
        """
        self.service_name = service_name
        self.server_id = self._generate_server_id()
        self._running = False
        self._broadcast_thread: Optional[threading.Thread] = None
        self._broadcast_count = 0

    @staticmethod
    def _generate_server_id() -> str:
        """Generate a unique server identifier."""
        import uuid
        return str(uuid.uuid4())[:8]

    def start_broadcasting(self, port: int, gpu_info: dict):
        """
        Start broadcasting the server's presence on the network.

        Spawns a daemon thread that sends heartbeat messages
        every BROADCAST_INTERVAL seconds.

        Args:
            port: TCP port the GPU service is listening on.
            gpu_info: Dictionary with server information to broadcast.
                      Should include: device, gpu_count, gpu_name, etc.

        Raises:
            RuntimeError: If already broadcasting.
        """
        if self._running:
            raise RuntimeError("Already broadcasting. Stop first.")

        self._running = True
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            args=(port, gpu_info),
            daemon=True,
            name="DiscoveryBroadcast"
        )
        self._broadcast_thread.start()

    def _broadcast_loop(self, port: int, gpu_info: dict):
        """
        Main broadcast loop.

        Sends a heartbeat every BROADCAST_INTERVAL seconds.
        Catches and logs errors but continues broadcasting.
        """
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Allow multicast loopback (useful for testing on same machine)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        except OSError:
            pass

        # Target address
        target = (MULTICAST_GROUP, MULTICAST_PORT)

        # Build message template
        message_template = {
            'service': self.service_name,
            'server_id': self.server_id,
            'port': port,
            'gpu_info': gpu_info,
            'timestamp': 0,
            'version': '3.0.0',
        }

        consecutive_errors = 0

        while self._running:
            try:
                # Update timestamp
                message_template['timestamp'] = time.time()

                # Serialize to JSON
                data = json.dumps(
                    message_template,
                    ensure_ascii=False
                ).encode('utf-8')

                # Send broadcast
                sock.sendto(data, target)

                self._broadcast_count += 1
                consecutive_errors = 0  # Reset error counter

            except OSError as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"Broadcast warning: {e}")
                # Don't spam logs for persistent errors

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 1:
                    print(f"Broadcast error: {e}")

            # Sleep until next broadcast
            time.sleep(BROADCAST_INTERVAL)

        sock.close()

    def stop(self):
        """
        Stop broadcasting.

        Signals the broadcast thread to exit and waits for it
        to finish (with a 3-second timeout).
        """
        if not self._running:
            return

        self._running = False

        if self._broadcast_thread and self._broadcast_thread.is_alive():
            self._broadcast_thread.join(timeout=3.0)

        self._broadcast_thread = None

    @property
    def broadcast_count(self) -> int:
        """Total number of broadcasts sent."""
        return self._broadcast_count

    @property
    def is_broadcasting(self) -> bool:
        """Whether the service is currently broadcasting."""
        return self._running