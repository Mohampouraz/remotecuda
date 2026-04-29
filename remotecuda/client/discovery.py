"""
Auto-Discovery Module — Pure Python Standard Library Only
==========================================================
Zero-dependency UDP multicast server discovery.

This module uses ONLY Python standard library:
    - socket     (UDP multicast)
    - json       (message parsing)
    - time       (timeouts)
    - threading  (background discovery)

Protocol:
    Servers broadcast heartbeat messages every 2 seconds on:
        Multicast Group: 239.255.100.100
        Multicast Port:  55556

    Message format (JSON):
    {
        "service": "RemoteCUDA",
        "server_id": "abc12345",
        "port": 55555,
        "version": "3.0.0",
        "device": "cuda:0",
        "gpu_count": 1,
        "timestamp": 1234567890.123
    }

    Clients listen for these broadcasts to discover available servers.
    After receiving a broadcast, the client verifies reachability
    with a TCP health check before returning the server address.

Usage:
    from remotecuda.client.discovery import discover_server

    server = discover_server()
    if server:
        host, port = server
        print(f"Found server at {host}:{port}")
    else:
        print("No server found")

    # Discover all servers
    from remotecuda.client.discovery import discover_all_servers
    servers = discover_all_servers()
    for host, port in servers:
        print(f"Server: {host}:{port}")
"""

import socket
import json
import time
import threading
from typing import Optional, Tuple, List, Dict


# ============================================================
#  Constants
# ============================================================

# Multicast configuration
MULTICAST_GROUP = '239.255.100.100'
MULTICAST_PORT = 55556
SERVICE_NAME = 'RemoteCUDA'
SERVICE_VERSION = '3.0'

# Timing
DEFAULT_DISCOVERY_TIMEOUT = 3.0  # seconds
VERIFY_TIMEOUT = 2.0             # seconds for TCP verification
BROADCAST_LISTEN_INTERVAL = 1.0  # seconds between socket polls

# Network
RECV_BUFFER_SIZE = 4096  # bytes
MULTICAST_TTL = 2        # restrict to local network


# ============================================================
#  Server Discovery Functions
# ============================================================

def discover_server(
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    verify: bool = True
) -> Optional[Tuple[str, int]]:
    """
    Discover a single RemoteCUDA server on the local network.

    Listens for UDP multicast heartbeat broadcasts from servers.
    Returns the first reachable server found.

    Pure Python implementation — no external dependencies.

    Args:
        timeout: Maximum time to wait for discovery (seconds).
                 Longer timeouts find more servers.
        verify: If True, verifies server reachability with a TCP
                health check before returning.

    Returns:
        Optional[Tuple[str, int]]:
            - (host, port) of the discovered server, or
            - None if no server was found within the timeout.

    Example:
        >>> server = discover_server()
        >>> if server:
        ...     host, port = server
        ...     from remotecuda.client.connection import ClientConnection
        ...     conn = ClientConnection(host, port)
        ...     conn.connect()
    """
    sock = _create_multicast_listener()

    if sock is None:
        return None

    start_time = time.time()
    result = None

    while time.time() - start_time < timeout:
        try:
            data, addr = sock.recvfrom(RECV_BUFFER_SIZE)

            message = _parse_broadcast(data)
            if message is None:
                continue

            server_host = addr[0]
            server_port = message.get('port', 55555)

            if verify:
                if _verify_server(server_host, server_port, timeout=VERIFY_TIMEOUT):
                    result = (server_host, server_port)
                    break
            else:
                result = (server_host, server_port)
                break

        except socket.timeout:
            continue
        except OSError:
            continue
        except Exception:
            continue

    _close_socket(sock)
    return result


def discover_all_servers(
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    verify: bool = True
) -> List[Tuple[str, int]]:
    """
    Discover all RemoteCUDA servers on the local network.

    Listens for broadcasts from multiple servers and returns
    all unique, reachable servers found.

    Args:
        timeout: Maximum time to listen for broadcasts (seconds).
        verify: If True, verifies each server's reachability.

    Returns:
        List[Tuple[str, int]]: List of (host, port) tuples.
                               Empty list if no servers found.

    Example:
        >>> servers = discover_all_servers()
        >>> for host, port in servers:
        ...     print(f"Server: {host}:{port}")
    """
    sock = _create_multicast_listener()

    if sock is None:
        return []

    start_time = time.time()
    servers: Dict[str, Tuple[str, int]] = {}
    seen: set = set()

    while time.time() - start_time < timeout:
        try:
            data, addr = sock.recvfrom(RECV_BUFFER_SIZE)

            message = _parse_broadcast(data)
            if message is None:
                continue

            server_key = f"{addr[0]}:{message.get('port', 55555)}"

            if server_key not in seen:
                seen.add(server_key)
                server_host = addr[0]
                server_port = message.get('port', 55555)

                if verify:
                    if _verify_server(server_host, server_port, timeout=VERIFY_TIMEOUT):
                        servers[server_key] = (server_host, server_port)
                else:
                    servers[server_key] = (server_host, server_port)

        except socket.timeout:
            continue
        except OSError:
            continue
        except Exception:
            continue

    _close_socket(sock)
    return list(servers.values())


def discover_with_info(
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    verify: bool = True
) -> List[dict]:
    """
    Discover servers with detailed information.

    Returns server metadata in addition to connection details.

    Args:
        timeout: Discovery timeout in seconds.
        verify: Whether to verify reachability.

    Returns:
        List[dict]: List of server info dictionaries:
        [
            {
                'host': '192.168.1.100',
                'port': 55555,
                'server_id': 'abc12345',
                'version': '3.0.0',
                'device': 'cuda:0',
                'gpu_count': 1,
                'gpu_name': 'NVIDIA RTX 4090',
            },
            ...
        ]
    """
    sock = _create_multicast_listener()

    if sock is None:
        return []

    start_time = time.time()
    servers: List[dict] = []
    seen: set = set()

    while time.time() - start_time < timeout:
        try:
            data, addr = sock.recvfrom(RECV_BUFFER_SIZE)

            message = _parse_broadcast(data)
            if message is None:
                continue

            server_key = f"{addr[0]}:{message.get('port', 55555)}"

            if server_key not in seen:
                seen.add(server_key)

                server_host = addr[0]
                server_port = message.get('port', 55555)

                if verify and not _verify_server(server_host, server_port):
                    continue

                servers.append({
                    'host': server_host,
                    'port': server_port,
                    'server_id': message.get('server_id', ''),
                    'version': message.get('version', ''),
                    'device': message.get('device', ''),
                    'gpu_count': message.get('gpu_count', 0),
                    'gpu_name': message.get('gpu_name', ''),
                    'hostname': message.get('hostname', ''),
                })

        except socket.timeout:
            continue
        except OSError:
            continue
        except Exception:
            continue

    _close_socket(sock)
    return servers


# ============================================================
#  Continuous Discovery (Background Thread)
# ============================================================

class NetworkScanner:
    """
    Background network scanner that continuously monitors for GPU servers.

    Runs a daemon thread that listens for server broadcasts and
    maintains an up-to-date registry of available servers.

    Features:
        - Continuous monitoring
        - Automatic server join/leave detection
        - Callback notifications
        - Server timeout detection

    Usage:
        scanner = NetworkScanner()
        scanner.on_server_found = lambda host, port, info: print(f"Found: {host}")
        scanner.on_server_lost = lambda host, port: print(f"Lost: {host}")
        scanner.start()

        # Get current servers
        servers = scanner.get_servers()

        scanner.stop()
    """

    def __init__(self, server_timeout: float = 10.0):
        """
        Initialize the network scanner.

        Args:
            server_timeout: Time in seconds before a server is considered
                           lost (no heartbeat received).
        """
        self.server_timeout = server_timeout
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._servers: Dict[str, dict] = {}

        # Callbacks
        self.on_server_found = None   # Callable(host, port, info)
        self.on_server_lost = None    # Callable(host, port)
        self.on_server_updated = None # Callable(host, port, info)

    def start(self):
        """
        Start the background scanner thread.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._scan_loop,
            daemon=True,
            name="NetworkScanner"
        )
        self._thread.start()

    def stop(self):
        """
        Stop the background scanner thread.

        Safe to call multiple times.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def get_servers(self) -> List[dict]:
        """
        Get the list of currently known servers.

        Returns:
            List[dict]: Server information dictionaries.
        """
        with self._lock:
            return [
                {
                    'host': info['host'],
                    'port': info['port'],
                    'server_id': info.get('server_id', ''),
                    'version': info.get('version', ''),
                    'device': info.get('device', ''),
                    'gpu_count': info.get('gpu_count', 0),
                    'gpu_name': info.get('gpu_name', ''),
                    'last_seen': info.get('last_seen', 0),
                    'is_verified': info.get('is_verified', False),
                }
                for info in self._servers.values()
            ]

    def get_server_count(self) -> int:
        """Get the number of known servers."""
        with self._lock:
            return len(self._servers)

    def _scan_loop(self):
        """Main scanning loop."""
        sock = _create_multicast_listener()
        if sock is None:
            return

        while self._running:
            try:
                data, addr = sock.recvfrom(RECV_BUFFER_SIZE)

                message = _parse_broadcast(data)
                if message is None:
                    continue

                server_key = f"{addr[0]}:{message.get('port', 55555)}"
                server_host = addr[0]
                server_port = message.get('port', 55555)

                is_new = False

                with self._lock:
                    if server_key not in self._servers:
                        is_new = True
                        self._servers[server_key] = {}

                    self._servers[server_key].update({
                        'host': server_host,
                        'port': server_port,
                        'server_id': message.get('server_id', ''),
                        'version': message.get('version', ''),
                        'device': message.get('device', ''),
                        'gpu_count': message.get('gpu_count', 0),
                        'gpu_name': message.get('gpu_name', ''),
                        'hostname': message.get('hostname', ''),
                        'last_seen': time.time(),
                    })

                if is_new and self.on_server_found:
                    try:
                        self.on_server_found(server_host, server_port, message)
                    except Exception:
                        pass

                if not is_new and self.on_server_updated:
                    try:
                        self.on_server_updated(server_host, server_port, message)
                    except Exception:
                        pass

            except socket.timeout:
                pass
            except OSError:
                pass
            except Exception:
                pass

            # Purge timed-out servers
            self._purge_stale_servers()

        _close_socket(sock)

    def _purge_stale_servers(self):
        """Remove servers that haven't broadcast recently."""
        now = time.time()
        stale_keys = []

        with self._lock:
            for key, info in self._servers.items():
                if now - info.get('last_seen', 0) > self.server_timeout:
                    stale_keys.append(key)

            for key in stale_keys:
                info = self._servers.pop(key)
                if self.on_server_lost:
                    try:
                        self.on_server_lost(info['host'], info['port'])
                    except Exception:
                        pass


# ============================================================
#  Internal Helpers
# ============================================================

def _create_multicast_listener() -> Optional[socket.socket]:
    """
    Create and configure a UDP multicast listener socket.

    Returns:
        Optional[socket.socket]: Configured socket, or None on failure.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Allow multiple listeners on the same port
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass  # Not available on all platforms

        # Bind to the multicast port
        try:
            sock.bind(('', MULTICAST_PORT))
        except OSError:
            sock.bind(('0.0.0.0', MULTICAST_PORT))

        # Join the multicast group
        try:
            mreq = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton('0.0.0.0')
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError:
            # Some platforms handle multicast differently — try to continue
            pass

        # Set multicast TTL
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
        except OSError:
            pass

        # Set timeout for non-blocking receive
        sock.settimeout(BROADCAST_LISTEN_INTERVAL)

        return sock

    except OSError as e:
        print(f"Warning: Failed to create multicast listener: {e}")
        return None


def _parse_broadcast(data: bytes) -> Optional[dict]:
    """
    Parse a received broadcast message.

    Args:
        data: Raw bytes received via UDP.

    Returns:
        Optional[dict]: Parsed message dictionary, or None if invalid.
    """
    try:
        message = json.loads(data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    # Validate service name
    if message.get('service') != SERVICE_NAME:
        return None

    # Validate version compatibility (major version must match)
    version = message.get('version', '0.0.0')
    try:
        major_version = int(version.split('.')[0])
    except (ValueError, IndexError):
        return None

    # Accept v3.x.x servers
    if major_version < 3:
        return None

    return message


def _verify_server(host: str, port: int, timeout: float = VERIFY_TIMEOUT) -> bool:
    """
    Verify that a server is reachable via TCP.

    Performs a quick TCP connect/disconnect to check if the server
    is actually listening and responsive.

    Args:
        host: Server hostname or IP.
        port: Server port.
        timeout: Connection timeout in seconds.

    Returns:
        bool: True if server accepts TCP connection.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False
    except Exception:
        return False


def _close_socket(sock: Optional[socket.socket]):
    """Safely close a socket."""
    if sock:
        try:
            sock.close()
        except Exception:
            pass