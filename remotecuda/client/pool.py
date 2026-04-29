"""
GPU Connection Pool Module — Pure Python Client
=================================================
Manages multiple GPU server connections for the client.

This module is optional but recommended for:
    - Multiple GPU server environments
    - Load balancing across servers
    - Automatic failover
    - Parallel task execution

Pure Python implementation — no external dependencies.

Architecture:
    ┌─────────────────────────────────────────┐
    │              Connection Pool             │
    │                                          │
    │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
    │  │ Server 1 │  │ Server 2 │  │ Ser N  │ │
    │  │ 10.0.0.1 │  │ 10.0.0.2 │  │  ...   │ │
    │  └────┬─────┘  └────┬─────┘  └───┬────┘ │
    │       │              │            │      │
    │  ┌────▼──────────────▼────────────▼────┐ │
    │  │         Load Balancer               │ │
    │  │  - Round Robin                      │ │
    │  │  - Least Connections                │ │
    │  │  - Memory Aware                     │ │
    │  └─────────────────────────────────────┘ │
    └─────────────────────────────────────────┘

Usage:
    from remotecuda.client.pool import ConnectionPool

    pool = ConnectionPool()
    pool.auto_discover()
    pool.connect_all()

    # Get best connection
    conn = pool.get_connection()

    # Or iterate all
    for conn in pool:
        result = conn.send_command('ping', {})

    pool.disconnect_all()
"""

import threading
import time
from typing import Dict, List, Optional, Iterator

from .connection import ClientConnection
from .discovery import discover_all_servers, NetworkScanner


class ConnectionPool:
    """
    Manages multiple GPU server connections.

    Provides load-balanced access to multiple servers,
    automatic health checking, and failover.

    Features:
        - Auto-discovery of servers on the network
        - Manual server addition
        - Load-balanced connection selection
        - Health monitoring
        - Thread-safe operations
        - Context manager support

    Attributes:
        connections (dict): Active connections keyed by "host:port".
        strategy (str): Load balancing strategy name.

    Usage:
        pool = ConnectionPool()
        pool.auto_discover(timeout=3.0)
        pool.connect_all()

        conn = pool.get_connection()
        result = conn.send_command('zeros', {'shape': [100, 100]})

        pool.disconnect_all()
    """

    # Load balancing strategies
    STRATEGY_ROUND_ROBIN = 'round_robin'
    STRATEGY_LEAST_CONNECTIONS = 'least_connections'
    STRATEGY_RANDOM = 'random'

    def __init__(self, strategy: str = STRATEGY_ROUND_ROBIN):
        """
        Initialize the connection pool.

        Args:
            strategy: Load balancing strategy.
                     Options: 'round_robin', 'least_connections', 'random'.
        """
        self._connections: Dict[str, ClientConnection] = {}
        self._lock = threading.RLock()
        self._strategy = strategy
        self._round_robin_index = 0
        self._scanner: Optional[NetworkScanner] = None
        self._auto_discover_running = False

        # Statistics
        self._stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'commands_routed': 0,
        }

    # ============================================================
    #  Properties
    # ============================================================

    @property
    def connection_count(self) -> int:
        """Number of connections in the pool."""
        return len(self._connections)

    @property
    def active_count(self) -> int:
        """Number of currently active (connected) connections."""
        return sum(1 for c in self._connections.values() if c.is_connected)

    @property
    def connections(self) -> Dict[str, ClientConnection]:
        """Dictionary of all connections."""
        return dict(self._connections)

    # ============================================================
    #  Server Discovery and Connection
    # ============================================================

    def auto_discover(self, timeout: float = 3.0) -> List[tuple]:
        """
        Discover GPU servers on the local network.

        Uses UDP multicast to find servers broadcasting their presence.

        Args:
            timeout: Time to wait for server responses (seconds).

        Returns:
            List[tuple]: List of (host, port) tuples found.

        Example:
            >>> pool = ConnectionPool()
            >>> servers = pool.auto_discover()
            >>> for host, port in servers:
            ...     print(f"Found: {host}:{port}")
        """
        servers = discover_all_servers(timeout=timeout)

        if servers:
            print(f"Discovered {len(servers)} server(s):")
            for host, port in servers:
                print(f"  {host}:{port}")
        else:
            print("No servers found on the network.")

        return servers

    def start_auto_discovery(self, callback=None):
        """
        Start continuous background server discovery.

        Automatically connects to newly discovered servers
        and removes disconnected ones.

        Args:
            callback: Optional function called when servers change.
                     callback(action, host, port)

        Example:
            >>> def on_change(action, host, port):
            ...     print(f"{action}: {host}:{port}")
            >>> pool.start_auto_discovery(callback=on_change)
        """
        if self._auto_discover_running:
            return

        self._auto_discover_running = True
        self._scanner = NetworkScanner()

        def on_found(host, port, info):
            if f"{host}:{port}" not in self._connections:
                try:
                    self.add_server(host, port)
                    if callback:
                        callback('added', host, port)
                except Exception:
                    pass

        def on_lost(host, port):
            key = f"{host}:{port}"
            if key in self._connections:
                self.remove_server(host, port)
                if callback:
                    callback('removed', host, port)

        self._scanner.on_server_found = on_found
        self._scanner.on_server_lost = on_lost
        self._scanner.start()

    def stop_auto_discovery(self):
        """Stop background auto-discovery."""
        self._auto_discover_running = False
        if self._scanner:
            self._scanner.stop()
            self._scanner = None

    def add_server(self, host: str, port: int = 55555) -> bool:
        """
        Add a server to the pool and connect to it.

        Args:
            host: Server IP address or hostname.
            port: Server port number.

        Returns:
            bool: True if connection was successful.

        Example:
            >>> pool.add_server('192.168.1.100', 55555)
        """
        key = f"{host}:{port}"

        with self._lock:
            if key in self._connections:
                if self._connections[key].is_connected:
                    return True
                else:
                    # Remove stale connection
                    del self._connections[key]

        try:
            conn = ClientConnection(host, port)
            conn.connect()
            with self._lock:
                self._connections[key] = conn
                self._stats['total_connections'] += 1
                self._stats['active_connections'] += 1
            return True
        except Exception as e:
            self._stats['failed_connections'] += 1
            raise ConnectionError(f"Failed to connect to {host}:{port}: {e}")

    def remove_server(self, host: str, port: int = 55555):
        """
        Remove a server from the pool.

        Args:
            host: Server IP or hostname.
            port: Server port.
        """
        key = f"{host}:{port}"

        with self._lock:
            conn = self._connections.pop(key, None)
            if conn:
                try:
                    conn.disconnect()
                except Exception:
                    pass
                self._stats['active_connections'] -= 1

    def connect_all(self, servers: List[tuple] = None):
        """
        Connect to all discovered or specified servers.

        Args:
            servers: List of (host, port) tuples.
                    If None, uses auto_discover().

        Example:
            >>> pool.connect_all()
            >>> # Or with specific servers:
            >>> pool.connect_all([('10.0.0.1', 55555), ('10.0.0.2', 55555)])
        """
        if servers is None:
            servers = self.auto_discover()

        for host, port in servers:
            try:
                self.add_server(host, port)
            except Exception as e:
                print(f"Warning: Could not connect to {host}:{port}: {e}")

    def disconnect_all(self):
        """Disconnect from all servers and stop auto-discovery."""
        self.stop_auto_discovery()

        with self._lock:
            for key, conn in list(self._connections.items()):
                try:
                    conn.disconnect()
                except Exception:
                    pass
            self._connections.clear()
            self._stats['active_connections'] = 0

    # ============================================================
    #  Connection Selection (Load Balancing)
    # ============================================================

    def get_connection(self, strategy: str = None) -> ClientConnection:
        """
        Get a connection using the configured load balancing strategy.

        Args:
            strategy: Override the default strategy for this call.
                     Options: 'round_robin', 'least_connections', 'random'.

        Returns:
            ClientConnection: The selected connection.

        Raises:
            RuntimeError: If no connections are available.

        Example:
            >>> conn = pool.get_connection()
            >>> result = conn.send_command('ping', {})
        """
        strategy = strategy or self._strategy

        with self._lock:
            active = {
                k: c for k, c in self._connections.items()
                if c.is_connected
            }

            if not active:
                raise RuntimeError(
                    "No active connections available. "
                    "Add servers with add_server() or connect_all()."
                )

            if strategy == self.STRATEGY_ROUND_ROBIN:
                conn = self._select_round_robin(active)
            elif strategy == self.STRATEGY_LEAST_CONNECTIONS:
                conn = self._select_least_connections(active)
            elif strategy == self.STRATEGY_RANDOM:
                conn = self._select_random(active)
            else:
                conn = self._select_round_robin(active)

            self._stats['commands_routed'] += 1
            return conn

    def _select_round_robin(self, active: dict) -> ClientConnection:
        """Round-robin selection."""
        keys = list(active.keys())
        if not keys:
            raise RuntimeError("No active connections")

        self._round_robin_index = self._round_robin_index % len(keys)
        selected = active[keys[self._round_robin_index]]
        self._round_robin_index += 1
        return selected

    def _select_least_connections(self, active: dict) -> ClientConnection:
        """Select connection with fewest commands sent."""
        best_key = None
        best_count = float('inf')

        for key, conn in active.items():
            count = conn.command_count
            if count < best_count:
                best_count = count
                best_key = key

        return active[best_key]

    def _select_random(self, active: dict) -> ClientConnection:
        """Random selection."""
        import random
        keys = list(active.keys())
        return active[random.choice(keys)]

    # ============================================================
    #  Health Check
    # ============================================================

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all connections.

        Returns:
            Dict[str, bool]: Connection health status keyed by "host:port".

        Example:
            >>> health = pool.health_check()
            >>> for key, ok in health.items():
            ...     print(f"{key}: {'OK' if ok else 'DOWN'}")
        """
        results = {}

        with self._lock:
            for key, conn in self._connections.items():
                results[key] = conn.ping()

        return results

    def remove_unhealthy(self):
        """Remove all unhealthy connections."""
        health = self.health_check()

        for key, ok in health.items():
            if not ok:
                host, port = key.split(':')
                self.remove_server(host, int(port))

    # ============================================================
    #  Statistics
    # ============================================================

    def get_stats(self) -> dict:
        """
        Get pool statistics.

        Returns:
            dict: Pool statistics.
        """
        with self._lock:
            return {
                **self._stats,
                'total_connections': len(self._connections),
                'active_connections': self.active_count,
                'strategy': self._strategy,
                'connections': {
                    key: {
                        'host': conn.host,
                        'port': conn.port,
                        'connected': conn.is_connected,
                        'commands': conn.command_count,
                        'duration': conn.connection_duration,
                    }
                    for key, conn in self._connections.items()
                },
            }

    # ============================================================
    #  Bulk Operations
    # ============================================================

    def broadcast_command(self, command: str, params: dict) -> Dict[str, any]:
        """
        Send a command to ALL connected servers.

        Useful for gathering info from all servers simultaneously.

        Args:
            command: Command to send.
            params: Command parameters.

        Returns:
            Dict[str, any]: Results keyed by "host:port".

        Example:
            >>> infos = pool.broadcast_command('info', {})
            >>> for server, info in infos.items():
            ...     print(f"{server}: {info.get('device')}")
        """
        results = {}

        with self._lock:
            connections = list(self._connections.items())

        for key, conn in connections:
            try:
                if conn.is_connected:
                    results[key] = conn.send_command(command, params)
                else:
                    results[key] = {'error': 'Not connected'}
            except Exception as e:
                results[key] = {'error': str(e)}

        return results

    # ============================================================
    #  Context Manager & Iteration
    # ============================================================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — auto-disconnect."""
        self.disconnect_all()
        return False

    def __iter__(self) -> Iterator[ClientConnection]:
        """Iterate over all connected endpoints."""
        with self._lock:
            connections = [
                c for c in self._connections.values()
                if c.is_connected
            ]
        return iter(connections)

    def __len__(self) -> int:
        """Number of connections in the pool."""
        return self.connection_count

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConnectionPool({self.active_count} active, "
            f"{self.connection_count} total, "
            f"strategy={self._strategy})"
        )

    def __str__(self) -> str:
        """User-friendly string."""
        return f"ConnectionPool with {self.active_count} active connection(s)"

    # ============================================================
    #  Destructor
    # ============================================================

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.disconnect_all()
        except Exception:
            pass