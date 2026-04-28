"""
GPU Pool Module
===============
Manages multiple GPU connections for parallel processing.
Automatically discovers and connects to all available GPU servers.
"""

import threading
import time
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future

from .connection import GPUConnection
from ..server.discovery import NetworkDiscovery


class GPUPool:
    """
    Manages a pool of GPU connections from multiple servers.
    
    Features:
    - Automatic server discovery
    - Load-balanced GPU allocation
    - Parallel operation execution
    - Health monitoring
    - Failover support
    
    Usage:
        pool = GPUPool()
        pool.auto_discover()  # Find all GPU servers on network
        pool.connect_all()    # Connect to all found servers
        
        # Execute operations in parallel
        results = pool.parallel_execute(operations)
    """
    
    def __init__(self, max_connections: int = 16):
        """
        Initialize GPU pool.
        
        Args:
            max_connections (int): Maximum number of GPU connections to maintain
        """
        self.max_connections = max_connections
        self.connections: Dict[str, GPUConnection] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_connections)
        
        # Discovery
        self.discovery = NetworkDiscovery()
        self.discovery.on_server_found = self._on_server_found
        self.discovery.on_server_lost = self._on_server_lost
        
        # Auto-discovery state
        self._auto_discover_running = False
    
    def auto_discover(self, timeout: float = 5.0) -> List[dict]:
        """
        Automatically discover GPU servers on the local network.
        
        Uses UDP multicast to find servers without manual configuration.
        
        Args:
            timeout (float): How long to wait for server responses
            
        Returns:
            List[dict]: List of discovered server information
        """
        print("🔍 Scanning network for GPU servers...")
        
        # Start discovery
        self.discovery.start_discovery()
        
        # Wait for servers to respond
        time.sleep(timeout)
        
        # Get discovered servers
        discovered = self.discovery.get_available_gpus()
        
        if discovered:
            print(f"✅ Found {len(discovered)} GPUs across {len(set(g['server_id'] for g in discovered))} server(s)")
            for gpu in discovered:
                print(f"   🎮 {gpu['server_ip']}:{gpu['server_port']} - GPU {gpu['gpu_index']}: {gpu['gpu_name']}")
        else:
            print("⚠️  No GPU servers found on the network.")
            print("   Make sure remotecuda-server is running on your GPU machines.")
        
        return discovered
    
    def connect_to(self, host: str, port: int = 55555) -> bool:
        """
        Connect to a specific GPU server.
        
        Args:
            host (str): Server IP or hostname
            port (int): Server port
            
        Returns:
            bool: True if connection successful
        """
        conn_id = f"{host}:{port}"
        
        with self._lock:
            if conn_id in self.connections:
                print(f"Already connected to {conn_id}")
                return True
            
            if len(self.connections) >= self.max_connections:
                print(f"Maximum connections ({self.max_connections}) reached")
                return False
        
        try:
            conn = GPUConnection(host, port)
            conn.connect()
            
            with self._lock:
                self.connections[conn_id] = conn
            
            print(f"✅ Connected to {conn_id}")
            print(f"   GPU: {conn.gpu_name} ({conn.total_memory_gb:.1f} GB)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to {host}:{port}: {e}")
            return False
    
    def connect_all_discovered(self) -> int:
        """
        Connect to all discovered GPU servers.
        
        Returns:
            int: Number of successful connections
        """
        discovered = self.discovery.get_available_gpus()
        connected = 0
        
        # Group by server to avoid duplicate connections
        servers = {}
        for gpu in discovered:
            key = f"{gpu['server_ip']}:{gpu['server_port']}"
            if key not in servers:
                servers[key] = gpu
        
        for key, gpu in servers.items():
            if self.connect_to(gpu['server_ip'], gpu['server_port']):
                connected += 1
        
        return connected
    
    def get_best_gpu(self) -> Optional[GPUConnection]:
        """
        Get the connection with the most available GPU memory.
        
        Returns:
            Optional[GPUConnection]: Best available connection, or None
        """
        best_conn = None
        best_memory = 0
        
        with self._lock:
            for conn in self.connections.values():
                if conn.is_connected and conn.free_memory > best_memory:
                    best_memory = conn.free_memory
                    best_conn = conn
        
        return best_conn
    
    def parallel_execute(self, operation: dict, inputs: List) -> List:
        """
        Execute the same operation on multiple inputs in parallel.
        
        Distributes work across available GPUs for maximum throughput.
        
        Args:
            operation (dict): Operation template
            inputs (List): List of input data (one per parallel execution)
            
        Returns:
            List: Results in the same order as inputs
        """
        if not self.connections:
            raise RuntimeError("No GPU connections available. Call connect_to() first.")
        
        # Get available connections
        with self._lock:
            active_connections = [c for c in self.connections.values() if c.is_connected]
        
        if not active_connections:
            raise RuntimeError("No active GPU connections")
        
        # Distribute work
        num_workers = len(active_connections)
        chunk_size = max(1, len(inputs) // num_workers)
        
        futures: List[Future] = []
        
        for i, conn in enumerate(active_connections):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_workers - 1 else len(inputs)
            
            if start_idx >= len(inputs):
                break
            
            chunk = inputs[start_idx:end_idx]
            
            # Submit work to executor
            future = self._executor.submit(
                self._execute_batch,
                conn,
                operation,
                chunk,
                start_idx
            )
            futures.append(future)
        
        # Collect results in order
        results = [None] * len(inputs)
        
        for future in futures:
            batch_results = future.result()
            for pos, result in batch_results:
                results[pos] = result
        
        return results
    
    def _execute_batch(self, conn: GPUConnection, operation: dict, inputs: List, start_idx: int) -> List:
        """
        Execute a batch of operations on a specific connection.
        """
        results = []
        for i, input_data in enumerate(inputs):
            op = operation.copy()
            op['data'] = input_data
            result = conn.execute(op)
            results.append((start_idx + i, result))
        return results
    
    def _on_server_found(self, server_id: str, info: dict):
        """
        Callback when a new server is discovered.
        """
        print(f"🔔 GPU server discovered: {info.get('ip')}:{info.get('port')}")
    
    def _on_server_lost(self, server_id: str, info: dict):
        """
        Callback when a server is lost.
        """
        print(f"🔕 GPU server lost: {info.get('ip')}")
        
        # Close connection if we had one
        key = f"{info.get('ip')}:{info.get('port')}"
        with self._lock:
            if key in self.connections:
                self.connections[key].disconnect()
                del self.connections[key]
    
    def get_stats(self) -> dict:
        """
        Get aggregate statistics for all connections.
        
        Returns:
            dict: Pool statistics
        """
        total_gpus = 0
        total_memory = 0
        free_memory = 0
        active_connections = 0
        
        with self._lock:
            for conn in self.connections.values():
                if conn.is_connected:
                    active_connections += 1
                    total_gpus += conn.gpu_count
                    total_memory += conn.total_memory
                    free_memory += conn.free_memory
        
        return {
            'active_connections': active_connections,
            'total_connections': len(self.connections),
            'total_gpus': total_gpus,
            'total_memory_gb': total_memory / (1024**3),
            'free_memory_gb': free_memory / (1024**3),
            'utilization': (1 - free_memory / total_memory) if total_memory > 0 else 0
        }
    
    def disconnect_all(self):
        """
        Disconnect from all GPU servers.
        """
        with self._lock:
            for conn in self.connections.values():
                conn.disconnect()
            self.connections.clear()
        
        self._executor.shutdown(wait=True)
        print("✅ All connections closed.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect_all()
        return False