"""
Network Discovery Module
========================
Enables automatic server discovery on the local network.
GPU servers broadcast their presence; clients find them automatically.
No manual IP configuration needed.
"""

import socket
import threading
import json
import time
import uuid
from typing import Dict, Optional, Callable


class NetworkDiscovery:
    """
    Zero-configuration network discovery for GPU servers.
    
    Uses UDP broadcasting to announce GPU availability and
    allow clients to discover servers without manual IP setup.
    
    Protocol:
        - Servers broadcast heartbeat every 2 seconds
        - Clients listen for broadcasts to find available GPUs
        - No central registry needed
    """
    
    # Multicast group for discovery (uses the IPv4 admin scope)
    MULTICAST_GROUP = '239.255.100.100'
    MULTICAST_PORT = 55556
    
    def __init__(self, service_name: str = 'RemoteCUDA'):
        """
        Initialize network discovery.
        
        Args:
            service_name (str): Identifier for this service on the network.
                              Used to filter relevant broadcasts.
        """
        self.service_name = service_name
        self.server_id = str(uuid.uuid4())[:8]  # Unique server identifier
        self._running = False
        self._broadcast_thread = None
        self._discovery_thread = None
        
        # Callback for when a server is discovered
        self.on_server_found: Optional[Callable] = None
        self.on_server_lost: Optional[Callable] = None
        
        # Track discovered servers
        self._known_servers: Dict[str, dict] = {}
        self._server_timeout = 6.0  # Seconds before considering a server lost
    
    def start_broadcasting(self, port: int, gpu_info: dict):
        """
        Start broadcasting this server's availability on the network.
        
        Sends periodic heartbeat messages containing server info
        so clients can discover and connect automatically.
        
        Args:
            port (int): The TCP port the GPU service is listening on
            gpu_info (dict): Information about available GPUs
        """
        self._running = True
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            args=(port, gpu_info),
            daemon=True
        )
        self._broadcast_thread.start()
    
    def _broadcast_loop(self, port: int, gpu_info: dict):
        """
        Main broadcast loop. Sends heartbeat every 2 seconds.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        
        message_template = {
            'service': self.service_name,
            'server_id': self.server_id,
            'port': port,
            'gpu_info': gpu_info,
            'timestamp': 0
        }
        
        while self._running:
            try:
                message_template['timestamp'] = time.time()
                data = json.dumps(message_template).encode('utf-8')
                sock.sendto(data, (self.MULTICAST_GROUP, self.MULTICAST_PORT))
            except Exception as e:
                print(f"Broadcast error: {e}")
            
            time.sleep(2)  # Broadcast interval
        
        sock.close()
    
    def start_discovery(self) -> Dict[str, dict]:
        """
        Start listening for server broadcasts.
        
        Returns immediately with currently known servers.
        The on_server_found callback will be called for new discoveries.
        
        Returns:
            Dict[str, dict]: Currently known servers {server_id: server_info}
        """
        self._running = True
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop,
            daemon=True
        )
        self._discovery_thread.start()
        return self._known_servers
    
    def _discovery_loop(self):
        """
        Main discovery loop. Listens for server broadcasts.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to multicast port
        sock.bind(('', self.MULTICAST_PORT))
        
        # Join multicast group
        mreq = socket.inet_aton(self.MULTICAST_GROUP) + socket.inet_aton('0.0.0.0')
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        sock.settimeout(1.0)
        
        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                
                # Filter: only our service
                if message.get('service') != self.service_name:
                    continue
                
                server_id = message['server_id']
                server_ip = addr[0]
                
                # Merge IP into server info
                message['ip'] = server_ip
                
                # Check if this is a new server
                is_new = server_id not in self._known_servers
                
                # Update server info
                self._known_servers[server_id] = message
                
                # Notify about new server
                if is_new and self.on_server_found:
                    self.on_server_found(server_id, message)
                    
            except socket.timeout:
                pass
            except Exception as e:
                if self._running:
                    print(f"Discovery error: {e}")
            
            # Clean up dead servers
            self._cleanup_dead_servers()
        
        sock.close()
    
    def _cleanup_dead_servers(self):
        """
        Remove servers that haven't broadcasted recently.
        """
        current_time = time.time()
        dead_servers = []
        
        for server_id, info in self._known_servers.items():
            if current_time - info['timestamp'] > self._server_timeout:
                dead_servers.append(server_id)
        
        for server_id in dead_servers:
            lost_info = self._known_servers.pop(server_id)
            if self.on_server_lost:
                self.on_server_lost(server_id, lost_info)
    
    def get_available_gpus(self) -> list:
        """
        Get a list of all available GPUs across all discovered servers.
        
        Returns:
            list: List of GPU descriptors:
                [{
                    'server_id': str,
                    'server_ip': str,
                    'server_port': int,
                    'gpu_index': int,
                    'gpu_name': str,
                    'total_memory': int,
                    'free_memory': int,
                    'utilization': float
                }, ...]
        """
        available_gpus = []
        
        for server_id, info in self._known_servers.items():
            for gpu_idx, gpu_data in enumerate(info.get('gpu_info', {}).get('gpus', [])):
                available_gpus.append({
                    'server_id': server_id,
                    'server_ip': info['ip'],
                    'server_port': info['port'],
                    'gpu_index': gpu_idx,
                    'gpu_name': gpu_data.get('name', 'Unknown'),
                    'total_memory': gpu_data.get('total_memory', 0),
                    'free_memory': gpu_data.get('free_memory', 0),
                    'utilization': gpu_data.get('utilization', 0.0)
                })
        
        return available_gpus
    
    def stop(self):
        """
        Stop all discovery and broadcasting threads.
        """
        self._running = False
        if self._broadcast_thread:
            self._broadcast_thread.join(timeout=3)
        if self._discovery_thread:
            self._discovery_thread.join(timeout=3)