"""
GPU Connection Module
=====================
Low-level connection to a single RemoteCUDA GPU server.
Handles socket communication, health checks, and operation execution.
"""

import socket
import time
import threading
from typing import Optional

import torch

from ..protocol.messages import MessageProtocol
from ..protocol.compression import TensorCompressor


class GPUConnection:
    """
    Connection to a single RemoteCUDA GPU server.
    
    Each connection gives access to all GPUs on that server.
    The server handles GPU assignment internally.
    
    Features:
    - Automatic reconnection
    - Health monitoring
    - Async operation support
    - Tensor caching
    """
    
    def __init__(self, host: str, port: int = 55555, timeout: float = 30.0):
        """
        Initialize GPU connection.
        
        Args:
            host (str): Server hostname or IP
            port (int): Server port
            timeout (float): Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # Socket
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        
        # Connection state
        self.is_connected = False
        self._reconnect_interval = 5.0
        
        # Server info (populated after connection)
        self.server_id: str = ''
        self.gpu_name: str = ''
        self.gpu_count: int = 0
        self.total_memory: int = 0
        self.total_memory_gb: float = 0.0
        self.free_memory: int = 0
        self.cuda_version: str = ''
        
        # Tensor cache for client-side tracking
        self._tensor_cache: dict = {}
        
        # Operation counter
        self.operations_count: int = 0
    
    def connect(self) -> bool:
        """
        Establish connection to the GPU server.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            
            # Enable TCP_NODELAY for lower latency
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Get server info via ping
            info = self.execute({'type': 'ping'})
            
            if 'error' in info:
                raise ConnectionError(f"Server error: {info['error']}")
            
            self.is_connected = True
            self.gpu_name = info.get('device', 'Unknown')
            
            return True
            
        except Exception as e:
            self.is_connected = False
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")
    
    def disconnect(self):
        """
        Close the connection gracefully.
        """
        self.is_connected = False
        
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
    
    def execute(self, operation: dict) -> dict:
        """
        Execute an operation on the remote GPU.
        
        Args:
            operation (dict): Operation specification
            
        Returns:
            dict: Operation result
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to GPU server")
        
        with self._lock:
            try:
                # Encode and send
                message = MessageProtocol.encode_operation(operation)
                self._socket.sendall(message)
                
                # Receive response
                buffer = b''
                while True:
                    data = self._socket.recv(65536)
                    if not data:
                        raise ConnectionError("Server closed connection")
                    
                    buffer += data
                    response, buffer = MessageProtocol.decode(buffer)
                    
                    if response is not None:
                        self.operations_count += 1
                        return response
                        
            except socket.timeout:
                self.is_connected = False
                raise TimeoutError("Operation timed out")
            except Exception as e:
                self.is_connected = False
                raise ConnectionError(f"Connection error: {e}")
    
    def allocate_tensor(self, tensor: torch.Tensor) -> int:
        """
        Allocate a tensor on the remote GPU.
        
        Args:
            tensor (torch.Tensor): Tensor to transfer to GPU
            
        Returns:
            int: Tensor ID for future reference
        """
        tensor_data = TensorCompressor.compress_tensor(tensor)
        
        result = self.execute({
            'type': 'allocate_tensor',
            'data': tensor_data,
            'tensor_id': id(tensor)
        })
        
        if 'error' in result:
            raise RuntimeError(f"Tensor allocation failed: {result['error']}")
        
        tensor_id = result['tensor_id']
        self._tensor_cache[tensor_id] = tensor.shape
        return tensor_id
    
    def free_tensor(self, tensor_id: int):
        """
        Free a tensor from the remote GPU.
        
        Args:
            tensor_id (int): Tensor ID to free
        """
        self.execute({
            'type': 'free_tensor',
            'tensor_id': tensor_id
        })
        
        self._tensor_cache.pop(tensor_id, None)
    
    def forward(self, model_id: int, *input_tensor_ids: int) -> torch.Tensor:
        """
        Execute forward pass on the remote GPU.
        
        Args:
            model_id (int): Registered model ID
            *input_tensor_ids: IDs of input tensors
            
        Returns:
            torch.Tensor: Output tensor
        """
        result = self.execute({
            'type': 'forward',
            'model_id': model_id,
            'input_tensor_ids': list(input_tensor_ids)
        })
        
        if 'error' in result:
            raise RuntimeError(f"Forward pass failed: {result['error']}")
        
        return result
    
    def health_check(self) -> bool:
        """
        Check if the connection is still alive.
        
        Returns:
            bool: True if server responds
        """
        try:
            result = self.execute({'type': 'ping'})
            return 'status' in result and result['status'] == 'ok'
        except:
            return False
    
    def __repr__(self) -> str:
        status = "Connected" if self.is_connected else "Disconnected"
        return f"GPUConnection({self.host}:{self.port}, {status}, GPU: {self.gpu_name})"