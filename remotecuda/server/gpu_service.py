"""
GPU Service Module
==================
The background service that runs on GPU-equipped machines.
Just run: remotecuda start
That's all you need to share your GPUs.
"""

import socket
import threading
import time
import json
import uuid
import signal
import sys
import os
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import torch

from .discovery import NetworkDiscovery
from ..protocol.messages import MessageProtocol
from ..protocol.compression import TensorCompressor


class GPUWorker:
    """
    Manages operations on a single physical GPU.
    One worker per GPU device.
    """
    
    def __init__(self, gpu_index: int):
        """
        Initialize worker for a specific GPU.
        
        Args:
            gpu_index (int): CUDA device index to manage
        """
        self.gpu_index = gpu_index
        self.device = torch.device(f'cuda:{gpu_index}')
        self.device_name = torch.cuda.get_device_name(gpu_index)
        
        # Memory tracking
        self.total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
        self._allocated_memory = 0
        
        # Object registry for this GPU
        self.tensors: Dict[int, torch.Tensor] = {}
        self.models: Dict[int, torch.nn.Module] = {}
        self.optimizers: Dict[int, torch.optim.Optimizer] = {}
        
        # CUDA stream for async operations
        self.stream = torch.cuda.Stream(device=self.device)
        
        # Operation statistics
        self.stats = {
            'operations_completed': 0,
            'bytes_transferred': 0,
            'total_compute_time': 0.0,
            'last_activity': time.time()
        }
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
    
    def get_info(self) -> dict:
        """
        Get current GPU information.
        
        Returns:
            dict: GPU state including memory and utilization
        """
        free_memory = self.total_memory - self._allocated_memory
        return {
            'index': self.gpu_index,
            'name': self.device_name,
            'total_memory': self.total_memory,
            'free_memory': free_memory,
            'allocated_memory': self._allocated_memory,
            'utilization': self._allocated_memory / self.total_memory if self.total_memory > 0 else 0,
            'stats': self.stats.copy()
        }
    
    def execute_operation(self, operation: dict) -> dict:
        """
        Execute a CUDA operation on this GPU.
        
        Args:
            operation (dict): Operation specification
            
        Returns:
            dict: Operation result
        """
        with self._lock:
            self.stats['last_activity'] = time.time()
            self.stats['operations_completed'] += 1
        
        op_type = operation.get('type')
        
        try:
            with torch.cuda.stream(self.stream):
                if op_type == 'allocate_tensor':
                    return self._op_allocate_tensor(operation)
                elif op_type == 'free_tensor':
                    return self._op_free_tensor(operation)
                elif op_type == 'forward':
                    return self._op_forward(operation)
                elif op_type == 'backward':
                    return self._op_backward(operation)
                elif op_type == 'optimizer_step':
                    return self._op_optimizer_step(operation)
                elif op_type == 'get_parameters':
                    return self._op_get_parameters(operation)
                elif op_type == 'ping':
                    return {'status': 'ok', 'device': self.device_name}
                else:
                    return {'error': f'Unknown operation: {op_type}'}
                    
        except Exception as e:
            return {'error': str(e), 'type': type(e).__name__}
    
    def _op_allocate_tensor(self, op: dict) -> dict:
        """Allocate a tensor on GPU from serialized data."""
        tensor_data = op['data']
        tensor_id = op.get('tensor_id', id(object()))
        
        # Deserialize and move to this GPU
        tensor = TensorCompressor.decompress_tensor(tensor_data, self.device)
        
        # Track memory
        memory_used = tensor.element_size() * tensor.nelement()
        self._allocated_memory += memory_used
        self.stats['bytes_transferred'] += memory_used
        
        # Register tensor
        self.tensors[tensor_id] = tensor
        
        return {
            'status': 'ok',
            'tensor_id': tensor_id,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        }
    
    def _op_free_tensor(self, op: dict) -> dict:
        """Free a registered tensor from GPU memory."""
        tensor_id = op.get('tensor_id')
        
        if tensor_id in self.tensors:
            tensor = self.tensors.pop(tensor_id)
            memory_freed = tensor.element_size() * tensor.nelement()
            self._allocated_memory -= memory_freed
            del tensor
        
        return {'status': 'ok'}
    
    def _op_forward(self, op: dict) -> dict:
        """Execute forward pass on a registered model."""
        model_id = op['model_id']
        input_ids = op['input_tensor_ids']
        
        model = self.models.get(model_id)
        if model is None:
            return {'error': f'Model {model_id} not found'}
        
        # Gather inputs
        inputs = []
        for tid in input_ids:
            tensor = self.tensors.get(tid)
            if tensor is None:
                return {'error': f'Input tensor {tid} not found'}
            inputs.append(tensor)
        
        # Execute forward pass
        start_time = time.time()
        with torch.no_grad() if op.get('no_grad', True) else torch.enable_grad():
            if len(inputs) == 1:
                output = model(inputs[0])
            else:
                output = model(*inputs)
        compute_time = time.time() - start_time
        
        self.stats['total_compute_time'] += compute_time
        
        # Register output tensor
        output_id = id(output) if not isinstance(output, tuple) else [id(o) for o in output]
        if not isinstance(output, tuple):
            self.tensors[output_id] = output
        else:
            for o, oid in zip(output, output_id):
                self.tensors[oid] = o
        
        # Serialize output
        output_data = TensorCompressor.compress_tensor(output) if not isinstance(output, tuple) else [
            TensorCompressor.compress_tensor(o) for o in output
        ]
        
        return {
            'status': 'ok',
            'output_data': output_data,
            'output_id': output_id,
            'compute_time_ms': compute_time * 1000
        }
    
    def _op_backward(self, op: dict) -> dict:
        """Execute backward pass."""
        tensor_id = op['tensor_id']
        gradient = op.get('gradient')  # Optional external gradient
        
        tensor = self.tensors.get(tensor_id)
        if tensor is None:
            return {'error': f'Tensor {tensor_id} not found'}
        
        start_time = time.time()
        if gradient is not None:
            grad_tensor = TensorCompressor.decompress_tensor(gradient, self.device)
            tensor.backward(grad_tensor)
        else:
            tensor.backward()
        compute_time = time.time() - start_time
        
        self.stats['total_compute_time'] += compute_time
        
        # Collect gradients from model parameters that have them
        gradient_data = {}
        for model_id, model in self.models.items():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradient_data[f'{model_id}_{name}'] = TensorCompressor.compress_tensor(param.grad)
        
        return {
            'status': 'ok',
            'gradients': gradient_data,
            'compute_time_ms': compute_time * 1000
        }
    
    def _op_optimizer_step(self, op: dict) -> dict:
        """Execute optimizer step."""
        optimizer_id = op['optimizer_id']
        
        optimizer = self.optimizers.get(optimizer_id)
        if optimizer is None:
            return {'error': f'Optimizer {optimizer_id} not found'}
        
        start_time = time.time()
        optimizer.step()
        optimizer.zero_grad()
        compute_time = time.time() - start_time
        
        self.stats['total_compute_time'] += compute_time
        
        return {
            'status': 'ok',
            'compute_time_ms': compute_time * 1000
        }
    
    def _op_get_parameters(self, op: dict) -> dict:
        """Get serialized model parameters."""
        model_id = op['model_id']
        
        model = self.models.get(model_id)
        if model is None:
            return {'error': f'Model {model_id} not found'}
        
        params = {}
        for name, param in model.named_parameters():
            params[name] = TensorCompressor.compress_tensor(param.data)
        
        return {
            'status': 'ok',
            'parameters': params
        }
    
    def register_model(self, model_id: int, model: torch.nn.Module):
        """Register a model for future operations."""
        self.models[model_id] = model.to(self.device)
    
    def register_optimizer(self, optimizer_id: int, optimizer: torch.optim.Optimizer):
        """Register an optimizer for future operations."""
        self.optimizers[optimizer_id] = optimizer
    
    def cleanup(self):
        """
        Clean up GPU memory and resources.
        """
        with self._lock:
            # Clear all registered objects
            for tensor in self.tensors.values():
                del tensor
            for model in self.models.values():
                del model
            for optimizer in self.optimizers.values():
                del optimizer
            
            self.tensors.clear()
            self.models.clear()
            self.optimizers.clear()
            self._allocated_memory = 0
            
            # Synchronize and clear CUDA cache
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()


class GPUService:
    """
    Main GPU service that runs as a background process.
    
    Usage:
        service = GPUService(port=55555)
        service.start()  # Blocks until stopped
        
    Or from command line:
        remotecuda start --port 55555
    """
    
    def __init__(self, port: int = 55555, host: str = '0.0.0.0'):
        """
        Initialize the GPU service.
        
        Args:
            port (int): TCP port for client connections
            host (str): Bind address (0.0.0.0 for all interfaces)
        """
        self.port = port
        self.host = host
        self.server_id = str(uuid.uuid4())[:8]
        self._running = False
        
        # Initialize GPU workers (one per physical GPU)
        self.gpu_count = torch.cuda.device_count()
        self.workers: Dict[int, GPUWorker] = {}
        self._init_workers()
        
        # Network discovery
        self.discovery = NetworkDiscovery()
        
        # Server socket
        self._server_socket: Optional[socket.socket] = None
        
        # Thread pool for handling client connections
        self._executor = ThreadPoolExecutor(max_workers=8)
        
        # Active client sessions
        self._sessions: Dict[str, dict] = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_workers(self):
        """
        Initialize GPU workers for each available CUDA device.
        """
        if self.gpu_count == 0:
            print("⚠️  WARNING: No CUDA devices found!")
            print("   Install CUDA and PyTorch with CUDA support.")
            return
        
        for i in range(self.gpu_count):
            worker = GPUWorker(i)
            self.workers[i] = worker
            print(f"   GPU {i}: {worker.device_name}")
    
    def _signal_handler(self, signum, frame):
        """
        Handle OS signals for graceful shutdown.
        """
        print("\n🛑 Shutdown signal received. Cleaning up...")
        self.stop()
        sys.exit(0)
    
    def get_gpu_info(self) -> dict:
        """
        Get information about all GPUs on this server.
        
        Returns:
            dict: Complete GPU information for broadcasting
        """
        gpus = []
        for idx, worker in self.workers.items():
            gpus.append(worker.get_info())
        
        return {
            'server_id': self.server_id,
            'hostname': socket.gethostname(),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'gpu_count': len(self.workers),
            'gpus': gpus
        }
    
    def start(self):
        """
        Start the GPU service.
        
        This method blocks until stop() is called.
        """
        if self.gpu_count == 0:
            print("❌ Cannot start service: No GPUs available.")
            return
        
        # Create server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)
        self._running = True
        
        # Start network discovery broadcasting
        gpu_info = self.get_gpu_info()
        self.discovery.start_broadcasting(self.port, gpu_info)
        
        # Display startup banner
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                 🚀 RemoteCUDA GPU Service                   ║
╠══════════════════════════════════════════════════════════════╣
║  Server ID:  {self.server_id:<46}║
║  Host:       {self.host}:{self.port:<41}║
║  GPUs:       {len(self.workers):<46}║
║  Status:     Ready - Waiting for connections               ║
╠══════════════════════════════════════════════════════════════╣
║  Your GPUs are now available on the network!               ║
║  Clients can connect and use them transparently.           ║
║                                                            ║
║  To stop: Press Ctrl+C                                     ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Print GPU details
        for idx, worker in self.workers.items():
            mem_gb = worker.total_memory / (1024**3)
            print(f"  🎮 GPU {idx}: {worker.device_name} ({mem_gb:.1f} GB)")
        print()
        
        # Main accept loop
        while self._running:
            try:
                self._server_socket.settimeout(1.0)
                client_socket, address = self._server_socket.accept()
                
                # Handle client in thread pool
                self._executor.submit(self._handle_client, client_socket, address)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"Accept error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """
        Handle a connected client session.
        
        Args:
            client_socket: Client socket
            address: Client address tuple (ip, port)
        """
        client_ip = address[0]
        session_id = str(uuid.uuid4())[:12]
        
        # Register session
        self._sessions[session_id] = {
            'ip': client_ip,
            'connected_at': time.time(),
            'gpu_assignment': None,  # Will be assigned based on load
            'operations_count': 0
        }
        
        print(f"✅ Client connected: {client_ip} (Session: {session_id})")
        
        # Assign GPU based on load balancing
        assigned_gpu = self._assign_gpu()
        self._sessions[session_id]['gpu_assignment'] = assigned_gpu
        
        # Get the worker for assigned GPU
        worker = self.workers[assigned_gpu]
        
        # Message buffer
        buffer = b''
        
        try:
            while self._running:
                data = client_socket.recv(65536)  # 64KB receive buffer
                if not data:
                    break
                
                buffer += data
                
                # Process all complete messages
                while True:
                    message, buffer = MessageProtocol.decode(buffer)
                    
                    if message is None:
                        break  # Incomplete message
                    
                    # Execute operation on assigned GPU
                    result = worker.execute_operation(message)
                    self._sessions[session_id]['operations_count'] += 1
                    
                    # Send response
                    response = MessageProtocol.encode_response(result)
                    client_socket.sendall(response)
                    
        except Exception as e:
            print(f"Session {session_id} error: {e}")
        finally:
            client_socket.close()
            del self._sessions[session_id]
            print(f"👋 Client disconnected: {client_ip} (Session: {session_id})")
    
    def _assign_gpu(self) -> int:
        """
        Assign a GPU to a new client based on current load.
        
        Uses simple load balancing: picks the GPU with the most free memory.
        
        Returns:
            int: Index of the assigned GPU
        """
        if not self.workers:
            return 0
        
        # Find GPU with most free memory
        best_gpu = 0
        best_free = 0
        
        for idx, worker in self.workers.items():
            free = worker.total_memory - worker._allocated_memory
            if free > best_free:
                best_free = free
                best_gpu = idx
        
        return best_gpu
    
    def stop(self):
        """
        Gracefully stop the GPU service.
        """
        if not self._running:
            return
        
        print("\n🧹 Cleaning up resources...")
        
        self._running = False
        
        # Stop discovery
        self.discovery.stop()
        
        # Clean up all GPU workers
        for worker in self.workers.values():
            worker.cleanup()
        
        # Close server socket
        if self._server_socket:
            self._server_socket.close()
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        print("✅ GPU Service stopped.")


def main():
    """
    Entry point for the GPU service command line.
    
    Usage:
        remotecuda-start --port 55555
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RemoteCUDA GPU Service - Share your GPUs on the network'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=55555,
        help='TCP port to listen on (default: 55555)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Bind address (default: 0.0.0.0 for all interfaces)'
    )
    
    args = parser.parse_args()
    
    # Create and start service
    service = GPUService(port=args.port, host=args.host)
    service.start()


if __name__ == '__main__':
    main()