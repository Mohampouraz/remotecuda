"""
GPU Service Module — Dual Backend (PyTorch + Direct Driver)
============================================================
Auto-detects available backend and starts computation service.

Backend Priority:
    1. PyTorch with CUDA → Best performance, full feature set
    2. PyTorch CPU-only → Full features, CPU computation
    3. Direct NVIDIA Driver (nvcuda.dll) → No PyTorch needed, GPU only
    4. Pure NumPy → No GPU, no PyTorch, CPU computation (last resort)

The server automatically selects the best available backend.
No configuration needed — just run: remotecuda start
"""

import socket
import threading
import time
import json
import uuid
import sys
import signal
import struct
import pickle
import zlib
from typing import Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .discovery import NetworkDiscovery


# ============================================================
#  Backend Detection
# ============================================================

BACKEND_PYTORCH_CUDA = 'pytorch_cuda'
BACKEND_PYTORCH_CPU = 'pytorch_cpu'
BACKEND_DIRECT_CUDA = 'direct_cuda'
BACKEND_NUMPY_ONLY = 'numpy_only'

def _detect_backend() -> Tuple[str, dict]:
    """
    Auto-detect the best available computation backend.
    
    Returns:
        Tuple[str, dict]: (backend_name, backend_info)
    """
    info = {
        'hostname': socket.gethostname(),
        'python_version': sys.version.split()[0],
        'numpy_version': np.__version__,
    }
    
    # Step 1: Try PyTorch with CUDA
    try:
        import torch
        info['pytorch_version'] = torch.__version__
        
        if torch.cuda.is_available():
            info['device'] = 'cuda:0'
            info['device_type'] = 'cuda'
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_mb'] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            info['cuda_version'] = torch.version.cuda
            info['backend'] = BACKEND_PYTORCH_CUDA
            print(f"Backend: PyTorch + CUDA ({info['gpu_name']})")
            return BACKEND_PYTORCH_CUDA, info
        else:
            info['device'] = 'cpu'
            info['device_type'] = 'cpu'
            info['gpu_count'] = 0
            info['backend'] = BACKEND_PYTORCH_CPU
            print("Backend: PyTorch CPU (no GPU detected)")
            return BACKEND_PYTORCH_CPU, info
            
    except ImportError:
        info['pytorch_version'] = None
    
    # Step 2: Try Direct NVIDIA Driver (no PyTorch needed)
    try:
        from .cuda_driver import CUDADriver
        driver = CUDADriver()
        device_count = driver.initialize()
        
        if device_count > 0:
            dev_info = driver.get_device_info(0)
            info['device'] = 'cuda:0'
            info['device_type'] = 'cuda'
            info['gpu_count'] = device_count
            info['gpu_name'] = dev_info.name
            info['gpu_memory_mb'] = dev_info.total_memory // (1024 * 1024)
            info['cuda_version'] = 'driver_only'
            info['backend'] = BACKEND_DIRECT_CUDA
            driver.shutdown()
            print(f"Backend: Direct NVIDIA Driver ({info['gpu_name']})")
            return BACKEND_DIRECT_CUDA, info
        else:
            driver.shutdown()
    except (ImportError, RuntimeError, OSError):
        pass
    
    # Step 3: Pure NumPy (last resort)
    info['device'] = 'cpu'
    info['device_type'] = 'cpu'
    info['gpu_count'] = 0
    info['backend'] = BACKEND_NUMPY_ONLY
    print("Backend: NumPy CPU (no GPU, no PyTorch)")
    return BACKEND_NUMPY_ONLY, info


# ============================================================
#  Message Protocol
# ============================================================

class MessageProtocol:
    """Binary protocol for client-server communication."""
    
    MAGIC = b'RCUD'
    VERSION = 1
    MSG_OPERATION = 1
    MSG_RESPONSE = 2
    FLAG_COMPRESSED = 0x01
    
    @staticmethod
    def encode_response(response: dict) -> bytes:
        payload = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        flags = 0
        if len(payload) > 1024:
            payload = zlib.compress(payload, level=3)
            flags |= MessageProtocol.FLAG_COMPRESSED
        
        header = struct.pack(
            '!4sBBHI',
            MessageProtocol.MAGIC,
            MessageProtocol.VERSION,
            flags,
            MessageProtocol.MSG_RESPONSE,
            len(payload)
        )
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return header + struct.pack('!I', crc) + payload
    
    @staticmethod
    def decode(buffer: bytes) -> Tuple[Optional[dict], bytes]:
        if len(buffer) < 16:
            return None, buffer
        
        magic = buffer[:4]
        if magic != MessageProtocol.MAGIC:
            return None, buffer[1:]
        
        version, flags, msg_type, payload_len = struct.unpack('!BBHI', buffer[4:12])
        total_needed = 16 + payload_len
        
        if len(buffer) < total_needed:
            return None, buffer
        
        received_crc = struct.unpack('!I', buffer[12:16])[0]
        payload = buffer[16:total_needed]
        
        calculated_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if received_crc != calculated_crc:
            return None, buffer[16:]
        
        if flags & MessageProtocol.FLAG_COMPRESSED:
            try:
                payload = zlib.decompress(payload)
            except zlib.error:
                return None, buffer[total_needed:]
        
        try:
            message = pickle.loads(payload)
        except pickle.UnpicklingError:
            return None, buffer[total_needed:]
        
        return message, buffer[total_needed:]


# ============================================================
#  Base Compute Engine
# ============================================================

class ComputeEngine:
    """Base class for compute backends."""
    
    def execute(self, command: str, params: dict) -> Any:
        raise NotImplementedError
    
    def shutdown(self):
        pass


# ============================================================
#  PyTorch Compute Engine
# ============================================================

class PyTorchComputeEngine(ComputeEngine):
    """PyTorch-based compute engine (GPU or CPU)."""
    
    def __init__(self):
        import torch
        self._torch = torch
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._tensors: Dict[int, dict] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._use_gpu = torch.cuda.is_available()
        
        if self._use_gpu:
            torch.cuda.init()
    
    def execute(self, command: str, params: dict) -> Any:
        handlers = {
            'ping': self._op_ping,
            'info': self._op_info,
            'zeros': self._op_zeros,
            'ones': self._op_ones,
            'full': self._op_full,
            'rand': self._op_rand,
            'randn': self._op_randn,
            'eye': self._op_eye,
            'arange': self._op_arange,
            'send_tensor': self._op_send_tensor,
            'get_tensor': self._op_get_tensor,
            'free_tensor': self._op_free_tensor,
            'add': self._op_add,
            'subtract': self._op_subtract,
            'multiply': self._op_multiply,
            'divide': self._op_divide,
            'matmul': self._op_matmul,
            'relu': self._op_relu,
            'sigmoid': self._op_sigmoid,
            'tanh': self._op_tanh,
            'softmax': self._op_softmax,
            'sum': self._op_sum,
            'mean': self._op_mean,
            'max': self._op_max,
            'min': self._op_min,
            'reshape': self._op_reshape,
            'transpose': self._op_transpose,
            'flatten': self._op_flatten,
            'abs': self._op_abs,
            'sqrt': self._op_sqrt,
            'exp': self._op_exp,
            'log': self._op_log,
            'pow': self._op_pow,
            'clamp': self._op_clamp,
        }
        
        handler = handlers.get(command)
        if handler is None:
            return {'error': f'Unknown command: {command}'}
        
        try:
            return handler(params)
        except Exception as e:
            return {'error': str(e), 'type': type(e).__name__}
    
    def _register(self, tensor) -> int:
        with self._lock:
            self._counter += 1
            tid = self._counter
            self._tensors[tid] = {
                'tensor': tensor,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype).replace('torch.', ''),
                'size': tensor.element_size() * tensor.nelement(),
            }
            return tid
    
    def _get(self, tid: int):
        with self._lock:
            entry = self._tensors.get(tid)
            if entry is None:
                raise KeyError(f"Tensor {tid} not found")
            return entry['tensor']
    
    def _encode(self, tensor) -> dict:
        import base64
        cpu = tensor.detach().cpu()
        raw = cpu.numpy().tobytes()
        encoded = base64.b64encode(raw).decode('ascii')
        dtype_str = str(tensor.dtype).replace('torch.', '')
        fmt_map = {'float32': 'f', 'float64': 'd', 'int32': 'i', 'int64': 'q', 'uint8': 'B'}
        return {
            'data': encoded,
            'shape': list(tensor.shape),
            'dtype': dtype_str,
            'format': fmt_map.get(dtype_str, 'f'),
            'size': len(raw),
        }
    
    def _decode(self, params: dict):
        import base64
        encoded = params.get('data', '')
        shape = tuple(params.get('shape', []))
        dtype_str = params.get('dtype', 'float32')
        raw = base64.b64decode(encoded)
        dtype_map = {'float32': np.float32, 'float64': np.float64, 'int32': np.int32, 'int64': np.int64, 'uint8': np.uint8}
        arr = np.frombuffer(raw, dtype=dtype_map.get(dtype_str, np.float32)).reshape(shape)
        return self._torch.from_numpy(arr.copy()).to(self._device)
    
    def _op_ping(self, p): return {'status': 'ok', 'device': str(self._device), 'backend': 'pytorch'}
    def _op_info(self, p): return {'status': 'ok', 'device': str(self._device), 'backend': 'pytorch', 'gpu': self._use_gpu, 'tensors': len(self._tensors)}
    
    def _op_zeros(self, p):
        shape = p.get('shape', [1])
        dtype = getattr(self._torch, p.get('dtype', 'float32'))
        return self._register(self._torch.zeros(shape, dtype=dtype, device=self._device))
    
    def _op_ones(self, p):
        shape = p.get('shape', [1])
        dtype = getattr(self._torch, p.get('dtype', 'float32'))
        return self._register(self._torch.ones(shape, dtype=dtype, device=self._device))
    
    def _op_full(self, p):
        shape = p.get('shape', [1])
        value = p.get('value', 0.0)
        dtype = getattr(self._torch, p.get('dtype', 'float32'))
        return self._register(self._torch.full(shape, value, dtype=dtype, device=self._device))
    
    def _op_rand(self, p):
        shape = p.get('shape', [1])
        return self._register(self._torch.rand(shape, device=self._device))
    
    def _op_randn(self, p):
        shape = p.get('shape', [1])
        return self._register(self._torch.randn(shape, device=self._device))
    
    def _op_eye(self, p):
        n = p.get('n', 1)
        m = p.get('m', n)
        return self._register(self._torch.eye(n, m, device=self._device))
    
    def _op_arange(self, p):
        start = p.get('start', 0)
        end = p.get('end', 1)
        step = p.get('step', 1)
        dtype = getattr(self._torch, p.get('dtype', 'float32'))
        return self._register(self._torch.arange(start, end, step, dtype=dtype, device=self._device))
    
    def _op_send_tensor(self, p):
        tensor = self._decode(p)
        return self._register(tensor)
    
    def _op_get_tensor(self, p):
        tid = p.get('tensor_id')
        tensor = self._get(tid)
        return self._encode(tensor)
    
    def _op_free_tensor(self, p):
        tid = p.get('tensor_id')
        with self._lock:
            self._tensors.pop(tid, None)
        return {'status': 'ok'}
    
    def _op_add(self, p): return self._register(self._get(p['a_id']) + self._get(p['b_id']))
    def _op_subtract(self, p): return self._register(self._get(p['a_id']) - self._get(p['b_id']))
    def _op_multiply(self, p): return self._register(self._get(p['a_id']) * self._get(p['b_id']))
    def _op_divide(self, p): return self._register(self._get(p['a_id']) / self._get(p['b_id']))
    def _op_matmul(self, p): return self._register(self._torch.matmul(self._get(p['a_id']), self._get(p['b_id'])))
    def _op_relu(self, p): return self._register(self._torch.relu(self._get(p['tensor_id'])))
    def _op_sigmoid(self, p): return self._register(self._torch.sigmoid(self._get(p['tensor_id'])))
    def _op_tanh(self, p): return self._register(self._torch.tanh(self._get(p['tensor_id'])))
    
    def _op_softmax(self, p):
        return self._register(self._torch.softmax(self._get(p['tensor_id']), dim=p.get('dim', -1)))
    
    def _op_sum(self, p):
        t = self._get(p['tensor_id'])
        dim = p.get('dim')
        return self._register(t.sum() if dim is None else t.sum(dim=dim, keepdim=p.get('keepdim', False)))
    
    def _op_mean(self, p):
        t = self._get(p['tensor_id'])
        dim = p.get('dim')
        return self._register(t.mean() if dim is None else t.mean(dim=dim, keepdim=p.get('keepdim', False)))
    
    def _op_max(self, p):
        t = self._get(p['tensor_id'])
        dim = p.get('dim')
        if dim is None: return self._register(t.max())
        result, _ = t.max(dim=dim, keepdim=p.get('keepdim', False))
        return self._register(result)
    
    def _op_min(self, p):
        t = self._get(p['tensor_id'])
        dim = p.get('dim')
        if dim is None: return self._register(t.min())
        result, _ = t.min(dim=dim, keepdim=p.get('keepdim', False))
        return self._register(result)
    
    def _op_reshape(self, p): return self._register(self._get(p['tensor_id']).reshape(p['shape']))
    def _op_transpose(self, p): return self._register(self._get(p['tensor_id']).transpose(p.get('dim0', 0), p.get('dim1', 1)))
    def _op_flatten(self, p): return self._register(self._torch.flatten(self._get(p['tensor_id'])))
    def _op_abs(self, p): return self._register(self._torch.abs(self._get(p['tensor_id'])))
    def _op_sqrt(self, p): return self._register(self._torch.sqrt(self._get(p['tensor_id'])))
    def _op_exp(self, p): return self._register(self._torch.exp(self._get(p['tensor_id'])))
    def _op_log(self, p): return self._register(self._torch.log(self._get(p['tensor_id'])))
    def _op_pow(self, p): return self._register(self._torch.pow(self._get(p['tensor_id']), p.get('exponent', 2.0)))
    def _op_clamp(self, p): return self._register(self._torch.clamp(self._get(p['tensor_id']), min=p.get('min'), max=p.get('max')))
    
    def shutdown(self):
        with self._lock:
            for entry in self._tensors.values():
                del entry['tensor']
            self._tensors.clear()
        if self._use_gpu:
            self._torch.cuda.empty_cache()


# ============================================================
#  Direct CUDA Compute Engine (No PyTorch)
# ============================================================

class DirectCUDAComputeEngine(ComputeEngine):
    """Direct NVIDIA Driver compute engine — no PyTorch needed."""
    
    def __init__(self):
        from .cuda_driver import CUDADriver
        from .memory_manager import GPUMemoryManager
        
        self._driver = CUDADriver()
        self._device_count = self._driver.initialize()
        self._device_index = 0
        self._memory = GPUMemoryManager(self._driver)
        self._tensors: Dict[int, dict] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._driver._set_device(0)
    
    def execute(self, command: str, params: dict) -> Any:
        handlers = {
            'ping': self._op_ping,
            'info': self._op_info,
            'zeros': self._op_zeros,
            'ones': self._op_ones,
            'full': self._op_full,
            'send_tensor': self._op_send_tensor,
            'get_tensor': self._op_get_tensor,
            'free_tensor': self._op_free_tensor,
            'add': self._op_add,
            'subtract': self._op_subtract,
            'multiply': self._op_multiply,
            'divide': self._op_divide,
            'relu': self._op_relu,
            'sum': self._op_sum,
            'mean': self._op_mean,
            'reshape': self._op_reshape,
            'abs': self._op_abs,
        }
        
        handler = handlers.get(command)
        if handler is None:
            return {'error': f'Command not supported in direct CUDA mode: {command}'}
        
        try:
            return handler(params)
        except Exception as e:
            return {'error': str(e), 'type': type(e).__name__}
    
    def _alloc(self, arr: np.ndarray) -> int:
        block_id = self._memory.allocate(arr.nbytes, owner_id='tensor')
        ptr = self._memory.get_ptr(block_id)
        self._driver.memcpy_host_to_device(ptr, arr.tobytes())
        
        with self._lock:
            self._counter += 1
            tid = self._counter
            self._tensors[tid] = {
                'block_id': block_id,
                'ptr': ptr,
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'size': arr.nbytes,
            }
            return tid
    
    def _get_arr(self, tid: int) -> np.ndarray:
        with self._lock:
            entry = self._tensors.get(tid)
            if entry is None:
                raise KeyError(f"Tensor {tid} not found")
        
        ptr = entry['ptr']
        size = entry['size']
        raw = self._driver.memcpy_device_to_host(ptr, size)
        return np.frombuffer(raw, dtype=np.dtype(entry['dtype'])).reshape(entry['shape']).copy()
    
    def _encode(self, arr: np.ndarray) -> dict:
        import base64
        raw = arr.tobytes()
        encoded = base64.b64encode(raw).decode('ascii')
        fmt_map = {'float32': 'f', 'float64': 'd', 'int32': 'i', 'int64': 'q', 'uint8': 'B'}
        return {
            'data': encoded,
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'format': fmt_map.get(str(arr.dtype), 'f'),
            'size': len(raw),
        }
    
    def _decode(self, params: dict) -> np.ndarray:
        import base64
        encoded = params.get('data', '')
        shape = tuple(params.get('shape', []))
        dtype_str = params.get('dtype', 'float32')
        raw = base64.b64decode(encoded)
        dtype_map = {'float32': np.float32, 'float64': np.float64, 'int32': np.int32, 'int64': np.int64, 'uint8': np.uint8}
        return np.frombuffer(raw, dtype=dtype_map.get(dtype_str, np.float32)).reshape(shape).copy()
    
    def _op_ping(self, p): return {'status': 'ok', 'device': f'cuda:{self._device_index}', 'backend': 'direct_cuda'}
    def _op_info(self, p): return {'status': 'ok', 'device': f'cuda:{self._device_index}', 'backend': 'direct_cuda', 'gpu': True}
    def _op_zeros(self, p): return self._alloc(np.zeros(p.get('shape', [1]), dtype=np.float32))
    def _op_ones(self, p): return self._alloc(np.ones(p.get('shape', [1]), dtype=np.float32))
    def _op_full(self, p): return self._alloc(np.full(p.get('shape', [1]), p.get('value', 0.0), dtype=np.float32))
    def _op_send_tensor(self, p): return self._alloc(self._decode(p))
    def _op_get_tensor(self, p): return self._encode(self._get_arr(p.get('tensor_id')))
    
    def _op_free_tensor(self, p):
        tid = p.get('tensor_id')
        with self._lock:
            entry = self._tensors.pop(tid, None)
        if entry:
            self._memory.free(entry['block_id'])
        return {'status': 'ok'}
    
    def _op_add(self, p):
        a = self._get_arr(p['a_id'])
        b = self._get_arr(p['b_id'])
        return self._alloc(a + b)
    
    def _op_subtract(self, p):
        a = self._get_arr(p['a_id'])
        b = self._get_arr(p['b_id'])
        return self._alloc(a - b)
    
    def _op_multiply(self, p):
        a = self._get_arr(p['a_id'])
        b = self._get_arr(p['b_id'])
        return self._alloc(a * b)
    
    def _op_divide(self, p):
        a = self._get_arr(p['a_id'])
        b = self._get_arr(p['b_id'])
        return self._alloc(a / b)
    
    def _op_relu(self, p):
        a = self._get_arr(p['tensor_id'])
        a[a < 0] = 0
        return self._alloc(a)
    
    def _op_sum(self, p):
        a = self._get_arr(p['tensor_id'])
        dim = p.get('dim')
        if dim is None:
            return self._alloc(np.array([a.sum()], dtype=a.dtype))
        return self._alloc(a.sum(axis=dim, keepdims=p.get('keepdim', False)))
    
    def _op_mean(self, p):
        a = self._get_arr(p['tensor_id'])
        dim = p.get('dim')
        if dim is None:
            return self._alloc(np.array([a.mean()], dtype=a.dtype))
        return self._alloc(a.mean(axis=dim, keepdims=p.get('keepdim', False)))
    
    def _op_reshape(self, p):
        a = self._get_arr(p['tensor_id'])
        return self._alloc(a.reshape(p['shape']))
    
    def _op_abs(self, p):
        a = self._get_arr(p['tensor_id'])
        return self._alloc(np.abs(a))
    
    def shutdown(self):
        with self._lock:
            for entry in self._tensors.values():
                try:
                    self._memory.free(entry['block_id'])
                except Exception:
                    pass
            self._tensors.clear()
        self._driver.shutdown()


# ============================================================
#  Main GPU Service
# ============================================================

class GPUService:
    """Dual-backend computation service."""
    
    def __init__(self, port: int = 55555, host: str = '0.0.0.0'):
        self.port = port
        self.host = host
        self.server_id = str(uuid.uuid4())[:8]
        self._running = False
        
        # Detect backend
        print("Detecting computation backend...")
        self._backend_name, self._backend_info = _detect_backend()
        
        # Create engine
        if self._backend_name in (BACKEND_PYTORCH_CUDA, BACKEND_PYTORCH_CPU):
            self._engine = PyTorchComputeEngine()
        elif self._backend_name == BACKEND_DIRECT_CUDA:
            self._engine = DirectCUDAComputeEngine()
        else:
            print("ERROR: No usable backend found!")
            print("Install PyTorch: pip install remotecuda[server]")
            print("Or install NVIDIA driver for direct GPU access.")
            sys.exit(1)
        
        # Discovery
        self._discovery = NetworkDiscovery()
        
        # Server
        self._server_socket: Optional[socket.socket] = None
        self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="Worker")
        self._sessions: Dict[str, dict] = {}
        self._session_counter = 0
    
    def start(self):
        if self._engine is None:
            print("ERROR: No compute engine available.")
            return
        
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)
        self._running = True
        
        # Start discovery
        gpu_info = self._backend_info.copy()
        gpu_info['server_id'] = self.server_id
        gpu_info['port'] = self.port
        gpu_info['version'] = '3.0.0'
        self._discovery.start_broadcasting(self.port, gpu_info)
        
        print(f"""
+==================================================================+
|              RemoteCUDA v3.0 — Computation Service              |
+==================================================================+
|  Server ID:   {self.server_id:<50}|
|  Backend:     {self._backend_name:<50}|
|  Device:      {self._backend_info.get('device', 'unknown'):<50}|
|  Host:        {self.host}:{self.port:<44}|
|  Discovery:   Enabled                                         |
+==================================================================+
|  Ready! Press Ctrl+C to stop.                                 |
+==================================================================+
        """)
        
        while self._running:
            try:
                self._server_socket.settimeout(1.0)
                client_socket, address = self._server_socket.accept()
                self._executor.submit(self._handle_client, client_socket, address)
            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    print(f"Accept error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        self._session_counter += 1
        session_id = f"S{self._session_counter:06d}"
        client_ip = address[0]
        
        self._sessions[session_id] = {'ip': client_ip, 'started': time.time()}
        
        buffer = b''
        client_socket.settimeout(30.0)
        
        try:
            while self._running:
                try:
                    data = client_socket.recv(65536)
                except socket.timeout:
                    continue
                
                if not data:
                    break
                
                buffer += data
                
                while True:
                    message, buffer = MessageProtocol.decode(buffer)
                    if message is None:
                        break
                    
                    command = message.get('command', '')
                    params = message.get('params', {})
                    request_id = message.get('id', 0)
                    
                    result = self._engine.execute(command, params)
                    
                    if isinstance(result, dict) and 'error' in result:
                        response = {'status': 'error', 'error': result['error'], 'id': request_id}
                    else:
                        response = {'status': 'ok', 'result': result, 'id': request_id}
                    
                    response_bytes = MessageProtocol.encode_response(response)
                    try:
                        client_socket.sendall(response_bytes)
                    except OSError:
                        return
                    
        except (OSError, ConnectionError):
            pass
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            self._sessions.pop(session_id, None)
    
    def stop(self):
        print("\nShutting down...")
        self._running = False
        self._discovery.stop()
        
        if self._server_socket:
            self._server_socket.close()
        
        self._executor.shutdown(wait=True, cancel_futures=True)
        
        if self._engine:
            self._engine.shutdown()
        
        print("Service stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='RemoteCUDA v3.0 — Computation Service')
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    
    service = GPUService(port=args.port, host=args.host)
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()


if __name__ == '__main__':
    main()