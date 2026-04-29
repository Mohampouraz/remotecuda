"""
GPU Worker Module
=================
Dedicated worker thread for managing operations on a single physical GPU.
Each GPU in the system gets its own worker for isolated, parallel execution.

The worker handles all aspects of GPU resource management:
    - Memory allocation and tracking
    - Operation execution and scheduling
    - Tensor lifecycle management
    - Performance monitoring and statistics
    - Error handling and recovery
    - Resource cleanup and defragmentation

This is the core execution unit on the server side - all actual
CUDA operations flow through GPU workers.
"""

import threading
import time
import queue
import uuid
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..protocol.compression import TensorCompressor


class WorkerStatus(Enum):
    """
    Lifecycle states of a GPU worker.
    """
    INITIALIZING = 'initializing'
    READY = 'ready'
    BUSY = 'busy'
    DRAINING = 'draining'       # Finishing current work, no new tasks
    MAINTENANCE = 'maintenance'  # Running cleanup/optimization
    ERROR = 'error'
    STOPPED = 'stopped'


@dataclass
class MemoryBlock:
    """
    Represents an allocated block of GPU memory.
    
    Tracks allocation details for memory management and defragmentation.
    """
    block_id: str
    tensor_id: int
    size_bytes: int
    address: int = 0  # Virtual address (for tracking, not actual pointer)
    allocated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    owner_task_id: Optional[str] = None
    is_pinned: bool = False
    
    def touch(self):
        """
        Update last access time.
        """
        self.last_accessed = time.time()
    
    @property
    def age_seconds(self) -> float:
        """
        Time since allocation in seconds.
        """
        return time.time() - self.allocated_at
    
    @property
    def idle_seconds(self) -> float:
        """
        Time since last access in seconds.
        """
        return time.time() - self.last_accessed


@dataclass
class WorkerTask:
    """
    A task queued for execution on this GPU worker.
    """
    task_id: str
    operation_type: str
    operation_data: dict
    priority: int = 0  # Lower number = higher priority
    submitted_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    
    # Result storage
    result: Optional[dict] = None
    error: Optional[str] = None
    completed: bool = False
    
    def __lt__(self, other: 'WorkerTask') -> bool:
        """
        Priority queue ordering.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.submitted_at < other.submitted_at


class GPUWorker:
    """
    Dedicated worker for a single physical GPU.
    
    Manages all operations on one GPU device, providing:
    - Isolated execution environment
    - Memory management with defragmentation
    - Object registry (tensors, models, optimizers)
    - Operation queuing and scheduling
    - Performance monitoring
    - Automatic cleanup
    
    Each worker runs in its own thread and manages exactly one
    CUDA device, enabling true parallel execution across multiple GPUs.
    
    Architecture:
        ┌─────────────────────────────────────────┐
        │              GPU Worker                 │
        │  ┌───────────────────────────────────┐  │
        │  │         Task Queue               │  │
        │  │  ┌─────┐ ┌─────┐ ┌─────┐        │  │
        │  │  │Task1│ │Task2│ │Task3│ ...    │  │
        │  │  └─────┘ └─────┘ └─────┘        │  │
        │  └───────────────┬───────────────────┘  │
        │                  │                      │
        │  ┌───────────────▼───────────────────┐  │
        │  │        Task Processor             │  │
        │  │  • Forward/Backward               │  │
        │  │  • Optimizer step                 │  │
        │  │  • Tensor operations              │  │
        │  └───────────────┬───────────────────┘  │
        │                  │                      │
        │  ┌───────────────▼───────────────────┐  │
        │  │     Memory Manager                │  │
        │  │  • Allocation tracking            │  │
        │  │  • Garbage collection             │  │
        │  │  • Defragmentation                │  │
        │  └───────────────────────────────────┘  │
        │                                          │
        │  ┌───────────────────────────────────┐  │
        │  │     Object Registry               │  │
        │  │  • Tensor handles                 │  │
        │  │  • Model references               │  │
        │  │  • Optimizer state                │  │
        │  └───────────────────────────────────┘  │
        └─────────────────────────────────────────┘
    """
    
    # Maximum memory utilization before triggering cleanup
    MEMORY_CLEANUP_THRESHOLD = 0.85  # 85%
    
    # Maximum idle time before tensor is considered stale
    MAX_TENSOR_IDLE_SECONDS = 300  # 5 minutes
    
    # Maximum number of registered tensors before cleanup
    MAX_REGISTERED_TENSORS = 10000
    
    def __init__(
        self,
        gpu_index: int,
        worker_id: Optional[str] = None,
        max_queue_size: int = 1000,
        enable_profiling: bool = False
    ):
        """
        Initialize a GPU worker.
        
        Args:
            gpu_index (int): CUDA device index to manage
            worker_id (str, optional): Unique worker identifier
            max_queue_size (int): Maximum pending tasks in queue
            enable_profiling (bool): Enable detailed performance profiling
        """
        # Worker identity
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.gpu_index = gpu_index
        self.device = torch.device(f'cuda:{gpu_index}')
        self.max_queue_size = max_queue_size
        self.enable_profiling = enable_profiling
        
        # GPU information
        self._init_gpu_info()
        
        # Worker state
        self.status = WorkerStatus.INITIALIZING
        self._running = False
        self._lock = threading.RLock()
        
        # Task management
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._active_task: Optional[WorkerTask] = None
        self._task_counter = 0
        self._completed_tasks: Dict[str, WorkerTask] = {}
        
        # Main worker thread
        self._worker_thread: Optional[threading.Thread] = None
        
        # CUDA stream for async operations
        self._stream = torch.cuda.Stream(device=self.device)
        
        # Object registries
        self._tensors: Dict[int, torch.Tensor] = {}
        self._models: Dict[int, nn.Module] = {}
        self._optimizers: Dict[int, optim.Optimizer] = {}
        self._buffers: Dict[str, torch.Tensor] = {}
        
        # Memory management
        self._memory_blocks: Dict[str, MemoryBlock] = {}
        self._allocated_memory = 0
        self._peak_memory = 0
        
        # CUDA events for profiling
        self._cuda_events: Dict[str, torch.cuda.Event] = {}
        
        # Statistics
        self.stats = {
            'tasks_received': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'operations': defaultdict(int),
            'total_compute_time_ms': 0.0,
            'total_bytes_transferred': 0,
            'peak_allocated_memory': 0,
            'current_allocated_memory': 0,
            'out_of_memory_events': 0,
            'tensor_allocations': 0,
            'tensor_frees': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_task_time_ms': 0.0,
            'uptime_seconds': 0.0,
            'last_error': None,
            'last_error_time': None
        }
        
        # Profiling data
        self._profile_data: Dict[str, List[float]] = defaultdict(list)
        
        # Background maintenance
        self._maintenance_thread: Optional[threading.Thread] = None
        self._maintenance_running = False
        
        # Startup time
        self._start_time = time.time()
    
    def _init_gpu_info(self):
        """
        Initialize GPU device information.
        """
        try:
            props = torch.cuda.get_device_properties(self.gpu_index)
            self.gpu_name = props.name
            self.total_memory = props.total_memory
            self.compute_capability = (props.major, props.minor)
            self.multi_processor_count = props.multi_processor_count
            
            # Initialize CUDA context
            torch.cuda.set_device(self.gpu_index)
            
            self.status = WorkerStatus.READY
            
        except Exception as e:
            self.status = WorkerStatus.ERROR
            raise RuntimeError(
                f"Failed to initialize GPU {self.gpu_index}: {e}"
            )
    
    @property
    def free_memory(self) -> int:
        """
        Available GPU memory in bytes.
        """
        return self.total_memory - self._allocated_memory
    
    @property
    def memory_pressure(self) -> float:
        """
        Memory utilization ratio (0.0-1.0).
        """
        if self.total_memory == 0:
            return 1.0
        return self._allocated_memory / self.total_memory
    
    @property
    def can_accept_tasks(self) -> bool:
        """
        Check if worker can accept new tasks.
        """
        if self.status not in [WorkerStatus.READY, WorkerStatus.BUSY]:
            return False
        if self._task_queue.full():
            return False
        if self.memory_pressure > self.MEMORY_CLEANUP_THRESHOLD:
            return False
        return True
    
    def start(self):
        """
        Start the worker thread.
        
        Begins processing tasks from the queue.
        """
        if self._running:
            return
        
        self._running = True
        self.status = WorkerStatus.READY
        
        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"GPUWorker-{self.gpu_index}"
        )
        self._worker_thread.start()
        
        # Start maintenance thread
        self._start_maintenance()
        
        print(f"✅ GPU Worker {self.worker_id} started")
        print(f"   Device: {self.gpu_name}")
        print(f"   Memory: {self.total_memory / (1024**3):.1f} GB")
        print(f"   CUDA Capability: {self.compute_capability[0]}.{self.compute_capability[1]}")
    
    def stop(self, drain: bool = True):
        """
        Stop the worker gracefully.
        
        Args:
            drain (bool): If True, complete pending tasks before stopping
        """
        if not self._running:
            return
        
        if drain:
            self.status = WorkerStatus.DRAINING
            print(f"⏳ Worker {self.worker_id} draining...")
            
            # Wait for queue to empty
            while not self._task_queue.empty():
                time.sleep(0.1)
        
        self._running = False
        self.status = WorkerStatus.STOPPED
        
        # Signal worker thread
        self._task_queue.put(WorkerTask(
            task_id='__SHUTDOWN__',
            operation_type='shutdown',
            operation_data={},
            priority=-1  # Highest priority
        ))
        
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        
        # Stop maintenance
        self._stop_maintenance()
        
        # Clean up all resources
        self._cleanup_all()
        
        print(f"✅ Worker {self.worker_id} stopped")
    
    def submit_task(
        self,
        operation_type: str,
        operation_data: dict,
        priority: int = 50,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None
    ) -> str:
        """
        Submit a task for execution on this GPU.
        
        Args:
            operation_type (str): Type of operation
            operation_data (dict): Operation parameters
            priority (int): Priority (0-100, lower = higher priority)
            timeout (float, optional): Task timeout in seconds
            callback (callable): Called on success with result dict
            error_callback (callable): Called on error with Exception
            
        Returns:
            str: Task ID for tracking
            
        Raises:
            RuntimeError: If worker cannot accept tasks
        """
        if not self.can_accept_tasks:
            raise RuntimeError(
                f"Worker {self.worker_id} cannot accept tasks "
                f"(status: {self.status.value})"
            )
        
        with self._lock:
            self._task_counter += 1
            task_id = f"{self.worker_id}_task_{self._task_counter:08d}"
        
        task = WorkerTask(
            task_id=task_id,
            operation_type=operation_type,
            operation_data=operation_data,
            priority=priority,
            timeout=timeout,
            callback=callback,
            error_callback=error_callback
        )
        
        self._task_queue.put(task)
        self.stats['tasks_received'] += 1
        
        return task_id
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> dict:
        """
        Wait for a specific task to complete.
        
        Args:
            task_id (str): Task ID
            timeout (float, optional): Maximum wait time
            
        Returns:
            dict: Task result
            
        Raises:
            ValueError: If task not found
            TimeoutError: If task doesn't complete
            RuntimeError: If task failed
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id in self._completed_tasks:
                    task = self._completed_tasks.pop(task_id)
                    
                    if task.error:
                        raise RuntimeError(f"Task failed: {task.error}")
                    return task.result or {}
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            
            time.sleep(0.001)
    
    def get_info(self) -> dict:
        """
        Get current worker and GPU information.
        
        Returns:
            dict: Worker status and GPU metrics
        """
        self.stats['uptime_seconds'] = time.time() - self._start_time
        self.stats['current_allocated_memory'] = self._allocated_memory
        self.stats['peak_allocated_memory'] = self._peak_memory
        
        return {
            'worker_id': self.worker_id,
            'gpu_index': self.gpu_index,
            'gpu_name': self.gpu_name,
            'status': self.status.value,
            'total_memory': self.total_memory,
            'free_memory': self.free_memory,
            'allocated_memory': self._allocated_memory,
            'memory_pressure': self.memory_pressure,
            'registered_tensors': len(self._tensors),
            'registered_models': len(self._models),
            'queue_size': self._task_queue.qsize(),
            'stats': dict(self.stats)
        }
    
    def _worker_loop(self):
        """
        Main worker processing loop.
        
        Continuously dequeues and executes tasks.
        """
        while self._running or not self._task_queue.empty():
            try:
                # Get next task with timeout
                task = self._task_queue.get(timeout=0.5)
                
                # Handle shutdown signal
                if task.operation_type == 'shutdown':
                    self._task_queue.task_done()
                    break
                
                # Execute task
                self._active_task = task
                self.status = WorkerStatus.BUSY
                
                self._execute_task(task)
                
                self._active_task = None
                self.status = WorkerStatus.READY
                
                self._task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {self.worker_id} loop error: {e}")
                traceback.print_exc()
    
    def _execute_task(self, task: WorkerTask):
        """
        Execute a single task.
        
        Args:
            task (WorkerTask): Task to execute
        """
        start_time = time.time()
        
        try:
            # Execute based on operation type
            result = self._execute_operation(
                task.operation_type,
                task.operation_data
            )
            
            # Update statistics
            elapsed = (time.time() - start_time) * 1000
            with self._lock:
                self.stats['tasks_completed'] += 1
                self.stats['operations'][task.operation_type] += 1
                self.stats['total_compute_time_ms'] += elapsed
                
                # Update running average
                n = self.stats['tasks_completed']
                self.stats['avg_task_time_ms'] = (
                    (self.stats['avg_task_time_ms'] * (n - 1) + elapsed) / n
                )
            
            # Store result
            task.result = result
            task.completed = True
            
            # Add to completed tasks
            with self._lock:
                self._completed_tasks[task.task_id] = task
            
            # Call success callback
            if task.callback:
                try:
                    task.callback(result)
                except Exception as cb_error:
                    print(f"Task callback error: {cb_error}")
            
            # Profile if enabled
            if self.enable_profiling:
                self._profile_data[f"op_{task.operation_type}"].append(elapsed)
        
        except Exception as e:
            # Handle failure
            error_msg = f"{type(e).__name__}: {str(e)}"
            task.error = error_msg
            task.completed = True
            
            with self._lock:
                self.stats['tasks_failed'] += 1
                self.stats['last_error'] = error_msg
                self.stats['last_error_time'] = time.time()
                
                # Track OOM events specifically
                if 'out of memory' in str(e).lower():
                    self.stats['out_of_memory_events'] += 1
            
            # Add to completed tasks
            with self._lock:
                self._completed_tasks[task.task_id] = task
            
            # Call error callback
            if task.error_callback:
                try:
                    task.error_callback(e)
                except Exception as cb_error:
                    print(f"Error callback failed: {cb_error}")
    
    def _execute_operation(self, op_type: str, data: dict) -> dict:
        """
        Execute a specific GPU operation.
        
        Args:
            op_type (str): Operation type
            data (dict): Operation parameters
            
        Returns:
            dict: Operation result
        """
        # Route to appropriate handler
        handlers = {
            'ping': self._op_ping,
            'get_info': self._op_get_info,
            'allocate_tensor': self._op_allocate_tensor,
            'free_tensor': self._op_free_tensor,
            'get_tensor_data': self._op_get_tensor_data,
            'forward': self._op_forward,
            'backward': self._op_backward,
            'optimizer_step': self._op_optimizer_step,
            'register_model': self._op_register_model,
            'unregister_model': self._op_unregister_model,
            'register_optimizer': self._op_register_optimizer,
            'get_parameters': self._op_get_parameters,
            'set_parameters': self._op_set_parameters,
            'train_step': self._op_train_step,
            'inference': self._op_inference,
            'clear_cache': self._op_clear_cache,
            'defragment': self._op_defragment,
            'get_memory_map': self._op_get_memory_map,
        }
        
        handler = handlers.get(op_type)
        if handler is None:
            return {'error': f'Unknown operation type: {op_type}'}
        
        return handler(data)
    
    def _op_ping(self, data: dict) -> dict:
        """
        Health check operation.
        """
        return {
            'status': 'ok',
            'worker_id': self.worker_id,
            'device': self.gpu_name,
            'memory_free': self.free_memory,
            'memory_total': self.total_memory,
            'queue_size': self._task_queue.qsize()
        }
    
    def _op_get_info(self, data: dict) -> dict:
        """
        Get detailed worker information.
        """
        return self.get_info()
    
    def _op_allocate_tensor(self, data: dict) -> dict:
        """
        Allocate a tensor on this GPU.
        """
        compressed_data = data.get('data')
        if compressed_data is None:
            return {'error': 'No tensor data provided'}
        
        try:
            # Decompress and move to GPU
            tensor = TensorCompressor.decompress_tensor(compressed_data, self.device)
            
            # Generate tensor ID
            tensor_id = data.get('tensor_id', id(tensor))
            
            # Calculate memory usage
            memory_used = tensor.element_size() * tensor.nelement()
            
            # Check memory availability
            if memory_used > self.free_memory * 0.95:
                # Try to free up memory
                self._garbage_collect()
                
                if memory_used > self.free_memory * 0.95:
                    return {
                        'error': 'Out of memory',
                        'requested_bytes': memory_used,
                        'free_bytes': self.free_memory
                    }
            
            # Register tensor
            with self._lock:
                self._tensors[tensor_id] = tensor
                self._allocated_memory += memory_used
                
                if self._allocated_memory > self._peak_memory:
                    self._peak_memory = self._allocated_memory
                
                # Create memory block record
                block = MemoryBlock(
                    block_id=f"block_{tensor_id}",
                    tensor_id=tensor_id,
                    size_bytes=memory_used
                )
                self._memory_blocks[block.block_id] = block
                
                self.stats['tensor_allocations'] += 1
                self.stats['total_bytes_transferred'] += len(compressed_data)
            
            return {
                'status': 'ok',
                'tensor_id': tensor_id,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'size_bytes': memory_used,
                'device': str(self.device)
            }
            
        except torch.cuda.OutOfMemoryError as e:
            return {
                'error': f'CUDA out of memory: {e}',
                'free_bytes': self.free_memory
            }
        except Exception as e:
            return {'error': f'Tensor allocation failed: {e}'}
    
    def _op_free_tensor(self, data: dict) -> dict:
        """
        Free a tensor from GPU memory.
        """
        tensor_id = data.get('tensor_id')
        
        with self._lock:
            tensor = self._tensors.pop(tensor_id, None)
            
            if tensor is not None:
                memory_freed = tensor.element_size() * tensor.nelement()
                self._allocated_memory = max(0, self._allocated_memory - memory_freed)
                del tensor
                
                # Remove memory block
                block_key = f"block_{tensor_id}"
                self._memory_blocks.pop(block_key, None)
                
                self.stats['tensor_frees'] += 1
                
                return {
                    'status': 'ok',
                    'freed_bytes': memory_freed
                }
        
        return {'status': 'ok', 'note': 'Tensor not found'}
    
    def _op_get_tensor_data(self, data: dict) -> dict:
        """
        Get serialized tensor data from GPU.
        """
        tensor_id = data.get('tensor_id')
        
        with self._lock:
            tensor = self._tensors.get(tensor_id)
            
            if tensor is None:
                return {'error': f'Tensor {tensor_id} not found'}
            
            # Touch memory block
            block_key = f"block_{tensor_id}"
            block = self._memory_blocks.get(block_key)
            if block:
                block.touch()
            
            # Serialize tensor
            compressed_data = TensorCompressor.compress_tensor(tensor)
        
        return {
            'status': 'ok',
            'data': compressed_data,
            'data_size': len(compressed_data),
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        }
    
    def _op_forward(self, data: dict) -> dict:
        """
        Execute forward pass on a registered model.
        """
        model_id = data.get('model_id')
        input_ids = data.get('input_tensor_ids', [])
        no_grad = data.get('no_grad', True)
        
        # Get model
        model = self._models.get(model_id)
        if model is None:
            return {'error': f'Model {model_id} not found'}
        
        # Gather inputs
        inputs = []
        for tid in input_ids:
            tensor = self._tensors.get(tid)
            if tensor is None:
                return {'error': f'Input tensor {tid} not found'}
            inputs.append(tensor)
        
        if not inputs:
            return {'error': 'No input tensors provided'}
        
        try:
            # Execute forward pass
            start_time = time.time()
            
            with torch.cuda.stream(self._stream):
                if no_grad:
                    with torch.no_grad():
                        output = model(*inputs) if len(inputs) > 1 else model(inputs[0])
                else:
                    output = model(*inputs) if len(inputs) > 1 else model(inputs[0])
            
            torch.cuda.synchronize(self.device)
            compute_time = (time.time() - start_time) * 1000
            
            # Handle multiple outputs
            if isinstance(output, tuple):
                output_ids = []
                output_data = []
                for i, out in enumerate(output):
                    oid = id(out)
                    output_ids.append(oid)
                    self._tensors[oid] = out
                    output_data.append(TensorCompressor.compress_tensor(out))
            else:
                output_id = id(output)
                output_ids = [output_id]
                self._tensors[output_id] = output
                output_data = [TensorCompressor.compress_tensor(output)]
            
            return {
                'status': 'ok',
                'output_ids': output_ids,
                'output_data': output_data[0] if len(output_data) == 1 else output_data,
                'compute_time_ms': compute_time
            }
            
        except Exception as e:
            return {'error': f'Forward pass failed: {e}'}
    
    def _op_backward(self, data: dict) -> dict:
        """
        Execute backward pass.
        """
        tensor_id = data.get('tensor_id')
        gradient_data = data.get('gradient')
        
        tensor = self._tensors.get(tensor_id)
        if tensor is None:
            return {'error': f'Tensor {tensor_id} not found'}
        
        try:
            start_time = time.time()
            
            with torch.cuda.stream(self._stream):
                if gradient_data is not None:
                    grad = TensorCompressor.decompress_tensor(gradient_data, self.device)
                    tensor.backward(gradient=grad)
                else:
                    tensor.backward()
            
            torch.cuda.synchronize(self.device)
            compute_time = (time.time() - start_time) * 1000
            
            # Collect computed gradients
            gradients = {}
            for model_id, model in self._models.items():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        key = f"{model_id}_{name}"
                        gradients[key] = TensorCompressor.compress_tensor(param.grad)
            
            return {
                'status': 'ok',
                'gradients': gradients,
                'compute_time_ms': compute_time
            }
            
        except Exception as e:
            return {'error': f'Backward pass failed: {e}'}
    
    def _op_optimizer_step(self, data: dict) -> dict:
        """
        Execute optimizer step.
        """
        optimizer_id = data.get('optimizer_id')
        
        optimizer = self._optimizers.get(optimizer_id)
        if optimizer is None:
            return {'error': f'Optimizer {optimizer_id} not found'}
        
        try:
            start_time = time.time()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize(self.device)
            compute_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'ok',
                'compute_time_ms': compute_time
            }
            
        except Exception as e:
            return {'error': f'Optimizer step failed: {e}'}
    
    def _op_register_model(self, data: dict) -> dict:
        """
        Register a model for future operations.
        """
        model_id = data.get('model_id')
        model_type = data.get('model_type', 'sequential')
        model_params = data.get('parameters', {})
        
        # Reconstruct model from parameters
        try:
            model = self._reconstruct_model(model_type, model_params)
            model = model.to(self.device)
            
            with self._lock:
                self._models[model_id] = model
            
            return {
                'status': 'ok',
                'model_id': model_id,
                'parameter_count': sum(p.numel() for p in model.parameters())
            }
            
        except Exception as e:
            return {'error': f'Model registration failed: {e}'}
    
    def _op_unregister_model(self, data: dict) -> dict:
        """
        Unregister and free a model.
        """
        model_id = data.get('model_id')
        
        with self._lock:
            model = self._models.pop(model_id, None)
            if model is not None:
                del model
        
        return {'status': 'ok'}
    
    def _op_register_optimizer(self, data: dict) -> dict:
        """
        Register an optimizer.
        """
        optimizer_id = data.get('optimizer_id')
        model_id = data.get('model_id')
        optimizer_type = data.get('optimizer_type', 'adam')
        optimizer_params = data.get('optimizer_params', {})
        
        model = self._models.get(model_id)
        if model is None:
            return {'error': f'Model {model_id} not found'}
        
        try:
            if optimizer_type == 'adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=optimizer_params.get('lr', 0.001)
                )
            elif optimizer_type == 'sgd':
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=optimizer_params.get('lr', 0.01)
                )
            else:
                return {'error': f'Unknown optimizer type: {optimizer_type}'}
            
            with self._lock:
                self._optimizers[optimizer_id] = optimizer
            
            return {
                'status': 'ok',
                'optimizer_id': optimizer_id
            }
            
        except Exception as e:
            return {'error': f'Optimizer registration failed: {e}'}
    
    def _op_get_parameters(self, data: dict) -> dict:
        """
        Get serialized model parameters.
        """
        model_id = data.get('model_id')
        
        model = self._models.get(model_id)
        if model is None:
            return {'error': f'Model {model_id} not found'}
        
        params = {}
        for name, param in model.named_parameters():
            params[name] = TensorCompressor.compress_tensor(param.data)
        
        return {
            'status': 'ok',
            'parameters': params
        }
    
    def _op_set_parameters(self, data: dict) -> dict:
        """
        Set model parameters from serialized data.
        """
        model_id = data.get('model_id')
        parameters = data.get('parameters', {})
        
        model = self._models.get(model_id)
        if model is None:
            return {'error': f'Model {model_id} not found'}
        
        try:
            state_dict = {}
            for name, compressed_param in parameters.items():
                state_dict[name] = TensorCompressor.decompress_tensor(
                    compressed_param,
                    self.device
                )
            
            model.load_state_dict(state_dict)
            return {'status': 'ok'}
            
        except Exception as e:
            return {'error': f'Parameter loading failed: {e}'}
    
    def _op_train_step(self, data: dict) -> dict:
        """
        Complete training step: forward + backward + optimizer step.
        """
        model_id = data.get('model_id')
        optimizer_id = data.get('optimizer_id')
        input_ids = data.get('input_ids', [])
        target_ids = data.get('target_ids', [])
        
        model = self._models.get(model_id)
        optimizer = self._optimizers.get(optimizer_id)
        
        if model is None:
            return {'error': f'Model {model_id} not found'}
        if optimizer is None:
            return {'error': f'Optimizer {optimizer_id} not found'}
        
        # Gather inputs and targets
        inputs = [self._tensors.get(tid) for tid in input_ids]
        targets = [self._tensors.get(tid) for tid in target_ids]
        
        if None in inputs or (targets and None in targets):
            return {'error': 'Input or target tensor not found'}
        
        try:
            start_time = time.time()
            
            # Training step
            optimizer.zero_grad()
            output = model(*inputs) if len(inputs) > 1 else model(inputs[0])
            
            if targets:
                loss_fn = nn.CrossEntropyLoss()
                target = targets[0] if len(targets) == 1 else targets
                loss = loss_fn(output, target)
            else:
                loss = output
            
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize(self.device)
            compute_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'ok',
                'loss': loss.item() if torch.is_tensor(loss) else loss,
                'compute_time_ms': compute_time
            }
            
        except Exception as e:
            return {'error': f'Training step failed: {e}'}
    
    def _op_inference(self, data: dict) -> dict:
        """
        Run inference with no gradient computation.
        """
        return self._op_forward({**data, 'no_grad': True})
    
    def _op_clear_cache(self, data: dict) -> dict:
        """
        Clear CUDA cache.
        """
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return {'status': 'ok'}
    
    def _op_defragment(self, data: dict) -> dict:
        """
        Defragment GPU memory.
        """
        # In CUDA, memory defragmentation is handled by the driver
        # but we can do some cleanup
        self._garbage_collect(aggressive=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        
        return {
            'status': 'ok',
            'free_memory_after': self.free_memory
        }
    
    def _op_get_memory_map(self, data: dict) -> dict:
        """
        Get memory allocation map.
        """
        blocks = []
        for block_id, block in self._memory_blocks.items():
            blocks.append({
                'block_id': block_id,
                'tensor_id': block.tensor_id,
                'size_bytes': block.size_bytes,
                'allocated_at': block.allocated_at,
                'last_accessed': block.last_accessed,
                'age_seconds': block.age_seconds,
                'idle_seconds': block.idle_seconds
            })
        
        return {
            'status': 'ok',
            'total_blocks': len(blocks),
            'total_allocated': self._allocated_memory,
            'free_memory': self.free_memory,
            'blocks': blocks
        }
    
    def _reconstruct_model(
        self,
        model_type: str,
        parameters: dict
    ) -> nn.Module:
        """
        Reconstruct a model from serialized parameters.
        
        Args:
            model_type (str): Type of model
            parameters (dict): Serialized model parameters
            
        Returns:
            nn.Module: Reconstructed model
        """
        # This is a simplified reconstruction
        # In production, you'd use model architecture definitions
        
        if model_type == 'linear':
            in_features = parameters.get('in_features', 784)
            out_features = parameters.get('out_features', 10)
            return nn.Linear(in_features, out_features)
        
        elif model_type == 'sequential':
            layers = parameters.get('layers', [])
            model = nn.Sequential()
            for i, layer_config in enumerate(layers):
                layer_type = layer_config.get('type')
                if layer_type == 'linear':
                    model.add_module(
                        f'linear_{i}',
                        nn.Linear(
                            layer_config.get('in_features', 784),
                            layer_config.get('out_features', 256)
                        )
                    )
                elif layer_type == 'relu':
                    model.add_module(f'relu_{i}', nn.ReLU())
                elif layer_type == 'dropout':
                    model.add_module(
                        f'dropout_{i}',
                        nn.Dropout(layer_config.get('p', 0.5))
                    )
            return model
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _garbage_collect(self, aggressive: bool = False):
        """
        Collect garbage and free unused tensors.
        
        Args:
            aggressive (bool): If True, free more aggressively
        """
        threshold = self.MAX_TENSOR_IDLE_SECONDS
        if aggressive:
            threshold = threshold / 3  # Much more aggressive
        
        current_time = time.time()
        to_free = []
        
        with self._lock:
            for block_id, block in self._memory_blocks.items():
                if block.is_pinned:
                    continue
                
                idle_time = current_time - block.last_accessed
                if idle_time > threshold:
                    to_free.append(block)
            
            for block in to_free:
                tensor = self._tensors.pop(block.tensor_id, None)
                if tensor is not None:
                    self._allocated_memory = max(
                        0,
                        self._allocated_memory - block.size_bytes
                    )
                    del tensor
                    self._memory_blocks.pop(block.block_id, None)
                    self.stats['tensor_frees'] += 1
        
        if to_free:
            torch.cuda.empty_cache()
    
    def _start_maintenance(self, interval: float = 30.0):
        """
        Start background maintenance thread.
        
        Args:
            interval (float): Maintenance interval in seconds
        """
        if self._maintenance_running:
            return
        
        self._maintenance_running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            args=(interval,),
            daemon=True
        )
        self._maintenance_thread.start()
    
    def _maintenance_loop(self, interval: float):
        """
        Background maintenance loop.
        """
        while self._maintenance_running:
            time.sleep(interval)
            
            if self.memory_pressure > self.MEMORY_CLEANUP_THRESHOLD:
                self._garbage_collect(aggressive=True)
            
            if len(self._tensors) > self.MAX_REGISTERED_TENSORS:
                self._garbage_collect(aggressive=True)
            
            # Clean up old completed tasks
            with self._lock:
                old_tasks = []
                for task_id, task in self._completed_tasks.items():
                    if time.time() - task.submitted_at > 300:  # 5 minutes
                        old_tasks.append(task_id)
                
                for task_id in old_tasks:
                    del self._completed_tasks[task_id]
    
    def _stop_maintenance(self):
        """
        Stop background maintenance.
        """
        self._maintenance_running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)
    
    def _cleanup_all(self):
        """
        Clean up all GPU resources.
        """
        with self._lock:
            # Clear registries
            for tensor in self._tensors.values():
                del tensor
            for model in self._models.values():
                del model
            for optimizer in self._optimizers.values():
                del optimizer
            for buffer_tensor in self._buffers.values():
                del buffer_tensor
            
            self._tensors.clear()
            self._models.clear()
            self._optimizers.clear()
            self._buffers.clear()
            self._memory_blocks.clear()
            self._allocated_memory = 0
        
        # Clear CUDA memory
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    def __repr__(self) -> str:
        return (
            f"GPUWorker(id={self.worker_id}, "
            f"device={self.gpu_name}, "
            f"status={self.status.value}, "
            f"memory={self._allocated_memory / (1024**2):.0f}MB/"
            f"{self.total_memory / (1024**3):.1f}GB)"
        )