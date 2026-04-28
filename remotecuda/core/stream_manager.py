"""
Stream Manager Module
=====================
Manages CUDA stream operations for asynchronous execution.
Enables pipelining of data transfers and computation for
maximum throughput across the network.
"""

import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import torch


class StreamPriority(Enum):
    """
    Priority levels for stream operations.
    """
    HIGH = 0    # Latency-sensitive operations
    NORMAL = 1  # Standard operations
    LOW = 2     # Background operations (prefetching, etc.)
    IDLE = 3    # Lowest priority, only when idle


@dataclass
class StreamOperation:
    """
    Represents a single operation in a CUDA stream.
    """
    operation_id: int
    op_type: str
    data: Any
    priority: StreamPriority = StreamPriority.NORMAL
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    timestamp: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    
    # Execution tracking
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = 'pending'  # pending, running, completed, failed
    result: Any = None
    error: Optional[str] = None


class OperationStream:
    """
    A single stream of operations executed sequentially on a GPU.
    
    Mimics CUDA stream semantics: operations within a stream
    are executed in order, but different streams can execute
    concurrently.
    
    Features:
        - Ordered execution
        - Priority queuing
        - Timeout handling
        - Error propagation
    """
    
    def __init__(self, stream_id: int, gpu_connection, max_queue_size: int = 100):
        """
        Initialize an operation stream.
        
        Args:
            stream_id (int): Unique stream identifier
            gpu_connection: GPU connection for executing operations
            max_queue_size (int): Maximum pending operations
        """
        self.stream_id = stream_id
        self.gpu = gpu_connection
        self.max_queue_size = max_queue_size
        
        # Operation queues per priority
        self._queues: Dict[StreamPriority, queue.PriorityQueue] = {
            StreamPriority.HIGH: queue.PriorityQueue(),
            StreamPriority.NORMAL: queue.PriorityQueue(),
            StreamPriority.LOW: queue.PriorityQueue(),
            StreamPriority.IDLE: queue.PriorityQueue()
        }
        
        # Completed operations waiting for result collection
        self._completed: Dict[int, StreamOperation] = {}
        
        # Execution thread
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Operation counter
        self._operation_counter = 0
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'total_compute_time': 0.0,
            'total_queue_time': 0.0
        }
    
    def submit(
        self,
        op_type: str,
        data: Any,
        priority: StreamPriority = StreamPriority.NORMAL,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        timeout: Optional[float] = None
    ) -> int:
        """
        Submit an operation to this stream.
        
        Args:
            op_type (str): Type of operation to execute
            data (Any): Operation data
            priority (StreamPriority): Operation priority
            callback (Callable, optional): Called on success with result
            error_callback (Callable, optional): Called on error with exception
            timeout (float, optional): Maximum execution time in seconds
            
        Returns:
            int: Operation ID for tracking
            
        Raises:
            RuntimeError: If stream is not running or queue is full
        """
        if not self._running:
            raise RuntimeError(f"Stream {self.stream_id} is not running")
        
        with self._lock:
            self._operation_counter += 1
            operation_id = self._operation_counter
            self.stats['submitted'] += 1
        
        # Create operation
        operation = StreamOperation(
            operation_id=operation_id,
            op_type=op_type,
            data=data,
            priority=priority,
            callback=callback,
            error_callback=error_callback,
            timeout=timeout
        )
        
        # Enqueue with priority
        # Priority queue uses (priority_value, sequence, operation)
        self._queues[priority].put((
            priority.value,
            operation_id,
            operation
        ))
        
        return operation_id
    
    def wait(self, operation_id: int, timeout: Optional[float] = None) -> Any:
        """
        Wait for a specific operation to complete.
        
        Args:
            operation_id (int): Operation ID to wait for
            timeout (float, optional): Maximum wait time in seconds
            
        Returns:
            Any: Operation result
            
        Raises:
            TimeoutError: If operation doesn't complete within timeout
            RuntimeError: If operation failed
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if operation_id in self._completed:
                    op = self._completed.pop(operation_id)
                    
                    if op.status == 'completed':
                        return op.result
                    else:
                        raise RuntimeError(f"Operation failed: {op.error}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Operation {operation_id} did not complete within {timeout}s"
                )
            
            time.sleep(0.001)  # Small sleep to prevent busy-waiting
    
    def synchronize(self):
        """
        Wait for all operations in this stream to complete.
        
        Blocks until all submitted operations have been processed.
        """
        while True:
            pending = False
            for q in self._queues.values():
                if not q.empty():
                    pending = True
                    break
            
            if not pending:
                break
            
            time.sleep(0.01)
    
    def start(self):
        """
        Start processing operations in this stream.
        """
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name=f"Stream-{self.stream_id}"
        )
        self._thread.start()
    
    def stop(self):
        """
        Stop processing operations.
        
        Completes any in-progress operation before stopping.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
    
    def _process_loop(self):
        """
        Main processing loop for this stream.
        
        Continuously dequeues and executes operations in priority order.
        """
        while self._running:
            operation = self._dequeue_operation()
            
            if operation is None:
                time.sleep(0.001)  # No operations, brief sleep
                continue
            
            # Record queue time
            queue_time = time.time() - operation.timestamp
            with self._lock:
                self.stats['total_queue_time'] += queue_time
            
            # Execute operation
            operation.started_at = time.time()
            operation.status = 'running'
            
            try:
                # Set timeout if specified
                if operation.timeout:
                    # In a real implementation, this would use async I/O
                    # with timeout. Simplified here for clarity.
                    pass
                
                # Execute on GPU
                result = self.gpu.execute({
                    'type': operation.op_type,
                    **operation.data
                })
                
                # Check for errors
                if 'error' in result:
                    raise RuntimeError(result['error'])
                
                # Success
                operation.status = 'completed'
                operation.result = result
                operation.completed_at = time.time()
                
                with self._lock:
                    self.stats['completed'] += 1
                    self.stats['total_compute_time'] += (
                        operation.completed_at - operation.started_at
                    )
                
                # Call success callback
                if operation.callback:
                    try:
                        operation.callback(result)
                    except Exception as e:
                        print(f"Callback error for operation {operation.operation_id}: {e}")
                
            except Exception as e:
                # Failure
                operation.status = 'failed'
                operation.error = str(e)
                operation.completed_at = time.time()
                
                with self._lock:
                    self.stats['failed'] += 1
                
                # Call error callback
                if operation.error_callback:
                    try:
                        operation.error_callback(e)
                    except Exception as cb_e:
                        print(f"Error callback error: {cb_e}")
            
            # Move to completed dict
            with self._lock:
                self._completed[operation.operation_id] = operation
    
    def _dequeue_operation(self) -> Optional[StreamOperation]:
        """
        Dequeue the highest priority operation.
        
        Checks queues in priority order: HIGH, NORMAL, LOW, IDLE.
        
        Returns:
            Optional[StreamOperation]: Next operation or None
        """
        priorities = [
            StreamPriority.HIGH,
            StreamPriority.NORMAL,
            StreamPriority.LOW,
            StreamPriority.IDLE
        ]
        
        for priority in priorities:
            queue_obj = self._queues[priority]
            if not queue_obj.empty():
                try:
                    _, _, operation = queue_obj.get_nowait()
                    return operation
                except queue.Empty:
                    continue
        
        return None


class StreamManager:
    """
    Manages multiple CUDA operation streams across GPU connections.
    
    Provides:
        - Stream creation per GPU connection
        - Load balancing across streams
        - Async operation submission
        - Stream synchronization
        - Performance monitoring
    
    This enables pipelining: while one operation executes on GPU,
    the next operation's data can be transferred over the network.
    
    Usage:
        manager = StreamManager(gpu_pool)
        
        # Submit async operations
        op_id = manager.submit('forward', data, gpu='gpu_0')
        
        # Do other work while GPU processes...
        
        # Wait for result
        result = manager.wait(op_id)
    """
    
    def __init__(self, gpu_pool, streams_per_gpu: int = 2):
        """
        Initialize the stream manager.
        
        Args:
            gpu_pool: GPU connection pool
            streams_per_gpu (int): Number of streams per GPU connection
        """
        self.gpu_pool = gpu_pool
        self.streams_per_gpu = streams_per_gpu
        
        # Streams organized by GPU connection
        self._streams: Dict[str, List[OperationStream]] = {}
        
        # Global stream counter
        self._stream_counter = 0
        
        # Operation tracking: op_id -> (stream, gpu_key)
        self._operation_map: Dict[int, tuple] = {}
        
        # Lock
        self._lock = threading.Lock()
    
    def initialize(self):
        """
        Create streams for all active GPU connections.
        """
        for key, gpu in self.gpu_pool.connections.items():
            if gpu.is_connected:
                self._streams[key] = []
                
                for i in range(self.streams_per_gpu):
                    stream_id = self._stream_counter
                    self._stream_counter += 1
                    
                    stream = OperationStream(stream_id, gpu)
                    stream.start()
                    
                    self._streams[key].append(stream)
    
    def submit(
        self,
        op_type: str,
        data: dict,
        gpu_key: Optional[str] = None,
        priority: StreamPriority = StreamPriority.NORMAL,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        timeout: Optional[float] = None
    ) -> int:
        """
        Submit an operation to a GPU stream.
        
        If no GPU is specified, the least loaded stream is chosen.
        
        Args:
            op_type (str): Operation type
            data (dict): Operation data
            gpu_key (str, optional): Specific GPU to use
            priority (StreamPriority): Operation priority
            callback (Callable, optional): Success callback
            error_callback (Callable, optional): Error callback
            timeout (float, optional): Operation timeout
            
        Returns:
            int: Operation ID
            
        Raises:
            RuntimeError: If no streams are available
        """
        if not self._streams:
            raise RuntimeError("Stream manager not initialized. Call initialize() first.")
        
        # Select stream
        if gpu_key:
            streams = self._streams.get(gpu_key)
            if not streams:
                raise ValueError(f"GPU {gpu_key} not found")
        else:
            # Load balancing: pick least loaded GPU
            gpu_key, streams = self._get_least_loaded_gpu()
        
        # Pick least loaded stream within the GPU
        stream = min(streams, key=lambda s: sum(
            not q.empty() for q in s._queues.values()
        ))
        
        # Submit operation
        operation_id = stream.submit(
            op_type=op_type,
            data=data,
            priority=priority,
            callback=callback,
            error_callback=error_callback,
            timeout=timeout
        )
        
        # Track operation
        with self._lock:
            self._operation_map[operation_id] = (stream, gpu_key)
        
        return operation_id
    
    def wait(self, operation_id: int, timeout: Optional[float] = None) -> Any:
        """
        Wait for a specific operation to complete.
        
        Args:
            operation_id (int): Operation ID
            timeout (float, optional): Maximum wait time
            
        Returns:
            Any: Operation result
        """
        with self._lock:
            if operation_id not in self._operation_map:
                raise ValueError(f"Unknown operation: {operation_id}")
            
            stream, _ = self._operation_map[operation_id]
        
        result = stream.wait(operation_id, timeout)
        
        # Clean up tracking
        with self._lock:
            self._operation_map.pop(operation_id, None)
        
        return result
    
    def synchronize_all(self):
        """
        Wait for all operations across all streams to complete.
        """
        for gpu_streams in self._streams.values():
            for stream in gpu_streams:
                stream.synchronize()
    
    def _get_least_loaded_gpu(self) -> tuple:
        """
        Find the GPU with the fewest pending operations.
        
        Returns:
            tuple: (gpu_key, list_of_streams)
        """
        best_key = None
        best_load = float('inf')
        
        for key, streams in self._streams.items():
            load = sum(
                sum(not q.empty() for q in stream._queues.values())
                for stream in streams
            )
            
            if load < best_load:
                best_load = load
                best_key = key
        
        return best_key, self._streams[best_key]
    
    def shutdown(self):
        """
        Stop all streams and clean up.
        """
        for gpu_streams in self._streams.values():
            for stream in gpu_streams:
                stream.stop()
        
        self._streams.clear()
        self._operation_map.clear()
    
    def get_stats(self) -> dict:
        """
        Get aggregate statistics across all streams.
        
        Returns:
            dict: Stream statistics
        """
        total_stats = {
            'total_streams': 0,
            'active_streams': 0,
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_queue_time': 0.0,
            'total_compute_time': 0.0,
            'per_gpu': {}
        }
        
        for key, streams in self._streams.items():
            gpu_stats = {
                'streams': len(streams),
                'submitted': 0,
                'completed': 0,
                'failed': 0
            }
            
            for stream in streams:
                total_stats['total_streams'] += 1
                if stream._running:
                    total_stats['active_streams'] += 1
                
                gpu_stats['submitted'] += stream.stats['submitted']
                gpu_stats['completed'] += stream.stats['completed']
                gpu_stats['failed'] += stream.stats['failed']
                
                total_stats['total_submitted'] += stream.stats['submitted']
                total_stats['total_completed'] += stream.stats['completed']
                total_stats['total_failed'] += stream.stats['failed']
                total_stats['total_queue_time'] += stream.stats['total_queue_time']
                total_stats['total_compute_time'] += stream.stats['total_compute_time']
            
            total_stats['per_gpu'][key] = gpu_stats
        
        return total_stats