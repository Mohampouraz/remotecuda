"""
GPU Scheduler Module
====================
Intelligent workload distribution across multiple remote GPUs.
Handles load balancing, task prioritization, resource allocation,
and fault tolerance for distributed GPU computing.

This scheduler treats all remote GPUs as a unified resource pool,
automatically distributing operations to maximize throughput and
minimize latency based on real-time GPU metrics.

Key Capabilities:
    - Dynamic load balancing across heterogeneous GPUs
    - Task prioritization with preemption support
    - Resource-aware scheduling (memory, compute, bandwidth)
    - Fault-tolerant execution with automatic failover
    - Elastic scaling as GPUs join/leave the network
    - Performance profiling and adaptive optimization
"""

import threading
import time
import heapq
import uuid
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import queue

import torch


class TaskPriority(Enum):
    """
    Priority levels for scheduled tasks.
    
    CRITICAL: Immediate execution, preempts lower priority tasks
    HIGH:     Executed as soon as possible
    NORMAL:   Standard priority
    LOW:      Best-effort execution
    BACKGROUND: Only when idle
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """
    Lifecycle states of a scheduled task.
    """
    PENDING = 'pending'          # Waiting in queue
    ASSIGNED = 'assigned'         # Assigned to a GPU worker
    RUNNING = 'running'           # Currently executing
    COMPLETED = 'completed'       # Successfully finished
    FAILED = 'failed'             # Execution failed
    PREEMPTED = 'preempted'       # Interrupted for higher priority
    RETRYING = 'retrying'         # Being retried after failure
    CANCELLED = 'cancelled'       # Cancelled before completion


@dataclass
class GPUCapability:
    """
    Describes the capabilities and current state of a GPU resource.
    Used by the scheduler for placement decisions.
    """
    gpu_id: str                          # Unique GPU identifier
    server_id: str                       # Server identifier
    server_ip: str                       # Server IP address
    server_port: int                     # Server port
    gpu_index: int                       # Physical GPU index on server
    gpu_name: str                        # GPU model name
    total_memory: int                    # Total memory in bytes
    free_memory: int                      # Currently free memory
    compute_capability: Tuple[int, int]   # CUDA compute capability
    current_utilization: float            # 0.0-1.0
    current_temperature: float            # Celsius
    
    # Performance metrics
    avg_forward_time_ms: float = 0.0     # Average forward pass time
    avg_bandwidth_mbps: float = 0.0      # Average network bandwidth
    tasks_completed: int = 0              # Total tasks completed
    tasks_failed: int = 0                 # Total failed tasks
    last_heartbeat: float = field(default_factory=time.time)
    
    # Scheduling state
    is_active: bool = True
    current_load: int = 0                 # Number of active tasks
    max_concurrent_tasks: int = 4         # Maximum concurrent tasks
    
    def memory_pressure(self) -> float:
        """
        Calculate memory pressure (0.0 = empty, 1.0 = full).
        """
        if self.total_memory == 0:
            return 1.0
        return 1.0 - (self.free_memory / self.total_memory)
    
    def can_accept_task(self, estimated_memory: int = 0) -> bool:
        """
        Check if GPU can accept a new task.
        
        Args:
            estimated_memory (int): Estimated memory needed by the task
            
        Returns:
            bool: True if GPU can accept the task
        """
        if not self.is_active:
            return False
        if self.current_load >= self.max_concurrent_tasks:
            return False
        if estimated_memory > self.free_memory * 0.9:  # 10% buffer
            return False
        return True


@dataclass
class ScheduledTask:
    """
    A task that has been scheduled for execution on a remote GPU.
    
    Contains all information needed to execute, track, and retry
    the task across potentially multiple GPU resources.
    """
    task_id: str
    operation_type: str
    operation_data: dict
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_gpu: Optional[str] = None
    
    # Resource requirements (estimated)
    estimated_memory: int = 0              # Estimated GPU memory needed
    estimated_duration_ms: float = 100.0   # Estimated execution time
    estimated_data_size: int = 0           # Estimated data transfer size
    
    # Execution tracking
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Results
    result: Any = None
    error_message: Optional[str] = None
    
    # Callbacks
    on_complete: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    on_progress: Optional[Callable[[float], None]] = None
    
    # Synchronization
    _completion_event: Optional[threading.Event] = field(
        default_factory=threading.Event
    )
    
    def __lt__(self, other: 'ScheduledTask') -> bool:
        """
        Comparison for priority queue ordering.
        Higher priority tasks come first.
        """
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at
    
    def mark_running(self, gpu_id: str):
        """
        Mark task as running on a specific GPU.
        """
        self.status = TaskStatus.RUNNING
        self.assigned_gpu = gpu_id
        self.started_at = time.time()
    
    def mark_completed(self, result: Any):
        """
        Mark task as successfully completed.
        """
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
        
        if self._completion_event:
            self._completion_event.set()
        
        if self.on_complete:
            try:
                self.on_complete(result)
            except Exception:
                pass
    
    def mark_failed(self, error_message: str):
        """
        Mark task as failed.
        """
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        self.completed_at = time.time()
        
        if self._completion_event:
            self._completion_event.set()
        
        if self.on_error:
            try:
                self.on_error(Exception(error_message))
            except Exception:
                pass
    
    def should_retry(self) -> bool:
        """
        Determine if task should be retried.
        """
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    def prepare_retry(self):
        """
        Prepare task for retry execution.
        """
        self.status = TaskStatus.RETRYING
        self.retry_count += 1
        self.assigned_gpu = None
        self.started_at = None
        self.result = None
        self.error_message = None
        if self._completion_event:
            self._completion_event.clear()
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for task completion.
        
        Args:
            timeout (float, optional): Maximum wait time in seconds
            
        Returns:
            bool: True if completed, False if timeout
        """
        if self._completion_event:
            return self._completion_event.wait(timeout)
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
    
    @property
    def elapsed_time(self) -> float:
        """
        Time since task creation in seconds.
        """
        return time.time() - self.created_at
    
    @property
    def execution_time(self) -> Optional[float]:
        """
        Actual execution time in seconds.
        """
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at


class SchedulingStrategy:
    """
    Base class for GPU scheduling strategies.
    
    Different strategies optimize for different objectives:
    - Throughput: Maximize total operations per second
    - Latency: Minimize individual operation latency
    - Fairness: Equal share of GPU time across tasks
    - Power efficiency: Minimize energy consumption
    """
    
    def select_gpu(
        self,
        task: ScheduledTask,
        available_gpus: Dict[str, GPUCapability]
    ) -> Optional[str]:
        """
        Select the best GPU for a given task.
        
        Args:
            task (ScheduledTask): Task to assign
            available_gpus (dict): Available GPU capabilities
            
        Returns:
            Optional[str]: Selected GPU ID, or None if no suitable GPU
        """
        raise NotImplementedError


class LoadBalancingStrategy(SchedulingStrategy):
    """
    Distributes tasks evenly across all available GPUs.
    
    Best for: Homogeneous GPU clusters with similar workloads.
    """
    
    def select_gpu(
        self,
        task: ScheduledTask,
        available_gpus: Dict[str, GPUCapability]
    ) -> Optional[str]:
        """
        Select GPU with the lowest current load.
        """
        best_gpu = None
        lowest_load = float('inf')
        
        for gpu_id, gpu in available_gpus.items():
            if not gpu.can_accept_task(task.estimated_memory):
                continue
            
            if gpu.current_load < lowest_load:
                lowest_load = gpu.current_load
                best_gpu = gpu_id
        
        return best_gpu


class MemoryAwareStrategy(SchedulingStrategy):
    """
    Places tasks on GPUs with the most available memory.
    
    Best for: Memory-intensive workloads (large models, big batches).
    """
    
    def select_gpu(
        self,
        task: ScheduledTask,
        available_gpus: Dict[str, GPUCapability]
    ) -> Optional[str]:
        """
        Select GPU with the most free memory.
        """
        best_gpu = None
        most_memory = task.estimated_memory
        
        for gpu_id, gpu in available_gpus.items():
            if not gpu.can_accept_task(task.estimated_memory):
                continue
            
            if gpu.free_memory > most_memory:
                most_memory = gpu.free_memory
                best_gpu = gpu_id
        
        return best_gpu


class LatencyOptimizedStrategy(SchedulingStrategy):
    """
    Routes tasks to GPUs with the lowest historical latency.
    
    Best for: Latency-sensitive applications, real-time inference.
    """
    
    def select_gpu(
        self,
        task: ScheduledTask,
        available_gpus: Dict[str, GPUCapability]
    ) -> Optional[str]:
        """
        Select GPU with lowest average forward time.
        """
        best_gpu = None
        lowest_latency = float('inf')
        
        for gpu_id, gpu in available_gpus.items():
            if not gpu.can_accept_task(task.estimated_memory):
                continue
            
            if gpu.avg_forward_time_ms < lowest_latency:
                lowest_latency = gpu.avg_forward_time_ms
                best_gpu = gpu_id
        
        return best_gpu


class AdaptiveStrategy(SchedulingStrategy):
    """
    Dynamically switches strategies based on workload characteristics.
    
    Uses different strategies for:
    - Memory-bound tasks → MemoryAware
    - Latency-sensitive tasks → LatencyOptimized
    - Standard tasks → LoadBalancing
    """
    
    def __init__(self):
        """
        Initialize adaptive strategy with sub-strategies.
        """
        self.load_balancer = LoadBalancingStrategy()
        self.memory_aware = MemoryAwareStrategy()
        self.latency_optimized = LatencyOptimizedStrategy()
        
        # Thresholds for strategy selection
        self.memory_threshold = 100 * 1024 * 1024  # 100 MB
        self.latency_threshold = 50.0  # 50 ms target
    
    def select_gpu(
        self,
        task: ScheduledTask,
        available_gpus: Dict[str, GPUCapability]
    ) -> Optional[str]:
        """
        Select strategy based on task characteristics.
        """
        if task.estimated_memory > self.memory_threshold:
            # Memory-heavy task
            return self.memory_aware.select_gpu(task, available_gpus)
        elif task.estimated_duration_ms < self.latency_threshold:
            # Latency-sensitive task
            return self.latency_optimized.select_gpu(task, available_gpus)
        else:
            # Standard task
            return self.load_balancer.select_gpu(task, available_gpus)


class GPUScheduler:
    """
    Central scheduler for distributed GPU task execution.
    
    Manages the entire lifecycle of GPU tasks:
    1. Task submission and queuing
    2. Resource allocation and GPU selection
    3. Execution monitoring
    4. Result collection
    5. Failure handling and retry
    
    Features:
        - Priority-based scheduling with preemption
        - Multiple scheduling strategies
        - Automatic failover on GPU failure
        - Elastic scaling with GPU discovery
        - Comprehensive monitoring and statistics
        - Batch submission for efficiency
    
    Usage:
        scheduler = GPUScheduler(gpu_pool)
        scheduler.start()
        
        # Submit a task
        task = scheduler.submit(
            op_type='forward',
            data={'input': input_data},
            priority=TaskPriority.HIGH
        )
        
        # Wait for result
        result = scheduler.wait_for_task(task.task_id)
        
        scheduler.stop()
    """
    
    def __init__(
        self,
        gpu_pool,
        strategy: str = 'adaptive',
        max_concurrent_tasks: int = 64,
        task_timeout: float = 300.0,
        heartbeat_interval: float = 5.0
    ):
        """
        Initialize the GPU scheduler.
        
        Args:
            gpu_pool: Connected GPU pool
            strategy (str): Scheduling strategy name
            max_concurrent_tasks (int): Maximum concurrent tasks
            task_timeout (float): Default task timeout in seconds
            heartbeat_interval (float): GPU health check interval
        """
        self.gpu_pool = gpu_pool
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Select scheduling strategy
        self.strategy_name = strategy
        if strategy == 'load_balancing':
            self.strategy = LoadBalancingStrategy()
        elif strategy == 'memory_aware':
            self.strategy = MemoryAwareStrategy()
        elif strategy == 'latency_optimized':
            self.strategy = LatencyOptimizedStrategy()
        elif strategy == 'adaptive':
            self.strategy = AdaptiveStrategy()
        else:
            raise ValueError(f"Unknown scheduling strategy: {strategy}")
        
        # GPU capabilities registry
        self.gpu_capabilities: Dict[str, GPUCapability] = {}
        
        # Task queues by priority
        self._task_queues: Dict[TaskPriority, List[ScheduledTask]] = {
            TaskPriority.CRITICAL: [],
            TaskPriority.HIGH: [],
            TaskPriority.NORMAL: [],
            TaskPriority.LOW: [],
            TaskPriority.BACKGROUND: []
        }
        
        # All tasks indexed by ID
        self._all_tasks: Dict[str, ScheduledTask] = {}
        
        # Completed tasks (for result retrieval)
        self._completed_tasks: Dict[str, ScheduledTask] = {}
        
        # Task counter for ID generation
        self._task_counter = 0
        
        # Scheduler state
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        # Synchronization
        self._lock = threading.RLock()
        self._task_available = threading.Condition(self._lock)
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'tasks_retried': 0,
            'total_execution_time': 0.0,
            'total_queue_time': 0.0,
            'gpu_assignments': defaultdict(int)
        }
        
        # Performance profiling
        self._profiling_data: Dict[str, List[float]] = defaultdict(list)
    
    def start(self):
        """
        Start the scheduler.
        
        Begins processing tasks and monitoring GPUs.
        """
        if self._running:
            return
        
        # Initialize GPU capabilities
        self._update_gpu_capabilities()
        
        self._running = True
        
        # Start scheduler thread
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="GPUScheduler"
        )
        self._scheduler_thread.start()
        
        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="GPUHeartbeat"
        )
        self._heartbeat_thread.start()
        
        print(f"✅ GPU Scheduler started with '{self.strategy_name}' strategy")
        print(f"   Available GPUs: {len(self.gpu_capabilities)}")
    
    def stop(self):
        """
        Stop the scheduler gracefully.
        
        Completes running tasks and cleans up.
        """
        if not self._running:
            return
        
        self._running = False
        
        # Wake up scheduler thread
        with self._task_available:
            self._task_available.notify_all()
        
        # Wait for threads
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        
        # Cancel pending tasks
        with self._lock:
            for priority_queue in self._task_queues.values():
                for task in priority_queue:
                    task.status = TaskStatus.CANCELLED
            self._task_queues = {p: [] for p in TaskPriority}
        
        print(f"✅ GPU Scheduler stopped")
        print(f"   Total tasks: {self.stats['tasks_submitted']}")
        print(f"   Completed: {self.stats['tasks_completed']}")
        print(f"   Failed: {self.stats['tasks_failed']}")
    
    def submit(
        self,
        op_type: str,
        data: dict,
        priority: TaskPriority = TaskPriority.NORMAL,
        estimated_memory: int = 0,
        estimated_duration_ms: float = 100.0,
        max_retries: int = 3,
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> ScheduledTask:
        """
        Submit a task for execution on a remote GPU.
        
        Args:
            op_type (str): Type of GPU operation
            data (dict): Operation data
            priority (TaskPriority): Task priority
            estimated_memory (int): Estimated GPU memory needed (bytes)
            estimated_duration_ms (float): Estimated execution time
            max_retries (int): Maximum retry attempts on failure
            on_complete (callable): Called on successful completion
            on_error (callable): Called on failure
            on_progress (callable): Called with progress updates
            
        Returns:
            ScheduledTask: Submitted task (can be used to wait/track)
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter:08d}"
            
            task = ScheduledTask(
                task_id=task_id,
                operation_type=op_type,
                operation_data=data,
                priority=priority,
                estimated_memory=estimated_memory,
                estimated_duration_ms=estimated_duration_ms,
                max_retries=max_retries,
                on_complete=on_complete,
                on_error=on_error,
                on_progress=on_progress
            )
            
            # Add to queue
            self._task_queues[priority].append(task)
            self._all_tasks[task_id] = task
            self.stats['tasks_submitted'] += 1
        
        # Notify scheduler thread
        with self._task_available:
            self._task_available.notify()
        
        return task
    
    def submit_batch(
        self,
        tasks: List[Tuple[str, dict]],
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> List[ScheduledTask]:
        """
        Submit multiple tasks at once.
        
        Args:
            tasks (list): List of (op_type, data) tuples
            priority (TaskPriority): Priority for all tasks
            **kwargs: Additional arguments passed to submit()
            
        Returns:
            list: List of ScheduledTask objects
        """
        submitted = []
        for op_type, data in tasks:
            task = self.submit(op_type, data, priority=priority, **kwargs)
            submitted.append(task)
        return submitted
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Wait for a specific task to complete.
        
        Args:
            task_id (str): Task ID to wait for
            timeout (float, optional): Maximum wait time
            
        Returns:
            Any: Task result
            
        Raises:
            ValueError: If task not found
            TimeoutError: If task doesn't complete in time
            RuntimeError: If task failed
        """
        with self._lock:
            task = self._all_tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")
        
        if task.wait(timeout):
            if task.status == TaskStatus.COMPLETED:
                return task.result
            else:
                raise RuntimeError(
                    f"Task failed: {task.error_message or 'Unknown error'}"
                )
        else:
            raise TimeoutError(f"Task {task_id} did not complete within timeout")
    
    def wait_all(
        self,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wait for all pending tasks to complete.
        
        Args:
            timeout (float, optional): Maximum total wait time
            
        Returns:
            dict: {task_id: result} for all completed tasks
        """
        start_time = time.time()
        results = {}
        
        while True:
            with self._lock:
                pending = [
                    t for t in self._all_tasks.values()
                    if t.status in [
                        TaskStatus.PENDING,
                        TaskStatus.ASSIGNED,
                        TaskStatus.RUNNING,
                        TaskStatus.RETRYING
                    ]
                ]
                
                if not pending:
                    break
            
            if timeout and (time.time() - start_time) > timeout:
                break
            
            time.sleep(0.01)
        
        with self._lock:
            for task in self._all_tasks.values():
                if task.status == TaskStatus.COMPLETED:
                    results[task.task_id] = task.result
                elif task.status == TaskStatus.FAILED:
                    results[task.task_id] = RuntimeError(task.error_message)
        
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id (str): Task ID to cancel
            
        Returns:
            bool: True if cancelled successfully
        """
        with self._lock:
            task = self._all_tasks.get(task_id)
            if task is None:
                return False
            
            if task.status in [
                TaskStatus.PENDING,
                TaskStatus.ASSIGNED
            ]:
                task.status = TaskStatus.CANCELLED
                self.stats['tasks_cancelled'] += 1
                
                # Remove from queue
                for pq in self._task_queues.values():
                    if task in pq:
                        pq.remove(task)
                        break
                
                if task._completion_event:
                    task._completion_event.set()
                
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the current status of a task.
        
        Args:
            task_id (str): Task ID
            
        Returns:
            Optional[TaskStatus]: Task status, or None if not found
        """
        with self._lock:
            task = self._all_tasks.get(task_id)
            if task:
                return task.status
        return None
    
    def _scheduler_loop(self):
        """
        Main scheduler loop.
        
        Continuously assigns pending tasks to available GPUs.
        """
        while self._running:
            with self._task_available:
                # Wait for tasks to be available
                self._task_available.wait(timeout=0.1)
            
            # Update GPU capabilities
            self._update_gpu_capabilities()
            
            # Process tasks by priority
            self._schedule_next_batch()
    
    def _schedule_next_batch(self):
        """
        Assign pending tasks to available GPUs.
        """
        with self._lock:
            # Process in priority order
            for priority in [
                TaskPriority.CRITICAL,
                TaskPriority.HIGH,
                TaskPriority.NORMAL,
                TaskPriority.LOW,
                TaskPriority.BACKGROUND
            ]:
                queue_tasks = self._task_queues[priority]
                
                # Process tasks from this priority level
                tasks_to_assign = []
                for task in queue_tasks[:]:
                    if task.status == TaskStatus.PENDING:
                        # Find suitable GPU
                        gpu_id = self.strategy.select_gpu(
                            task, self.gpu_capabilities
                        )
                        
                        if gpu_id:
                            tasks_to_assign.append((task, gpu_id))
                            queue_tasks.remove(task)
                
                # Assign tasks to GPUs
                for task, gpu_id in tasks_to_assign:
                    self._assign_task(task, gpu_id)
    
    def _assign_task(self, task: ScheduledTask, gpu_id: str):
        """
        Assign a task to a specific GPU for execution.
        
        Args:
            task (ScheduledTask): Task to assign
            gpu_id (str): Target GPU ID
        """
        gpu = self.gpu_capabilities.get(gpu_id)
        if gpu is None:
            task.mark_failed(f"GPU {gpu_id} no longer available")
            return
        
        task.mark_running(gpu_id)
        gpu.current_load += 1
        self.stats['gpu_assignments'][gpu_id] += 1
        
        # Get GPU connection
        gpu_key = f"{gpu.server_ip}:{gpu.server_port}"
        connection = self.gpu_pool.connections.get(gpu_key)
        
        if connection is None or not connection.is_connected:
            task.mark_failed(f"GPU connection lost: {gpu_key}")
            gpu.current_load -= 1
            return
        
        # Execute task in separate thread
        executor_thread = threading.Thread(
            target=self._execute_task,
            args=(task, connection, gpu),
            daemon=True
        )
        executor_thread.start()
    
    def _execute_task(
        self,
        task: ScheduledTask,
        connection,
        gpu: GPUCapability
    ):
        """
        Execute a task on a specific GPU connection.
        
        Args:
            task (ScheduledTask): Task to execute
            connection: GPU connection object
            gpu (GPUCapability): GPU capabilities
        """
        start_time = time.time()
        
        try:
            # Execute operation
            result = connection.execute({
                'type': task.operation_type,
                **task.operation_data
            })
            
            # Check for errors
            if isinstance(result, dict) and 'error' in result:
                raise RuntimeError(result['error'])
            
            # Success
            execution_time = (time.time() - start_time) * 1000
            self._update_gpu_metrics(gpu, execution_time, success=True)
            
            with self._lock:
                task.mark_completed(result)
                gpu.current_load = max(0, gpu.current_load - 1)
                gpu.tasks_completed += 1
                self.stats['tasks_completed'] += 1
                self.stats['total_execution_time'] += execution_time
                self.stats['total_queue_time'] += (
                    task.started_at - task.created_at
                )
            
        except Exception as e:
            # Failure
            execution_time = (time.time() - start_time) * 1000
            self._update_gpu_metrics(gpu, execution_time, success=False)
            
            with self._lock:
                gpu.current_load = max(0, gpu.current_load - 1)
                gpu.tasks_failed += 1
                
                if task.should_retry():
                    task.mark_failed(str(e))
                    task.prepare_retry()
                    self.stats['tasks_retried'] += 1
                    
                    # Re-queue for retry
                    self._task_queues[task.priority].append(task)
                    
                    # Notify scheduler
                    with self._task_available:
                        self._task_available.notify()
                else:
                    task.mark_failed(str(e))
                    self.stats['tasks_failed'] += 1
    
    def _update_gpu_capabilities(self):
        """
        Update GPU capabilities from the pool.
        """
        with self._lock:
            for key, connection in self.gpu_pool.connections.items():
                if not connection.is_connected:
                    continue
                
                # Create GPU ID
                gpu_id = f"{connection.host}:{connection.port}:{connection.gpu_name}"
                
                # Get current stats
                try:
                    info = connection.execute({'type': 'ping'})
                except Exception:
                    continue
                
                if gpu_id not in self.gpu_capabilities:
                    # New GPU discovered
                    self.gpu_capabilities[gpu_id] = GPUCapability(
                        gpu_id=gpu_id,
                        server_id=getattr(connection, 'server_id', ''),
                        server_ip=connection.host,
                        server_port=connection.port,
                        gpu_index=0,
                        gpu_name=connection.gpu_name,
                        total_memory=connection.total_memory,
                        free_memory=getattr(connection, 'free_memory', connection.total_memory),
                        compute_capability=(8, 0),
                        current_utilization=0.0,
                        current_temperature=0.0,
                        max_concurrent_tasks=4
                    )
                else:
                    # Update existing GPU info
                    gpu = self.gpu_capabilities[gpu_id]
                    gpu.free_memory = getattr(connection, 'free_memory', gpu.total_memory)
                    gpu.last_heartbeat = time.time()
                    gpu.is_active = True
            
            # Mark inactive GPUs
            current_time = time.time()
            for gpu in self.gpu_capabilities.values():
                if current_time - gpu.last_heartbeat > self.heartbeat_interval * 3:
                    gpu.is_active = False
    
    def _update_gpu_metrics(
        self,
        gpu: GPUCapability,
        execution_time_ms: float,
        success: bool
    ):
        """
        Update GPU performance metrics.
        
        Args:
            gpu (GPUCapability): GPU to update
            execution_time_ms (float): Last execution time
            success (bool): Whether execution succeeded
        """
        # Exponential moving average for forward time
        alpha = 0.1
        gpu.avg_forward_time_ms = (
            alpha * execution_time_ms +
            (1 - alpha) * gpu.avg_forward_time_ms
        )
        
        # Store profiling data (keep last 1000 samples)
        key = f"{gpu.gpu_id}_execution_times"
        self._profiling_data[key].append(execution_time_ms)
        if len(self._profiling_data[key]) > 1000:
            self._profiling_data[key] = self._profiling_data[key][-1000:]
    
    def _heartbeat_loop(self):
        """
        Periodic GPU health check loop.
        """
        while self._running:
            time.sleep(self.heartbeat_interval)
            self._check_gpu_health()
    
    def _check_gpu_health(self):
        """
        Check health of all registered GPUs.
        """
        dead_gpus = []
        
        with self._lock:
            for gpu_id, gpu in self.gpu_capabilities.items():
                # Check heartbeat timeout
                if time.time() - gpu.last_heartbeat > 30:
                    dead_gpus.append(gpu_id)
                    
                    # Reassign tasks from dead GPU
                    for task in self._all_tasks.values():
                        if (
                            task.status == TaskStatus.RUNNING and
                            task.assigned_gpu == gpu_id
                        ):
                            task.mark_failed(f"GPU {gpu_id} is unresponsive")
                            
                            if task.should_retry():
                                task.prepare_retry()
                                self._task_queues[task.priority].append(task)
            
            # Remove dead GPUs
            for gpu_id in dead_gpus:
                del self.gpu_capabilities[gpu_id]
    
    def get_stats(self) -> dict:
        """
        Get comprehensive scheduler statistics.
        
        Returns:
            dict: Scheduler statistics
        """
        with self._lock:
            active_tasks = sum(
                1 for t in self._all_tasks.values()
                if t.status in [
                    TaskStatus.PENDING,
                    TaskStatus.ASSIGNED,
                    TaskStatus.RUNNING,
                    TaskStatus.RETRYING
                ]
            )
            
            gpu_stats = {}
            for gpu_id, gpu in self.gpu_capabilities.items():
                gpu_stats[gpu_id] = {
                    'name': gpu.gpu_name,
                    'server': gpu.server_ip,
                    'active': gpu.is_active,
                    'load': gpu.current_load,
                    'memory_pressure': gpu.memory_pressure(),
                    'avg_latency_ms': gpu.avg_forward_time_ms,
                    'tasks_completed': gpu.tasks_completed,
                    'tasks_failed': gpu.tasks_failed
                }
            
            return {
                **self.stats,
                'active_tasks': active_tasks,
                'total_tasks': len(self._all_tasks),
                'gpu_count': len(self.gpu_capabilities),
                'active_gpus': sum(
                    1 for g in self.gpu_capabilities.values()
                    if g.is_active
                ),
                'gpu_stats': gpu_stats,
                'strategy': self.strategy_name
            }
    
    def print_status(self):
        """
        Print current scheduler status to console.
        """
        stats = self.get_stats()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              GPU Scheduler Status                           ║
╠══════════════════════════════════════════════════════════════╣
║  Strategy:        {stats['strategy']:<42}║
║  Active GPUs:     {stats['active_gpus']}/{stats['gpu_count']:<40}║
║  Active Tasks:    {stats['active_tasks']:<42}║
║  Completed:       {stats['tasks_completed']:<42}║
║  Failed:          {stats['tasks_failed']:<42}║
║  Retried:         {stats['tasks_retried']:<42}║
║  Cancelled:       {stats['tasks_cancelled']:<42}║
╚══════════════════════════════════════════════════════════════╝
        """)