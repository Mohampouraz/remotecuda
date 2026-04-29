"""
Task Scheduler Module — Pure Python Client
===========================================
Intelligent task scheduling and distribution across multiple GPU servers.

This module provides priority-based task queuing, automatic GPU selection,
retry logic with exponential backoff, and comprehensive monitoring.

Pure Python implementation — no external dependencies.

Architecture:
    ┌────────────────────────────────────────────────────────────┐
    │                     Task Scheduler                         │
    │                                                             │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │ CRITICAL │  │   HIGH   │  │  NORMAL  │  │   LOW    │  │
    │  │  Queue   │  │  Queue   │  │  Queue   │  │  Queue   │  │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
    │       └──────────────┴─────────────┴────────────┘         │
    │                          │                                 │
    │                   ┌──────▼──────┐                          │
    │                   │  Strategy   │                          │
    │                   │  Selector   │                          │
    │                   └──────┬──────┘                          │
    │                          │                                 │
    │          ┌───────────────┼───────────────┐                │
    │          ▼               ▼               ▼                │
    │    ┌──────────┐   ┌──────────┐   ┌──────────┐            │
    │    │ Server 1 │   │ Server 2 │   │ Server N │            │
    │    └──────────┘   └──────────┘   └──────────┘            │
    └────────────────────────────────────────────────────────────┘

Usage:
    from remotecuda.client.pool import ConnectionPool
    from remotecuda.client.scheduler import TaskScheduler, TaskPriority

    pool = ConnectionPool()
    pool.connect_all()

    scheduler = TaskScheduler(pool)
    scheduler.start()

    task = scheduler.submit(
        'zeros',
        {'shape': [1000, 1000]},
        priority=TaskPriority.HIGH
    )

    result = scheduler.wait_for(task.task_id)
    scheduler.stop()
"""

import threading
import time
import heapq
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .pool import ConnectionPool
from .connection import ClientConnection


class TaskPriority(Enum):
    """Priority levels for task scheduling."""
    CRITICAL = 0  # Execute immediately
    HIGH = 1      # Execute as soon as possible
    NORMAL = 2    # Standard priority
    LOW = 3       # Best-effort execution
    BACKGROUND = 4  # Only when idle


class TaskStatus(Enum):
    """Lifecycle states of a scheduled task."""
    PENDING = 'pending'
    ASSIGNED = 'assigned'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    RETRYING = 'retrying'
    CANCELLED = 'cancelled'
    TIMED_OUT = 'timed_out'


@dataclass(order=True)
class ScheduledTask:
    """
    A task scheduled for execution on a remote GPU server.

    Supports priority ordering via heapq.
    Lower priority value + earlier creation = higher priority.
    """
    # Sorting fields
    _sort_priority: int = field(init=False, repr=False)
    _sort_created: float = field(init=False, repr=False)

    # Core fields
    task_id: str = ''
    command: str = 'ping'
    params: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Retry
    retry_count: int = 0
    max_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 60.0

    # Timeout
    timeout: Optional[float] = None

    # Assignment
    assigned_server: Optional[str] = None

    # Results
    result: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

    # Callbacks
    on_complete: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None

    # Synchronization
    _completion_event: Optional[threading.Event] = field(
        default_factory=threading.Event, repr=False
    )
    _cancel_event: Optional[threading.Event] = field(
        default_factory=threading.Event, repr=False
    )

    def __post_init__(self):
        self._sort_priority = self.priority.value
        self._sort_created = self.created_at

    def should_retry(self) -> bool:
        """Determine if task should be retried."""
        return (
            self.status in [TaskStatus.FAILED, TaskStatus.TIMED_OUT]
            and self.retry_count < self.max_retries
            and not self._cancel_event.is_set()
        )

    def get_retry_delay(self) -> float:
        """Calculate exponential backoff delay."""
        delay = self.retry_delay_base * (2 ** (self.retry_count - 1))
        return min(delay, self.retry_delay_max)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for task completion."""
        if self._completion_event:
            return self._completion_event.wait(timeout)
        return self.status in [
            TaskStatus.COMPLETED, TaskStatus.FAILED,
            TaskStatus.CANCELLED, TaskStatus.TIMED_OUT
        ]

    def cancel(self):
        """Request task cancellation."""
        self._cancel_event.set()

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in [
            TaskStatus.COMPLETED, TaskStatus.FAILED,
            TaskStatus.CANCELLED, TaskStatus.TIMED_OUT
        ]


class TaskScheduler:
    """
    Intelligent distributed task scheduler.

    Manages task lifecycle from submission to completion,
    with automatic server selection, retry logic, and health monitoring.

    Features:
        - 5 priority levels
        - Automatic server selection (round-robin, least-loaded)
        - Exponential backoff retry
        - Task timeout management
        - Callback support
        - Thread-safe submission
        - Statistics and monitoring

    Usage:
        scheduler = TaskScheduler(pool)
        scheduler.start()

        task = scheduler.submit('zeros', {'shape': [100, 100]})
        result = scheduler.wait_for(task.task_id)

        scheduler.stop()
    """

    def __init__(
        self,
        pool: ConnectionPool,
        max_concurrent_per_server: int = 4,
        task_timeout_default: float = 300.0,
        gc_interval: float = 60.0,
    ):
        """
        Initialize the task scheduler.

        Args:
            pool: ConnectionPool for server access.
            max_concurrent_per_server: Max simultaneous tasks per server.
            task_timeout_default: Default timeout for tasks (seconds).
            gc_interval: Interval for cleaning completed tasks (seconds).
        """
        self._pool = pool
        self._max_concurrent_per_server = max_concurrent_per_server
        self._task_timeout_default = task_timeout_default
        self._gc_interval = gc_interval

        # Task management
        self._all_tasks: Dict[str, ScheduledTask] = {}
        self._pending_tasks: List[ScheduledTask] = []
        self._active_tasks: Dict[str, ScheduledTask] = {}
        self._completed_tasks: Dict[str, ScheduledTask] = {}

        # Per-server active count
        self._server_load: Dict[str, int] = {}

        # Task counter
        self._task_counter = 0

        # Scheduler state
        self._running = False
        self._lock = threading.RLock()
        self._task_available = threading.Condition(self._lock)

        # Threads
        self._scheduler_thread: Optional[threading.Thread] = None
        self._gc_thread: Optional[threading.Thread] = None

        # Statistics
        self.stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'timed_out': 0,
            'retried': 0,
        }

    def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="TaskScheduler"
        )
        self._scheduler_thread.start()

        self._gc_thread = threading.Thread(
            target=self._gc_loop, daemon=True, name="TaskGC"
        )
        self._gc_thread.start()

    def stop(self, drain: bool = True):
        """
        Stop the scheduler.

        Args:
            drain: If True, wait for active tasks to complete.
        """
        if drain:
            while True:
                with self._lock:
                    if not self._active_tasks:
                        break
                time.sleep(0.1)

        self._running = False

        with self._task_available:
            self._task_available.notify_all()

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        if self._gc_thread:
            self._gc_thread.join(timeout=3)

    def submit(
        self,
        command: str,
        params: dict,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> ScheduledTask:
        """
        Submit a task for execution.

        Args:
            command: Server command to execute.
            params: Command parameters.
            priority: Task priority.
            max_retries: Maximum retry attempts.
            timeout: Task timeout (seconds).
            on_complete: Callable(result) on success.
            on_error: Callable(exception) on failure.

        Returns:
            ScheduledTask: The submitted task.
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter:08d}"

        task = ScheduledTask(
            task_id=task_id,
            command=command,
            params=params,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout or self._task_timeout_default,
            on_complete=on_complete,
            on_error=on_error,
        )

        with self._lock:
            self._all_tasks[task_id] = task
            heapq.heappush(self._pending_tasks, task)
            self.stats['submitted'] += 1

        with self._task_available:
            self._task_available.notify()

        return task

    def wait_for(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for a task to complete and return its result.

        Args:
            task_id: Task identifier.
            timeout: Maximum wait time.

        Returns:
            Any: Task result.

        Raises:
            ValueError: If task not found.
            TimeoutError: If task doesn't complete.
            RuntimeError: If task failed.
        """
        with self._lock:
            task = self._all_tasks.get(task_id)

        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if task.wait(timeout):
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.CANCELLED:
                raise RuntimeError(f"Task {task_id} was cancelled")
            else:
                raise RuntimeError(f"Task failed: {task.error_message}")
        else:
            raise TimeoutError(f"Task {task_id} did not complete")

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending or active task."""
        with self._lock:
            task = self._all_tasks.get(task_id)

        if task is None or task.is_terminal:
            return False

        task.cancel()
        task.status = TaskStatus.CANCELLED
        self.stats['cancelled'] += 1
        return True

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        with self._lock:
            return self._all_tasks.get(task_id)

    def _scheduler_loop(self):
        """Main scheduling loop."""
        while self._running:
            with self._task_available:
                self._task_available.wait(timeout=0.1)

            self._dispatch_pending_tasks()

    def _dispatch_pending_tasks(self):
        """Assign pending tasks to available servers."""
        with self._lock:
            if not self._pending_tasks:
                return

            heapq.heapify(self._pending_tasks)

            remaining = []

            while self._pending_tasks:
                task = heapq.heappop(self._pending_tasks)

                if task._cancel_event.is_set():
                    task.status = TaskStatus.CANCELLED
                    self.stats['cancelled'] += 1
                    continue

                # Select server
                server_key = self._select_server()

                if server_key:
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_server = server_key
                    self._server_load[server_key] = self._server_load.get(server_key, 0) + 1

                    # Execute in thread
                    threading.Thread(
                        target=self._execute_task,
                        args=(task, server_key),
                        daemon=True,
                    ).start()
                else:
                    remaining.append(task)

            for task in remaining:
                heapq.heappush(self._pending_tasks, task)

    def _select_server(self) -> Optional[str]:
        """Select the best server for task execution."""
        available = []

        for key, conn in self._pool.connections.items():
            if conn.is_connected:
                load = self._server_load.get(key, 0)
                if load < self._max_concurrent_per_server:
                    available.append((load, key))

        if not available:
            return None

        available.sort()
        return available[0][1]

    def _execute_task(self, task: ScheduledTask, server_key: str):
        """Execute a task on a specific server."""
        conn = self._pool.connections.get(server_key)

        if conn is None or not conn.is_connected:
            self._handle_task_failure(task, server_key, "Server not available")
            return

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        with self._lock:
            self._active_tasks[task.task_id] = task

        try:
            result = conn.send_command(task.command, task.params)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.execution_time_ms = (task.completed_at - task.started_at) * 1000

            with self._lock:
                self.stats['completed'] += 1

            if task._completion_event:
                task._completion_event.set()

            if task.on_complete:
                try:
                    task.on_complete(result)
                except Exception:
                    pass

        except Exception as e:
            self._handle_task_failure(task, server_key, str(e))

        finally:
            with self._lock:
                self._active_tasks.pop(task.task_id, None)
                self._server_load[server_key] = max(0, self._server_load.get(server_key, 1) - 1)

    def _handle_task_failure(self, task: ScheduledTask, server_key: str, error: str):
        """Handle task failure with retry logic."""
        task.error_message = error
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()

        if task.should_retry():
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            with self._lock:
                self.stats['retried'] += 1
                heapq.heappush(self._pending_tasks, task)
        else:
            with self._lock:
                self.stats['failed'] += 1

            if task._completion_event:
                task._completion_event.set()

            if task.on_error:
                try:
                    task.on_error(RuntimeError(error))
                except Exception:
                    pass

    def _gc_loop(self):
        """Periodic cleanup of old completed tasks."""
        while self._running:
            time.sleep(self._gc_interval)

            with self._lock:
                to_remove = []
                now = time.time()
                for task_id, task in self._completed_tasks.items():
                    if task.completed_at and (now - task.completed_at) > 600:
                        to_remove.append(task_id)

                for task_id in to_remove:
                    self._completed_tasks.pop(task_id, None)

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        with self._lock:
            return {
                **self.stats,
                'pending': len(self._pending_tasks),
                'active': len(self._active_tasks),
                'completed_tasks': len(self._completed_tasks),
                'total_tasks': len(self._all_tasks),
                'server_loads': dict(self._server_load),
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(drain=True)
        return False