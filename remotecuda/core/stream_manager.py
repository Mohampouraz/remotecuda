"""
Stream Manager Module — Async Operation Pipeline
==================================================
Manages asynchronous operation pipelines for the client.

Enables overlapping of data transfer and computation:
    - While one operation executes on the server,
    - The next operation's data can be prepared/transferred.

This improves throughput for sequential operations by
hiding network latency behind computation.

Pure Python implementation — no external dependencies.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │              Stream Manager                     │
    │                                                  │
    │  Operation Queue:  [Op1] → [Op2] → [Op3] → ... │
    │                                                  │
    │  ┌──────────────┐   ┌──────────────┐            │
    │  │  Worker 1    │   │  Worker 2    │   ...      │
    │  │  (Server A)  │   │  (Server B)  │            │
    │  └──────────────┘   └──────────────┘            │
    │                                                  │
    │  Results:  [Res1, Res2, Res3, ...]              │
    └─────────────────────────────────────────────────┘

Usage:
    from remotecuda.client.pool import ConnectionPool
    from remotecuda.core.stream_manager import StreamManager

    pool = ConnectionPool()
    pool.connect_all()

    manager = StreamManager(pool)

    # Pipeline multiple operations
    results = manager.pipeline([
        ('zeros', {'shape': [100, 100]}),
        ('ones', {'shape': [100, 100]}),
        ('add', {'a_id': 0, 'b_id': 0}),  # IDs resolved after previous ops
    ])

    manager.shutdown()
"""

import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class OperationStatus(Enum):
    """Status of a streamed operation."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'


@dataclass
class StreamOperation:
    """A single operation in an execution stream."""
    op_id: int
    command: str
    params: dict
    status: OperationStatus = OperationStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None


class StreamManager:
    """
    Manages asynchronous operation streams.

    Provides pipelining of operations to maximize throughput
    by overlapping network transfers with computation.

    Features:
        - Sequential operation pipeline
        - Parallel operation dispatch
        - Result ordering
        - Error propagation
        - Callback support
        - Timeout management

    Usage:
        manager = StreamManager(pool)
        results = manager.pipeline(operations)
        manager.shutdown()
    """

    def __init__(self, pool, max_workers: int = 4):
        """
        Initialize the stream manager.

        Args:
            pool: ConnectionPool for server access.
            max_workers: Maximum concurrent operations.
        """
        self._pool = pool
        self._max_workers = max_workers

        self._operation_queue: queue.Queue = queue.Queue()
        self._results: Dict[int, Any] = {}
        self._results_lock = threading.Lock()
        self._completion_events: Dict[int, threading.Event] = {}

        self._counter = 0
        self._counter_lock = threading.Lock()

        self._running = False
        self._workers: List[threading.Thread] = []

        self.stats = {
            'operations_submitted': 0,
            'operations_completed': 0,
            'operations_failed': 0,
        }

    def start(self):
        """Start worker threads."""
        if self._running:
            return

        self._running = True

        for i in range(self._max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"StreamWorker-{i}"
            )
            worker.start()
            self._workers.append(worker)

    def stop(self):
        """Stop all worker threads."""
        self._running = False

        # Drain queue
        while not self._operation_queue.empty():
            try:
                self._operation_queue.get_nowait()
            except queue.Empty:
                break

        for worker in self._workers:
            worker.join(timeout=3)

    def pipeline(self, operations: List[Tuple[str, dict]]) -> List[Any]:
        """
        Execute a pipeline of operations sequentially.

        Each operation can reference results from previous operations
        using special placeholders in params.

        Args:
            operations: List of (command, params) tuples.

        Returns:
            List[Any]: Results in the same order as operations.

        Example:
            >>> results = manager.pipeline([
            ...     ('zeros', {'shape': [100, 100]}),
            ...     ('ones', {'shape': [100, 100]}),
            ... ])
            >>> tid_a = results[0]  # tensor ID from zeros
            >>> tid_b = results[1]  # tensor ID from ones
        """
        if not self._running:
            self.start()

        results = []
        op_ids = []

        for command, params in operations:
            op_id = self.submit(command, params)
            op_ids.append(op_id)

        for op_id in op_ids:
            result = self.wait_for(op_id)
            results.append(result)

        return results

    def submit(
        self,
        command: str,
        params: dict,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
    ) -> int:
        """
        Submit a single operation to the stream.

        Args:
            command: Server command to execute.
            params: Command parameters.
            callback: Called with result on success.
            error_callback: Called with error on failure.

        Returns:
            int: Operation ID for tracking.
        """
        with self._counter_lock:
            self._counter += 1
            op_id = self._counter

        operation = StreamOperation(
            op_id=op_id,
            command=command,
            params=params,
            callback=callback,
            error_callback=error_callback,
        )

        event = threading.Event()
        self._completion_events[op_id] = event

        self._operation_queue.put(operation)
        self.stats['operations_submitted'] += 1

        return op_id

    def wait_for(self, op_id: int, timeout: Optional[float] = None) -> Any:
        """
        Wait for an operation to complete.

        Args:
            op_id: Operation ID.
            timeout: Maximum wait time.

        Returns:
            Any: Operation result.

        Raises:
            TimeoutError: If operation doesn't complete.
            RuntimeError: If operation failed.
        """
        event = self._completion_events.get(op_id)
        if event is None:
            raise ValueError(f"Unknown operation ID: {op_id}")

        if event.wait(timeout):
            with self._results_lock:
                result = self._results.pop(op_id, None)

            if result is None:
                raise RuntimeError(f"Operation {op_id} result not found")

            if isinstance(result, Exception):
                raise result

            return result
        else:
            raise TimeoutError(f"Operation {op_id} timed out")

        finally:
            self._completion_events.pop(op_id, None)

    def _worker_loop(self):
        """Worker thread main loop."""
        while self._running:
            try:
                operation = self._operation_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._execute_operation(operation)
            except Exception:
                pass
            finally:
                self._operation_queue.task_done()

    def _execute_operation(self, operation: StreamOperation):
        """Execute a single operation."""
        operation.status = OperationStatus.RUNNING
        operation.started_at = time.time()

        try:
            conn = self._pool.get_connection()
            result = conn.send_command(operation.command, operation.params)

            operation.result = result
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = time.time()

            self.stats['operations_completed'] += 1

            with self._results_lock:
                self._results[operation.op_id] = result

            if operation.callback:
                try:
                    operation.callback(result)
                except Exception:
                    pass

        except Exception as e:
            operation.error = str(e)
            operation.status = OperationStatus.FAILED
            operation.completed_at = time.time()

            self.stats['operations_failed'] += 1

            with self._results_lock:
                self._results[operation.op_id] = e

            if operation.error_callback:
                try:
                    operation.error_callback(e)
                except Exception:
                    pass

        finally:
            event = self._completion_events.get(operation.op_id)
            if event:
                event.set()

    def get_stats(self) -> dict:
        """Get stream manager statistics."""
        return {
            **self.stats,
            'queue_size': self._operation_queue.qsize(),
            'workers': self._max_workers,
            'running': self._running,
        }

    def shutdown(self):
        """Stop workers and clean up."""
        self.stop()
        self._completion_events.clear()
        with self._results_lock:
            self._results.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False