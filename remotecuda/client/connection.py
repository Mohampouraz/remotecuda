"""
Client Connection Module — Pure Python Standard Library Only
=============================================================
Zero-dependency TCP client for RemoteCUDA v3.0 server.

This module uses ONLY Python standard library modules:
    - socket       (TCP communication)
    - json         (message serialization)
    - struct       (binary length prefix)
    - base64       (tensor data encoding)
    - threading    (thread safety)
    - time         (timeouts and timing)

ABSOLUTELY NO external dependencies:
    - No NumPy
    - No PyTorch
    - No pickle (security risk)
    - No third-party packages whatsoever

Protocol Specification:
=======================
The client and server communicate via a simple length-prefixed JSON protocol
over TCP. All tensor data is transmitted as base64-encoded raw bytes with
shape and dtype metadata.

Message Structure (Client → Server):
    [4-byte length prefix (network byte order)] [JSON payload (UTF-8)]

    JSON payload format:
    {
        "command": "operation_name",     // String: command to execute
        "params": {},                     // Object: command parameters
        "id": 12345                       // Integer: request ID for tracking
    }

Message Structure (Server → Client):
    [4-byte length prefix (network byte order)] [JSON payload (UTF-8)]

    JSON payload format (success):
    {
        "status": "ok",                  // String: response status
        "result": <value>,               // Any JSON value: response data
        "id": 12345                      // Integer: matching request ID
    }

    JSON payload format (error):
    {
        "status": "error",               // String: error status
        "error": "Error message",        // String: human-readable error
        "code": "ERROR_CODE",            // String: machine-readable code
        "id": 12345                      // Integer: matching request ID
    }

Tensor Data Encoding:
    Tensors are transferred as JSON objects with base64-encoded raw bytes:
    {
        "data": "base64_encoded_string",  // String: base64(raw_bytes)
        "shape": [100, 100],              // Array: tensor dimensions
        "dtype": "float32",               // String: data type
        "format": "f",                    // String: struct format char
        "size": 40000                     // Integer: raw bytes size
    }

Supported Data Types:
    - float32  (struct format: 'f', 4 bytes)
    - float64  (struct format: 'd', 8 bytes)
    - int32    (struct format: 'i', 4 bytes)
    - int64    (struct format: 'q', 8 bytes)
    - uint8    (struct format: 'B', 1 byte)
    - int8     (struct format: 'b', 1 byte)

Usage Example:
    from remotecuda.client.connection import ClientConnection

    # Connect to server
    conn = ClientConnection('192.168.1.100', 55555)
    conn.connect()

    # Send commands
    info = conn.send_command('info', {})
    tensor_id = conn.send_command('zeros', {'shape': [100, 100]})
    data = conn.send_command('get_tensor', {'tensor_id': tensor_id})

    # Disconnect
    conn.disconnect()

Error Handling:
    - ConnectionError: Network issues, server unreachable
    - TimeoutError: Command execution timeout
    - RuntimeError: Server-side execution errors
    - ValueError: Invalid parameters

Thread Safety:
    All public methods are thread-safe via internal locking.
    Multiple threads can share a single connection safely.
"""

import socket
import json
import struct
import base64
import time
import threading
from typing import Optional, Dict, Any, Tuple, List, Union


# ============================================================
#  Constants
# ============================================================

# Socket configuration
DEFAULT_PORT = 55555
DEFAULT_TIMEOUT = 30.0
RECV_CHUNK_SIZE = 65536  # 64KB receive chunks
MAX_MESSAGE_SIZE = 1024 * 1024 * 100  # 100MB max message size

# Message format
LENGTH_PREFIX_SIZE = 4  # 32-bit unsigned integer, network byte order
LENGTH_FORMAT = '!I'    # struct format for length prefix
JSON_ENCODING = 'utf-8'

# Response status values
STATUS_OK = 'ok'
STATUS_ERROR = 'error'

# Error codes
ERR_CONNECTION_LOST = 'CONNECTION_LOST'
ERR_TIMEOUT = 'TIMEOUT'
ERR_SERVER_ERROR = 'SERVER_ERROR'
ERR_INVALID_RESPONSE = 'INVALID_RESPONSE'
ERR_MESSAGE_TOO_LARGE = 'MESSAGE_TOO_LARGE'
ERR_NOT_CONNECTED = 'NOT_CONNECTED'


# ============================================================
#  Custom Exceptions
# ============================================================

class RemoteCUDAError(Exception):
    """Base exception for all RemoteCUDA client errors."""
    pass


class ConnectionLostError(RemoteCUDAError):
    """Raised when the connection to the server is lost."""
    pass


class ServerError(RemoteCUDAError):
    """Raised when the server returns an error response."""
    
    def __init__(self, message: str, code: str = ERR_SERVER_ERROR):
        self.code = code
        super().__init__(message)


class MessageTooLargeError(RemoteCUDAError):
    """Raised when a message exceeds the maximum allowed size."""
    pass


# ============================================================
#  Data Type Utilities
# ============================================================

# Mapping from dtype strings to struct format characters and byte sizes
DTYPE_MAP = {
    'float32': {'fmt': 'f', 'size': 4, 'name': 'float32'},
    'float64': {'fmt': 'd', 'size': 8, 'name': 'float64'},
    'int32':   {'fmt': 'i', 'size': 4, 'name': 'int32'},
    'int64':   {'fmt': 'q', 'size': 8, 'name': 'int64'},
    'uint8':   {'fmt': 'B', 'size': 1, 'name': 'uint8'},
    'int8':    {'fmt': 'b', 'size': 1, 'name': 'int8'},
    'int16':   {'fmt': 'h', 'size': 2, 'name': 'int16'},
    'uint16':  {'fmt': 'H', 'size': 2, 'name': 'uint16'},
    'uint32':  {'fmt': 'I', 'size': 4, 'name': 'uint32'},
    'uint64':  {'fmt': 'Q', 'size': 8, 'name': 'uint64'},
}

# Reverse mapping from struct format to dtype
FMT_TO_DTYPE = {v['fmt']: k for k, v in DTYPE_MAP.items()}


def get_dtype_info(dtype: str) -> dict:
    """
    Get information about a supported data type.

    Args:
        dtype: Data type string (e.g., 'float32', 'int64').

    Returns:
        dict: {'fmt': struct_format, 'size': bytes_per_element, 'name': canonical_name}

    Raises:
        ValueError: If dtype is not supported.
    """
    info = DTYPE_MAP.get(dtype)
    if info is None:
        raise ValueError(
            f"Unsupported dtype: {dtype}. "
            f"Supported types: {list(DTYPE_MAP.keys())}"
        )
    return info


def encode_tensor_data(data_list: list, dtype: str = 'float32') -> str:
    """
    Encode a flat Python list as base64-encoded raw bytes.

    Pure Python — no NumPy needed.

    Args:
        data_list: Flat list of numbers (floats or ints).
        dtype: Data type string.

    Returns:
        str: Base64-encoded string.

    Raises:
        ValueError: If data_list contains invalid values or dtype is unsupported.
    """
    dtype_info = get_dtype_info(dtype)
    fmt = dtype_info['fmt']
    
    raw_bytes = b''
    
    try:
        if dtype in ('float32', 'float64'):
            for val in data_list:
                raw_bytes += struct.pack(fmt, float(val))
        elif dtype in ('int32', 'int64', 'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64'):
            for val in data_list:
                raw_bytes += struct.pack(fmt, int(val))
        else:
            raise ValueError(f"Unsupported dtype for encoding: {dtype}")
    except (struct.error, OverflowError) as e:
        raise ValueError(f"Data encoding error: {e}. Check that values match dtype {dtype}.")
    
    return base64.b64encode(raw_bytes).decode('ascii')


def decode_tensor_data(encoded: str, dtype: str, shape: list) -> list:
    """
    Decode base64-encoded tensor data back to a flat Python list.

    Pure Python — no NumPy needed.

    Args:
        encoded: Base64-encoded string.
        dtype: Data type string.
        shape: Original tensor shape (for validation).

    Returns:
        list: Flat list of decoded values.

    Raises:
        ValueError: If encoding or dtype is invalid.
    """
    dtype_info = get_dtype_info(dtype)
    fmt = dtype_info['fmt']
    element_size = dtype_info['size']
    
    try:
        raw_bytes = base64.b64decode(encoded)
    except Exception as e:
        raise ValueError(f"Base64 decode error: {e}")
    
    expected_size = 1
    for dim in shape:
        expected_size *= dim
    expected_bytes = expected_size * element_size
    
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Data size mismatch: got {len(raw_bytes)} bytes, "
            f"expected {expected_bytes} bytes for shape {shape} with dtype {dtype}"
        )
    
    num_elements = len(raw_bytes) // element_size
    data = []
    
    try:
        if dtype in ('float32', 'float64'):
            for i in range(num_elements):
                val = struct.unpack_from(fmt, raw_bytes, i * element_size)[0]
                data.append(float(val))
        elif dtype in ('int32', 'int64', 'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64'):
            for i in range(num_elements):
                val = struct.unpack_from(fmt, raw_bytes, i * element_size)[0]
                data.append(int(val))
        else:
            raise ValueError(f"Unsupported dtype for decoding: {dtype}")
    except struct.error as e:
        raise ValueError(f"Data decoding error: {e}")
    
    return data


# ============================================================
#  Client Connection Class
# ============================================================

class ClientConnection:
    """
    Pure Python TCP client for RemoteCUDA v3.0 server.

    Provides a simple, thread-safe interface for sending commands
    to a RemoteCUDA computation server and receiving responses.

    All tensor data is transmitted as base64-encoded JSON strings.
    No external dependencies required on the client side.

    Features:
        - JSON protocol (human-readable, secure, no pickle)
        - Length-prefixed messages for reliable TCP streaming
        - Thread-safe with internal locking
        - Automatic reconnection support
        - Comprehensive error handling
        - Configurable timeouts
        - Command tracking with sequence IDs

    Usage:
        # Basic usage
        conn = ClientConnection('192.168.1.100', 55555)
        conn.connect()
        result = conn.send_command('zeros', {'shape': [100, 100]})
        conn.disconnect()

        # Context manager (auto connect/disconnect)
        with ClientConnection('192.168.1.100', 55555) as conn:
            info = conn.send_command('info', {})
            tensor = conn.send_command('ones', {'shape': [50, 50]})

        # With error handling
        try:
            conn.connect()
            result = conn.send_command('matmul', {'a_id': 1, 'b_id': 2})
        except ConnectionLostError:
            print("Server connection lost")
        except ServerError as e:
            print(f"Server error: {e}")

    Attributes:
        host (str): Server hostname or IP address.
        port (int): Server port number.
        timeout (float): Socket timeout in seconds.
        is_connected (bool): Whether currently connected to server.
        command_count (int): Number of commands sent (read-only).
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT
    ):
        """
        Initialize a client connection.

        Args:
            host: Server IP address or hostname.
            port: Server port number (default: 55555).
            timeout: Socket timeout in seconds for connect and receive operations.
                     Commands may take longer; this is the network timeout,
                     not the command execution timeout.

        Raises:
            ValueError: If host is empty or port is invalid.
        """
        if not host or not host.strip():
            raise ValueError("Host must not be empty")
        if port < 1 or port > 65535:
            raise ValueError(f"Invalid port: {port}. Must be 1-65535.")
        if timeout <= 0:
            raise ValueError(f"Invalid timeout: {timeout}. Must be positive.")

        self.host = host.strip()
        self.port = port
        self.timeout = timeout

        # Internal state
        self._socket: Optional[socket.socket] = None
        self._lock = threading.RLock()
        self._is_connected = False
        self._command_counter = 0
        self._total_bytes_sent = 0
        self._total_bytes_received = 0
        self._connected_at: Optional[float] = None

    # ============================================================
    #  Properties
    # ============================================================

    @property
    def is_connected(self) -> bool:
        """
        Check if the client is currently connected to the server.

        Returns:
            bool: True if connected and socket is alive.
        """
        return self._is_connected and self._socket is not None

    @property
    def command_count(self) -> int:
        """
        Total number of commands sent since connection.

        Returns:
            int: Command count.
        """
        return self._command_counter

    @property
    def bytes_sent(self) -> int:
        """
        Total bytes sent to server.

        Returns:
            int: Bytes sent.
        """
        return self._total_bytes_sent

    @property
    def bytes_received(self) -> int:
        """
        Total bytes received from server.

        Returns:
            int: Bytes received.
        """
        return self._total_bytes_received

    @property
    def connection_duration(self) -> Optional[float]:
        """
        Duration of current connection in seconds.

        Returns:
            Optional[float]: Connection duration, or None if not connected.
        """
        if self._connected_at is None:
            return None
        return time.time() - self._connected_at

    # ============================================================
    #  Connection Management
    # ============================================================

    def connect(self):
        """
        Establish a TCP connection to the RemoteCUDA server.

        This method blocks until the connection is established or fails.
        Uses TCP_NODELAY for low-latency communication.

        Raises:
            ConnectionError: If the server cannot be reached.
            TimeoutError: If the connection attempt times out.
            RuntimeError: If already connected.
            OSError: For low-level socket errors.

        Note:
            Safe to call multiple times — subsequent calls are no-ops
            if already connected.
        """
        if self._is_connected:
            return

        try:
            # Create socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Set socket options
            self._socket.settimeout(self.timeout)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            # Set keepalive options (platform-dependent)
            try:
                # TCP_KEEPIDLE: time before starting keepalive probes (seconds)
                self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                # TCP_KEEPINTVL: interval between keepalive probes (seconds)
                self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                # TCP_KEEPCNT: number of probes before declaring dead
                self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
            except (AttributeError, OSError):
                # Platform doesn't support these options — that's OK
                pass

            # Connect
            self._socket.connect((self.host, self.port))

            # Success
            self._is_connected = True
            self._connected_at = time.time()
            self._command_counter = 0
            self._total_bytes_sent = 0
            self._total_bytes_received = 0

        except socket.timeout:
            self._cleanup_socket()
            raise TimeoutError(
                f"Connection to {self.host}:{self.port} timed out. "
                f"Make sure the server is running ('remotecuda start') "
                f"and the address is correct."
            )
        except ConnectionRefusedError:
            self._cleanup_socket()
            raise ConnectionError(
                f"Connection refused by {self.host}:{self.port}. "
                f"Ensure 'remotecuda start' is running on the server "
                f"and no firewall is blocking the connection."
            )
        except socket.gaierror:
            self._cleanup_socket()
            raise ConnectionError(
                f"Could not resolve hostname: {self.host}. "
                f"Check the server address."
            )
        except OSError as e:
            self._cleanup_socket()
            raise ConnectionError(f"Socket error connecting to {self.host}:{self.port}: {e}")
        except Exception as e:
            self._cleanup_socket()
            raise ConnectionError(f"Unexpected error connecting: {e}")

    def disconnect(self):
        """
        Gracefully close the connection to the server.

        This method is safe to call multiple times and will not raise
        exceptions even if the socket is already closed or broken.

        All resources are properly cleaned up.
        """
        self._is_connected = False
        self._connected_at = None
        self._cleanup_socket()

    def _cleanup_socket(self):
        """
        Internal helper to safely close and remove the socket.

        Never raises exceptions — all errors are silently handled
        to ensure cleanup always succeeds.
        """
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except (OSError, Exception):
                pass  # Socket may already be closed

            try:
                self._socket.close()
            except (OSError, Exception):
                pass  # Best-effort close

            self._socket = None

    def reconnect(self):
        """
        Disconnect and reconnect to the server.

        Useful after a connection loss or when the server restarts.

        Raises:
            Same exceptions as connect().
        """
        self.disconnect()
        self.connect()

    # ============================================================
    #  Command Execution
    # ============================================================

    def send_command(self, command: str, params: dict) -> Any:
        """
        Send a command to the server and return the response.

        This is the primary method for interacting with the RemoteCUDA server.
        All operations (tensor creation, computation, query) use this method.

        The command and parameters are serialized to JSON, prefixed with
        a 4-byte length header, and sent over TCP. The response is read
        and parsed similarly.

        Args:
            command: Command name string. Supported commands:
                - 'info': Get server information
                - 'ping': Health check
                - 'zeros': Create zero tensor
                - 'ones': Create one tensor
                - 'full': Create constant tensor
                - 'send_tensor': Transfer tensor data to server
                - 'get_tensor': Retrieve tensor data from server
                - 'free_tensor': Free a tensor
                - 'add': Element-wise addition
                - 'subtract': Element-wise subtraction
                - 'multiply': Element-wise multiplication
                - 'divide': Element-wise division
                - 'matmul': Matrix multiplication
                - 'relu': ReLU activation
                - 'sigmoid': Sigmoid activation
                - 'tanh': Tanh activation
                - 'softmax': Softmax
                - 'sum': Sum reduction
                - 'mean': Mean reduction
                - 'reshape': Reshape tensor
                - 'transpose': Transpose tensor

            params: Command parameters as a dictionary. Parameters vary
                    by command — see individual command documentation.

        Returns:
            Any: Server response. The type depends on the command:
                - 'info': dict with server information
                - tensor creation: int (tensor ID)
                - computation: int (result tensor ID)
                - 'get_tensor': dict with 'data', 'shape', 'dtype'
                - 'ping': dict with 'status'

        Raises:
            ConnectionLostError: If not connected or connection is lost
                                 during the operation.
            ServerError: If the server returns an error response.
            TimeoutError: If the operation times out.
            MessageTooLargeError: If the response exceeds the maximum size.
            ValueError: If command or params are invalid.

        Example:
            >>> conn = ClientConnection('10.0.0.5', 55555)
            >>> conn.connect()
            >>>
            >>> # Create tensors
            >>> a = conn.send_command('ones', {'shape': [100, 100]})
            >>> b = conn.send_command('ones', {'shape': [100, 100]})
            >>>
            >>> # Compute
            >>> c = conn.send_command('add', {'a_id': a, 'b_id': b})
            >>>
            >>> # Get result
            >>> result = conn.send_command('get_tensor', {'tensor_id': c})
            >>> print(result['data'][0])  # Should be 2.0
            >>>
            >>> # Cleanup
            >>> conn.disconnect()
        """
        if not command or not isinstance(command, str):
            raise ValueError("Command must be a non-empty string")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Params must be a dictionary")

        if not self.is_connected:
            raise ConnectionLostError(
                "Not connected to server. Call connect() first."
            )

        with self._lock:
            # Increment command counter for tracking
            self._command_counter += 1
            command_id = self._command_counter

            # Build the request message
            request = {
                'command': command,
                'params': params,
                'id': command_id,
                'timestamp': time.time(),
            }

            # Serialize to JSON
            try:
                json_data = json.dumps(request, ensure_ascii=False).encode(JSON_ENCODING)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Failed to serialize command to JSON: {e}")

            # Check message size
            if len(json_data) > MAX_MESSAGE_SIZE:
                raise MessageTooLargeError(
                    f"Request message too large: {len(json_data)} bytes. "
                    f"Maximum is {MAX_MESSAGE_SIZE} bytes."
                )

            # Prefix with 4-byte length header (network byte order)
            length_prefix = struct.pack(LENGTH_FORMAT, len(json_data))
            message = length_prefix + json_data

            # Send the message
            try:
                self._socket.sendall(message)
                self._total_bytes_sent += len(message)
            except socket.timeout:
                self._handle_connection_loss("Send timed out")
                raise TimeoutError("Send operation timed out")
            except (OSError, ConnectionError) as e:
                self._handle_connection_loss(f"Send failed: {e}")
                raise ConnectionLostError(f"Connection lost during send: {e}")

            # Receive the response
            try:
                # First, read the 4-byte length prefix
                length_bytes = self._recv_exact(LENGTH_PREFIX_SIZE)

                if len(length_bytes) < LENGTH_PREFIX_SIZE:
                    self._handle_connection_loss("Server closed connection")
                    raise ConnectionLostError(
                        "Server closed connection before sending response"
                    )

                response_length = struct.unpack(LENGTH_FORMAT, length_bytes)[0]

                # Validate response length
                if response_length > MAX_MESSAGE_SIZE:
                    raise MessageTooLargeError(
                        f"Response message too large: {response_length} bytes. "
                        f"Maximum is {MAX_MESSAGE_SIZE} bytes."
                    )
                if response_length == 0:
                    raise ConnectionLostError(
                        "Server sent empty response (length = 0)"
                    )

                # Read the response body
                response_bytes = self._recv_exact(response_length)

                if len(response_bytes) < response_length:
                    self._handle_connection_loss("Incomplete response")
                    raise ConnectionLostError(
                        f"Incomplete response: got {len(response_bytes)} bytes, "
                        f"expected {response_length} bytes"
                    )

                self._total_bytes_received += LENGTH_PREFIX_SIZE + len(response_bytes)

            except socket.timeout:
                self._handle_connection_loss("Receive timed out")
                raise TimeoutError("Receive operation timed out")
            except (OSError, ConnectionError) as e:
                self._handle_connection_loss(f"Receive failed: {e}")
                raise ConnectionLostError(f"Connection lost during receive: {e}")
            except (MessageTooLargeError, ConnectionLostError):
                raise
            except Exception as e:
                self._handle_connection_loss(f"Unexpected receive error: {e}")
                raise ConnectionLostError(f"Unexpected error: {e}")

            # Parse JSON response
            try:
                response = json.loads(response_bytes.decode(JSON_ENCODING))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ConnectionLostError(
                    f"Failed to parse server response: {e}. "
                    f"Response may be corrupted."
                )

            # Check response structure
            if not isinstance(response, dict):
                raise ConnectionLostError(
                    f"Invalid response type: expected dict, got {type(response).__name__}"
                )

            # Check for server errors
            if response.get('status') == STATUS_ERROR:
                error_msg = response.get('error', 'Unknown server error')
                error_code = response.get('code', ERR_SERVER_ERROR)
                raise ServerError(error_msg, code=error_code)

            # Check for success status
            if response.get('status') != STATUS_OK:
                # Some commands return just the result without status wrapper
                if 'result' in response:
                    return response['result']
                # Return the entire response if no status field
                return response

            # Return the result
            return response.get('result', response)

    # ============================================================
    #  Internal Helpers
    # ============================================================

    def _recv_exact(self, n: int) -> bytes:
        """
        Receive exactly n bytes from the socket.

        Handles partial receives by looping until all bytes are received.
        This is necessary because TCP is a stream protocol and may
        deliver data in chunks smaller than the requested size.

        Args:
            n: Exact number of bytes to receive.

        Returns:
            bytes: Received data (may be less than n on error/disconnect).

        Raises:
            socket.timeout: If a receive operation times out.
            OSError: For other socket errors.
        """
        data = b''
        remaining = n

        while remaining > 0:
            # Receive up to the remaining bytes, but no more than chunk size
            chunk_size = min(remaining, RECV_CHUNK_SIZE)

            try:
                chunk = self._socket.recv(chunk_size)
            except socket.timeout:
                # Return what we have on timeout
                break
            except OSError:
                # Return what we have on error
                break

            if not chunk:
                # Socket closed by remote
                break

            data += chunk
            remaining -= len(chunk)

        return data

    def _handle_connection_loss(self, reason: str):
        """
        Handle detection of connection loss.

        Marks the connection as disconnected and cleans up the socket.

        Args:
            reason: Human-readable reason for logging.
        """
        self._is_connected = False
        self._connected_at = None
        self._cleanup_socket()

    # ============================================================
    #  Health Check
    # ============================================================

    def ping(self) -> bool:
        """
        Check if the server is responsive.

        Sends a lightweight ping command and checks the response.

        Returns:
            bool: True if server responds correctly.

        Note:
            Does not raise exceptions on failure — returns False instead.
        """
        try:
            if not self.is_connected:
                return False

            response = self.send_command('ping', {})
            return response.get('status') == 'ok'
        except Exception:
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get detailed server information.

        Returns:
            dict: Server information including:
                - 'device': Current compute device ('cuda:0' or 'cpu')
                - 'gpu_count': Number of GPUs available
                - 'gpu_name': GPU model name (if CUDA available)
                - 'active_tensors': Number of tensors in memory
                - 'memory_used_mb': Memory used in MB
                - 'memory_total_mb': Total memory in MB
                - 'host': Server hostname
                - 'version': Server version
                - 'uptime_seconds': Server uptime

        Raises:
            ConnectionLostError: If not connected.
            ServerError: If server returns an error.
        """
        return self.send_command('info', {})

    # ============================================================
    #  Context Manager Support
    # ============================================================

    def __enter__(self):
        """
        Context manager entry — automatically connects.

        Usage:
            with ClientConnection('10.0.0.5') as conn:
                result = conn.send_command('ping', {})

        Returns:
            self: The connected ClientConnection instance.
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit — automatically disconnects.

        Always disconnects, even if an exception occurred inside the block.
        """
        self.disconnect()
        return False  # Do not suppress exceptions

    # ============================================================
    #  String Representation
    # ============================================================

    def __repr__(self) -> str:
        """
        Human-readable representation of the connection.

        Returns:
            str: Connection description.
        """
        state = "connected" if self._is_connected else "disconnected"
        cmds = f"{self._command_counter} commands" if self._is_connected else ""
        duration = (
            f", {self.connection_duration:.0f}s"
            if self._is_connected and self.connection_duration
            else ""
        )
        return (
            f"ClientConnection({self.host}:{self.port}, {state}"
            f"{duration}{', ' + cmds if cmds else ''})"
        )

    def __str__(self) -> str:
        """
        User-friendly string representation.

        Returns:
            str: Connection status.
        """
        if self._is_connected:
            return (
                f"Connected to RemoteCUDA server at {self.host}:{self.port} "
                f"({self.connection_duration:.0f}s, {self._command_counter} commands)"
            )
        else:
            return f"Disconnected from {self.host}:{self.port}"

    # ============================================================
    #  Destructor
    # ============================================================

    def __del__(self):
        """
        Destructor — ensures socket is cleaned up on garbage collection.

        Note:
            It's better to explicitly call disconnect() rather than
            relying on the destructor.
        """
        try:
            self.disconnect()
        except Exception:
            pass