"""
Message Protocol — Server Side
==============================
Handles encoding/decoding of JSON messages with length-prefix framing.

This is the server-side implementation. The client uses
pure Python JSON + struct directly (see client/connection.py).
"""

import json
import struct
from typing import Tuple, Optional, Any


# Protocol constants
LENGTH_FORMAT = '!I'       # 4-byte unsigned int, network byte order
LENGTH_SIZE = 4            # Size of length prefix in bytes
JSON_ENCODING = 'utf-8'
MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100 MB


def encode_message(data: Any) -> bytes:
    """
    Encode a Python object as a length-prefixed JSON message.

    Args:
        data: Any JSON-serializable Python object.

    Returns:
        bytes: Encoded message ready for socket.send().

    Example:
        >>> msg = encode_message({'status': 'ok', 'result': 42})
        >>> isinstance(msg, bytes)
        True
    """
    json_data = json.dumps(data, ensure_ascii=False).encode(JSON_ENCODING)
    length_prefix = struct.pack(LENGTH_FORMAT, len(json_data))
    return length_prefix + json_data


def decode_message(data: bytes) -> Tuple[Optional[dict], bytes]:
    """
    Decode a length-prefixed JSON message from a byte buffer.

    Handles partial messages by returning the remaining buffer
    when a complete message cannot be extracted.

    Args:
        data: Accumulated receive buffer.

    Returns:
        Tuple[Optional[dict], bytes]:
            - Decoded message (dict), or None if incomplete
            - Remaining buffer after consuming the message

    Example:
        >>> buffer = recv_data()
        >>> msg, buffer = decode_message(buffer)
        >>> if msg:
        ...     process(msg)
    """
    if len(data) < LENGTH_SIZE:
        return None, data

    msg_length = struct.unpack(LENGTH_FORMAT, data[:LENGTH_SIZE])[0]

    if msg_length > MAX_MESSAGE_SIZE:
        # Skip invalid message
        return None, data[LENGTH_SIZE:]

    total_needed = LENGTH_SIZE + msg_length
    if len(data) < total_needed:
        return None, data

    json_bytes = data[LENGTH_SIZE:total_needed]

    try:
        message = json.loads(json_bytes.decode(JSON_ENCODING))
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Malformed message — skip and continue
        return None, data[total_needed:]

    return message, data[total_needed:]