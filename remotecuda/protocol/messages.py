"""
Message Protocol Module
=======================
Defines the binary protocol for client-server communication.
Optimized for low latency and high throughput.
"""

import struct
import pickle
import zlib
from typing import Optional, Tuple, Any


class MessageProtocol:
    """
    Binary protocol for RemoteCUDA communication.
    
    Message Format:
        [Magic: 4 bytes][Version: 1 byte][Flags: 1 byte]
        [Message Type: 2 bytes][Payload Length: 4 bytes]
        [CRC32: 4 bytes][Payload: variable]
    
    Total Header: 16 bytes
    """
    
    MAGIC = b'RCUD'  # RemoteCUDA magic bytes
    VERSION = 1
    
    # Message types
    MSG_OPERATION = 1     # Client -> Server: Execute operation
    MSG_RESPONSE = 2      # Server -> Client: Operation result
    MSG_STREAM_START = 3  # Start of streaming data
    MSG_STREAM_DATA = 4   # Streaming data chunk
    MSG_STREAM_END = 5    # End of streaming data
    MSG_HEARTBEAT = 6     # Keep-alive
    MSG_ERROR = 7         # Error response
    
    # Flags
    FLAG_COMPRESSED = 0x01
    FLAG_ENCRYPTED = 0x02
    FLAG_STREAM = 0x04
    
    @staticmethod
    def encode_operation(operation: dict, compress: bool = True) -> bytes:
        """
        Encode an operation request message.
        
        Args:
            operation (dict): Operation data
            compress (bool): Whether to compress the payload
            
        Returns:
            bytes: Encoded message ready for socket send
        """
        payload = pickle.dumps(operation)
        flags = 0
        
        if compress:
            payload = zlib.compress(payload, level=3)  # Level 3 = balanced speed/compression
            flags |= MessageProtocol.FLAG_COMPRESSED
        
        return MessageProtocol._build_message(
            MessageProtocol.MSG_OPERATION,
            flags,
            payload
        )
    
    @staticmethod
    def encode_response(response: dict, compress: bool = True) -> bytes:
        """
        Encode a response message.
        
        Args:
            response (dict): Response data
            compress (bool): Whether to compress the payload
            
        Returns:
            bytes: Encoded message
        """
        payload = pickle.dumps(response)
        flags = 0
        
        if compress:
            payload = zlib.compress(payload, level=3)
            flags |= MessageProtocol.FLAG_COMPRESSED
        
        return MessageProtocol._build_message(
            MessageProtocol.MSG_RESPONSE,
            flags,
            payload
        )
    
    @staticmethod
    def _build_message(msg_type: int, flags: int, payload: bytes) -> bytes:
        """
        Build a complete message with header and payload.
        """
        header = struct.pack(
            '!4sBBHI',
            MessageProtocol.MAGIC,
            MessageProtocol.VERSION,
            flags,
            msg_type,
            len(payload)
        )
        
        # CRC32 for data integrity
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        crc_bytes = struct.pack('!I', crc)
        
        return header + crc_bytes + payload
    
    @staticmethod
    def decode(buffer: bytes) -> Tuple[Optional[dict], bytes]:
        """
        Decode a message from a byte buffer.
        
        Args:
            buffer (bytes): Accumulated receive buffer
            
        Returns:
            Tuple[Optional[dict], bytes]: (decoded_message, remaining_buffer)
        """
        # Need at least the header
        if len(buffer) < 16:
            return None, buffer
        
        # Parse header
        magic = buffer[:4]
        if magic != MessageProtocol.MAGIC:
            # Invalid magic bytes - skip one byte and try again
            return None, buffer[1:]
        
        version, flags, msg_type, payload_len = struct.unpack(
            '!BBHI',
            buffer[4:12]
        )
        
        # Check version compatibility
        if version != MessageProtocol.VERSION:
            raise ValueError(f"Unsupported protocol version: {version}")
        
        # Need CRC + payload
        total_needed = 16 + payload_len
        if len(buffer) < total_needed:
            return None, buffer
        
        # Extract CRC and payload
        received_crc = struct.unpack('!I', buffer[12:16])[0]
        payload = buffer[16:total_needed]
        
        # Verify CRC
        calculated_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if received_crc != calculated_crc:
            # Data corruption - skip this message
            return None, buffer[16:]
        
        # Decompress if needed
        if flags & MessageProtocol.FLAG_COMPRESSED:
            try:
                payload = zlib.decompress(payload)
            except zlib.error:
                return None, buffer[total_needed:]
        
        # Deserialize
        try:
            message = pickle.loads(payload)
        except pickle.UnpicklingError:
            return None, buffer[total_needed:]
        
        return message, buffer[total_needed:]