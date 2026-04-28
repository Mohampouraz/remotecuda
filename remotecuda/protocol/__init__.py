"""
RemoteCUDA Protocol Package
===========================
Communication protocols for client-server interaction.
Handles message encoding, compression, and data integrity.
"""

from .messages import MessageProtocol
from .compression import TensorCompressor

__all__ = ['MessageProtocol', 'TensorCompressor']