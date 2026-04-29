#!/usr/bin/env python3
"""
Connection Tests — Pure Python Client
======================================
Tests for the ClientConnection class.

These tests can run without a server to verify
the client code is correct. For integration tests
that require a server, set the environment variable:

    REMOTECUDA_TEST_SERVER=192.168.1.100:55555

Usage:
    python -m pytest tests/test_connection.py -v
    REMOTECUDA_TEST_SERVER=localhost:55555 python -m pytest tests/test_connection.py -v
"""

import os
import sys
import unittest
import struct
import json
import base64


class TestClientConnection(unittest.TestCase):
    """Tests for the ClientConnection class."""

    @classmethod
    def setUpClass(cls):
        """Check if we have a test server available."""
        cls.server_addr = os.environ.get('REMOTECUDA_TEST_SERVER', '')
        cls.has_server = bool(cls.server_addr)

    def setUp(self):
        """Import the client connection module."""
        from remotecuda.client.connection import ClientConnection
        self.ClientConnection = ClientConnection

    def test_import(self):
        """Test that the module imports correctly."""
        from remotecuda.client import connection
        self.assertTrue(hasattr(connection, 'ClientConnection'))

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        conn = self.ClientConnection('localhost', 55555)
        self.assertEqual(conn.host, 'localhost')
        self.assertEqual(conn.port, 55555)
        self.assertFalse(conn.is_connected)

    def test_init_invalid_host(self):
        """Test initialization with invalid host."""
        with self.assertRaises(ValueError):
            self.ClientConnection('', 55555)

        with self.assertRaises(ValueError):
            self.ClientConnection('   ', 55555)

    def test_init_invalid_port(self):
        """Test initialization with invalid port."""
        with self.assertRaises(ValueError):
            self.ClientConnection('localhost', 0)

        with self.assertRaises(ValueError):
            self.ClientConnection('localhost', 65536)

        with self.assertRaises(ValueError):
            self.ClientConnection('localhost', -1)

    def test_init_invalid_timeout(self):
        """Test initialization with invalid timeout."""
        with self.assertRaises(ValueError):
            self.ClientConnection('localhost', 55555, timeout=0)

        with self.assertRaises(ValueError):
            self.ClientConnection('localhost', 55555, timeout=-1)

    def test_disconnect_when_not_connected(self):
        """Test that disconnect() is safe when not connected."""
        conn = self.ClientConnection('localhost', 55555)
        conn.disconnect()  # Should not raise
        self.assertFalse(conn.is_connected)

    def test_repr(self):
        """Test string representation."""
        conn = self.ClientConnection('10.0.0.1', 55555, timeout=30)
        repr_str = repr(conn)
        self.assertIn('10.0.0.1', repr_str)
        self.assertIn('55555', repr_str)
        self.assertIn('disconnected', repr_str)

    def test_context_manager(self):
        """Test context manager support."""
        conn = self.ClientConnection('10.0.0.1', 55555, timeout=1)
        # Context manager should not raise on failed connect
        try:
            with conn:
                pass
        except Exception:
            pass
        self.assertFalse(conn.is_connected)

    def test_command_count_initial(self):
        """Test that command count starts at 0."""
        conn = self.ClientConnection('localhost', 55555)
        self.assertEqual(conn.command_count, 0)

    def test_bytes_counters_initial(self):
        """Test that byte counters start at 0."""
        conn = self.ClientConnection('localhost', 55555)
        self.assertEqual(conn.bytes_sent, 0)
        self.assertEqual(conn.bytes_received, 0)

    def test_connection_duration_none_when_disconnected(self):
        """Test that connection duration is None when disconnected."""
        conn = self.ClientConnection('localhost', 55555)
        self.assertIsNone(conn.connection_duration)


class TestEncodingFunctions(unittest.TestCase):
    """Tests for tensor data encoding/decoding functions."""

    def test_encode_float32(self):
        """Test encoding float32 data."""
        from remotecuda.client.connection import encode_tensor_data
        data = [1.0, 2.0, 3.0, 4.0]
        encoded = encode_tensor_data(data, 'float32')
        self.assertIsInstance(encoded, str)

        # Decode to verify
        raw = base64.b64decode(encoded)
        values = struct.unpack('4f', raw)
        self.assertEqual(values, (1.0, 2.0, 3.0, 4.0))

    def test_encode_float64(self):
        """Test encoding float64 data."""
        from remotecuda.client.connection import encode_tensor_data
        data = [1.0, 2.0, 3.0]
        encoded = encode_tensor_data(data, 'float64')
        self.assertIsInstance(encoded, str)

        raw = base64.b64decode(encoded)
        values = struct.unpack('3d', raw)
        self.assertEqual(values, (1.0, 2.0, 3.0))

    def test_encode_int32(self):
        """Test encoding int32 data."""
        from remotecuda.client.connection import encode_tensor_data
        data = [1, 2, 3, 4, 5]
        encoded = encode_tensor_data(data, 'int32')
        self.assertIsInstance(encoded, str)

        raw = base64.b64decode(encoded)
        values = struct.unpack('5i', raw)
        self.assertEqual(values, (1, 2, 3, 4, 5))

    def test_decode_float32(self):
        """Test decoding float32 data."""
        from remotecuda.client.connection import decode_tensor_data

        data = [1.0, 2.0, 3.0, 4.0]
        raw = struct.pack('4f', *data)
        encoded = base64.b64encode(raw).decode('ascii')

        decoded = decode_tensor_data(encoded, 'float32', [4])
        self.assertEqual(decoded, [1.0, 2.0, 3.0, 4.0])

    def test_decode_int32(self):
        """Test decoding int32 data."""
        from remotecuda.client.connection import decode_tensor_data

        data = [10, 20, 30]
        raw = struct.pack('3i', *data)
        encoded = base64.b64encode(raw).decode('ascii')

        decoded = decode_tensor_data(encoded, 'int32', [3])
        self.assertEqual(decoded, [10, 20, 30])

    def test_dtype_info(self):
        """Test getting dtype information."""
        from remotecuda.client.connection import get_dtype_info

        info = get_dtype_info('float32')
        self.assertEqual(info['fmt'], 'f')
        self.assertEqual(info['size'], 4)

        info = get_dtype_info('float64')
        self.assertEqual(info['fmt'], 'd')
        self.assertEqual(info['size'], 8)

        info = get_dtype_info('int32')
        self.assertEqual(info['fmt'], 'i')
        self.assertEqual(info['size'], 4)

        with self.assertRaises(ValueError):
            get_dtype_info('invalid_dtype')

    def test_encode_decode_roundtrip_float32(self):
        """Test full roundtrip: encode → decode."""
        from remotecuda.client.connection import (
            encode_tensor_data,
            decode_tensor_data,
        )

        original = [1.5, 2.5, 3.5, 4.5, 5.5]
        encoded = encode_tensor_data(original, 'float32')
        decoded = decode_tensor_data(encoded, 'float32', [5])

        for o, d in zip(original, decoded):
            self.assertAlmostEqual(o, d, places=5)

    def test_encode_decode_roundtrip_int32(self):
        """Test full roundtrip for int32."""
        from remotecuda.client.connection import (
            encode_tensor_data,
            decode_tensor_data,
        )

        original = [100, 200, 300, 400]
        encoded = encode_tensor_data(original, 'int32')
        decoded = decode_tensor_data(encoded, 'int32', [4])

        self.assertEqual(original, decoded)


class TestDiscovery(unittest.TestCase):
    """Tests for the server discovery module."""

    def test_import_discovery(self):
        """Test that the discovery module imports correctly."""
        from remotecuda.client import discovery
        self.assertTrue(hasattr(discovery, 'discover_server'))
        self.assertTrue(hasattr(discovery, 'discover_all_servers'))
        self.assertTrue(hasattr(discovery, 'discover_with_info'))
        self.assertTrue(hasattr(discovery, 'NetworkScanner'))

    def test_discover_all_servers_returns_list(self):
        """Test that discover_all_servers returns a list."""
        from remotecuda.client.discovery import discover_all_servers
        servers = discover_all_servers(timeout=1.0)
        self.assertIsInstance(servers, list)

    def test_network_scanner_init(self):
        """Test NetworkScanner initialization."""
        from remotecuda.client.discovery import NetworkScanner
        scanner = NetworkScanner(server_timeout=10.0)
        self.assertIsNotNone(scanner)
        self.assertEqual(scanner.server_timeout, 10.0)
        self.assertEqual(scanner.get_server_count(), 0)

    def test_network_scanner_lifecycle(self):
        """Test NetworkScanner start/stop."""
        from remotecuda.client.discovery import NetworkScanner
        scanner = NetworkScanner(server_timeout=1.0)
        scanner.start()
        scanner.stop()
        self.assertEqual(scanner.get_server_count(), 0)


class TestExceptions(unittest.TestCase):
    """Tests for custom exception classes."""

    def test_connection_lost_error(self):
        """Test ConnectionLostError."""
        from remotecuda.client.connection import ConnectionLostError
        err = ConnectionLostError("Test error")
        self.assertIsInstance(err, Exception)
        self.assertEqual(str(err), "Test error")

    def test_server_error(self):
        """Test ServerError."""
        from remotecuda.client.connection import ServerError
        err = ServerError("Server failed", code="TEST_ERROR")
        self.assertEqual(err.code, "TEST_ERROR")
        self.assertEqual(str(err), "Server failed")

    def test_message_too_large_error(self):
        """Test MessageTooLargeError."""
        from remotecuda.client.connection import MessageTooLargeError
        err = MessageTooLargeError("Message too big")
        self.assertIsInstance(err, Exception)


class TestIntegrationWithServer(unittest.TestCase):
    """
    Integration tests that require a running server.

    Set REMOTECUDA_TEST_SERVER=host:port to enable.
    """

    @classmethod
    def setUpClass(cls):
        cls.server_addr = os.environ.get('REMOTECUDA_TEST_SERVER', '')
        cls.has_server = bool(cls.server_addr)

        if cls.has_server:
            parts = cls.server_addr.split(':')
            cls.server_host = parts[0]
            cls.server_port = int(parts[1]) if len(parts) > 1 else 55555

    def setUp(self):
        if not self.has_server:
            self.skipTest("No test server configured")

        from remotecuda.client.connection import ClientConnection
        self.conn = ClientConnection(self.server_host, self.server_port)

    def test_connect_and_ping(self):
        """Test connecting to server and sending ping."""
        self.conn.connect()
        self.assertTrue(self.conn.is_connected)

        response = self.conn.send_command('ping', {})
        self.assertEqual(response.get('status'), 'ok')

        self.conn.disconnect()
        self.assertFalse(self.conn.is_connected)

    def test_create_zeros_tensor(self):
        """Test creating a zeros tensor."""
        self.conn.connect()

        tid = self.conn.send_command('zeros', {'shape': [10, 10]})
        self.assertIsInstance(tid, int)
        self.assertGreater(tid, 0)

        # Cleanup
        self.conn.send_command('free_tensor', {'tensor_id': tid})
        self.conn.disconnect()

    def test_create_and_retrieve_tensor(self):
        """Test creating a tensor and retrieving its data."""
        self.conn.connect()

        # Create ones tensor
        tid = self.conn.send_command('ones', {'shape': [5, 5]})

        # Get tensor data
        response = self.conn.send_command('get_tensor', {'tensor_id': tid})
        self.assertIn('data', response)
        self.assertEqual(response['shape'], [5, 5])

        # Decode and check
        import base64
        import struct
        raw = base64.b64decode(response['data'])
        values = struct.unpack('25f', raw)
        for v in values:
            self.assertAlmostEqual(v, 1.0, places=5)

        # Cleanup
        self.conn.send_command('free_tensor', {'tensor_id': tid})
        self.conn.disconnect()

    def test_add_tensors(self):
        """Test tensor addition."""
        self.conn.connect()

        a = self.conn.send_command('ones', {'shape': [10, 10]})
        b = self.conn.send_command('ones', {'shape': [10, 10]})
        c = self.conn.send_command('add', {'a_id': a, 'b_id': b})

        response = self.conn.send_command('get_tensor', {'tensor_id': c})

        import base64
        import struct
        raw = base64.b64decode(response['data'])
        values = struct.unpack('100f', raw)
        for v in values:
            self.assertAlmostEqual(v, 2.0, places=5)

        self.conn.send_command('free_tensor', {'tensor_id': a})
        self.conn.send_command('free_tensor', {'tensor_id': b})
        self.conn.send_command('free_tensor', {'tensor_id': c})
        self.conn.disconnect()

    def test_matmul(self):
        """Test matrix multiplication."""
        self.conn.connect()

        # Identity matrix
        a = self.conn.send_command('eye', {'n': 5, 'm': 5})
        b = self.conn.send_command('ones', {'shape': [5, 3]})
        c = self.conn.send_command('matmul', {'a_id': a, 'b_id': b})

        response = self.conn.send_command('get_tensor', {'tensor_id': c})
        self.assertEqual(response['shape'], [5, 3])

        import base64
        import struct
        raw = base64.b64decode(response['data'])
        values = struct.unpack('15f', raw)
        for v in values:
            self.assertAlmostEqual(v, 1.0, places=5)

        self.conn.send_command('free_tensor', {'tensor_id': a})
        self.conn.send_command('free_tensor', {'tensor_id': b})
        self.conn.send_command('free_tensor', {'tensor_id': c})
        self.conn.disconnect()

    def test_server_info(self):
        """Test getting server information."""
        self.conn.connect()

        info = self.conn.send_command('info', {})
        self.assertIn('device', info)
        self.assertIn('pytorch_version', info)

        self.conn.disconnect()


if __name__ == '__main__':
    unittest.main()