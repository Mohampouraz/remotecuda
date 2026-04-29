"""
CUDA Driver Interface — Pure ctypes, Zero Dependencies
======================================================
Direct communication with NVIDIA GPU through the display driver.
No CUDA Toolkit. No PyTorch. No compiled extensions.

Architecture:
    Python ctypes → nvcuda.dll (Windows) / libcuda.so (Linux)
    Direct GPU control via NVIDIA Driver API.

Required on server:
    - Python 3.8+
    - NumPy
    - NVIDIA display driver (any recent version)

NOT required on server:
    - CUDA Toolkit
    - PyTorch
    - cuDNN
    - Any other ML framework
"""

import ctypes
import platform
import sys
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import threading

# ============================================================
#  CUDA Error Codes
# ============================================================
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
CUDA_ERROR_LAUNCH_FAILED = 719
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
CUDA_ERROR_NOT_FOUND = 500

_ERROR_NAMES = {v: k for k, v in vars().items() if k.startswith('CUDA_ERROR_')}


def _check(result: int):
    if result != CUDA_SUCCESS:
        name = _ERROR_NAMES.get(result, f"UNKNOWN_{result}")
        raise RuntimeError(f"CUDA Error {result}: {name}")


# ============================================================
#  CUDA Driver API Structures (ctypes)
# ============================================================

class _CUdevice_v1(ctypes.Structure):
    pass

class _CUcontext_v1(ctypes.Structure):
    pass

class _CUmodule_v1(ctypes.Structure):
    pass

class _CUfunction_v1(ctypes.Structure):
    pass

class _CUstream_v1(ctypes.Structure):
    pass

# Opaque handles
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64
CUsize = ctypes.c_size_t


# ============================================================
#  Device Info Dataclass
# ============================================================

@dataclass
class GPUDeviceInfo:
    """Complete information about a single GPU device."""
    index: int
    name: str
    total_memory: int
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    max_threads_per_block: int
    max_block_dim: Tuple[int, int, int]
    max_grid_dim: Tuple[int, int, int]
    clock_rate_khz: int
    memory_bus_width: int
    pci_bus_id: int
    pci_device_id: int
    
    @property
    def total_memory_mb(self) -> int:
        return self.total_memory // (1024 * 1024)
    
    @property
    def total_memory_gb(self) -> float:
        return self.total_memory / (1024 ** 3)
    
    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'name': self.name,
            'total_memory': self.total_memory,
            'total_memory_gb': round(self.total_memory_gb, 1),
            'compute_capability': f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            'multiprocessor_count': self.multiprocessor_count,
            'max_threads_per_block': self.max_threads_per_block,
            'clock_rate_khz': self.clock_rate_khz,
        }


# ============================================================
#  Main CUDADriver Class
# ============================================================

class CUDADriver:
    """
    Zero-dependency NVIDIA GPU driver interface.
    
    Communicates directly with nvcuda.dll / libcuda.so via ctypes.
    No CUDA Toolkit, no PyTorch, no compiled code required.
    
    Usage:
        driver = CUDADriver()
        driver.initialize()
        
        info = driver.get_device_info(0)
        print(info.name)
        
        ptr = driver.mem_alloc(1024 * 1024)  # Allocate 1MB on GPU
        # ... use GPU ...
        driver.mem_free(ptr)
        
        driver.shutdown()
    """
    
    # Singleton instance
    _instance: Optional['CUDADriver'] = None
    _lock = threading.Lock()
    
    # Device attribute codes (from cuda.h)
    _CU_DEVICE_ATTRIBUTE = {
        'MAX_THREADS_PER_BLOCK': 1,
        'MAX_BLOCK_DIM_X': 2,
        'MAX_BLOCK_DIM_Y': 3,
        'MAX_BLOCK_DIM_Z': 4,
        'MAX_GRID_DIM_X': 5,
        'MAX_GRID_DIM_Y': 6,
        'MAX_GRID_DIM_Z': 7,
        'MAX_SHARED_MEMORY_PER_BLOCK': 8,
        'TOTAL_CONSTANT_MEMORY': 9,
        'WARP_SIZE': 10,
        'MAX_PITCH': 11,
        'MAX_REGISTERS_PER_BLOCK': 12,
        'CLOCK_RATE': 13,
        'TEXTURE_ALIGNMENT': 14,
        'GPU_OVERLAP': 15,
        'MULTIPROCESSOR_COUNT': 16,
        'KERNEL_EXEC_TIMEOUT': 17,
        'INTEGRATED': 18,
        'CAN_MAP_HOST_MEMORY': 19,
        'COMPUTE_MODE': 20,
        'MAXIMUM_TEXTURE1D_WIDTH': 21,
        'MAXIMUM_TEXTURE2D_WIDTH': 22,
        'MAXIMUM_TEXTURE2D_HEIGHT': 23,
        'MAXIMUM_TEXTURE3D_WIDTH': 24,
        'MAXIMUM_TEXTURE3D_HEIGHT': 25,
        'MAXIMUM_TEXTURE3D_DEPTH': 26,
        'MAXIMUM_TEXTURE2D_LAYERED_WIDTH': 27,
        'MAXIMUM_TEXTURE2D_LAYERED_HEIGHT': 28,
        'MAXIMUM_TEXTURE2D_LAYERED_LAYERS': 29,
        'SURFACE_ALIGNMENT': 30,
        'CONCURRENT_KERNELS': 31,
        'ECC_ENABLED': 32,
        'PCI_BUS_ID': 33,
        'PCI_DEVICE_ID': 34,
        'TCC_DRIVER': 35,
        'MEMORY_CLOCK_RATE': 36,
        'GLOBAL_MEMORY_BUS_WIDTH': 37,
        'L2_CACHE_SIZE': 38,
        'MAX_THREADS_PER_MULTIPROCESSOR': 39,
        'ASYNC_ENGINE_COUNT': 40,
        'UNIFIED_ADDRESSING': 41,
        'MAXIMUM_TEXTURE1D_LAYERED_WIDTH': 42,
        'MAXIMUM_TEXTURE1D_LAYERED_LAYERS': 43,
        'CAN_TEX2D_GATHER': 44,
        'MAXIMUM_TEXTURE2D_GATHER_WIDTH': 45,
        'MAXIMUM_TEXTURE2D_GATHER_HEIGHT': 46,
        'MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE': 47,
        'MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE': 48,
        'MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE': 49,
        'PCI_DOMAIN_ID': 50,
        'TEXTURE_PITCH_ALIGNMENT': 51,
        'MAXIMUM_TEXTURECUBEMAP_WIDTH': 52,
        'MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH': 53,
        'MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS': 54,
        'MAXIMUM_SURFACE1D_WIDTH': 55,
        'MAXIMUM_SURFACE2D_WIDTH': 56,
        'MAXIMUM_SURFACE2D_HEIGHT': 57,
        'MAXIMUM_SURFACE3D_WIDTH': 58,
        'MAXIMUM_SURFACE3D_HEIGHT': 59,
        'MAXIMUM_SURFACE3D_DEPTH': 60,
        'MAXIMUM_SURFACE1D_LAYERED_WIDTH': 61,
        'MAXIMUM_SURFACE1D_LAYERED_LAYERS': 62,
        'MAXIMUM_SURFACE2D_LAYERED_WIDTH': 63,
        'MAXIMUM_SURFACE2D_LAYERED_HEIGHT': 64,
        'MAXIMUM_SURFACE2D_LAYERED_LAYERS': 65,
        'MAXIMUM_SURFACECUBEMAP_WIDTH': 66,
        'MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH': 67,
        'MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS': 68,
        'MAXIMUM_TEXTURE1D_LINEAR_WIDTH': 69,
        'MAXIMUM_TEXTURE2D_LINEAR_WIDTH': 70,
        'MAXIMUM_TEXTURE2D_LINEAR_HEIGHT': 71,
        'MAXIMUM_TEXTURE2D_LINEAR_PITCH': 72,
        'MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH': 73,
        'MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT': 74,
        'COMPUTE_CAPABILITY_MAJOR': 75,
        'COMPUTE_CAPABILITY_MINOR': 76,
        'MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH': 77,
        'STREAM_PRIORITIES_SUPPORTED': 78,
        'GLOBAL_L1_CACHE_SUPPORTED': 79,
        'LOCAL_L1_CACHE_SUPPORTED': 80,
        'MAX_SHARED_MEMORY_PER_MULTIPROCESSOR': 81,
        'MAX_REGISTERS_PER_MULTIPROCESSOR': 82,
        'MANAGED_MEMORY': 83,
        'MULTI_GPU_BOARD': 84,
        'MULTI_GPU_BOARD_GROUP_ID': 85,
    }
    
    def __init__(self):
        self._lib: Optional[ctypes.CDLL] = None
        self._initialized: bool = False
        self._device_count: int = 0
        self._current_context: Optional[int] = None
        self._contexts: Dict[int, CUcontext] = {}
        self._streams: Dict[int, List[CUstream]] = {}
        self._memory_allocations: Dict[int, int] = {}  # ptr → size
    
    @classmethod
    def get_instance(cls) -> 'CUDADriver':
        """Get or create singleton driver instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def initialize(self) -> int:
        """
        Initialize the CUDA driver.
        
        Returns:
            int: Number of GPU devices found.
            
        Raises:
            RuntimeError: If NVIDIA driver not found or no GPUs detected.
        """
        if self._initialized:
            return self._device_count
        
        # Load NVIDIA driver library
        self._lib = self._load_driver_library()
        self._setup_function_signatures()
        
        # Initialize CUDA
        result = self._lib.cuInit(0)
        if result != CUDA_SUCCESS:
            if result == CUDA_ERROR_NO_DEVICE:
                raise RuntimeError(
                    "No NVIDIA GPU found.\n"
                    "Please install NVIDIA display driver from nvidia.com"
                )
            elif result == CUDA_ERROR_NOT_INITIALIZED:
                raise RuntimeError(
                    "NVIDIA driver found but CUDA init failed.\n"
                    "Try reinstalling the latest driver from nvidia.com"
                )
            else:
                _check(result)
        
        # Get device count
        count = ctypes.c_int()
        _check(self._lib.cuDeviceGetCount(ctypes.byref(count)))
        self._device_count = count.value
        
        if self._device_count == 0:
            raise RuntimeError(
                "No CUDA-capable GPU detected.\n"
                "Check that your NVIDIA GPU is properly connected."
            )
        
        # Create contexts for all devices
        for i in range(self._device_count):
            device = ctypes.c_int()
            _check(self._lib.cuDeviceGet(ctypes.byref(device), i))
            
            ctx = CUcontext()
            _check(self._lib.cuCtxCreate(ctypes.byref(ctx), 0, device))
            self._contexts[i] = ctx
            
            # Create default stream
            stream = CUstream()
            _check(self._lib.cuStreamCreate(ctypes.byref(stream), 0))
            self._streams[i] = [stream]
        
        self._initialized = True
        self._set_device(0)  # Default to device 0
        
        return self._device_count
    
    def _load_driver_library(self) -> ctypes.CDLL:
        """Load the NVIDIA driver shared library."""
        system = platform.system()
        
        if system == 'Windows':
            names = ['nvcuda.dll', 'nvcuda64.dll']
        elif system == 'Linux':
            names = ['libcuda.so', 'libcuda.so.1', 'libcuda.so.550']
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")
        
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        
        raise RuntimeError(
            "NVIDIA driver library not found.\n"
            f"Tried: {', '.join(names)}\n"
            "Please install the latest NVIDIA display driver from:\n"
            "https://www.nvidia.com/Download/index.aspx"
        )
    
    def _setup_function_signatures(self):
        """Configure ctypes function signatures for all CUDA API calls."""
        L = self._lib
        
        # ---- Device ----
        L.cuInit.argtypes = [ctypes.c_uint]
        L.cuInit.restype = ctypes.c_int
        
        L.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.cuDeviceGetCount.restype = ctypes.c_int
        
        L.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.cuDeviceGet.restype = ctypes.c_int
        
        L.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.cuDeviceGetName.restype = ctypes.c_int
        
        L.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(CUsize), ctypes.c_int]
        L.cuDeviceTotalMem_v2.restype = ctypes.c_int
        
        L.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        L.cuDeviceGetAttribute.restype = ctypes.c_int
        
        # ---- Context ----
        L.cuCtxCreate_v2.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, ctypes.c_int]
        L.cuCtxCreate_v2.restype = ctypes.c_int
        
        L.cuCtxSetCurrent.argtypes = [CUcontext]
        L.cuCtxSetCurrent.restype = ctypes.c_int
        
        L.cuCtxGetCurrent.argtypes = [ctypes.POINTER(CUcontext)]
        L.cuCtxGetCurrent.restype = ctypes.c_int
        
        L.cuCtxSynchronize.argtypes = []
        L.cuCtxSynchronize.restype = ctypes.c_int
        
        L.cuCtxDestroy_v2.argtypes = [CUcontext]
        L.cuCtxDestroy_v2.restype = ctypes.c_int
        
        # ---- Memory ----
        L.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CUdeviceptr), CUsize]
        L.cuMemAlloc_v2.restype = ctypes.c_int
        
        L.cuMemFree_v2.argtypes = [CUdeviceptr]
        L.cuMemFree_v2.restype = ctypes.c_int
        
        L.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.c_void_p, CUsize]
        L.cuMemcpyHtoD_v2.restype = ctypes.c_int
        
        L.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, CUsize]
        L.cuMemcpyDtoH_v2.restype = ctypes.c_int
        
        L.cuMemcpyDtoD_v2.argtypes = [CUdeviceptr, CUdeviceptr, CUsize]
        L.cuMemcpyDtoD_v2.restype = ctypes.c_int
        
        L.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(CUsize), ctypes.POINTER(CUsize)]
        L.cuMemGetInfo_v2.restype = ctypes.c_int
        
        # ---- Stream ----
        L.cuStreamCreate.argtypes = [ctypes.POINTER(CUstream), ctypes.c_uint]
        L.cuStreamCreate.restype = ctypes.c_int
        
        L.cuStreamDestroy_v2.argtypes = [CUstream]
        L.cuStreamDestroy_v2.restype = ctypes.c_int
        
        L.cuStreamSynchronize.argtypes = [CUstream]
        L.cuStreamSynchronize.restype = ctypes.c_int
    
    # ============================================================
    #  Device Management
    # ============================================================
    
    @property
    def device_count(self) -> int:
        return self._device_count
    
    def _set_device(self, device_index: int):
        """Switch to a specific device context."""
        if device_index not in self._contexts:
            raise ValueError(f"Device {device_index} not initialized")
        ctx = self._contexts[device_index]
        _check(self._lib.cuCtxSetCurrent(ctx))
        self._current_context = device_index
    
    def get_device_info(self, device_index: int) -> GPUDeviceInfo:
        """Get detailed GPU device information."""
        self._set_device(device_index)
        
        device = ctypes.c_int()
        _check(self._lib.cuDeviceGet(ctypes.byref(device), device_index))
        
        # Name
        name_buf = ctypes.create_string_buffer(256)
        _check(self._lib.cuDeviceGetName(name_buf, 256, device))
        name = name_buf.value.decode('utf-8', errors='replace').strip('\x00')
        
        # Total memory
        total_mem = CUsize()
        _check(self._lib.cuDeviceTotalMem_v2(ctypes.byref(total_mem), device))
        
        # Attribute helper
        def attr(code):
            val = ctypes.c_int()
            _check(self._lib.cuDeviceGetAttribute(ctypes.byref(val), code, device))
            return val.value
        
        A = self._CU_DEVICE_ATTRIBUTE
        
        return GPUDeviceInfo(
            index=device_index,
            name=name,
            total_memory=total_mem.value,
            compute_capability=(
                attr(A['COMPUTE_CAPABILITY_MAJOR']),
                attr(A['COMPUTE_CAPABILITY_MINOR'])
            ),
            multiprocessor_count=attr(A['MULTIPROCESSOR_COUNT']),
            max_threads_per_block=attr(A['MAX_THREADS_PER_BLOCK']),
            max_block_dim=(
                attr(A['MAX_BLOCK_DIM_X']),
                attr(A['MAX_BLOCK_DIM_Y']),
                attr(A['MAX_BLOCK_DIM_Z'])
            ),
            max_grid_dim=(
                attr(A['MAX_GRID_DIM_X']),
                attr(A['MAX_GRID_DIM_Y']),
                attr(A['MAX_GRID_DIM_Z'])
            ),
            clock_rate_khz=attr(A['CLOCK_RATE']),
            memory_bus_width=attr(A['GLOBAL_MEMORY_BUS_WIDTH']),
            pci_bus_id=attr(A['PCI_BUS_ID']),
            pci_device_id=attr(A['PCI_DEVICE_ID'])
        )
    
    def get_all_device_info(self) -> List[GPUDeviceInfo]:
        """Get information about all available GPUs."""
        return [self.get_device_info(i) for i in range(self._device_count)]
    
    # ============================================================
    #  Memory Management
    # ============================================================
    
    def mem_alloc(self, size: int) -> int:
        """
        Allocate memory on the current GPU device.
        
        Args:
            size: Number of bytes to allocate
            
        Returns:
            int: GPU memory pointer (device pointer)
        """
        ptr = CUdeviceptr()
        _check(self._lib.cuMemAlloc_v2(ctypes.byref(ptr), CUsize(size)))
        
        ptr_val = ptr.value
        self._memory_allocations[ptr_val] = size
        return ptr_val
    
    def mem_free(self, ptr: int):
        """Free GPU memory."""
        try:
            _check(self._lib.cuMemFree_v2(CUdeviceptr(ptr)))
            self._memory_allocations.pop(ptr, None)
        except Exception:
            pass
    
    def mem_get_info(self) -> Tuple[int, int]:
        """Get current memory usage: (free_bytes, total_bytes)."""
        free = CUsize()
        total = CUsize()
        _check(self._lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)))
        return free.value, total.value
    
    def memcpy_host_to_device(self, device_ptr: int, host_data: bytes):
        """Copy data from host (CPU) to device (GPU)."""
        size = len(host_data)
        c_buffer = ctypes.create_string_buffer(host_data, size)
        _check(self._lib.cuMemcpyHtoD_v2(
            CUdeviceptr(device_ptr),
            ctypes.cast(c_buffer, ctypes.c_void_p),
            CUsize(size)
        ))
    
    def memcpy_device_to_host(self, device_ptr: int, size: int) -> bytes:
        """Copy data from device (GPU) to host (CPU)."""
        host_buffer = ctypes.create_string_buffer(size)
        _check(self._lib.cuMemcpyDtoH_v2(
            ctypes.cast(host_buffer, ctypes.c_void_p),
            CUdeviceptr(device_ptr),
            CUsize(size)
        ))
        return host_buffer.raw
    
    def memcpy_device_to_device(self, dst_ptr: int, src_ptr: int, size: int):
        """Copy data between two GPU memory regions."""
        _check(self._lib.cuMemcpyDtoD_v2(
            CUdeviceptr(dst_ptr),
            CUdeviceptr(src_ptr),
            CUsize(size)
        ))
    
    # ============================================================
    #  Stream Management
    # ============================================================
    
    def stream_create(self, device_index: int) -> int:
        """Create a new CUDA stream on the specified device."""
        self._set_device(device_index)
        stream = CUstream()
        _check(self._lib.cuStreamCreate(ctypes.byref(stream), 0))
        stream_val = stream.value
        
        if device_index not in self._streams:
            self._streams[device_index] = []
        self._streams[device_index].append(stream)
        
        return stream_val
    
    def stream_synchronize(self, stream_ptr: int):
        """Wait for all operations in a stream to complete."""
        _check(self._lib.cuStreamSynchronize(CUstream(stream_ptr)))
    
    def stream_destroy(self, stream_ptr: int):
        """Destroy a CUDA stream."""
        try:
            _check(self._lib.cuStreamDestroy_v2(CUstream(stream_ptr)))
        except Exception:
            pass
    
    def synchronize(self):
        """Synchronize the current device context."""
        _check(self._lib.cuCtxSynchronize())
    
    # ============================================================
    #  Shutdown
    # ============================================================
    
    def shutdown(self):
        """Release all GPU resources and driver handles."""
        if not self._initialized:
            return
        
        for device_idx in list(self._streams.keys()):
            self._set_device(device_idx)
            
            # Destroy streams
            for stream in self._streams.get(device_idx, []):
                self.stream_destroy(stream.value if isinstance(stream, CUstream) else stream)
            
            # Free all memory
            for ptr in list(self._memory_allocations.keys()):
                self.mem_free(ptr)
            
            # Synchronize
            try:
                self._lib.cuCtxSynchronize()
            except Exception:
                pass
        
        # Destroy contexts
        for ctx in self._contexts.values():
            try:
                self._lib.cuCtxDestroy_v2(ctx)
            except Exception:
                pass
        
        self._contexts.clear()
        self._streams.clear()
        self._memory_allocations.clear()
        self._initialized = False
        self._current_context = None