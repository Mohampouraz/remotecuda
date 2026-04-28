"""
Automatic CUDA Hook Module
==========================
Transparently intercepts PyTorch CUDA calls and redirects to remote GPUs.
This is the magic that makes existing code work without changes.
"""

import torch
import torch.nn as nn
from typing import Optional

from .pool import GPUPool
from .connection import GPUConnection


class AutoCUDAHook:
    """
    Automatic CUDA hook that intercepts PyTorch CUDA operations.
    
    When installed, all .cuda(), .to('cuda'), and related calls
    are automatically redirected to available remote GPUs.
    
    Usage:
        hook = AutoCUDAHook(gpu_pool)
        hook.install()     # Activate interception
        
        # Your existing code - unchanged!
        model = MyModel().cuda()
        x = torch.randn(10, 10).cuda()
        
        hook.uninstall()   # Restore original behavior
    """
    
    def __init__(self, gpu_pool: GPUPool):
        """
        Initialize the hook system.
        
        Args:
            gpu_pool (GPUPool): Connected GPU pool for execution
        """
        self.gpu_pool = gpu_pool
        self._installed = False
        
        # Save originals for restoration
        self._originals = {}
        
        # Track which GPU each object lives on
        self._object_gpu: dict = {}
        
        # Tensor proxy registry
        self._tensor_proxies: dict = {}
        self._model_proxies: dict = {}
    
    def install(self):
        """
        Install hooks on PyTorch CUDA methods.
        """
        if self._installed:
            return
        
        # Save original methods
        self._originals = {
            'Tensor.to': torch.Tensor.to,
            'Tensor.cuda': torch.Tensor.cuda,
            'Tensor.cpu': torch.Tensor.cpu,
            'Module.to': nn.Module.to,
            'Module.cuda': nn.Module.cuda,
            'Module.cpu': nn.Module.cpu,
            'Module.forward': nn.Module.__call__,
        }
        
        # Install hooks
        torch.Tensor.to = self._hooked_tensor_to
        torch.Tensor.cuda = self._hooked_tensor_cuda
        torch.Tensor.cpu = self._hooked_tensor_cpu
        nn.Module.to = self._hooked_module_to
        nn.Module.cuda = self._hooked_module_cuda
        nn.Module.cpu = self._hooked_module_cpu
        nn.Module.__call__ = self._hooked_module_call
        
        self._installed = True
        print("✅ AutoCUDA hooks installed. CUDA operations will use remote GPUs.")
    
    def uninstall(self):
        """
        Restore original PyTorch methods.
        """
        if not self._installed:
            return
        
        torch.Tensor.to = self._originals['Tensor.to']
        torch.Tensor.cuda = self._originals['Tensor.cuda']
        torch.Tensor.cpu = self._originals['Tensor.cpu']
        nn.Module.to = self._originals['Module.to']
        nn.Module.cuda = self._originals['Module.cuda']
        nn.Module.cpu = self._originals['Module.cpu']
        nn.Module.__call__ = self._originals['Module.forward']
        
        self._installed = False
        print("✅ AutoCUDA hooks removed. Original PyTorch behavior restored.")
    
    def _get_best_gpu(self) -> GPUConnection:
        """
        Get the best available GPU connection.
        """
        gpu = self.gpu_pool.get_best_gpu()
        if gpu is None:
            raise RuntimeError("No GPU connections available!")
        return gpu
    
    def _hooked_tensor_to(self, tensor, *args, **kwargs):
        """
        Intercepted tensor.to() method.
        """
        device = args[0] if args else kwargs.get('device', 'cpu')
        device_str = str(device)
        
        if 'cuda' in device_str:
            # Redirect to remote GPU
            return self._send_tensor_to_gpu(tensor)
        elif 'cpu' in device_str:
            # Move to CPU (local)
            return self._originals['Tensor.to'](tensor, 'cpu')
        else:
            # Other device types - local
            return self._originals['Tensor.to'](tensor, *args, **kwargs)
    
    def _hooked_tensor_cuda(self, tensor, *args, **kwargs):
        """
        Intercepted tensor.cuda() method.
        """
        return self._send_tensor_to_gpu(tensor)
    
    def _hooked_tensor_cpu(self, tensor, *args, **kwargs):
        """
        Intercepted tensor.cpu() method.
        """
        # Check if this is a proxy tensor
        if id(tensor) in self._tensor_proxies:
            return self._tensor_proxies[id(tensor)]['cpu_copy']
        return self._originals['Tensor.cpu'](tensor)
    
    def _hooked_module_to(self, module, *args, **kwargs):
        """
        Intercepted module.to() method.
        """
        device = args[0] if args else kwargs.get('device', 'cpu')
        device_str = str(device)
        
        if 'cuda' in device_str:
            return self._send_module_to_gpu(module)
        elif 'cpu' in device_str:
            return self._originals['Module.to'](module, 'cpu')
        else:
            return self._originals['Module.to'](module, *args, **kwargs)
    
    def _hooked_module_cuda(self, module, *args, **kwargs):
        """
        Intercepted module.cuda() method.
        """
        return self._send_module_to_gpu(module)
    
    def _hooked_module_cpu(self, module, *args, **kwargs):
        """
        Intercepted module.cpu() method.
        """
        if id(module) in self._model_proxies:
            return self._model_proxies[id(module)]['cpu_copy']
        return self._originals['Module.cpu'](module)
    
    def _hooked_module_call(self, module, *args, **kwargs):
        """
        Intercepted module forward call.
        """
        # Check if this is a remote model
        if id(module) in self._model_proxies:
            return self._remote_forward(module, *args, **kwargs)
        else:
            return self._originals['Module.forward'](module, *args, **kwargs)
    
    def _send_tensor_to_gpu(self, tensor):
        """
        Transfer a tensor to remote GPU and return a proxy.
        """
        gpu = self._get_best_gpu()
        
        try:
            tensor_id = gpu.allocate_tensor(tensor)
        except Exception as e:
            print(f"⚠️  Failed to allocate on remote GPU: {e}")
            return tensor  # Fall back to original tensor
        
        # Create proxy (lazy evaluation for GPU tensors)
        proxy_info = {
            'tensor_id': tensor_id,
            'gpu': gpu,
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'cpu_copy': tensor.cpu()  # Keep CPU copy for fallback
        }
        
        self._tensor_proxies[id(tensor)] = proxy_info
        
        # Mark tensor as being on remote GPU
        # This is a bit of a hack but works for most cases
        tensor._remotecuda_gpu = True
        tensor._remotecuda_tensor_id = tensor_id
        
        return tensor
    
    def _send_module_to_gpu(self, module):
        """
        Register a module on remote GPU.
        """
        gpu = self._get_best_gpu()
        
        # For now, keep a CPU copy and execute remotely
        cpu_copy = self._originals['Module.to'](module, 'cpu')
        
        model_info = {
            'gpu': gpu,
            'cpu_copy': cpu_copy,
            'model_id': id(module)
        }
        
        self._model_proxies[id(module)] = model_info
        
        # Register module parameters on GPU
        for name, param in module.named_parameters():
            gpu.allocate_tensor(param.data)
        
        return module
    
    def _remote_forward(self, module, *args, **kwargs):
        """
        Execute forward pass on remote GPU.
        """
        model_info = self._model_proxies[id(module)]
        gpu = model_info['gpu']
        model_id = model_info['model_id']
        
        # Collect input tensor IDs
        input_ids = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and hasattr(arg, '_remotecuda_tensor_id'):
                input_ids.append(arg._remotecuda_tensor_id)
            elif isinstance(arg, torch.Tensor):
                # Tensor not on remote GPU yet - send it
                tid = gpu.allocate_tensor(arg)
                input_ids.append(tid)
            else:
                raise ValueError(f"Unsupported argument type: {type(arg)}")
        
        # Execute forward pass on remote GPU
        result = gpu.forward(model_id, *input_ids)
        
        return result