"""
RemoteCUDA - Transparent Remote GPU Access
==========================================
Use remote GPUs over local network without changing your PyTorch code.

Quick Start:
    # On GPU machine:
    remotecuda start
    
    # On your machine:
    python
    >>> import remotecuda
    >>> remotecuda.init()  # Auto-discovers and connects
    >>> model = MyModel().cuda()  # Uses remote GPU!
"""

__version__ = "2.0.0"
__author__ = "RemoteCUDA Team"

from .client.pool import GPUPool
from .client.hook import AutoCUDAHook

# Global singleton pool
_global_pool: GPUPool = None
_global_hook: AutoCUDAHook = None


def init(auto_discover: bool = True, server: str = None, port: int = 55555):
    """
    Initialize RemoteCUDA on the client side.
    
    Automatically discovers and connects to GPU servers on the network.
    Installs CUDA hooks so all PyTorch .cuda() calls use remote GPUs.
    
    Args:
        auto_discover (bool): Automatically scan network for GPU servers
        server (str, optional): Specific server to connect to
        port (int): Port number (default: 55555)
    
    Usage:
        import remotecuda
        remotecuda.init()  # Auto-discover and connect
        
        # Your existing code now uses remote GPUs!
        model = MyModel().cuda()
    """
    global _global_pool, _global_hook
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🚀 RemoteCUDA v1.0 - Initializing              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create GPU pool
    _global_pool = GPUPool()
    
    if auto_discover:
        # Find and connect to all available GPU servers
        _global_pool.auto_discover()
        _global_pool.connect_all_discovered()
    elif server:
        # Connect to specific server
        _global_pool.connect_to(server, port)
    
    # Check if we have any connections
    stats = _global_pool.get_stats()
    if stats['active_connections'] == 0:
        print("⚠️  No GPU connections established.")
        print("   Make sure remotecuda-server is running on your GPU machines.")
        print("   Or manually connect: remotecuda.connect('server_ip')")
        return False
    
    # Install CUDA hooks
    _global_hook = AutoCUDAHook(_global_pool)
    _global_hook.install()
    
    print(f"""
✅ RemoteCUDA initialized!
   Connected to {stats['active_connections']} server(s)
   Total GPUs: {stats['total_gpus']}
   Total Memory: {stats['total_memory_gb']:.1f} GB
   Free Memory: {stats['free_memory_gb']:.1f} GB

   Your .cuda() calls will now use remote GPUs automatically!
    """)
    
    return True


def connect(host: str, port: int = 55555):
    """
    Manually connect to a specific GPU server.
    
    Args:
        host (str): Server IP or hostname
        port (int): Server port
    """
    global _global_pool
    
    if _global_pool is None:
        _global_pool = GPUPool()
    
    _global_pool.connect_to(host, port)


def shutdown():
    """
    Gracefully shut down RemoteCUDA.
    
    Disconnects from all servers and restores original PyTorch behavior.
    """
    global _global_pool, _global_hook
    
    if _global_hook:
        _global_hook.uninstall()
    
    if _global_pool:
        _global_pool.disconnect_all()
    
    print("✅ RemoteCUDA shut down.")


def status():
    """
    Display current RemoteCUDA status.
    """
    global _global_pool, _global_hook
    
    if _global_pool is None:
        print("RemoteCUDA not initialized. Run remotecuda.init() first.")
        return
    
    stats = _global_pool.get_stats()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              RemoteCUDA Status                              ║
╠══════════════════════════════════════════════════════════════╣
║  Active Connections: {stats['active_connections']:<37}║
║  Total GPUs:         {stats['total_gpus']:<37}║
║  Total Memory:       {stats['total_memory_gb']:.1f} GB{' ' * 27}║
║  Free Memory:        {stats['free_memory_gb']:.1f} GB{' ' * 27}║
║  Utilization:        {stats['utilization']:.1%}{' ' * 32}║
║  Hooks Active:       {str(_global_hook is not None and _global_hook._installed):<37}║
╚══════════════════════════════════════════════════════════════╝
    """)