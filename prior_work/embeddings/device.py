"""
NPU Device Manager

Handles device selection and management for NPU-accelerated training.
Uses DirectML backend for Windows NPU support (AMD, Intel, NVIDIA, Qualcomm).

This module provides:
- Automatic NPU detection
- Fallback to CPU if NPU unavailable
- Device context management
- Tensor conversions between NumPy and PyTorch
"""

import numpy as np
import torch
from typing import Optional, Union
import warnings


class DeviceManager:
    """
    Manages device selection for training (NPU, GPU, or CPU).

    Attributes:
        device: PyTorch device object
        device_type: String indicating device type ('npu', 'cuda', 'cpu')
        is_available: Whether hardware acceleration is available
    """

    def __init__(self, prefer_npu: bool = True, verbose: bool = True):
        """
        Initialize device manager.

        Args:
            prefer_npu: Whether to prefer NPU over other devices
            verbose: Print device information
        """
        self.device = None
        self.device_type = None
        self.is_available = False

        self._detect_device(prefer_npu, verbose)

    def _optimize_cpu(self):
        """Enable CPU optimizations for training."""
        # Use all available threads
        torch.set_num_threads(torch.get_num_threads())

        # Enable MKL-DNN if available
        if torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = True

        # Enable OpenMP if available
        if torch.backends.openmp.is_available():
            # Already enabled by default
            pass

    def _detect_device(self, prefer_npu: bool, verbose: bool):
        """Detect and select best available device."""

        # Try DirectML (NPU) first if preferred
        if prefer_npu:
            try:
                import torch_directml
                self.device = torch_directml.device()
                self.device_type = 'directml'
                self.is_available = True
                if verbose:
                    print(f"[NPU] DirectML device detected and enabled")
                    print(f"[NPU] Device: {self.device}")
                return
            except ImportError:
                if verbose:
                    print("[INFO] torch-directml not available")
                    print("[INFO] Note: DirectML support may not be available for Python 3.14+")
                    print("[INFO] Falling back to CUDA/CPU")
            except Exception as e:
                if verbose:
                    warnings.warn(f"DirectML initialization failed: {e}")

        # Fallback to CUDA if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_type = 'cuda'
            self.is_available = True
            if verbose:
                print(f"[GPU] CUDA device detected: {torch.cuda.get_device_name(0)}")
            return

        # Fallback to CPU with optimizations
        self.device = torch.device('cpu')
        self.device_type = 'cpu'
        self.is_available = False

        # Enable CPU optimizations
        self._optimize_cpu()

        if verbose:
            num_threads = torch.get_num_threads()
            print(f"[CPU] No hardware acceleration available, using CPU")
            print(f"[CPU] Enabled {num_threads} threads for parallel computation")
            print(f"[CPU] Training will be slower. Consider installing torch-directml for GPU support.")

    def to_tensor(self, array: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Convert NumPy array to PyTorch tensor on the managed device.

        Args:
            array: NumPy array to convert
            dtype: Target dtype (default: torch.float32)

        Returns:
            PyTorch tensor on device
        """
        if dtype is None:
            dtype = torch.float32

        tensor = torch.from_numpy(array).to(dtype)
        return tensor.to(self.device)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to NumPy array.

        Args:
            tensor: PyTorch tensor to convert

        Returns:
            NumPy array
        """
        return tensor.detach().cpu().numpy()

    def empty(self, *size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create empty tensor on device."""
        return torch.empty(*size, dtype=dtype, device=self.device)

    def zeros(self, *size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create zero tensor on device."""
        return torch.zeros(*size, dtype=dtype, device=self.device)

    def ones(self, *size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create ones tensor on device."""
        return torch.ones(*size, dtype=dtype, device=self.device)

    def randn(self, *size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create random normal tensor on device."""
        return torch.randn(*size, dtype=dtype, device=self.device)

    def get_device_stats(self) -> dict:
        """
        Get device statistics (memory usage, etc.).

        Returns:
            Dict with device stats
        """
        stats = {
            'device_type': self.device_type,
            'device': str(self.device),
            'hardware_acceleration': self.is_available,
        }

        if self.device_type == 'cuda':
            stats['memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            stats['memory_reserved'] = torch.cuda.memory_reserved() / 1024**2
            stats['device_name'] = torch.cuda.get_device_name(0)

        return stats

    def synchronize(self):
        """Synchronize device (wait for all operations to complete)."""
        if self.device_type == 'cuda':
            torch.cuda.synchronize()
        # DirectML doesn't require explicit synchronization

    def empty_cache(self):
        """Empty device cache to free memory."""
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()

    def __repr__(self):
        return f"DeviceManager(device={self.device}, type={self.device_type}, " \
               f"hardware_acceleration={self.is_available})"


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(prefer_npu: bool = True, verbose: bool = False) -> DeviceManager:
    """
    Get or create global device manager instance.

    Args:
        prefer_npu: Whether to prefer NPU over other devices
        verbose: Print device information

    Returns:
        DeviceManager instance
    """
    global _global_device_manager

    if _global_device_manager is None:
        _global_device_manager = DeviceManager(prefer_npu=prefer_npu, verbose=verbose)

    return _global_device_manager


def reset_device_manager():
    """Reset global device manager (useful for testing)."""
    global _global_device_manager
    _global_device_manager = None


# Convenience functions
def get_device(prefer_npu: bool = True) -> torch.device:
    """Get the current device."""
    return get_device_manager(prefer_npu=prefer_npu).device


def to_device(array: Union[np.ndarray, torch.Tensor],
              prefer_npu: bool = True) -> torch.Tensor:
    """Convert array/tensor to device."""
    dm = get_device_manager(prefer_npu=prefer_npu)

    if isinstance(array, np.ndarray):
        return dm.to_tensor(array)
    elif isinstance(array, torch.Tensor):
        return array.to(dm.device)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(array)}")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to NumPy array."""
    return get_device_manager().to_numpy(tensor)
