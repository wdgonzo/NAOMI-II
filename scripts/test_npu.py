"""
Test NPU Setup

Quick script to verify that NPU (DirectML) is working correctly.
"""

import sys
sys.path.insert(0, '.')

from src.embeddings.device import DeviceManager, get_device_manager
import torch
import time


def test_npu_detection():
    """Test that NPU is detected."""
    print("="*60)
    print("NPU DETECTION TEST")
    print("="*60)

    dm = DeviceManager(prefer_npu=True, verbose=True)

    print(f"\nDevice Manager:")
    print(f"  Device: {dm.device}")
    print(f"  Device Type: {dm.device_type}")
    print(f"  Hardware Acceleration: {dm.is_available}")

    stats = dm.get_device_stats()
    print(f"\nDevice Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return dm


def test_npu_performance(dm: DeviceManager):
    """Test NPU performance with matrix operations."""
    print("\n" + "="*60)
    print("NPU PERFORMANCE TEST")
    print("="*60)

    sizes = [100, 500, 1000, 2000]

    for size in sizes:
        # Create random matrices
        a = dm.randn(size, size)
        b = dm.randn(size, size)

        # Warm-up
        _ = torch.matmul(a, b)
        dm.synchronize()

        # Time matrix multiplication
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        dm.synchronize()
        elapsed = time.time() - start

        print(f"\nMatrix size: {size}x{size}")
        print(f"  10 iterations: {elapsed:.4f}s")
        print(f"  Per operation: {elapsed/10*1000:.2f}ms")


def test_npu_memory(dm: DeviceManager):
    """Test NPU memory management."""
    print("\n" + "="*60)
    print("NPU MEMORY TEST")
    print("="*60)

    print(f"\nAllocating large tensors...")

    tensors = []
    for i in range(10):
        size = 1000
        t = dm.randn(size, size)
        tensors.append(t)
        print(f"  Allocated tensor {i+1}: {size}x{size} = {size*size*4/1024/1024:.2f}MB")

    print(f"\nTotal allocated: {len(tensors)} tensors")

    # Free memory
    del tensors
    dm.empty_cache()
    print(f"Memory freed")


def main():
    """Run all NPU tests."""
    print("\n" + "="*60)
    print("NAOMI-II NPU SETUP VERIFICATION")
    print("="*60 + "\n")

    try:
        # Test 1: Detection
        dm = test_npu_detection()

        if not dm.is_available:
            print("\n[WARNING] No hardware acceleration available!")
            print("Install torch-directml for NPU support:")
            print("  pip install torch-directml")
            return False

        # Test 2: Performance
        test_npu_performance(dm)

        # Test 3: Memory
        test_npu_memory(dm)

        print("\n" + "="*60)
        print("ALL TESTS PASSED - NPU IS READY!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n[ERROR] NPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
