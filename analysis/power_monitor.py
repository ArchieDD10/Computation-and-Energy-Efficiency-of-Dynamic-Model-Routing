"""
Power Monitoring Module - Measure GPU/CPU power consumption
Author: Archie Deguzman
Purpose: Track power usage for different routing strategies
"""

import time
import threading
import os
import sys
from typing import Optional, Dict, List
import warnings

# Try to find NVML DLL on Windows
if sys.platform == 'win32':
    nvml_paths = [
        r"C:\Windows\System32\nvml.dll",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
    ]
    for path in nvml_paths:
        if os.path.exists(path):
            os.environ['PATH'] = os.path.dirname(path) + os.pathsep + os.environ.get('PATH', '')
            break

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn("pynvml not installed. Install with: pip install nvidia-ml-py3")


class PowerMonitor:
    """Monitor GPU power consumption during inference"""

    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.thread = None
        self.gpu_handle = None

        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
                # Test if we can read power
                pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                print("[INFO] NVIDIA GPU power monitoring enabled")
            except Exception as e:
                print(f"[WARNING] Could not initialize GPU monitoring: {e}")
                self.gpu_handle = None
        else:
            print("[WARNING] pynvml not available. Install with: pip install nvidia-ml-py3")

    def _monitor_loop(self, interval_ms: float = 10):
        """Background thread that samples power every interval_ms"""
        interval_sec = interval_ms / 1000.0

        while self.monitoring:
            try:
                if self.gpu_handle:
                    # Power is returned in milliwatts, convert to watts
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    power_w = power_mw / 1000.0

                    self.samples.append({
                        'timestamp': time.time(),
                        'power_watts': power_w
                    })
            except Exception as e:
                print(f"[ERROR] Power sampling failed: {e}")
                break

            time.sleep(interval_sec)

    def start(self, interval_ms: float = 10):
        """Start monitoring power in background thread"""
        if not self.gpu_handle:
            print("[WARNING] GPU not available, cannot monitor power")
            return False

        self.monitoring = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval_ms,), daemon=True)
        self.thread.start()
        return True

    def stop(self) -> Dict:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if not self.samples:
            return {
                'num_samples': 0,
                'avg_power_watts': 0.0,
                'max_power_watts': 0.0,
                'min_power_watts': 0.0,
                'duration_sec': 0.0,
                'energy_joules': 0.0,
                'energy_wh': 0.0
            }

        powers = [s['power_watts'] for s in self.samples]
        times = [s['timestamp'] for s in self.samples]

        duration_sec = times[-1] - times[0] if len(times) > 1 else 0.0
        avg_power = sum(powers) / len(powers)

        # Energy = Power × Time (in joules/watt-seconds)
        energy_joules = avg_power * duration_sec
        energy_wh = energy_joules / 3600.0  # Convert to watt-hours

        return {
            'num_samples': len(self.samples),
            'avg_power_watts': avg_power,
            'max_power_watts': max(powers),
            'min_power_watts': min(powers),
            'duration_sec': duration_sec,
            'energy_joules': energy_joules,
            'energy_wh': energy_wh
        }

    def __del__(self):
        """Cleanup NVML on deletion"""
        if NVML_AVAILABLE and self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# Context manager for easy use
class PowerMeasurement:
    """Context manager for measuring power consumption"""

    def __init__(self, interval_ms: float = 10):
        self.monitor = PowerMonitor()
        self.interval_ms = interval_ms
        self.stats = None

    def __enter__(self):
        self.monitor.start(self.interval_ms)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stats = self.monitor.stop()
        return False

    def get_stats(self) -> Dict:
        """Get power statistics after measurement"""
        return self.stats if self.stats else {}


def test_power_monitoring():
    """Test power monitoring functionality"""
    print("=" * 60)
    print("POWER MONITORING TEST")
    print("=" * 60)

    monitor = PowerMonitor()

    if not monitor.gpu_handle:
        print("\n[ERROR] No GPU available for testing")
        return

    print("\nMonitoring GPU power for 3 seconds...")
    monitor.start(interval_ms=10)

    # Simulate some work
    import torch
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device='cuda')
        for _ in range(100):
            y = torch.matmul(x, x)

    time.sleep(3.0)
    stats = monitor.stop()

    print("\nPower Statistics:")
    print(f"  Samples collected: {stats['num_samples']}")
    print(f"  Duration: {stats['duration_sec']:.2f} seconds")
    print(f"  Average power: {stats['avg_power_watts']:.2f} W")
    print(f"  Max power: {stats['max_power_watts']:.2f} W")
    print(f"  Min power: {stats['min_power_watts']:.2f} W")
    print(f"  Total energy: {stats['energy_joules']:.2f} J ({stats['energy_wh']:.6f} Wh)")
    print("\n" + "=" * 60)


# Example usage with context manager
def example_with_context_manager():
    """Example using context manager"""
    print("\nExample with context manager:")

    with PowerMeasurement(interval_ms=10) as pm:
        # Your code here
        import torch
        if torch.cuda.is_available():
            x = torch.randn(500, 500, device='cuda')
            y = torch.matmul(x, x)
        time.sleep(1.0)

    stats = pm.get_stats()
    print(f"Energy consumed: {stats['energy_joules']:.2f} J")


if __name__ == "__main__":
    test_power_monitoring()
    example_with_context_manager()
