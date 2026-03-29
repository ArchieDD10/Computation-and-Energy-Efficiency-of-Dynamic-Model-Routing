"""
Alternative Power Monitoring Solutions
Author: Archie Deguzman
Purpose: Multiple approaches to measure power consumption when NVML isn't available
"""

import subprocess
import time
import json
import re
from typing import Dict, Optional


class AlternativePowerMonitor:
    """Power monitoring using multiple fallback methods"""

    def __init__(self):
        self.method = self._detect_best_method()
        print(f"[INFO] Using power monitoring method: {self.method}")

    def _detect_best_method(self) -> str:
        """Detect the best available power monitoring method"""

        # Method 1: Try nvidia-smi (most reliable)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return "nvidia-smi"
        except Exception:
            pass

        # Method 2: Try nvidia-ml-py with error handling
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(handle)
            pynvml.nvmlShutdown()
            return "pynvml"
        except Exception:
            pass

        # Method 3: Fallback to estimation
        return "estimation"

    def get_gpu_power(self) -> Optional[float]:
        """Get current GPU power consumption in watts"""

        if self.method == "nvidia-smi":
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    power_str = result.stdout.strip()
                    if power_str and power_str != "N/A":
                        return float(power_str)
            except Exception as e:
                print(f"[WARNING] nvidia-smi error: {e}")

        elif self.method == "pynvml":
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                pynvml.nvmlShutdown()
                return power_mw / 1000.0  # Convert to watts
            except Exception as e:
                print(f"[WARNING] pynvml error: {e}")

        # Fallback: Return None (will use estimation)
        return None

    def estimate_power_from_utilization(self, duration_sec: float) -> Dict:
        """Estimate power based on GPU utilization and typical power draw"""
        try:
            # Get GPU utilization
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
                gpu_util = float(gpu_util.strip('%'))
                mem_used = float(mem_used)
                mem_total = float(mem_total)
                mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0

                # Estimate power based on utilization (rough approximation)
                base_power = 20  # Idle power (watts)
                max_power = 150  # Typical gaming GPU max power
                estimated_power = base_power + (max_power - base_power) * (gpu_util / 100.0)

                return {
                    'method': 'estimation',
                    'estimated_power_watts': estimated_power,
                    'gpu_utilization_pct': gpu_util,
                    'memory_utilization_pct': mem_util,
                    'energy_estimate_joules': estimated_power * duration_sec,
                    'confidence': 'low'
                }
        except Exception as e:
            print(f"[WARNING] Utilization estimation failed: {e}")

        # Ultimate fallback
        return {
            'method': 'fallback',
            'estimated_power_watts': 75.0,  # Conservative estimate
            'energy_estimate_joules': 75.0 * duration_sec,
            'confidence': 'very_low'
        }


class SimpleInferenceProfiler:
    """Simple profiler for measuring inference time and estimated power"""

    def __init__(self):
        self.power_monitor = AlternativePowerMonitor()
        self.start_time = None
        self.end_time = None
        self.power_samples = []

    def start_profiling(self):
        """Start profiling session"""
        self.start_time = time.time()
        self.power_samples = []

        # Take initial power reading
        power = self.power_monitor.get_gpu_power()
        if power is not None:
            self.power_samples.append(power)

    def end_profiling(self) -> Dict:
        """End profiling and return metrics"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0

        # Take final power reading
        power = self.power_monitor.get_gpu_power()
        if power is not None:
            self.power_samples.append(power)

        if self.power_samples:
            avg_power = sum(self.power_samples) / len(self.power_samples)
            energy_joules = avg_power * duration

            return {
                'duration_sec': duration,
                'power_method': self.power_monitor.method,
                'avg_power_watts': avg_power,
                'energy_joules': energy_joules,
                'energy_wh': energy_joules / 3600.0,
                'power_samples': len(self.power_samples),
                'confidence': 'medium'
            }
        else:
            # Use estimation
            estimation = self.power_monitor.estimate_power_from_utilization(duration)
            estimation['duration_sec'] = duration
            return estimation


def profile_model_inference(model_pipe, test_texts, model_name: str) -> Dict:
    """Profile a model's inference time and power consumption"""

    profiler = SimpleInferenceProfiler()

    print(f"  [INFO] Profiling {model_name} model...")
    profiler.start_profiling()

    # Run inference on test texts
    latencies = []
    for text in test_texts:
        start = time.time()
        _ = model_pipe(text)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)

    power_stats = profiler.end_profiling()

    # Combine timing and power stats
    inference_stats = {
        'model': model_name,
        'num_inferences': len(test_texts),
        'avg_latency_ms': sum(latencies) / len(latencies),
        'total_latency_ms': sum(latencies),
        'min_latency_ms': min(latencies),
        'max_latency_ms': max(latencies),
    }

    # Add power stats
    inference_stats.update(power_stats)

    return inference_stats


def quick_power_test():
    """Quick test of power monitoring capabilities"""
    print("=" * 60)
    print("POWER MONITORING CAPABILITY TEST")
    print("=" * 60)

    monitor = AlternativePowerMonitor()

    print(f"\nMethod detected: {monitor.method}")

    # Test power reading
    power = monitor.get_gpu_power()
    if power is not None:
        print(f"Current GPU power: {power:.1f}W")
    else:
        print("Direct power reading not available, using estimation")

        # Test estimation
        estimation = monitor.estimate_power_from_utilization(1.0)
        print(f"Estimated power: {estimation['estimated_power_watts']:.1f}W (confidence: {estimation['confidence']})")
        print(f"GPU utilization: {estimation.get('gpu_utilization_pct', 'N/A')}%")

    print("=" * 60)


if __name__ == "__main__":
    quick_power_test()