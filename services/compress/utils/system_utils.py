import psutil
import gc
import time
import os


def monitor_memory_usage(logging_enabled=True):
    """Monitor and log memory usage."""
    try:
        memory_info = psutil.virtual_memory()
        if logging_enabled:
            print(f"Memory usage: {memory_info.percent:.1f}% ({memory_info.used / (1024**3):.1f}GB / {memory_info.total / (1024**3):.1f}GB)")
        
        if memory_info.percent > 80:
            if logging_enabled:
                print("High memory usage detected, forcing garbage collection...")
            gc.collect()
            
        return memory_info.percent
    except Exception as e:
        if logging_enabled:
            print(f"Error monitoring memory: {e}")
        return 0


class ProgressTracker:
    def __init__(self, total_scenes, logging_enabled=True):
        self.total_scenes = total_scenes
        self.completed_scenes = 0
        self.failed_scenes = 0
        self.logging_enabled = logging_enabled
        self.start_time = time.time()
    
    def update(self, success=True):
        if success:
            self.completed_scenes += 1
        else:
            self.failed_scenes += 1
        
        if self.logging_enabled:
            elapsed = time.time() - self.start_time
            total_processed = self.completed_scenes + self.failed_scenes
            progress = (total_processed / self.total_scenes) * 100
            
            if total_processed > 0:
                avg_time_per_scene = elapsed / total_processed
                eta = avg_time_per_scene * (self.total_scenes - total_processed)
                print(f"Progress: {total_processed}/{self.total_scenes} ({progress:.1f}%) "
                      f"- ETA: {eta/60:.1f}m - Failed: {self.failed_scenes}")

