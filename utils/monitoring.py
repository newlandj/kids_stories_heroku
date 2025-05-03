import time
import logging
import contextlib
from typing import Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger("kids-story-lambda")

class MetricsCollector:
    """Utility for collecting performance metrics during story generation"""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.metrics = defaultdict(float)
        self.counters = defaultdict(int)
        self.start_time = time.time()
    
    @contextlib.asynccontextmanager
    async def timer_async(self, operation_name: str):
        """Asynchronous context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.metrics[f"{operation_name}_time"] = round(elapsed, 3)
            logger.debug(f"[{self.request_id}] {operation_name} completed in {elapsed:.3f}s")
    
    def increment(self, counter_name: str, value: int = 1):
        """Increment a counter metric"""
        self.counters[counter_name] += value
    
    def record_value(self, metric_name: str, value: float):
        """Record a specific metric value"""
        self.metrics[metric_name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        result = dict(self.metrics)
        
        # Add counters to metrics
        for counter_name, value in self.counters.items():
            result[counter_name] = value
            
        # Add total execution time
        result["total_elapsed"] = round(time.time() - self.start_time, 3)
        
        return result
