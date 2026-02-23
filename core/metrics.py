"""
Metrics classes for WIZ Intelligence Pipeline.
Structured metric collection and storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
import time


class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, name: str):
        """
        Initialize base metric.
        
        Args:
            name: Metric name
        """
        self.name = name
    
    @abstractmethod
    def get_value(self) -> Any:
        """Get current metric value."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric to initial state."""
        pass


class CounterMetric(BaseMetric):
    """
    Counter metric for counting events.
    
    Examples: blink_count, breath_count, error_count
    """
    
    def __init__(self, name: str):
        """
        Initialize counter metric.
        
        Args:
            name: Counter name
        """
        super().__init__(name)
        self._count: int = 0
    
    def increment(self, value: int = 1) -> None:
        """
        Increment counter.
        
        Args:
            value: Amount to increment (default 1)
        """
        self._count += value
    
    def get_value(self) -> int:
        """Get current count."""
        return self._count
    
    def reset(self) -> None:
        """Reset counter to zero."""
        self._count = 0


class TimerMetric(BaseMetric):
    """
    Timer metric for measuring durations.
    
    Examples: task_duration, model_load_time, inference_time
    """
    
    def __init__(self, name: str):
        """
        Initialize timer metric.
        
        Args:
            name: Timer name
        """
        super().__init__(name)
        self._start_time: float = 0.0
        self._total_time: float = 0.0
        self._is_running: bool = False
    
    def start(self) -> None:
        """Start timing."""
        if self._is_running:
            raise RuntimeError(f"Timer {self.name} already running")
        
        self._start_time = time.time()
        self._is_running = True
    
    def stop(self) -> float:
        """
        Stop timing and return duration.
        
        Returns:
            Duration in seconds
        """
        if not self._is_running:
            raise RuntimeError(f"Timer {self.name} not running")
        
        duration = time.time() - self._start_time
        self._total_time += duration
        self._is_running = False
        return duration
    
    def get_value(self) -> float:
        """Get total accumulated time."""
        return self._total_time
    
    def get_current_duration(self) -> float:
        """
        Get current duration if timer is running.
        
        Returns:
            Current duration in seconds, or 0 if not running
        """
        if self._is_running:
            return time.time() - self._start_time
        return 0.0
    
    def reset(self) -> None:
        """Reset timer."""
        self._start_time = 0.0
        self._total_time = 0.0
        self._is_running = False


class GaugeMetric(BaseMetric):
    """
    Gauge metric for point-in-time values.
    
    Examples: memory_usage, cpu_percent, current_fps
    """
    
    def __init__(self, name: str, initial_value: float = 0.0):
        """
        Initialize gauge metric.
        
        Args:
            name: Gauge name
            initial_value: Initial gauge value
        """
        super().__init__(name)
        self._value: float = initial_value
        self._history: List[float] = []
    
    def set_value(self, value: float) -> None:
        """
        Set gauge value.
        
        Args:
            value: New gauge value
        """
        self._value = value
        self._history.append(value)
    
    def get_value(self) -> float:
        """Get current gauge value."""
        return self._value
    
    def get_history(self) -> List[float]:
        """Get value history."""
        return self._history.copy()
    
    def get_max(self) -> float:
        """Get maximum recorded value."""
        return max(self._history) if self._history else 0.0
    
    def get_average(self) -> float:
        """Get average of recorded values."""
        return sum(self._history) / len(self._history) if self._history else 0.0
    
    def reset(self) -> None:
        """Reset gauge."""
        self._value = 0.0
        self._history = []


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""
    
    task_name: str
    duration: float
    success: bool
    error_message: str = ""
    events_detected: int = 0
    model_load_time: float = 0.0
    inference_time: float = 0.0


@dataclass
class PipelineMetrics:
    """Overall pipeline execution metrics."""
    
    total_duration: float
    task_metrics: List[TaskMetrics]
    total_events_detected: int
    failure_count: int
    success_rate: float
    
    def add_task_metrics(self, task_metrics: TaskMetrics) -> None:
        """Add task metrics to pipeline metrics."""
        self.task_metrics.append(task_metrics)
        self.total_events_detected += task_metrics.events_detected
        
        if not task_metrics.success:
            self.failure_count += 1
        
        # Update success rate
        total_tasks = len(self.task_metrics)
        successful_tasks = total_tasks - self.failure_count
        self.success_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0


class MetricsCollector:
    """
    Centralized metrics collection and management.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, CounterMetric] = {}
        self._timers: Dict[str, TimerMetric] = {}
        self._gauges: Dict[str, GaugeMetric] = {}
    
    def get_counter(self, name: str) -> CounterMetric:
        """
        Get or create counter metric.
        
        Args:
            name: Counter name
            
        Returns:
            CounterMetric instance
        """
        if name not in self._counters:
            self._counters[name] = CounterMetric(name)
        return self._counters[name]
    
    def get_timer(self, name: str) -> TimerMetric:
        """
        Get or create timer metric.
        
        Args:
            name: Timer name
            
        Returns:
            TimerMetric instance
        """
        if name not in self._timers:
            self._timers[name] = TimerMetric(name)
        return self._timers[name]
    
    def get_gauge(self, name: str) -> GaugeMetric:
        """
        Get or create gauge metric.
        
        Args:
            name: Gauge name
            
        Returns:
            GaugeMetric instance
        """
        if name not in self._gauges:
            self._gauges[name] = GaugeMetric(name)
        return self._gauges[name]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all current metric values.
        
        Returns:
            Dictionary of all metrics and their values
        """
        metrics = {}
        
        # Add counters
        for name, counter in self._counters.items():
            metrics[f"counter.{name}"] = counter.get_value()
        
        # Add timers
        for name, timer in self._timers.items():
            metrics[f"timer.{name}"] = timer.get_value()
        
        # Add gauges
        for name, gauge in self._gauges.items():
            metrics[f"gauge.{name}"] = gauge.get_value()
            metrics[f"gauge.{name}.max"] = gauge.get_max()
            metrics[f"gauge.{name}.avg"] = gauge.get_average()
        
        return metrics
    
    def reset_all(self) -> None:
        """Reset all metrics."""
        for counter in self._counters.values():
            counter.reset()
        
        for timer in self._timers.values():
            timer.reset()
        
        for gauge in self._gauges.values():
            gauge.reset()