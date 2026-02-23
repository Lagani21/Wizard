"""
Pipeline Monitor for WIZ Intelligence Pipeline.
Tracks execution metrics, performance, and failures.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .logger import Logger
from .metrics import (
    MetricsCollector, 
    TaskMetrics, 
    PipelineMetrics, 
    CounterMetric, 
    TimerMetric,
    GaugeMetric
)


@dataclass
class FailureRecord:
    """Record of a task failure."""
    
    task_name: str
    error_message: str
    timestamp: float
    duration_before_failure: float = 0.0


class PipelineMonitor:
    """
    Pipeline execution monitor.
    
    Tracks task performance, detections, failures, and provides
    comprehensive execution summaries.
    """
    
    def __init__(self, logger: Logger):
        """
        Initialize pipeline monitor.
        
        Args:
            logger: Logger instance for structured logging
        """
        self._logger = logger
        self._metrics = MetricsCollector()
        
        # Pipeline state
        self._pipeline_start_time: float = 0.0
        self._pipeline_end_time: float = 0.0
        self._current_task: Optional[str] = None
        self._task_start_times: Dict[str, float] = {}
        
        # Execution tracking
        self._task_metrics: List[TaskMetrics] = []
        self._failures: List[FailureRecord] = []
        
        # Detection counters
        self._detection_counts: Dict[str, int] = {
            'blink_events': 0,
            'breath_events': 0,
            'scenes': 0,
            'tone_segments': 0,
            'summaries': 0,
            'speakers': 0,
            'transcript_words': 0
        }
        
        # Performance timers
        self._task_timers: Dict[str, TimerMetric] = {}
        
        self._logger.log_info("Pipeline monitor initialized")
    
    def start_pipeline(self) -> None:
        """Mark start of pipeline execution."""
        self._pipeline_start_time = time.time()
        self._pipeline_timer = self._metrics.get_timer("pipeline.total_duration")
        self._pipeline_timer.start()
        
        self._logger.log_info("Pipeline execution started")
    
    def end_pipeline(self) -> None:
        """Mark end of pipeline execution."""
        self._pipeline_end_time = time.time()
        
        if hasattr(self, '_pipeline_timer'):
            self._pipeline_timer.stop()
        
        # Log final metrics
        total_duration = self._pipeline_end_time - self._pipeline_start_time
        self._logger.log_metric("pipeline.total_duration", total_duration, "s")
        
        self._logger.log_info("Pipeline execution completed")
    
    def start_task(self, task_name: str) -> None:
        """
        Mark start of task execution.
        
        Args:
            task_name: Name of the task starting
        """
        self._current_task = task_name
        self._task_start_times[task_name] = time.time()
        
        # Start task timer
        timer = self._metrics.get_timer(f"task.{task_name}.duration")
        timer.start()
        
        self._logger.log_task_start(task_name)
    
    def end_task(self, task_name: str, success: bool = True, error_message: str = "") -> float:
        """
        Mark end of task execution.
        
        Args:
            task_name: Name of the completed task
            success: Whether task completed successfully
            error_message: Error message if task failed
            
        Returns:
            Task duration in seconds
        """
        if task_name not in self._task_start_times:
            self._logger.log_warning(f"Task {task_name} end called without start")
            return 0.0
        
        # Calculate duration
        duration = time.time() - self._task_start_times[task_name]
        
        # Stop timer
        timer = self._metrics.get_timer(f"task.{task_name}.duration")
        timer.stop()
        
        # Create task metrics
        task_metrics = TaskMetrics(
            task_name=task_name,
            duration=duration,
            success=success,
            error_message=error_message
        )
        self._task_metrics.append(task_metrics)
        
        # Record failure if applicable
        if not success:
            failure = FailureRecord(
                task_name=task_name,
                error_message=error_message,
                timestamp=time.time(),
                duration_before_failure=duration
            )
            self._failures.append(failure)
            
            # Update failure counter
            failure_counter = self._metrics.get_counter("pipeline.failures")
            failure_counter.increment()
            
            self._logger.log_error(f"Task failed: {error_message}", task_name)
        
        self._logger.log_task_end(task_name, duration)
        self._logger.log_metric(f"{task_name}.duration", duration, "s")
        
        # Clear current task
        if self._current_task == task_name:
            self._current_task = None
        
        return duration
    
    def record_task_duration(self, task_name: str, duration: float) -> None:
        """
        Record task duration (for external timing).
        
        Args:
            task_name: Task name
            duration: Duration in seconds
        """
        self._logger.log_metric(f"{task_name}.duration", duration, "s")
        
        # Update timer metric
        timer = self._metrics.get_timer(f"task.{task_name}.duration")
        timer._total_time += duration  # Direct update for external timing
    
    def record_failure(self, task_name: str, error_message: str) -> None:
        """
        Record task failure.
        
        Args:
            task_name: Name of failed task
            error_message: Failure description
        """
        failure = FailureRecord(
            task_name=task_name,
            error_message=error_message,
            timestamp=time.time()
        )
        self._failures.append(failure)
        
        # Update failure counter
        failure_counter = self._metrics.get_counter("pipeline.failures")
        failure_counter.increment()
        
        self._logger.log_error(f"Task failure recorded: {error_message}", task_name)
    
    def record_model_load_time(self, model_name: str, load_time: float) -> None:
        """
        Record model loading time.
        
        Args:
            model_name: Name of the model
            load_time: Load time in seconds
        """
        self._logger.log_metric(f"{model_name}.load_time", load_time, "s")
        
        # Update model load timer
        timer = self._metrics.get_timer(f"model.{model_name}.load_time")
        timer._total_time = load_time
    
    def record_inference_time(self, model_name: str, inference_time: float) -> None:
        """
        Record model inference time.
        
        Args:
            model_name: Name of the model
            inference_time: Inference time in seconds
        """
        self._logger.log_metric(f"{model_name}.inference_time", inference_time, "s")
        
        # Update inference timer
        timer = self._metrics.get_timer(f"model.{model_name}.inference_time")
        timer._total_time += inference_time
    
    def record_detection_count(self, detection_type: str, count: int) -> None:
        """
        Record number of detected events.
        
        Args:
            detection_type: Type of detection (e.g., 'blink_events', 'breath_events')
            count: Number of detected events
        """
        self._detection_counts[detection_type] = count
        
        # Update counter metric
        counter = self._metrics.get_counter(f"detection.{detection_type}")
        counter._count = count  # Set absolute value
        
        self._logger.log_metric(f"detection.{detection_type}", count)
    
    def record_metric(self, name: str, value: float) -> None:
        """
        Record a general metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self._logger.log_metric(name, value)
        
        # Update gauge metric
        gauge = self._metrics.get_gauge(name)
        gauge.set_value(value)
    
    def record_memory_usage(self, usage_mb: float) -> None:
        """
        Record memory usage.
        
        Args:
            usage_mb: Memory usage in MB
        """
        gauge = self._metrics.get_gauge("system.memory_usage_mb")
        gauge.set_value(usage_mb)
        
        self._logger.log_metric("system.memory_usage", usage_mb, "MB")
    
    def get_pipeline_metrics(self) -> PipelineMetrics:
        """
        Get comprehensive pipeline metrics.
        
        Returns:
            PipelineMetrics with complete execution data
        """
        total_duration = (
            self._pipeline_end_time - self._pipeline_start_time
            if self._pipeline_end_time > 0 else 0.0
        )
        
        total_events = sum(self._detection_counts.values())
        failure_count = len(self._failures)
        total_tasks = len(self._task_metrics)
        success_rate = (
            ((total_tasks - failure_count) / total_tasks * 100)
            if total_tasks > 0 else 100.0
        )
        
        return PipelineMetrics(
            total_duration=total_duration,
            task_metrics=self._task_metrics.copy(),
            total_events_detected=total_events,
            failure_count=failure_count,
            success_rate=success_rate
        )
    
    def get_summary_report(self) -> str:
        """
        Generate structured summary report.
        
        Returns:
            Formatted summary report string
        """
        if self._pipeline_start_time == 0:
            return "Pipeline has not been executed"
        
        # Calculate total runtime
        total_duration = (
            self._pipeline_end_time - self._pipeline_start_time
            if self._pipeline_end_time > 0 
            else time.time() - self._pipeline_start_time
        )
        
        # Format duration
        def format_duration(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = seconds % 60
                return f"{minutes}m {secs:.0f}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h {minutes}m"
        
        # Build report
        lines = [
            "PIPELINE EXECUTION SUMMARY",
            "------------------------------------",
            f"Total Runtime: {format_duration(total_duration)}"
        ]
        
        # Tasks section
        if self._task_metrics:
            lines.append("")
            lines.append("Tasks:")
            
            for task_metrics in self._task_metrics:
                status = "✓" if task_metrics.success else "✗"
                duration_str = format_duration(task_metrics.duration)
                lines.append(f"- {task_metrics.task_name}: {duration_str} {status}")
                
                if not task_metrics.success and task_metrics.error_message:
                    lines.append(f"  Error: {task_metrics.error_message}")
        
        # Detections section
        total_detections = sum(self._detection_counts.values())
        if total_detections > 0:
            lines.append("")
            lines.append("Detected:")
            
            detection_labels = {
                'blink_events': 'Blink events',
                'breath_events': 'Breath events', 
                'scenes': 'Scenes',
                'tone_segments': 'Tone segments',
                'summaries': 'Summaries',
                'speakers': 'Speakers',
                'transcript_words': 'Transcript words'
            }
            
            for detection_type, count in self._detection_counts.items():
                if count > 0:
                    label = detection_labels.get(detection_type, detection_type)
                    lines.append(f"- {label}: {count}")
        
        # Failures section
        lines.append("")
        if self._failures:
            lines.append("Failures:")
            for failure in self._failures:
                lines.append(f"- {failure.task_name}: {failure.error_message}")
        else:
            lines.append("Failures: None")
        
        # Performance summary
        success_count = len([tm for tm in self._task_metrics if tm.success])
        total_tasks = len(self._task_metrics)
        if total_tasks > 0:
            success_rate = (success_count / total_tasks) * 100
            lines.append("")
            lines.append(f"Success Rate: {success_rate:.1f}% ({success_count}/{total_tasks})")
        
        return "\n".join(lines)
    
    def log_summary(self) -> None:
        """Log the summary report."""
        summary = self.get_summary_report()
        
        # Split into lines and log each
        for line in summary.split("\n"):
            if line.strip():
                self._logger.log_info(line)
    
    def reset(self) -> None:
        """Reset monitor state for new pipeline execution."""
        self._pipeline_start_time = 0.0
        self._pipeline_end_time = 0.0
        self._current_task = None
        self._task_start_times.clear()
        
        self._task_metrics.clear()
        self._failures.clear()
        
        for key in self._detection_counts:
            self._detection_counts[key] = 0
        
        self._metrics.reset_all()
        
        self._logger.log_info("Monitor reset for new execution")