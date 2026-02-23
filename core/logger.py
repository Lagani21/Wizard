"""
Structured Logger for WIZ Intelligence Pipeline.
Centralized logging with deterministic output format.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO
from contextlib import contextmanager


class Logger:
    """
    Centralized logger for WIZ Intelligence Pipeline.
    
    Provides structured logging with consistent format and optional file output.
    Thread-safe and deterministic.
    """
    
    def __init__(self, log_file_path: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            log_file_path: Optional path to log file. If None, logs only to console.
        """
        self._log_file: Optional[TextIO] = None
        self._current_task: Optional[str] = None
        
        if log_file_path:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, 'a', encoding='utf-8')
    
    def _format_timestamp(self) -> str:
        """Generate consistent timestamp format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _write_log(self, level: str, message: str, task_name: Optional[str] = None) -> None:
        """
        Write structured log entry.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, METRIC)
            message: Log message
            task_name: Optional task name context
        """
        timestamp = self._format_timestamp()
        task_context = f"[{task_name or self._current_task or 'Pipeline'}]"
        log_entry = f"[{timestamp}] [{level}] {task_context} {message}"
        
        # Write to console
        print(log_entry)
        
        # Write to file if configured
        if self._log_file:
            self._log_file.write(log_entry + "\n")
            self._log_file.flush()
    
    def log_info(self, message: str, task_name: Optional[str] = None) -> None:
        """
        Log informational message.
        
        Args:
            message: Information message
            task_name: Optional task context
        """
        self._write_log("INFO", message, task_name)
    
    def log_warning(self, message: str, task_name: Optional[str] = None) -> None:
        """
        Log warning message.
        
        Args:
            message: Warning message
            task_name: Optional task context
        """
        self._write_log("WARNING", message, task_name)
    
    def log_error(self, message: str, task_name: Optional[str] = None) -> None:
        """
        Log error message.
        
        Args:
            message: Error message
            task_name: Optional task context
        """
        self._write_log("ERROR", message, task_name)
    
    def log_task_start(self, task_name: str) -> None:
        """
        Log task start event.
        
        Args:
            task_name: Name of starting task
        """
        self._current_task = task_name
        self._write_log("INFO", "Started", task_name)
    
    def log_task_end(self, task_name: str, duration: float) -> None:
        """
        Log task completion event.
        
        Args:
            task_name: Name of completed task
            duration: Task duration in seconds
        """
        self._write_log("INFO", f"Completed in {duration:.2f}s", task_name)
        if self._current_task == task_name:
            self._current_task = None
    
    def log_metric(self, name: str, value: float, unit: str = "") -> None:
        """
        Log metric value.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit (e.g., 's', 'ms', 'MB')
        """
        unit_str = unit if unit else ""
        self._write_log("METRIC", f"{name}={value:.2f}{unit_str}")
    
    @contextmanager
    def task_context(self, task_name: str):
        """
        Context manager for task logging.
        
        Args:
            task_name: Task name for context
            
        Usage:
            with logger.task_context("BlinkTask"):
                # Task execution
                pass
        """
        start_time = time.time()
        self.log_task_start(task_name)
        
        try:
            yield
        except Exception as e:
            self.log_error(f"Failed: {str(e)}", task_name)
            raise
        finally:
            duration = time.time() - start_time
            self.log_task_end(task_name, duration)
    
    def close(self) -> None:
        """Close log file if open."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()