"""
Base task abstraction for the WIZ Intelligence Pipeline.
"""

from abc import ABC, abstractmethod
import time
from typing import Optional
from .context import PipelineContext
from .logger import Logger
from .monitor import PipelineMonitor


class BaseTask(ABC):
    """
    Abstract base class for all pipeline tasks.
    
    Each task must:
    - Accept PipelineContext with logger and monitor
    - Store results back into context
    - Use structured logging through injected logger
    - Record metrics through injected monitor
    - Handle failures gracefully
    """
    
    def __init__(self, task_name: str) -> None:
        """
        Initialize the task with a name for logging.
        
        Args:
            task_name: Human-readable name for this task
        """
        self.task_name = task_name
    
    def execute(self, context: PipelineContext) -> None:
        """
        Execute the task with structured logging and monitoring.
        
        Args:
            context: Pipeline context containing shared state, logger, and monitor
        """
        # Get logger and monitor from context
        logger = context.logger
        monitor = context.monitor
        
        if not logger or not monitor:
            raise RuntimeError(f"Task {self.task_name} requires logger and monitor in context")
        
        # Start monitoring and logging
        monitor.start_task(self.task_name)
        
        success = True
        error_message = ""
        
        try:
            # Execute task implementation
            self._run(context)
            
            # Record successful completion
            logger.log_info(f"Task completed successfully", self.task_name)
            
        except Exception as e:
            # Handle task failure
            success = False
            error_message = str(e)
            
            logger.log_error(f"Task failed: {error_message}", self.task_name)
            
            # Continue pipeline execution for non-critical failures
            if self._is_critical_failure(e):
                raise
        
        finally:
            # Always end task monitoring
            duration = monitor.end_task(self.task_name, success, error_message)
            
            # Record task-specific metrics
            self._record_task_metrics(context, duration)
    
    def _is_critical_failure(self, exception: Exception) -> bool:
        """
        Determine if failure should stop entire pipeline.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if failure is critical and should stop pipeline
        """
        # Override in subclasses for task-specific failure handling
        # By default, all failures are non-critical (pipeline continues)
        return False
    
    def _record_task_metrics(self, context: PipelineContext, duration: float) -> None:
        """
        Record task-specific metrics.
        
        Override in subclasses to record detection counts, model performance, etc.
        
        Args:
            context: Pipeline context
            duration: Task execution duration
        """
        # Base implementation - subclasses can override
        pass
    
    @abstractmethod
    def _run(self, context: PipelineContext) -> None:
        """
        Implement the core logic of the task.
        
        Args:
            context: Pipeline context containing shared state
        """
        pass