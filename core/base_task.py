"""
Base task abstraction for the WIZ Intelligence Pipeline.
"""

from abc import ABC, abstractmethod
import time
import logging
from typing import Any
from .context import PipelineContext


class BaseTask(ABC):
    """
    Abstract base class for all pipeline tasks.
    
    Each task must:
    - Accept PipelineContext
    - Store results back into context
    - Log execution time
    - Not directly call other tasks
    """
    
    def __init__(self, task_name: str) -> None:
        """
        Initialize the task with a name for logging.
        
        Args:
            task_name: Human-readable name for this task
        """
        self.task_name = task_name
        self.logger = logging.getLogger(f"wiz.tasks.{task_name}")
    
    def execute(self, context: PipelineContext) -> None:
        """
        Execute the task with timing and logging.
        
        Args:
            context: Pipeline context containing shared state
        """
        start_time = time.perf_counter()
        self.logger.info(f"Starting task: {self.task_name}")
        
        try:
            self._run(context)
            execution_time = time.perf_counter() - start_time
            self.logger.info(f"Completed task: {self.task_name} in {execution_time:.3f}s")
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.logger.error(f"Failed task: {self.task_name} after {execution_time:.3f}s - {str(e)}")
            raise
    
    @abstractmethod
    def _run(self, context: PipelineContext) -> None:
        """
        Implement the core logic of the task.
        
        Args:
            context: Pipeline context containing shared state
        """
        pass