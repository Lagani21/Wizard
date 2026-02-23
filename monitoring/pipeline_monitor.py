"""
Pipeline Monitoring and Logging System for WIZ Video Intelligence
"""

import os
import time
import psutil
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import queue

@dataclass
class ProcessingMetrics:
    """Metrics for a single processing session"""
    session_id: str
    video_path: str
    video_size_mb: float
    video_duration_seconds: float
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    
    # Task metrics
    blink_detection_time_seconds: Optional[float] = None
    breath_detection_time_seconds: Optional[float] = None
    speech_processing_time_seconds: Optional[float] = None
    tone_detection_time_seconds: Optional[float] = None
    context_summary_time_seconds: Optional[float] = None
    
    # Results
    blinks_detected: int = 0
    breaths_detected: int = 0
    speakers_detected: int = 0
    scenes_generated: int = 0
    
    # Resource usage
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Errors
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def total_processing_time_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def processing_speed_ratio(self) -> Optional[float]:
        """Ratio of processing time to video duration (lower is faster)"""
        if self.total_processing_time_seconds and self.video_duration_seconds:
            return self.total_processing_time_seconds / self.video_duration_seconds
        return None

class PipelineMonitor:
    """Monitors pipeline performance and collects metrics"""
    
    def __init__(self, log_dir: str = "logs", max_sessions: int = 1000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.max_sessions = max_sessions
        self.current_sessions: Dict[str, ProcessingMetrics] = {}
        self.completed_sessions: deque = deque(maxlen=max_sessions)
        
        # Resource monitoring
        self.resource_monitor_active = False
        self.resource_queue = queue.Queue()
        
        # Setup logging
        self.setup_logging()
        
        # Performance stats
        self.stats = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'failed_sessions': 0,
            'average_processing_time': 0.0,
            'average_processing_speed_ratio': 0.0
        }
        
        self.logger.info("Pipeline monitor initialized")
    
    def setup_logging(self):
        """Setup structured logging"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup main logger
        self.logger = logging.getLogger('wiz.monitor')
        self.logger.setLevel(logging.INFO)
        
        # File handler for detailed logs
        detailed_handler = logging.FileHandler(self.log_dir / 'pipeline_detailed.log')
        detailed_handler.setLevel(logging.DEBUG)
        detailed_handler.setFormatter(detailed_formatter)
        
        # File handler for simple logs
        simple_handler = logging.FileHandler(self.log_dir / 'pipeline_simple.log')
        simple_handler.setLevel(logging.INFO)
        simple_handler.setFormatter(simple_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(detailed_handler)
        self.logger.addHandler(simple_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def start_session(self, session_id: str, video_path: str) -> ProcessingMetrics:
        """Start monitoring a new processing session"""
        try:
            # Get video metadata
            video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            # Try to get video duration using opencv
            video_duration = 0.0
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        video_duration = frame_count / fps
                    cap.release()
            except Exception as e:
                self.logger.warning(f"Could not determine video duration: {e}")
            
            # Create metrics object
            metrics = ProcessingMetrics(
                session_id=session_id,
                video_path=video_path,
                video_size_mb=video_size_mb,
                video_duration_seconds=video_duration,
                start_time=datetime.now()
            )
            
            self.current_sessions[session_id] = metrics
            self.stats['total_sessions'] += 1
            
            # Start resource monitoring
            self.start_resource_monitoring(session_id)
            
            self.logger.info(f"Started session {session_id} for {Path(video_path).name} "
                           f"({video_size_mb:.2f}MB, {video_duration:.1f}s)")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error starting session {session_id}: {e}")
            raise
    
    def log_task_completion(self, session_id: str, task_name: str, duration_seconds: float, 
                           results: Optional[Dict[str, Any]] = None):
        """Log completion of a specific task"""
        if session_id not in self.current_sessions:
            self.logger.warning(f"Session {session_id} not found for task {task_name}")
            return
        
        metrics = self.current_sessions[session_id]
        
        # Store timing
        if task_name == "blink_detection":
            metrics.blink_detection_time_seconds = duration_seconds
            if results:
                metrics.blinks_detected = results.get('blink_count', 0)
                
        elif task_name == "breath_detection":
            metrics.breath_detection_time_seconds = duration_seconds
            if results:
                metrics.breaths_detected = results.get('breath_count', 0)
                
        elif task_name == "speech_processing":
            metrics.speech_processing_time_seconds = duration_seconds
            if results:
                metrics.speakers_detected = results.get('speaker_count', 0)
                
        elif task_name == "tone_detection":
            metrics.tone_detection_time_seconds = duration_seconds
            
        elif task_name == "context_summary":
            metrics.context_summary_time_seconds = duration_seconds
            if results:
                metrics.scenes_generated = results.get('scene_count', 0)
        
        self.logger.info(f"Task {task_name} completed in {duration_seconds:.2f}s for session {session_id}")
    
    def log_error(self, session_id: str, error_message: str, task_name: Optional[str] = None):
        """Log an error for a session"""
        if session_id in self.current_sessions:
            self.current_sessions[session_id].errors.append(error_message)
        
        error_context = f"[{task_name}] " if task_name else ""
        self.logger.error(f"Session {session_id}: {error_context}{error_message}")
    
    def end_session(self, session_id: str, status: str = "completed"):
        """End monitoring for a session"""
        if session_id not in self.current_sessions:
            self.logger.warning(f"Session {session_id} not found for ending")
            return
        
        metrics = self.current_sessions[session_id]
        metrics.end_time = datetime.now()
        metrics.status = status
        
        # Stop resource monitoring
        self.stop_resource_monitoring(session_id)
        
        # Move to completed sessions
        self.completed_sessions.append(metrics)
        del self.current_sessions[session_id]
        
        # Update stats
        if status == "completed":
            self.stats['successful_sessions'] += 1
        elif status == "failed":
            self.stats['failed_sessions'] += 1
        
        self.update_average_stats()
        
        # Log session summary
        total_time = metrics.total_processing_time_seconds
        speed_ratio = metrics.processing_speed_ratio
        
        self.logger.info(
            f"Session {session_id} {status}: "
            f"Total time: {total_time:.2f}s, "
            f"Speed ratio: {speed_ratio:.2f}x, "
            f"Blinks: {metrics.blinks_detected}, "
            f"Breaths: {metrics.breaths_detected}, "
            f"Speakers: {metrics.speakers_detected}, "
            f"Errors: {len(metrics.errors)}"
        )
        
        # Save session to file
        self.save_session_metrics(metrics)
    
    def start_resource_monitoring(self, session_id: str):
        """Start monitoring CPU and memory usage"""
        def monitor_resources():
            process = psutil.Process()
            metrics = self.current_sessions.get(session_id)
            if not metrics:
                return
            
            while session_id in self.current_sessions:
                try:
                    cpu_percent = process.cpu_percent()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    
                    metrics.peak_cpu_percent = max(metrics.peak_cpu_percent, cpu_percent)
                    metrics.peak_memory_mb = max(metrics.peak_memory_mb, memory_mb)
                    
                    time.sleep(1)  # Monitor every second
                except Exception:
                    break
        
        if not self.resource_monitor_active:
            thread = threading.Thread(target=monitor_resources, daemon=True)
            thread.start()
            self.resource_monitor_active = True
    
    def stop_resource_monitoring(self, session_id: str):
        """Stop resource monitoring for a session"""
        # Resource monitoring will stop when session is removed from current_sessions
        pass
    
    def update_average_stats(self):
        """Update average performance statistics"""
        if not self.completed_sessions:
            return
        
        # Calculate averages from recent sessions
        recent_sessions = list(self.completed_sessions)[-100:]  # Last 100 sessions
        
        processing_times = [s.total_processing_time_seconds for s in recent_sessions 
                          if s.total_processing_time_seconds is not None]
        
        speed_ratios = [s.processing_speed_ratio for s in recent_sessions 
                       if s.processing_speed_ratio is not None]
        
        if processing_times:
            self.stats['average_processing_time'] = sum(processing_times) / len(processing_times)
        
        if speed_ratios:
            self.stats['average_processing_speed_ratio'] = sum(speed_ratios) / len(speed_ratios)
    
    def save_session_metrics(self, metrics: ProcessingMetrics):
        """Save session metrics to JSON file"""
        try:
            metrics_file = self.log_dir / f"session_{metrics.session_id}.json"
            
            # Convert to dict with proper datetime serialization
            metrics_dict = asdict(metrics)
            metrics_dict['start_time'] = metrics.start_time.isoformat()
            if metrics.end_time:
                metrics_dict['end_time'] = metrics.end_time.isoformat()
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving session metrics: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        return {
            'current_time': datetime.now().isoformat(),
            'active_sessions': len(self.current_sessions),
            'total_sessions': self.stats['total_sessions'],
            'successful_sessions': self.stats['successful_sessions'],
            'failed_sessions': self.stats['failed_sessions'],
            'success_rate': (self.stats['successful_sessions'] / max(1, self.stats['total_sessions'])) * 100,
            'average_processing_time': self.stats['average_processing_time'],
            'average_speed_ratio': self.stats['average_processing_speed_ratio'],
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        }
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent completed sessions"""
        recent = list(self.completed_sessions)[-limit:]
        
        sessions_data = []
        for metrics in recent:
            session_dict = asdict(metrics)
            session_dict['start_time'] = metrics.start_time.isoformat()
            if metrics.end_time:
                session_dict['end_time'] = metrics.end_time.isoformat()
            sessions_data.append(session_dict)
        
        return sessions_data

# Global monitor instance
_monitor_instance: Optional[PipelineMonitor] = None

def get_monitor() -> PipelineMonitor:
    """Get the global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PipelineMonitor()
    return _monitor_instance

def initialize_monitoring(log_dir: str = "logs") -> PipelineMonitor:
    """Initialize the global monitoring system"""
    global _monitor_instance
    _monitor_instance = PipelineMonitor(log_dir=log_dir)
    return _monitor_instance