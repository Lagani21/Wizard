"""
Main pipeline for the WIZ Intelligence Pipeline.
"""

import os
import cv2
import numpy as np
import time
import gc
import threading
import psutil
from typing import List, Optional, Dict, Any

# Import core components
try:
    from .context import PipelineContext, VideoMetadata
    from .base_task import BaseTask
    from .logger import Logger
    from .monitor import PipelineMonitor
except ImportError:
    from core.context import PipelineContext, VideoMetadata
    from core.base_task import BaseTask
    from core.logger import Logger
    from core.monitor import PipelineMonitor

# Optional .wiz persistence — non-critical; pipeline still works without it
WizWriter = None
get_wiz_path_for_video = None
try:
    from ..wiz.writer import WizWriter
    from ..wiz.format import get_wiz_path_for_video
except ImportError:
    try:
        from wiz.writer import WizWriter
        from wiz.format import get_wiz_path_for_video
    except ImportError:
        pass

# Import tasks — all optional; missing tasks are skipped at runtime
BlinkTask = None
BreathTask = None
TranscriptionTask = None
DiarizationTask = None
AlignmentTask = None
VideoMAETask = None
ToneDetectionTask = None
ContextSummaryTask = None
AudioExtractor = None

try:
    from ..tasks.blink_task import BlinkTask
    from ..tasks.breath_task import BreathTask
    from ..audio.audio_extractor import AudioExtractor
except ImportError:
    try:
        from tasks.blink_task import BlinkTask
        from tasks.breath_task import BreathTask
        from audio.audio_extractor import AudioExtractor
    except ImportError:
        print("⚠️  Some pipeline components not available - basic functionality only")

try:
    from ..tasks.transcription_task import TranscriptionTask
    from ..tasks.diarization_task import DiarizationTask
    from ..tasks.alignment_task import AlignmentTask
    from ..tasks.video_mae_task import VideoMAETask
    from ..tasks.tone_detection_task import ToneDetectionTask
    from ..tasks.context_summary_task import ContextSummaryTask
except ImportError:
    try:
        from tasks.transcription_task import TranscriptionTask
        from tasks.diarization_task import DiarizationTask
        from tasks.alignment_task import AlignmentTask
        from tasks.video_mae_task import VideoMAETask
        from tasks.tone_detection_task import ToneDetectionTask
        from tasks.context_summary_task import ContextSummaryTask
    except ImportError:
        pass  # Advanced features not available

class Pipeline:
    """
    Main pipeline coordinator for WIZ Intelligence Processing.
    
    Orchestrates the execution of detection and speech processing tasks
    in a deterministic, object-oriented manner.
    """
    
    def __init__(
        self,
        logger: Optional[Logger] = None,
        monitor: Optional[PipelineMonitor] = None,
        # Per-task overrides (None → create default)
        transcription_task: Optional[TranscriptionTask] = None,
        diarization_task: Optional[DiarizationTask] = None,
        alignment_task: Optional[AlignmentTask] = None,
        video_mae_task: Optional[VideoMAETask] = None,
        tone_detection_task: Optional[ToneDetectionTask] = None,
        context_summary_task: Optional[ContextSummaryTask] = None,
        blink_task: Optional[BlinkTask] = None,
        breath_task: Optional[BreathTask] = None,
        run_mode: str = "full",
        # Kept for backwards compatibility — ignored in full mode
        enable_speech_processing: bool = True,
        enable_tone_detection: bool = True,
        enable_context_summary: bool = True,
        enable_video_mae: bool = True,
    ) -> None:
        """
        Initialize the WIZ Intelligence Pipeline.

        All tasks always run in full mode in this fixed order:
          transcription → diarization → alignment → video_mae →
          tone_detection → context_summary → blink → breath

        In 'lite' mode video_mae, tone_detection, and context_summary are
        skipped to reduce memory pressure.

        Args:
            logger:               Structured logger (default created if None)
            monitor:              Performance monitor (default created if None)
            transcription_task:   Override for transcription (Whisper)
            diarization_task:     Override for diarization (Pyannote)
            alignment_task:       Override for transcript/speaker alignment
            video_mae_task:       Override for Video MAE captioning
            tone_detection_task:  Override for emotional tone detection
            context_summary_task: Override for LLM scene summarization
            blink_task:           Override for blink detection (Apple Vision)
            breath_task:          Override for breath detection
            run_mode:             'full' (all tasks) or 'lite' (core tasks only)
        """
        # ── Infrastructure ─────────────────────────────────────────────────
        self.logger  = logger  if logger  is not None else Logger()
        self.monitor = monitor if monitor is not None else PipelineMonitor(self.logger)

        if run_mode not in ("full", "lite"):
            raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'full' or 'lite'")
        self.run_mode = run_mode
        self._configure_system_for_mode()

        lite = (run_mode == "lite")
        if lite:
            self.logger.log_info("LIGHTWEIGHT MODE ENABLED")
            self.logger.log_info("   - Video MAE, tone detection, context summaries disabled")
            self.logger.log_info("   - Sequential execution enforced")
            self.logger.log_info("   - Memory cleanup optimized")

        # ── Speech tasks (always-on) ────────────────────────────────────────
        if transcription_task is not None:
            self.transcription_task = transcription_task
        elif TranscriptionTask is not None:
            model_size = "small" if lite else "base"
            self.transcription_task = TranscriptionTask.create_default(model_size=model_size)
        else:
            self.transcription_task = None
            self.logger.log_warning("TranscriptionTask not available")

        if diarization_task is not None:
            self.diarization_task = diarization_task
        elif DiarizationTask is not None:
            self.diarization_task = DiarizationTask.create_default()
        else:
            self.diarization_task = None
            self.logger.log_warning("DiarizationTask not available")

        if alignment_task is not None:
            self.alignment_task = alignment_task
        elif AlignmentTask is not None:
            self.alignment_task = AlignmentTask.create_default()
        else:
            self.alignment_task = None
            self.logger.log_warning("AlignmentTask not available")

        # ── Video MAE (disabled in lite mode) ──────────────────────────────
        if lite:
            self.video_mae_task = None
        elif video_mae_task is not None:
            self.video_mae_task = video_mae_task
        elif VideoMAETask is not None:
            self.video_mae_task = VideoMAETask.create_default()
        else:
            self.video_mae_task = None
            self.logger.log_warning("VideoMAETask not available")

        # ── Tone detection (disabled in lite mode) ─────────────────────────
        if lite:
            self.tone_detection_task = None
        elif tone_detection_task is not None:
            self.tone_detection_task = tone_detection_task
        elif ToneDetectionTask is not None:
            self.tone_detection_task = ToneDetectionTask.create_default()
        else:
            self.tone_detection_task = None
            self.logger.log_warning("ToneDetectionTask not available")

        # ── Context summary (disabled in lite mode) ────────────────────────
        if lite:
            self.context_summary_task = None
        elif context_summary_task is not None:
            self.context_summary_task = context_summary_task
        elif ContextSummaryTask is not None:
            self.context_summary_task = ContextSummaryTask.create_default()
        else:
            self.context_summary_task = None
            self.logger.log_warning("ContextSummaryTask not available")

        # ── Physical detection tasks (always-on, run at end) ──────────────
        if blink_task is not None:
            self.blink_task = blink_task
        elif BlinkTask is not None:
            self.blink_task = BlinkTask.create_default()
        else:
            self.blink_task = None
            self.logger.log_warning("BlinkTask not available")

        if breath_task is not None:
            self.breath_task = breath_task
        elif BreathTask is not None:
            self.breath_task = BreathTask.create_default()
        else:
            self.breath_task = None
            self.logger.log_warning("BreathTask not available")

        # ── Audio extractor ────────────────────────────────────────────────
        self.audio_extractor = AudioExtractor()

        # ── Canonical task execution order ─────────────────────────────────
        # transcription → diarization → alignment → video_mae →
        # tone → context_summary → blink → breath
        self.tasks: List[BaseTask] = [
            self.transcription_task,
            self.diarization_task,
            self.alignment_task,
            self.video_mae_task,
            self.tone_detection_task,
            self.context_summary_task,
            self.blink_task,
            self.breath_task,
        ]

        active = [t.task_name for t in self.tasks if t is not None]
        self.logger.log_info(f"WIZ Intelligence Pipeline initialised: {', '.join(active)}")
    
    def _configure_system_for_mode(self) -> None:
        """
        Configure system settings based on run mode for optimal performance.
        """
        if self.run_mode == "lite":
            try:
                # Limit PyTorch threads for Apple Silicon
                import torch
                torch.set_num_threads(4)
                self.logger.log_info("   - PyTorch threads limited to 4")
            except ImportError:
                pass
            
            # Set OpenMP threads via environment variable
            os.environ["OMP_NUM_THREADS"] = "4"
            self.logger.log_info("   - OpenMP threads limited to 4")
            
            # Additional thread limiting for NumPy/BLAS
            os.environ["MKL_NUM_THREADS"] = "4"
            os.environ["OPENBLAS_NUM_THREADS"] = "4"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
            os.environ["NUMEXPR_NUM_THREADS"] = "4"
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Get current system memory usage.
        
        Returns:
            Dictionary with memory usage statistics in MB
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                "process_rss_mb": memory_info.rss / (1024 * 1024),
                "process_vms_mb": memory_info.vms / (1024 * 1024),
                "system_available_mb": system_memory.available / (1024 * 1024),
                "system_percent": system_memory.percent
            }
        except Exception:
            return {"error": "Memory stats unavailable"}
    
    def _aggressive_memory_cleanup(self) -> None:
        """
        Perform aggressive memory cleanup between tasks.
        """
        if self.run_mode == "lite":
            # Force garbage collection
            collected = gc.collect()
            
            # Log memory usage if available
            memory_stats = self._get_memory_usage()
            if "error" not in memory_stats:
                self.logger.log_info(
                    f"   Memory cleanup: {collected} objects collected, "
                    f"Process RSS: {memory_stats['process_rss_mb']:.1f}MB"
                )
    
    def _log_resource_usage(self, task_name: str, execution_time: float) -> None:
        """
        Log resource usage after task execution.
        
        Args:
            task_name: Name of the completed task
            execution_time: Task execution time in seconds
        """
        memory_stats = self._get_memory_usage()
        thread_count = threading.active_count()
        
        self.logger.log_info(
            f"Resource usage - {task_name}: "
            f"Time={execution_time:.2f}s, Threads={thread_count}"
        )
        
        if "error" not in memory_stats:
            self.logger.log_info(
                f"   Memory: Process={memory_stats['process_rss_mb']:.1f}MB, "
                f"System={memory_stats['system_percent']:.1f}% used"
            )
    
    def _validate_video_path(self, video_path: str) -> None:
        """
        Validate the input video path.
        
        Args:
            video_path: Path to video file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If file is not a valid video format
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(video_path):
            raise ValueError(f"Path is not a file: {video_path}")
        
        # Check file extension (basic validation)
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        _, ext = os.path.splitext(video_path.lower())
        
        if ext not in valid_extensions:
            self.logger.log_warning(f"Uncommon video extension: {ext}")
    
    def _extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata object with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            metadata = VideoMetadata(
                path=os.path.abspath(video_path),
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration_seconds
            )
            
            self.logger.log_info(
                f"Video metadata: {width}x{height}, {fps:.2f}fps, "
                f"{total_frames} frames, {duration_seconds:.2f}s"
            )
            
            return metadata
            
        finally:
            cap.release()
    
    def _extract_audio_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract mono audio from video file at 16kHz.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Mono audio waveform at 16kHz
        """
        return self.audio_extractor.extract_audio_from_video(video_path)
    
    def run(self, video_path: str) -> PipelineContext:
        """
        Main entry point for the WIZ Intelligence Pipeline.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            PipelineContext containing all detection results and infrastructure
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file is invalid or tasks fail
        """
        self.logger.log_info(f"Starting WIZ Intelligence Pipeline for: {video_path}")
        
        # Start pipeline monitoring
        self.monitor.start_pipeline()
        
        # Validate input
        self._validate_video_path(video_path)
        
        # Initialize context with logging and monitoring infrastructure
        context = PipelineContext(
            logger=self.logger,
            monitor=self.monitor
        )
        
        # Add run mode to context for task-level optimizations
        context.run_mode = self.run_mode
        
        try:
            # Extract video metadata with monitoring
            self.monitor.start_task("MetadataExtraction")
            try:
                self.logger.log_info("Extracting video metadata...")
                context.video_metadata = self._extract_video_metadata(video_path)
                self.monitor.end_task("MetadataExtraction", success=True)
            except Exception as e:
                self.monitor.end_task("MetadataExtraction", success=False, error_message=str(e))
                raise
            
            # Extract audio with monitoring
            self.monitor.start_task("AudioExtraction")
            try:
                self.logger.log_info("Extracting audio from video...")
                context.audio_waveform = self._extract_audio_from_video(video_path)
                self.monitor.end_task("AudioExtraction", success=True)
            except Exception as e:
                self.monitor.end_task("AudioExtraction", success=False, error_message=str(e))
                raise
            
            # Execute all pipeline tasks sequentially with resource monitoring
            for task in self.tasks:
                if task is not None:  # Skip None tasks
                    try:
                        # Record start time for resource monitoring
                        task_start_time = time.time()
                        
                        # Execute task with memory monitoring
                        if self.run_mode == "lite":
                            self.logger.log_info(f"🔄 Sequential execution: {task.task_name}")
                        
                        task.execute(context)
                        
                        # Calculate execution time
                        execution_time = time.time() - task_start_time
                        
                        # Log resource usage
                        self._log_resource_usage(task.task_name, execution_time)
                        
                        # Perform aggressive memory cleanup in lite mode
                        self._aggressive_memory_cleanup()
                        
                        # Record detection counts after each task
                        self._record_detection_metrics(context)
                        
                    except Exception as task_error:
                        # Task execution handles its own monitoring through BaseTask
                        # Pipeline continues for non-critical failures
                        self.logger.log_warning(f"Task {task.task_name} failed, continuing pipeline: {str(task_error)}")
                        
                        # Still perform cleanup even if task failed
                        if self.run_mode == "lite":
                            self._aggressive_memory_cleanup()
            
            # Record final pipeline metrics
            self._record_final_metrics(context)

            # Persist results to .wiz file (non-critical)
            wiz_path = self._write_database(video_path, context)
            if wiz_path:
                context.processing_metadata['wiz_path'] = wiz_path

            # End pipeline monitoring
            self.monitor.end_pipeline()

            # Log structured summary
            self.monitor.log_summary()

            self.logger.log_info("WIZ Intelligence Pipeline completed successfully")
            return context
            
        except Exception as e:
            self.monitor.end_pipeline()
            self.logger.log_error(f"WIZ Intelligence Pipeline failed: {str(e)}")
            raise
    
    def _write_database(self, video_path: str, context: PipelineContext) -> Optional[str]:
        """
        Persist completed pipeline context to a .wiz file.

        Non-critical: any failure is logged as a warning and the pipeline
        result is still returned to the caller.

        Returns:
            Absolute path to the written .wiz file, or None on failure.
        """
        if WizWriter is None or get_wiz_path_for_video is None:
            self.logger.log_warning("WizWriter not available — skipping .wiz persistence")
            return None
        if context.video_metadata is None:
            self.logger.log_warning("No video_metadata in context — skipping .wiz write")
            return None
        try:
            wiz_path = get_wiz_path_for_video(video_path, output_dir="results")
            abs_path = WizWriter().write(context, wiz_path)
            self.logger.log_info(f"Results persisted to: {abs_path}")
            return abs_path
        except Exception as exc:
            self.logger.log_warning(f".wiz write failed (non-critical): {exc}")
            return None

    def _record_detection_metrics(self, context: PipelineContext) -> None:
        """
        Record detection counts to monitor.
        
        Args:
            context: Pipeline context with detection results
        """
        # Record current detection counts
        self.monitor.record_detection_count('blink_events', len(context.blink_events))
        self.monitor.record_detection_count('breath_events', len(context.breath_events))
        self.monitor.record_detection_count('transcript_words', len(context.transcript_words))
        self.monitor.record_detection_count('tone_segments', len(context.tone_events))
        self.monitor.record_detection_count('summaries', len(context.scene_summaries))
        
        # Calculate unique speakers
        if context.speaker_segments:
            unique_speakers = len(set(seg.speaker_id for seg in context.speaker_segments))
            self.monitor.record_detection_count('speakers', unique_speakers)
    
    def _record_final_metrics(self, context: PipelineContext) -> None:
        """
        Record final pipeline execution metrics.
        
        Args:
            context: Pipeline context with final results
        """
        # Record final detection counts
        self._record_detection_metrics(context)
        
        # Record video metadata metrics
        if context.video_metadata:
            self.monitor.record_metric("video.duration_seconds", context.video_metadata.duration_seconds)
            self.monitor.record_metric("video.fps", context.video_metadata.fps)
            self.monitor.record_metric("video.width", context.video_metadata.width)
            self.monitor.record_metric("video.height", context.video_metadata.height)
    
    def _get_task_results(self, task_name: str, context: PipelineContext) -> Dict[str, Any]:
        """Get results from a completed task for monitoring"""
        results = {}
        
        if task_name == "blink_detection":
            results['blink_count'] = len(context.blink_events)
            
        elif task_name == "breath_detection":
            results['breath_count'] = len(context.breath_events)
            
        elif task_name == "speech_processing" or task_name == "transcription":
            if hasattr(context, 'aligned_segments') and context.aligned_segments:
                speakers = set(seg.speaker_id for seg in context.aligned_segments)
                results['speaker_count'] = len(speakers)
                results['transcript_segments'] = len(context.aligned_segments)
                
        elif task_name == "tone_detection":
            if hasattr(context, 'tone_events') and context.tone_events:
                results['tone_events_count'] = len(context.tone_events)
                
        elif task_name == "context_summary":
            if hasattr(context, 'scene_summaries') and context.scene_summaries:
                results['scene_count'] = len(context.scene_summaries)
        
        return results
    
    def _log_pipeline_results(self, context: PipelineContext) -> None:
        """
        Log the final pipeline results.
        
        Args:
            context: Pipeline context with results
        """
        self.logger.log_info("=== PIPELINE RESULTS ===")
        self.logger.log_info(f"Blink events detected: {len(context.blink_events)}")
        self.logger.log_info(f"Breath events detected: {len(context.breath_events)}")
        
        if context.blink_events:
            total_blink_time = sum(event.duration_ms for event in context.blink_events)
            avg_blink_confidence = np.mean([event.confidence for event in context.blink_events])
            self.logger.log_info(f"Total blink duration: {total_blink_time:.1f}ms")
            self.logger.log_info(f"Average blink confidence: {avg_blink_confidence:.3f}")
        
        if context.breath_events:
            total_breath_time = sum(event.duration_ms for event in context.breath_events)
            avg_breath_confidence = np.mean([event.confidence for event in context.breath_events])
            self.logger.log_info(f"Total breath duration: {total_breath_time:.1f}ms")
            self.logger.log_info(f"Average breath confidence: {avg_breath_confidence:.3f}")
        
        self.logger.log_info("========================")
    
    def get_results_summary(self, context: PipelineContext) -> dict:
        """
        Get a summary of pipeline results.
        
        Args:
            context: Pipeline context with results
            
        Returns:
            Dictionary with results summary
        """
        summary = {
            'video_metadata': {
                'path': context.video_metadata.path if context.video_metadata else None,
                'duration_s': context.video_metadata.duration_seconds if context.video_metadata else 0,
                'fps': context.video_metadata.fps if context.video_metadata else 0,
                'resolution': f"{context.video_metadata.width}x{context.video_metadata.height}" if context.video_metadata else None
            },
            'blink_detection': {
                'total_events': len(context.blink_events),
                'events': [
                    {
                        'start_frame': event.start_frame,
                        'end_frame': event.end_frame,
                        'duration_ms': event.duration_ms,
                        'confidence': event.confidence
                    }
                    for event in context.blink_events
                ]
            },
            'breath_detection': {
                'total_events': len(context.breath_events),
                'events': [
                    {
                        'start_time': event.start_time,
                        'end_time': event.end_time,
                        'duration_ms': event.duration_ms,
                        'confidence': event.confidence
                    }
                    for event in context.breath_events
                ]
            },
            'processing_metadata': context.processing_metadata
        }
        
        # Add speech processing results if available
        if self.enable_speech_processing:
            summary['speech_processing'] = {
                'transcription': {
                    'total_words': len(context.transcript_words),
                    'total_segments': len(context.transcript_segments),
                    'words': [
                        {
                            'text': word.text,
                            'start_time': word.start_time,
                            'end_time': word.end_time,
                            'confidence': word.confidence
                        }
                        for word in context.transcript_words
                    ],
                    'segments': [
                        {
                            'text': segment.text,
                            'start_time': segment.start_time,
                            'end_time': segment.end_time,
                            'word_count': len(segment.words)
                        }
                        for segment in context.transcript_segments
                    ]
                },
                'diarization': {
                    'total_segments': len(context.speaker_segments),
                    'num_speakers': len(set(seg.speaker_id for seg in context.speaker_segments)),
                    'segments': [
                        {
                            'speaker_id': segment.speaker_id,
                            'start_time': segment.start_time,
                            'end_time': segment.end_time
                        }
                        for segment in context.speaker_segments
                    ]
                },
                'alignment': {
                    'total_segments': len(context.aligned_segments),
                    'segments': [
                        {
                            'speaker_id': segment.speaker_id,
                            'text': segment.text,
                            'start_time': segment.start_time,
                            'end_time': segment.end_time,
                            'word_count': len(segment.words)
                        }
                        for segment in context.aligned_segments
                    ]
                }
            }
        
        # Add tone detection results if available
        if self.enable_tone_detection:
            summary['tone_detection'] = {
                'total_events': len(context.tone_events),
                'events': [
                    {
                        'scene_id': event.scene_id,
                        'start_time': event.start_time,
                        'end_time': event.end_time,
                        'tone_label': event.tone_label,
                        'confidence': event.confidence
                    }
                    for event in context.tone_events
                ]
            }
        
        # Add context summary results if available
        if self.enable_context_summary:
            summary['context_summary'] = {
                'total_summaries': len(context.scene_summaries),
                'summaries': [
                    {
                        'scene_id': summary_obj.scene_id,
                        'start_time': summary_obj.start_time,
                        'end_time': summary_obj.end_time,
                        'summary_text': summary_obj.summary_text,
                        'tone_label': summary_obj.tone_label,
                        'key_speakers': summary_obj.key_speakers,
                        'confidence': summary_obj.confidence
                    }
                    for summary_obj in context.scene_summaries
                ]
            }
        
        return summary