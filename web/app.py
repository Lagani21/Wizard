"""
Flask web application for WIZ Video Intelligence Pipeline
"""

import os
import uuid
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

# Import the existing pipeline with flexible path handling
import sys
from pathlib import Path

# Directories relative to this file (works regardless of cwd)
WEB_DIR = Path(__file__).parent
ROOT_DIR = WEB_DIR.parent

# Add project root and web directory to path
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(WEB_DIR))

# Try to import search engine
try:
    from wiz.search import SearchEngine
    SEARCH_AVAILABLE = True
except ImportError:
    SearchEngine = None
    SEARCH_AVAILABLE = False

# Try to import benchmark
try:
    from wiz.benchmark import run_benchmark_json
    BENCHMARK_AVAILABLE = True
except ImportError:
    run_benchmark_json = None
    BENCHMARK_AVAILABLE = False

# Try to import pipeline with multiple strategies
PIPELINE_AVAILABLE = False
Pipeline = None
PipelineContext = None

try:
    # Try importing from parent directory
    from core.pipeline import Pipeline
    from core.context import PipelineContext
    PIPELINE_AVAILABLE = True
    print("✅ WIZ Pipeline imported successfully")
except ImportError:
    try:
        # Try importing with sys.path modification
        sys.path.insert(0, '..')
        from core.pipeline import Pipeline
        from core.context import PipelineContext
        PIPELINE_AVAILABLE = True
        print("✅ WIZ Pipeline imported successfully (fallback)")
    except ImportError as e:
        print(f"⚠️  Pipeline import failed: {e}")
        print("🔧 Creating mock pipeline for testing")
        PIPELINE_AVAILABLE = False
        
        # Create mock classes for development
        class MockVideoMetadata:
            def __init__(self):
                self.duration_seconds = 30.0
                self.fps = 30
                self.width = 640
                self.height = 480
        
        class MockEvent:
            def __init__(self, start_time=0, confidence=0.8):
                self.start_time = start_time
                self.confidence = confidence
                
        class MockBlinkEvent:
            def __init__(self, start_frame=0, confidence=0.8):
                self.start_frame = start_frame
                self.end_frame = start_frame + 3
                self.duration_ms = 100.0
                self.confidence = confidence
        
        class MockContext:
            def __init__(self):
                self.video_metadata = MockVideoMetadata()
                self.blink_events = [MockBlinkEvent(30), MockBlinkEvent(90)]
                self.breath_events = [MockEvent(5.5), MockEvent(12.3), MockEvent(18.7)]
                self.transcript_words = []
                self.speaker_segments = []
                self.aligned_segments = []
                self.tone_events = []
                self.scene_summaries = []
        
        class Pipeline:
            def __init__(self, **kwargs):
                print(f"🔧 Mock pipeline created with options: {kwargs}")
                
            def run(self, video_path):
                print(f"🎬 Mock processing video: {Path(video_path).name}")
                return MockContext()
        
        class PipelineContext:
            pass


class VideoProcessor:
    """Handles video processing tasks in background"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.results_dir = WEB_DIR / 'results'
        self.uploads_dir = WEB_DIR / 'uploads'

        # Create directories if they don't exist
        self.results_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
        
        # Load offline configuration if available
        self.offline_config = self._load_offline_config()
        self.is_offline_mode = self.offline_config.get('offline_mode', False)
        
        print(f"🔧 Video processor initialized (Offline mode: {self.is_offline_mode})")
    
    def _load_offline_config(self) -> Dict[str, Any]:
        """Load offline configuration from setup"""
        config_path = Path.home() / '.wiz_models' / 'offline_config.json'
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load offline config: {e}")
        
        # Default configuration for offline operation
        return {
            'offline_mode': True,
            'models': {
                'enable_speech_processing': True,
                'enable_tone_detection': True, 
                'enable_context_summary': True,
                'llm_backend': 'mock'  # Mock LLM for offline
            }
        }
    
    def create_task(self, video_path: str) -> str:
        """Create a new processing task with logging"""
        task_id = str(uuid.uuid4())
        
        # Log task creation
        try:
            logger = get_logger() if MONITORING_AVAILABLE else None
            if logger:
                logger.log_info(f"Creating processing task for: {Path(video_path).name}")
        except:
            pass
        
        self.tasks[task_id] = {
            'id': task_id,
            'status': 'pending',
            'progress': 0,
            'message': 'Task created',
            'video_path': video_path,
            'created_at': datetime.now().isoformat(),
            'results': None,
            'wiz_path': None,
            'error': None
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=self._process_video, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        try:
            if logger:
                logger.log_info(f"Background processing started for task: {task_id}")
        except:
            pass
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and progress"""
        return self.tasks.get(task_id)
    
    def get_task_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task results"""
        task = self.tasks.get(task_id)
        if task and task['status'] == 'completed':
            return task['results']
        return None
    
    def _process_video(self, task_id: str):
        """Process video in background thread"""
        task = self.tasks[task_id]
        
        try:
            task['status'] = 'processing'
            task['progress'] = 10
            task['message'] = 'Initializing pipeline...'
            
            # Get video path from task
            video_path = task['video_path']
            
            # Create pipeline with offline-optimized settings
            pipeline = None
            if not PIPELINE_AVAILABLE:
                # Mock processing for development/testing
                task['progress'] = 50
                task['message'] = 'Mock processing (pipeline not available)...'
                time.sleep(2)  # Simulate processing
                context = Pipeline().run(video_path)  # This will return mock data
                print("⚠️  Using mock processing - pipeline not fully available")
            else:
                # Real pipeline processing with structured logging
                model_config = self.offline_config.get('models', {})
                
                # Get global logger and monitor
                logger = get_logger()
                monitor = get_monitor()
                
                # Create pipeline with logging and monitoring
                pipeline = Pipeline(
                    logger=logger,
                    monitor=monitor,
                )
                
                if self.is_offline_mode:
                    logger.log_info("Running in offline mode - all processing local")
                
                # Process the video with structured monitoring
                logger.log_info(f"Starting video processing: {video_path}")
                context = pipeline.run(video_path)
                logger.log_info("Video processing completed successfully")

            # Capture .wiz path written by the pipeline (may be None)
            task['wiz_path'] = (
                context.processing_metadata.get('wiz_path')
                if hasattr(context, 'processing_metadata') else None
            )

            task['progress'] = 20
            task['message'] = 'Processing video...'
            
            task['progress'] = 80
            task['message'] = 'Generating results...'
            
            # Convert results to JSON-serializable format
            results = self._format_results(context, pipeline)

            # Include .wiz file reference if it was written
            if task.get('wiz_path'):
                results['wiz_path'] = task['wiz_path']

            # Build waveform from actual extracted audio; fall back to mock
            if hasattr(context, 'audio_waveform') and context.audio_waveform is not None:
                results['audio_waveform'] = self._build_waveform(context.audio_waveform)
            else:
                results['audio_waveform'] = self._generate_mock_waveform()

            task['progress'] = 100
            task['message'] = 'Processing completed'
            task['status'] = 'completed'
            task['results'] = results
            
            # Save results to file
            results_file = self.results_dir / f"{task_id}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            # Log processing failure
            try:
                logger = get_logger() if MONITORING_AVAILABLE else None
                monitor = get_monitor() if MONITORING_AVAILABLE else None
                
                if logger:
                    logger.log_error(f"Video processing failed for task {task_id}: {str(e)}")
                    logger.log_metric("processing.failure", 1)
                
                if monitor:
                    monitor.record_failure("VideoProcessing", str(e))
            except:
                pass
            
            print(f"Error processing video {task_id}: {str(e)}")
            task['status'] = 'failed'
            task['error'] = str(e)
            task['message'] = f'Processing failed: {str(e)}'
    
    def _format_results(self, context: PipelineContext, pipeline: Pipeline) -> Dict[str, Any]:
        """Format pipeline results for JSON serialization"""
        results = {
            'video_metadata': {
                'duration': getattr(context.video_metadata, 'duration_seconds', 0),
                'fps': getattr(context.video_metadata, 'fps', 30),
                'width': getattr(context.video_metadata, 'width', 0),
                'height': getattr(context.video_metadata, 'height', 0)
            },
            'blink_events': [],
            'breath_events': [],
            'transcript_words': [],
            'speaker_segments': [],
            'aligned_segments': [],
            'tone_events': [],
            'scene_summaries': [],
            'video_captions': [],
        }
        
        # Format blink events
        if hasattr(context, 'blink_events') and context.blink_events:
            results['blink_events'] = [
                {
                    'start_frame': event.start_frame,
                    'end_frame': event.end_frame,
                    'duration_ms': event.duration_ms,
                    'confidence': event.confidence
                }
                for event in context.blink_events
            ]
        
        # Format breath events
        if hasattr(context, 'breath_events') and context.breath_events:
            results['breath_events'] = [
                {
                    'start_time': event.start_time,
                    'end_time': event.end_time,
                    'duration_ms': event.duration_ms,
                    'confidence': event.confidence
                }
                for event in context.breath_events
            ]
        
        # Format transcript words
        if hasattr(context, 'transcript_words') and context.transcript_words:
            results['transcript_words'] = [
                {
                    'text': word.text,
                    'start_time': word.start_time,
                    'end_time': word.end_time,
                    'confidence': word.confidence
                }
                for word in context.transcript_words
            ]
        
        # Format speaker segments
        if hasattr(context, 'speaker_segments') and context.speaker_segments:
            results['speaker_segments'] = [
                {
                    'speaker_id': segment.speaker_id,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time
                }
                for segment in context.speaker_segments
            ]
        
        # Format aligned segments
        if hasattr(context, 'aligned_segments') and context.aligned_segments:
            results['aligned_segments'] = [
                {
                    'speaker_id': segment.speaker_id,
                    'text': segment.text,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'words': [
                        {
                            'text': word.text,
                            'start_time': word.start_time,
                            'end_time': word.end_time,
                            'confidence': word.confidence
                        }
                        for word in segment.words
                    ]
                }
                for segment in context.aligned_segments
            ]
        
        # Format tone events
        if hasattr(context, 'tone_events') and context.tone_events:
            results['tone_events'] = [
                {
                    'scene_id': event.scene_id,
                    'start_time': event.start_time,
                    'end_time': event.end_time,
                    'tone_label': event.tone_label,
                    'confidence': event.confidence
                }
                for event in context.tone_events
            ]
        
        # Format scene summaries
        if hasattr(context, 'scene_summaries') and context.scene_summaries:
            results['scene_summaries'] = [
                {
                    'scene_id': summary.scene_id,
                    'start_time': summary.start_time,
                    'end_time': summary.end_time,
                    'summary_text': summary.summary_text,
                    'tone_label': summary.tone_label,
                    'key_speakers': summary.key_speakers,
                    'confidence': summary.confidence
                }
                for summary in context.scene_summaries
            ]

        # Format video captions (VideoMAE)
        if hasattr(context, 'video_captions') and context.video_captions:
            results['video_captions'] = [
                {
                    'window_id': cap.window_id,
                    'start_time': cap.start_time,
                    'end_time': cap.end_time,
                    'caption': cap.caption,
                    'confidence': cap.confidence,
                }
                for cap in context.video_captions
            ]

        return results
    
    def _generate_mock_waveform(self) -> list:
        """Generate mock waveform data for visualization"""
        samples = 1000
        x = np.linspace(0, 4 * np.pi, samples)
        waveform = (np.sin(x) + 0.3 * np.sin(3 * x) + 0.1 * np.random.randn(samples)) * 0.5
        return waveform.tolist()

    def _build_waveform(self, audio: np.ndarray, points: int = 1000) -> list:
        """Downsample audio to a fixed number of amplitude envelope points."""
        if len(audio) == 0:
            return self._generate_mock_waveform()
        chunk = max(1, len(audio) // points)
        envelope = []
        for i in range(points):
            start = i * chunk
            end = min(start + chunk, len(audio))
            chunk_data = audio[start:end]
            peak = float(np.max(np.abs(chunk_data))) if len(chunk_data) else 0.0
            # Alternate sign so it looks like a waveform mirror
            envelope.append(peak if i % 2 == 0 else -peak)
        return envelope


# Initialize new structured monitoring system
try:
    # Import our new logging and monitoring classes
    from core.logger import Logger
    from core.monitor import PipelineMonitor
    
    # Create global logger and monitor instances
    app_logger = Logger(log_file_path=str(WEB_DIR / 'logs' / 'web_pipeline.log'))
    app_monitor = PipelineMonitor(app_logger)
    
    print("✅ New structured monitoring system initialized")
    
    def get_monitor():
        return app_monitor
        
    def get_logger():
        return app_logger
        
    MONITORING_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️  New monitoring system not available: {e}")
    print("🔧 Using fallback monitoring")
    
    class FallbackMonitor:
        def get_pipeline_metrics(self):
            return None
        def get_summary_report(self):
            return "Monitoring not available"
        def _metrics(self):
            return type('MockMetrics', (), {'get_all_metrics': lambda: {}})()
    
    get_monitor = lambda: FallbackMonitor()
    get_logger = lambda: None
    MONITORING_AVAILABLE = False

# Initialize Flask app and processor
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
processor = VideoProcessor()

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main page"""
    return send_file(str(WEB_DIR / 'index.html'))


@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files from web directory"""
    return send_from_directory(str(WEB_DIR), filename)


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Upload and analyze video with monitoring"""
    
    # Get logger for upload monitoring
    logger = get_logger() if MONITORING_AVAILABLE else None
    monitor = get_monitor() if MONITORING_AVAILABLE else None
    
    if logger:
        logger.log_info("Video upload request received")
    
    # Check if file is present
    if 'video' not in request.files:
        if logger:
            logger.log_error("No video file provided in request")
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    # Check if file is selected and valid
    if file.filename == '':
        if logger:
            logger.log_error("No file selected in upload")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        if logger:
            logger.log_error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Log upload start with file info
        filename = secure_filename(file.filename)
        if logger:
            logger.log_info(f"Starting upload: {filename}")
            
        # Calculate file size if possible
        file_size_mb = 0
        try:
            file.seek(0, 2)  # Seek to end
            file_size_bytes = file.tell()
            file_size_mb = file_size_bytes / (1024 * 1024)
            file.seek(0)  # Reset to beginning
            
            if logger:
                logger.log_metric("upload.file_size", file_size_mb, "MB")
        except:
            pass
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        video_path = processor.uploads_dir / unique_filename
        
        if logger:
            logger.log_info(f"Saving file to: {unique_filename}")
        
        # Monitor upload task
        if monitor:
            monitor.start_task("VideoUpload")
        
        file.save(str(video_path))
        
        # Log successful upload
        if logger:
            logger.log_info(f"File uploaded successfully: {file_size_mb:.1f}MB")
            logger.log_metric("upload.success", 1)
            
        if monitor:
            monitor.end_task("VideoUpload", success=True)
            if file_size_mb > 0:
                monitor.record_metric("upload.file_size_mb", file_size_mb)
        
        # Create processing task
        task_id = processor.create_task(str(video_path))
        
        if logger:
            logger.log_info(f"Processing task created: {task_id}")
        
        return jsonify({
            'task_id': task_id,
            'message': 'Video uploaded successfully, processing started'
        })
        
    except Exception as e:
        # Log upload failure
        if logger:
            logger.log_error(f"Upload failed: {str(e)}")
            logger.log_metric("upload.failure", 1)
        
        if monitor:
            monitor.end_task("VideoUpload", success=False, error_message=str(e))
        
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    """Get processing progress"""
    
    task = processor.get_task_status(task_id)
    
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'progress': task['progress'],
        'message': task['message'],
        'error': task.get('error')
    })


@app.route('/api/results/<task_id>')
def get_results(task_id):
    """Get processing results"""
    
    results = processor.get_task_results(task_id)
    
    if not results:
        task = processor.get_task_status(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        elif task['status'] == 'failed':
            return jsonify({'error': task.get('error', 'Processing failed')}), 500
        else:
            return jsonify({'error': 'Results not ready'}), 202
    
    return jsonify(results)


@app.route('/api/uploads/<task_id>/video')
def serve_video(task_id):
    """Stream back the uploaded video for a given task (used to restore player after refresh)."""
    task = processor.get_task_status(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404

    video_path = task.get('video_path')
    if not video_path or not Path(video_path).exists():
        return jsonify({'error': 'Video file not found'}), 404

    return send_file(video_path, mimetype='video/mp4', conditional=True)


@app.route('/api/results/<task_id>/db')
def download_db(task_id):
    """Download the .wiz file produced for a completed task"""
    task = processor.get_task_status(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404

    wiz_path = task.get('wiz_path')
    if not wiz_path or not Path(wiz_path).exists():
        return jsonify({'error': '.wiz file not available'}), 404

    return send_file(
        wiz_path,
        as_attachment=True,
        download_name=Path(wiz_path).name,
        mimetype='application/octet-stream'
    )


@app.route('/api/results/<task_id>/search')
def search_wiz(task_id):
    """
    Query the .wiz file for a completed task.

    Query params (all optional, at least one required):
      speaker  — speaker ID, e.g. SPEAKER_01
      topic    — keyword/phrase, e.g. "machine learning"
      emotion  — tone label, e.g. confident
      safe_cuts — any truthy value to return safe cut points
      no_blink  — combine with speaker+topic to exclude blink windows

    Returns list of matching segments with timecodes.
    """
    if not SEARCH_AVAILABLE:
        return jsonify({'error': 'Search not available'}), 503

    task = processor.get_task_status(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404

    wiz_path = task.get('wiz_path')
    if not wiz_path or not Path(wiz_path).exists():
        return jsonify({'error': '.wiz file not available — reprocess the video'}), 404

    speaker   = request.args.get('speaker')
    topic     = request.args.get('topic')
    emotion   = request.args.get('emotion')
    safe_cuts = request.args.get('safe_cuts')
    no_blink  = request.args.get('no_blink')

    # Optional time-window filter (seconds, float)
    def _parse_float(key: str):
        v = request.args.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    t_start = _parse_float('time_start')
    t_end   = _parse_float('time_end')

    engine = SearchEngine(wiz_path)

    if safe_cuts:
        results = engine.find_safe_cuts(time_start=t_start, time_end=t_end)
    elif emotion:
        results = engine.find_emotion(emotion, time_start=t_start, time_end=t_end)
    elif speaker and topic and no_blink:
        results = engine.find_person_topic_no_blink(speaker, topic, time_start=t_start, time_end=t_end)
    elif speaker and topic:
        results = engine.find_person_topic(speaker, topic, time_start=t_start, time_end=t_end)
    elif speaker or topic:
        kwargs = {}
        if speaker:
            kwargs['speaker'] = speaker
        if topic:
            kwargs['topic'] = topic
        results = engine.query(**kwargs)
    else:
        return jsonify({
            'error': 'Provide at least one of: speaker, topic, emotion, safe_cuts',
            'stats': engine.stats(),
        }), 400

    return jsonify({
        'query': dict(request.args),
        'count': len(results),
        'stats': engine.stats(),
        'results': [
            {
                'time_start': r.time_start,
                'time_end':   r.time_end,
                'duration':   round(r.duration, 3),
                'timecode':   r.timecode(engine._fps),
                'speaker':    r.speaker,
                'transcript': r.transcript,
                'emotion':    r.emotion,
                'score':      round(r.score, 3),
            }
            for r in results
        ],
    })


@app.route('/api/tasks')
def list_tasks():
    """List all tasks (for debugging)"""
    tasks = [
        {
            'id': task['id'],
            'status': task['status'],
            'progress': task['progress'],
            'message': task['message'],
            'created_at': task['created_at']
        }
        for task in processor.tasks.values()
    ]

    return jsonify({'tasks': tasks})


@app.route('/api/monitoring/tasks')
def monitoring_tasks():
    """Pipeline-specific task summary for the monitoring dashboard."""
    rows = []
    for task in processor.tasks.values():
        results = task.get('results') or {}
        meta    = results.get('video_metadata') or {}
        fps     = meta.get('fps') or 0

        # Counts straight from results
        blinks   = len(results.get('blink_events') or [])
        breaths  = len(results.get('breath_events') or [])
        scenes   = len(results.get('scene_summaries') or [])
        captions = len(results.get('video_captions') or [])

        speakers = set()
        for seg in (results.get('aligned_segments') or []):
            if seg.get('speaker_id'):
                speakers.add(seg['speaker_id'])

        duration = meta.get('duration')

        rows.append({
            'id':          task['id'],
            'video':       Path(task.get('video_path', '')).name,
            'status':      task['status'],
            'created_at':  task.get('created_at'),
            'duration_s':  round(duration, 1) if duration else None,
            'fps':         round(fps, 2) if fps else None,
            'blinks':      blinks,
            'breaths':     breaths,
            'speakers':    len(speakers),
            'scenes':      scenes,
            'captions':    captions,
            'wiz_path':    task.get('wiz_path'),
        })

    # Most recent first
    rows.sort(key=lambda r: r['created_at'] or '', reverse=True)
    return jsonify({'tasks': rows})


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413


@app.route('/api/monitoring/status')
def get_monitoring_status():
    """Get system monitoring status using new structured monitoring"""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({'error': 'Monitoring not available'}), 503
            
        monitor = get_monitor()
        
        # Get pipeline metrics
        pipeline_metrics = monitor.get_pipeline_metrics()
        
        # Get all collected metrics
        all_metrics = monitor._metrics.get_all_metrics()
        
        # Build status response
        status = {
            'status': 'active',
            'uptime_ms': int(time.time() * 1000),
            'pipeline_metrics': {
                'total_duration': pipeline_metrics.total_duration if pipeline_metrics else 0,
                'total_events_detected': pipeline_metrics.total_events_detected if pipeline_metrics else 0,
                'success_rate': pipeline_metrics.success_rate if pipeline_metrics else 100.0,
                'failure_count': pipeline_metrics.failure_count if pipeline_metrics else 0
            },
            'system_metrics': {
                'memory_usage_mb': all_metrics.get('gauge.system.memory_usage_mb', 0),
                'cpu_percent': 0,  # Could add psutil integration here
                'disk_usage_mb': 0
            },
            'performance_metrics': {
                'avg_processing_time': all_metrics.get('timer.pipeline.total_duration', 0),
                'tasks_completed': len(pipeline_metrics.task_metrics) if pipeline_metrics else 0,
                'speed_ratio': all_metrics.get('gauge.task.speed_ratio', 0)
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': f'Monitoring error: {str(e)}'}), 500

@app.route('/api/monitoring/sessions')
def get_recent_sessions():
    """Get recent processing sessions from structured monitoring"""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({'error': 'Monitoring not available'}), 503
            
        limit = request.args.get('limit', 20, type=int)
        monitor = get_monitor()
        
        # Get pipeline metrics which contains session data
        pipeline_metrics = monitor.get_pipeline_metrics()
        
        # Build sessions from task metrics
        sessions = []
        if pipeline_metrics and pipeline_metrics.task_metrics:
            # Create a session entry
            session = {
                'id': 'current_session',
                'start_time': datetime.now().isoformat(),
                'duration_ms': int(pipeline_metrics.total_duration * 1000),
                'status': 'completed' if pipeline_metrics.failure_count == 0 else 'partial_failure',
                'video_path': 'processed_video',
                'tasks_completed': len(pipeline_metrics.task_metrics),
                'tasks_failed': pipeline_metrics.failure_count,
                'events_detected': pipeline_metrics.total_events_detected,
                'success_rate': pipeline_metrics.success_rate,
                'tasks': [
                    {
                        'name': task.task_name,
                        'duration_ms': int(task.duration * 1000),
                        'status': 'completed' if task.success else 'failed',
                        'error': task.error_message if not task.success else None,
                        'events_detected': task.events_detected
                    }
                    for task in pipeline_metrics.task_metrics
                ]
            }
            sessions.append(session)
        
        return jsonify({'sessions': sessions})
        
    except Exception as e:
        return jsonify({'error': f'Monitoring error: {str(e)}'}), 500

@app.route('/api/monitoring/logs')
def get_logs():
    """Get recent log entries from structured logging system"""
    try:
        log_type = request.args.get('type', 'web')  # web or test
        lines = request.args.get('lines', 100, type=int)
        
        # Try different log files from our structured logging system
        possible_log_files = [
            Path('logs') / f'{log_type}_pipeline.log',
            Path('logs') / 'web_pipeline.log',
            Path('logs') / 'test_pipeline.log',
            Path('logs') / 'wiz_pipeline.log'
        ]
        
        logs = []
        for log_file in possible_log_files:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    file_logs = f.readlines()[-lines:]
                    logs.extend([line.strip() for line in file_logs if line.strip()])
                break
        
        # If no structured logs found, create some sample data
        if not logs:
            logs = [
                '[2026-02-21 20:05:25] [INFO] [Pipeline] Structured logging system active',
                '[2026-02-21 20:05:25] [INFO] [Pipeline] No recent processing sessions',
                '[2026-02-21 20:05:25] [INFO] [Pipeline] Ready for video analysis'
            ]
        
        return jsonify({'logs': logs})
        
    except Exception as e:
        return jsonify({'error': f'Log retrieval error: {str(e)}'}), 500

@app.route('/api/monitoring/benchmark', methods=['POST'])
def run_search_benchmark():
    """
    Run a graph-vs-SQL search benchmark and return structured results.

    Body (JSON, all optional):
        hours  — hours of synthetic footage to generate (default 0.1)
        runs   — query iterations per method (default 20)
    """
    if not BENCHMARK_AVAILABLE:
        return jsonify({'error': 'Benchmark not available'}), 503

    body  = request.get_json(silent=True) or {}
    hours = float(body.get('hours', 0.1))
    runs  = int(body.get('runs', 20))

    # Clamp to sensible limits so the request doesn't hang
    hours = max(0.05, min(hours, 2.0))
    runs  = max(5,    min(runs,  100))

    try:
        result = run_benchmark_json(hours=hours, runs=runs)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/monitoring/summary')
def get_monitoring_summary():
    """Get structured monitoring summary report"""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({'error': 'Monitoring not available'}), 503
            
        monitor = get_monitor()
        summary_report = monitor.get_summary_report()
        
        return jsonify({
            'summary_report': summary_report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Summary error: {str(e)}'}), 500

@app.route('/monitoring')
def monitoring_dashboard():
    """Serve monitoring dashboard"""
    return send_file(str(WEB_DIR / 'monitoring.html'))

@app.route('/benchmark')
def benchmark_page():
    """Serve standalone search benchmark page"""
    return send_file(str(WEB_DIR / 'benchmark.html'))

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle general exceptions"""
    print(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting WIZ Video Intelligence Web Interface...")
    print("Open your browser to http://localhost:5555")

    # Create required directories
    (WEB_DIR / 'uploads').mkdir(exist_ok=True)
    (WEB_DIR / 'results').mkdir(exist_ok=True)
    (WEB_DIR / 'logs').mkdir(exist_ok=True)

    # Run Flask app on port 5555 to avoid conflicts
    app.run(host='0.0.0.0', port=5555, debug=True, threaded=True)