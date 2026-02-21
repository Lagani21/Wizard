#!/usr/bin/env python3
"""
Example usage of the WIZ Intelligence Pipeline with context summary generation.

This script demonstrates how to use the pipeline with AI-powered scene summaries
and shows how to access and interpret the generated summaries.
"""

import os
import logging
from Wizard import Pipeline, LocalLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_summary_results(results):
    """Print context summary results in a readable format."""
    if 'context_summary' not in results:
        print("No context summary results available.")
        return
    
    summary_data = results['context_summary']
    summaries = summary_data['summaries']
    
    print(f"\n{'='*60}")
    print("AI-GENERATED SCENE SUMMARIES")
    print(f"{'='*60}")
    print(f"Total scenes: {summary_data['total_summaries']}")
    
    if not summaries:
        print("No scene summaries generated.")
        return
    
    # Print each summary
    print(f"\nScene Breakdown:")
    print("-" * 60)
    
    for i, summary in enumerate(summaries, 1):
        duration = summary['end_time'] - summary['start_time']
        speakers_str = ", ".join(summary['key_speakers']) if summary['key_speakers'] else "Unknown"
        
        print(f"\n{i}. {summary['scene_id'].upper()}")
        print(f"   Time: {summary['start_time']:.1f}s - {summary['end_time']:.1f}s ({duration:.1f}s)")
        print(f"   Tone: {summary['tone_label'].title()} (confidence: {summary['confidence']:.3f})")
        print(f"   Key Speakers: {speakers_str}")
        print(f"   Summary: {summary['summary_text']}")
    
    # Print metadata if available
    if 'context_summary' in results.get('processing_metadata', {}):
        metadata = results['processing_metadata']['context_summary']
        
        print(f"\n{'='*60}")
        print("SUMMARY GENERATION DETAILS")
        print(f"{'='*60}")
        
        if 'model_info' in metadata:
            model = metadata['model_info']
            print(f"LLM Backend: {model.get('backend', 'Unknown')}")
            print(f"Model Type: {model.get('type', 'Unknown')}")
            if 'model_name' in model:
                print(f"Model Name: {model['model_name']}")
            elif 'model_path' in model:
                print(f"Model Path: {model['model_path']}")
        
        if 'scene_config' in metadata:
            config = metadata['scene_config']
            print(f"Scene Duration: {config.get('scene_duration_seconds', 'N/A')}s")
            print(f"Max Tokens: {config.get('max_tokens', 'N/A')}")
            print(f"Total Scenes: {config.get('total_scenes', 'N/A')}")
        
        if 'summary_stats' in metadata:
            stats = metadata['summary_stats']
            print(f"Successful Summaries: {stats.get('successful_summaries', 'N/A')}")
            print(f"Failed Summaries: {stats.get('failed_summaries', 'N/A')}")
            print(f"Average Summary Length: {stats.get('avg_summary_length', 0):.1f} characters")


def analyze_narrative_structure(summaries):
    """Analyze the narrative structure of the video."""
    if not summaries:
        return
    
    print(f"\n{'='*60}")
    print("NARRATIVE ANALYSIS")
    print(f"{'='*60}")
    
    # Track tone progression
    tone_progression = [s['tone_label'] for s in summaries]
    tone_counts = {}
    for tone in tone_progression:
        tone_counts[tone] = tone_counts.get(tone, 0) + 1
    
    print(f"Tone Progression: {' → '.join(tone_progression)}")
    
    # Identify tone shifts
    tone_shifts = []
    for i in range(len(tone_progression) - 1):
        if tone_progression[i] != tone_progression[i + 1]:
            tone_shifts.append(f"Scene {i+1}→{i+2}: {tone_progression[i]} → {tone_progression[i+1]}")
    
    if tone_shifts:
        print(f"\nTone Shifts ({len(tone_shifts)} detected):")
        for shift in tone_shifts:
            print(f"  • {shift}")
    else:
        print("\nNo major tone shifts detected - consistent emotional arc")
    
    # Speaker analysis
    all_speakers = set()
    for summary in summaries:
        all_speakers.update(summary['key_speakers'])
    
    if all_speakers:
        print(f"\nKey Participants: {', '.join(sorted(all_speakers))}")
        
        # Find scenes where speakers change
        speaker_changes = []
        for i in range(len(summaries) - 1):
            curr_speakers = set(summaries[i]['key_speakers'])
            next_speakers = set(summaries[i + 1]['key_speakers'])
            
            if curr_speakers != next_speakers:
                speaker_changes.append(f"Scene {i+2}: {', '.join(next_speakers) if next_speakers else 'No speakers'}")
        
        if speaker_changes:
            print(f"\nSpeaker Changes:")
            for change in speaker_changes:
                print(f"  • {change}")
    
    # Highlight interesting scenes
    high_confidence_scenes = [s for s in summaries if s['confidence'] > 0.8]
    if high_confidence_scenes:
        print(f"\nHigh-Confidence Scenes ({len(high_confidence_scenes)} scenes):")
        for scene in high_confidence_scenes:
            print(f"  • {scene['scene_id']}: {scene['tone_label'].title()} (conf: {scene['confidence']:.3f})")


def main():
    """Main example function."""
    print("WIZ Intelligence Pipeline - AI Scene Summary Example")
    print("=" * 60)
    
    # Check for test videos
    test_video_dir = "test_videos"
    if not os.path.exists(test_video_dir):
        print(f"ERROR: {test_video_dir} directory not found!")
        print("Please create the test_videos directory and add some video files.")
        print("See TEST_INSTRUCTIONS.md for more details.")
        return
    
    # Find video files
    video_files = []
    for file in os.listdir(test_video_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            video_files.append(os.path.join(test_video_dir, file))
    
    if not video_files:
        print(f"No video files found in {test_video_dir}/")
        print("Please add some video files and try again.")
        print("See TEST_INSTRUCTIONS.md for more details.")
        return
    
    # Use first video file
    video_path = video_files[0]
    print(f"Processing video: {video_path}")
    
    # Create pipeline with context summary enabled
    print("\nInitializing pipeline with AI scene summarization...")
    
    # Note: Uses mock LLM by default for testing
    # For production, specify: llm_backend="llama_cpp", model_path="/path/to/model.gguf"
    # Or: llm_backend="mlx", model_name="mlx-community/Llama-3.2-3B-Instruct-4bit"
    
    pipeline = Pipeline(
        enable_speech_processing=True,   # Required for meaningful summaries
        enable_tone_detection=True,      # Enhances summary quality
        enable_context_summary=True      # Enable AI scene summaries
    )
    
    try:
        # Run the pipeline
        print("Running pipeline...")
        print("NOTE: Using mock LLM for development. See code comments for production setup.")
        results = pipeline.run(video_path)
        
        # Print basic results
        print(f"\nPipeline completed successfully!")
        print(f"Video duration: {results['video_metadata']['duration_seconds']:.1f} seconds")
        print(f"Video resolution: {results['video_metadata']['width']}x{results['video_metadata']['height']}")
        
        # Print detection results summary
        print(f"\nProcessing Results:")
        print(f"  • Blink events: {results['blink_detection']['total_events']}")
        print(f"  • Breath events: {results['breath_detection']['total_events']}")
        
        if 'speech_processing' in results:
            speech = results['speech_processing']
            print(f"  • Transcript words: {speech['transcription']['total_words']}")
            print(f"  • Speaker segments: {speech['diarization']['total_segments']}")
            print(f"  • Unique speakers: {speech['diarization']['num_speakers']}")
        
        if 'tone_detection' in results:
            tone = results['tone_detection']
            print(f"  • Tone events: {tone['total_events']}")
        
        # Print context summary results
        print_summary_results(results)
        
        # Analyze narrative structure
        if 'context_summary' in results and results['context_summary']['summaries']:
            analyze_narrative_structure(results['context_summary']['summaries'])
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"\nERROR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your video file is valid and not corrupted")
        print("2. Check if you have enough disk space")
        print("3. Verify that all dependencies are installed (see requirements.txt)")
        print("4. For production LLM setup, ensure model files are downloaded")
        print("5. Try with a shorter video file first")
        return
    
    print(f"\n{'='*60}")
    print("CONTEXT SUMMARY FEATURES")
    print(f"{'='*60}")
    print("The context summary system provides:")
    print("• Scene-level AI summaries using local LLM")
    print("• Structured prompts with multimodal features")
    print("• Support for multiple LLM backends (mock/llama.cpp/MLX)")
    print("• Editorial-quality narrative analysis")
    print("• Tone-aware scene descriptions")
    print("• Speaker interaction analysis")
    
    print(f"\nSupported LLM backends:")
    print("• Mock: Development/testing (no model required)")
    print("• llama.cpp: GGUF quantized models (CPU optimized)")
    print("• MLX: Apple Silicon optimized models")
    
    print(f"\nExample production setup:")
    print("```python")
    print("# Using llama.cpp")
    print("pipeline = Pipeline(")
    print('    enable_context_summary=True,')
    print('    context_summary_task=ContextSummaryTask.create_default(')
    print('        llm_backend="llama_cpp",')
    print('        model_path="/path/to/model.gguf"')
    print('    )')
    print(")")
    print("")
    print("# Using Apple MLX")
    print("pipeline = Pipeline(")
    print('    enable_context_summary=True,')
    print('    context_summary_task=ContextSummaryTask.create_default(')
    print('        llm_backend="mlx",')
    print('        model_name="mlx-community/Llama-3.2-3B-Instruct-4bit"')
    print('    )')
    print(")")
    print("```")


if __name__ == "__main__":
    main()