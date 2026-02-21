#!/usr/bin/env python3
"""
Example usage of the WIZ Intelligence Pipeline with emotional tone detection.

This script demonstrates how to use the pipeline with tone detection enabled
and shows how to access and interpret tone detection results.
"""

import os
import logging
import numpy as np
from Wizard import Pipeline, ToneClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_tone_results(results):
    """Print tone detection results in a readable format."""
    if 'tone_detection' not in results:
        print("No tone detection results available.")
        return
    
    tone_data = results['tone_detection']
    events = tone_data['events']
    
    print(f"\n{'='*50}")
    print("EMOTIONAL TONE DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Total tone events: {tone_data['total_events']}")
    
    if not events:
        print("No tone events detected.")
        return
    
    # Group events by tone
    tone_groups = {}
    for event in events:
        tone = event['tone_label']
        if tone not in tone_groups:
            tone_groups[tone] = []
        tone_groups[tone].append(event)
    
    # Print summary
    print(f"\nTone Distribution:")
    for tone, tone_events in tone_groups.items():
        count = len(tone_events)
        percentage = (count / len(events)) * 100
        avg_confidence = np.mean([e['confidence'] for e in tone_events])
        total_duration = sum(e['end_time'] - e['start_time'] for e in tone_events)
        print(f"  {tone.upper()}: {count} events ({percentage:.1f}%, avg confidence: {avg_confidence:.3f}, {total_duration:.1f}s)")
    
    # Print timeline
    print(f"\nTone Timeline:")
    print(f"{'Time Range':<15} {'Scene ID':<12} {'Tone':<15} {'Confidence':<10}")
    print("-" * 65)
    
    for event in events:
        time_range = f"{event['start_time']:.1f}-{event['end_time']:.1f}s"
        print(f"{time_range:<15} {event['scene_id']:<12} {event['tone_label']:<15} {event['confidence']:.3f}")
    
    # Print metadata if available
    if 'tone_detection' in results.get('processing_metadata', {}):
        metadata = results['processing_metadata']['tone_detection']
        
        print(f"\nDetection Configuration:")
        if 'classifier_info' in metadata:
            classifier = metadata['classifier_info']
            print(f"  Classifier Type: {classifier.get('type', 'unknown')}")
            print(f"  Description: {classifier.get('description', 'N/A')}")
        
        if 'window_config' in metadata:
            window_config = metadata['window_config']
            print(f"  Window Size: {window_config.get('window_size_seconds', 'N/A')}s")
            print(f"  Window Overlap: {window_config.get('window_overlap_seconds', 'N/A')}s")
            print(f"  Total Windows: {window_config.get('total_windows', 'N/A')}")
        
        if 'detection_stats' in metadata:
            stats = metadata['detection_stats']
            print(f"  Overall Confidence: {stats.get('overall_avg_confidence', 0):.3f}")
            print(f"  Dominant Tone: {stats.get('dominant_tone', 'N/A')}")
            print(f"  Tone Changes: {stats.get('tone_changes', 0)}")


def main():
    """Main example function."""
    print("WIZ Intelligence Pipeline - Tone Detection Example")
    print("=" * 50)
    
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
    
    # Create pipeline with tone detection enabled
    print("\nInitializing pipeline with tone detection...")
    pipeline = Pipeline(
        enable_speech_processing=True,   # Required for tone detection
        enable_tone_detection=True
    )
    
    try:
        # Run the pipeline
        print("Running pipeline...")
        results = pipeline.run(video_path)
        
        # Print basic results
        print(f"\nPipeline completed successfully!")
        print(f"Video duration: {results['video_metadata']['duration_seconds']:.1f} seconds")
        print(f"Video resolution: {results['video_metadata']['width']}x{results['video_metadata']['height']}")
        
        # Print detection results summary
        print(f"\nDetection Results Summary:")
        print(f"  Blink events: {results['blink_detection']['total_events']}")
        print(f"  Breath events: {results['breath_detection']['total_events']}")
        
        if 'speech_processing' in results:
            speech = results['speech_processing']
            print(f"  Transcript words: {speech['transcription']['total_words']}")
            print(f"  Speaker segments: {speech['diarization']['total_segments']}")
            print(f"  Unique speakers: {speech['diarization']['num_speakers']}")
        
        # Print tone detection results
        print_tone_results(results)
        
        # Additional analysis examples
        if 'tone_detection' in results and results['tone_detection']['events']:
            print(f"\n{'='*50}")
            print("TONE ANALYSIS EXAMPLES")
            print(f"{'='*50}")
            
            events = results['tone_detection']['events']
            
            # Find most confident predictions
            most_confident = max(events, key=lambda x: x['confidence'])
            print(f"\nMost confident prediction:")
            print(f"  {most_confident['tone_label'].upper()} at {most_confident['start_time']:.1f}-{most_confident['end_time']:.1f}s (confidence: {most_confident['confidence']:.3f})")
            
            # Find emotional peaks
            emotional_tones = ['excited', 'tense', 'confrontational']
            emotional_events = [e for e in events if e['tone_label'] in emotional_tones]
            
            if emotional_events:
                print(f"\nEmotional peaks detected:")
                for event in emotional_events[:3]:  # Show top 3
                    print(f"  {event['tone_label'].upper()} at {event['start_time']:.1f}-{event['end_time']:.1f}s (confidence: {event['confidence']:.3f})")
            
            # Tone transitions
            transitions = []
            for i in range(len(events) - 1):
                if events[i]['tone_label'] != events[i+1]['tone_label']:
                    transitions.append((events[i], events[i+1]))
            
            if transitions:
                print(f"\nTone transitions ({len(transitions)} detected):")
                for prev_event, next_event in transitions[:3]:  # Show first 3
                    print(f"  {prev_event['tone_label'].upper()} → {next_event['tone_label'].upper()} at {next_event['start_time']:.1f}s")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"\nERROR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your video file is valid and not corrupted")
        print("2. Check if you have enough disk space")
        print("3. Verify that all dependencies are installed (see requirements.txt)")
        print("4. Try with a shorter video file first")
        return
    
    print(f"\n{'='*50}")
    print("TONE DETECTION FEATURES")
    print(f"{'='*50}")
    print("The tone detection system analyzes:")
    print("• Text features: speech rate, sentiment, speaker interaction")  
    print("• Audio features: energy, spectral characteristics, prosodics")
    print("• Visual features: motion patterns, scene intensity, stability")
    print("• Supported tones: calm, tense, excited, somber, neutral, confrontational")
    print("• Classifier options: rule-based (default) or MLP")
    print("\nFor more details, see the implementation in:")
    print("• features/ - Feature extraction modules")
    print("• models/tone_classifier.py - Classification logic")
    print("• tasks/tone_detection_task.py - Pipeline integration")


if __name__ == "__main__":
    main()