#!/usr/bin/env python3
"""
Example usage of the WIZ Intelligence Pipeline.

This script demonstrates how to use the pipeline to detect blink and breath events
from a video file.
"""

import sys
import json
from pathlib import Path
from core.pipeline import Pipeline


def main():
    """Main example function."""
    if len(sys.argv) != 2:
        print("Usage: python example.py <video_path>")
        print("Example: python example.py input_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Validate input file exists
    if not Path(video_path).exists():
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)
    
    print("=" * 50)
    print("WIZ Intelligence Pipeline - Simple Detection")
    print("Blink Detection + Breath Detection")
    print("=" * 50)
    
    try:
        # Create pipeline with default configuration
        print("Initializing pipeline...")
        pipeline = Pipeline()
        
        # Run the pipeline
        print(f"Processing video: {video_path}")
        print("This may take a few moments...")
        
        context = pipeline.run(video_path)
        
        # Get results summary
        results = pipeline.get_results_summary(context)
        
        # Display results
        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)
        
        # Video information
        video_info = results['video_metadata']
        print(f"Video: {Path(video_info['path']).name}")
        print(f"Duration: {video_info['duration_s']:.2f} seconds")
        print(f"Resolution: {video_info['resolution']}")
        print(f"FPS: {video_info['fps']:.2f}")
        
        # Blink detection results
        blink_info = results['blink_detection']
        print(f"\nBlink Events Detected: {blink_info['total_events']}")
        
        if blink_info['total_events'] > 0:
            print("Blink Details:")
            for i, event in enumerate(blink_info['events'], 1):
                print(f"  {i}. Frame {event['start_frame']}-{event['end_frame']} "
                      f"({event['duration_ms']:.1f}ms, confidence: {event['confidence']:.3f})")
        
        # Breath detection results
        breath_info = results['breath_detection']
        print(f"\nBreath Events Detected: {breath_info['total_events']}")
        
        if breath_info['total_events'] > 0:
            print("Breath Details:")
            for i, event in enumerate(breath_info['events'], 1):
                print(f"  {i}. {event['start_time']:.2f}s-{event['end_time']:.2f}s "
                      f"({event['duration_ms']:.1f}ms, confidence: {event['confidence']:.3f})")
        
        # Save results to JSON file
        output_file = f"results_{Path(video_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
        
        print("\n" + "=" * 50)
        print("Pipeline execution completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Pipeline execution failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()