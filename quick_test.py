#!/usr/bin/env python3
"""
Quick test script for rapid development testing.

Use this for fast iteration when making changes to the pipeline.
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import Pipeline


def quick_test(video_path: str) -> None:
    """
    Run a quick test on a single video file.
    
    Args:
        video_path: Path to video file to test
    """
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"🧪 Quick Test: {Path(video_path).name}")
    print("=" * 40)
    
    try:
        # Initialize pipeline
        pipeline = Pipeline()
        
        # Run pipeline
        print("⚡ Running pipeline...")
        start_time = time.time()
        context = pipeline.run(video_path)
        execution_time = time.time() - start_time
        
        # Print results
        print(f"✅ Pipeline completed in {execution_time:.2f}s")
        print()
        
        # Video info
        if context.video_metadata:
            print(f"📹 Video: {context.video_metadata.width}x{context.video_metadata.height}")
            print(f"📊 Duration: {context.video_metadata.duration_seconds:.2f}s")
            print(f"🎬 FPS: {context.video_metadata.fps:.2f}")
            processing_fps = context.video_metadata.total_frames / execution_time if execution_time > 0 else 0
            print(f"⚡ Processing: {processing_fps:.1f} fps")
            print()
        
        # Detection results
        print(f"👁️  Blinks: {len(context.blink_events)}")
        if context.blink_events:
            avg_blink_conf = sum(e.confidence for e in context.blink_events) / len(context.blink_events)
            avg_blink_duration = sum(e.duration_ms for e in context.blink_events) / len(context.blink_events)
            print(f"   Average confidence: {avg_blink_conf:.3f}")
            print(f"   Average duration: {avg_blink_duration:.1f}ms")
            
            # Show first few blinks
            print("   Recent blinks:")
            for i, blink in enumerate(context.blink_events[:3]):
                print(f"     {i+1}. Frame {blink.start_frame}-{blink.end_frame} ({blink.duration_ms:.1f}ms, conf: {blink.confidence:.3f})")
            if len(context.blink_events) > 3:
                print(f"     ... and {len(context.blink_events) - 3} more")
        
        print()
        print(f"🫁 Breaths: {len(context.breath_events)}")
        if context.breath_events:
            avg_breath_conf = sum(e.confidence for e in context.breath_events) / len(context.breath_events)
            avg_breath_duration = sum(e.duration_ms for e in context.breath_events) / len(context.breath_events)
            print(f"   Average confidence: {avg_breath_conf:.3f}")
            print(f"   Average duration: {avg_breath_duration:.1f}ms")
            
            # Show first few breaths
            print("   Recent breaths:")
            for i, breath in enumerate(context.breath_events[:3]):
                print(f"     {i+1}. {breath.start_time:.2f}s-{breath.end_time:.2f}s ({breath.duration_ms:.1f}ms, conf: {breath.confidence:.3f})")
            if len(context.breath_events) > 3:
                print(f"     ... and {len(context.breath_events) - 3} more")
        
        print()
        print("🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {type(e).__name__}: {str(e)}")
        import traceback
        print("\n🔍 Full error traceback:")
        traceback.print_exc()


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Quick Test Script for WIZ Intelligence Pipeline")
        print()
        print("Usage: python quick_test.py <video_path>")
        print()
        print("Examples:")
        print("  python quick_test.py test_videos/your_video.mp4")
        print("  python quick_test.py /path/to/your/video.mov")
        print()
        
        # Show available test videos if test_videos directory exists
        test_videos_dir = Path("test_videos")
        if test_videos_dir.exists():
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            video_files = [
                f for f in test_videos_dir.iterdir()
                if f.is_file() and f.suffix.lower() in video_extensions
            ]
            
            if video_files:
                print("Available test videos:")
                for video_file in sorted(video_files):
                    print(f"  python quick_test.py {video_file}")
                print()
        
        sys.exit(1)
    
    video_path = sys.argv[1]
    quick_test(video_path)


if __name__ == "__main__":
    main()