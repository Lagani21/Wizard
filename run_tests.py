#!/usr/bin/env python3
"""
Simple test runner for the WIZ Intelligence Pipeline.

Automatically detects available test videos and provides options for testing.
"""

import sys
import os
from pathlib import Path
import subprocess


def check_test_videos() -> list:
    """Check for available test videos."""
    test_videos_dir = Path("test_videos")
    
    if not test_videos_dir.exists():
        return []
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    video_files = [
        f for f in test_videos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    return sorted(video_files)


def run_command(command: list) -> int:
    """Run a command and return exit code."""
    try:
        result = subprocess.run(command, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Failed to run command: {e}")
        return 1


def main():
    """Main function."""
    print("🧪 WIZ Intelligence Pipeline Test Runner")
    print("=" * 50)
    
    # Check for test videos
    video_files = check_test_videos()
    
    if not video_files:
        print("📹 No test videos found!")
        print()
        print("Please add your video files to the test_videos/ directory:")
        print("  mkdir -p test_videos")
        print("  cp /path/to/your/videos/*.mp4 test_videos/")
        print()
        print("Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm")
        return 1
    
    print(f"📹 Found {len(video_files)} test videos:")
    for i, video_file in enumerate(video_files, 1):
        print(f"   {i}. {video_file.name}")
    print()
    
    # Interactive menu
    while True:
        print("Choose an option:")
        print("  1. Run quick test on a single video")
        print("  2. Run full test suite on all videos")
        print("  3. List available videos")
        print("  q. Quit")
        print()
        
        choice = input("Enter choice (1-3, q): ").strip().lower()
        
        if choice == 'q' or choice == 'quit':
            print("👋 Goodbye!")
            return 0
        
        elif choice == '1':
            # Quick test on single video
            print("\nAvailable videos:")
            for i, video_file in enumerate(video_files, 1):
                print(f"   {i}. {video_file.name}")
            
            try:
                video_choice = input(f"\nEnter video number (1-{len(video_files)}): ").strip()
                video_index = int(video_choice) - 1
                
                if 0 <= video_index < len(video_files):
                    selected_video = video_files[video_index]
                    print(f"\n🚀 Running quick test on {selected_video.name}...")
                    print("-" * 40)
                    
                    exit_code = run_command([sys.executable, "quick_test.py", str(selected_video)])
                    
                    if exit_code == 0:
                        print("\n✅ Quick test completed successfully!")
                    else:
                        print("\n❌ Quick test failed!")
                else:
                    print("❌ Invalid video number")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n⚠️  Test cancelled")
        
        elif choice == '2':
            # Full test suite
            print(f"\n🚀 Running full test suite on {len(video_files)} videos...")
            print("-" * 50)
            
            exit_code = run_command([sys.executable, "test_pipeline.py"])
            
            if exit_code == 0:
                print("\n🎉 All tests passed!")
            else:
                print("\n❌ Some tests failed. Check the output above.")
        
        elif choice == '3':
            # List videos with details
            print("\n📹 Available test videos:")
            for video_file in video_files:
                try:
                    size_mb = video_file.stat().st_size / (1024 * 1024)
                    print(f"   - {video_file.name} ({size_mb:.1f} MB)")
                except:
                    print(f"   - {video_file.name}")
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or q")
        
        print("\n" + "-" * 50)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Test runner interrupted. Goodbye!")
        sys.exit(1)