# Testing the WIZ Intelligence Pipeline

This document explains how to test the pipeline with your own videos.

## Quick Start

1. **Add your test videos** to the `test_videos/` directory:
   ```bash
   mkdir test_videos
   # Copy your video files here
   cp /path/to/your/videos/*.mp4 test_videos/
   ```

2. **Run a quick test** on a single video:
   ```bash
   python quick_test.py test_videos/your_video.mp4
   ```

3. **Run the full test suite** on all videos:
   ```bash
   python test_pipeline.py
   ```

## Test Scripts

### `quick_test.py` - Fast Development Testing
- **Purpose**: Quick validation during development
- **Usage**: `python quick_test.py <video_path>`
- **Output**: Immediate results with detection counts and basic stats
- **Best for**: Rapid iteration when making code changes

### `test_pipeline.py` - Comprehensive Testing
- **Purpose**: Full validation with accuracy checks and performance benchmarks
- **Usage**: `python test_pipeline.py` (automatically finds all videos in `test_videos/`)
- **Output**: Detailed report with validation results and JSON export
- **Best for**: Full validation before releases

## Test Video Requirements

### Supported Formats
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`

### Recommended Test Videos
For comprehensive testing, include videos with:

1. **Basic functionality videos**:
   - Clear face visibility (for blink detection)
   - Audible speech/breathing (for breath detection)
   - Duration: 5-30 seconds
   - Good lighting conditions

2. **Edge case videos**:
   - Poor lighting conditions
   - Multiple faces in frame
   - No clear face visible
   - Silent audio
   - Very short videos (< 3 seconds)
   - Very long videos (> 1 minute)

3. **Quality variations**:
   - Different resolutions (480p, 720p, 1080p)
   - Different frame rates (24fps, 30fps, 60fps)
   - Different aspect ratios

## Test Output

### Quick Test Output
```
🧪 Quick Test: sample_video.mp4
========================================
⚡ Running pipeline...
✅ Pipeline completed in 2.34s

📹 Video: 1920x1080
📊 Duration: 15.30s
🎬 FPS: 30.00
⚡ Processing: 196.2 fps

👁️  Blinks: 8
   Average confidence: 0.876
   Average duration: 145.2ms
   Recent blinks:
     1. Frame 45-52 (233.3ms, conf: 0.892)
     2. Frame 89-95 (200.0ms, conf: 0.834)
     3. Frame 156-163 (233.3ms, conf: 0.912)

🫁 Breaths: 12
   Average confidence: 0.634
   Average duration: 412.3ms
   Recent breaths:
     1. 1.23s-1.67s (440.0ms, conf: 0.692)
     2. 3.45s-3.89s (440.0ms, conf: 0.578)
     3. 6.12s-6.48s (360.0ms, conf: 0.723)

🎉 Test completed successfully!
```

### Full Test Suite Output
```
🧪 WIZ Intelligence Pipeline Test Suite
==================================================
📹 Found 3 test videos:
   - sample1.mp4
   - sample2.mov  
   - edge_case.avi

Testing: sample1.mp4
  Running basic functionality test...
    ✓ Success: 8 blinks, 12 breaths in 2.34s
  Running accuracy validation...
    ✓ Validation passed: 8 blinks, 12 breaths
  Running performance benchmark...
    Run 1/3: 2.31s
    Run 2/3: 2.37s
    Run 3/3: 2.29s
    ✓ Avg: 2.32s, Min: 2.29s, Max: 2.37s
    Processing rate: 196.8 fps

📊 TEST SUMMARY
==================================================
Tests Run: 9
Successful: 9 (100.0%)
Failed: 0
Total Execution Time: 45.67s
Average Test Time: 5.07s

🔍 DETECTION RESULTS
Total Blinks Detected: 24
Total Breaths Detected: 31

💾 Results saved to test_results.json
🎉 All tests passed!
```

## Interpreting Results

### Success Criteria
- ✅ **Basic Functionality**: Pipeline runs without errors
- ✅ **Accuracy Validation**: Event timing and confidence values are reasonable
- ✅ **Performance**: Processing speed is acceptable

### Common Issues
- ❌ **No face detected**: Video may have poor lighting or no visible faces
- ❌ **No audio**: Video may be silent or audio extraction failed  
- ❌ **High error rate**: Confidence thresholds may need adjustment
- ❌ **Slow processing**: Large videos or system performance issues

### Validation Checks
The test suite automatically validates:
- Event timing consistency (start < end)
- Confidence values in valid range [0, 1]
- Reasonable detection rates
- Event durations within expected ranges

## Continuous Testing Workflow

1. **Before making changes**: Run `python test_pipeline.py` to establish baseline
2. **During development**: Use `python quick_test.py <video>` for fast feedback
3. **After changes**: Run full test suite to ensure no regressions
4. **Before release**: Verify all tests pass with your complete video set

## Adding New Test Videos

Simply copy videos to the `test_videos/` directory:
```bash
# Add videos from different sources
cp ~/Downloads/meeting_video.mp4 test_videos/
cp ~/Desktop/interview.mov test_videos/
cp /path/to/edge_cases/*.avi test_videos/
```

The test scripts will automatically discover and test all supported video files.