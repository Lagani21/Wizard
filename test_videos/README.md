# Test Videos Directory

Place your test video files here for pipeline testing.

## Supported Formats
- `.mp4`
- `.avi` 
- `.mov`
- `.mkv`
- `.wmv`
- `.flv`
- `.webm`

## Usage

1. **Add your videos:**
   ```bash
   cp /path/to/your/videos/*.mp4 .
   ```

2. **Run tests:**
   ```bash
   # Interactive test runner
   python ../run_tests.py
   
   # Quick test single video
   python ../quick_test.py your_video.mp4
   
   # Full test suite  
   python ../test_pipeline.py
   ```

## Recommended Test Videos

For comprehensive testing, include:

- **Basic videos**: Clear faces, audible audio, 5-30 seconds
- **Edge cases**: Poor lighting, multiple faces, silent audio
- **Variations**: Different resolutions, frame rates, aspect ratios

The test scripts will automatically discover all video files in this directory.