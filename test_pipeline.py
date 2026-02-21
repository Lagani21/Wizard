#!/usr/bin/env python3
"""
Comprehensive test suite for the WIZ Intelligence Pipeline.

This script tests functionality, accuracy, and performance of the pipeline
across multiple test videos and scenarios.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import Pipeline
from core.context import PipelineContext, BlinkEvent, BreathEvent


@dataclass
class TestResult:
    """Results from a single test run."""
    video_path: str
    test_name: str
    success: bool
    execution_time_s: float
    blink_count: int
    breath_count: int
    avg_blink_confidence: float
    avg_breath_confidence: float
    error_message: Optional[str] = None
    video_duration_s: float = 0.0
    processing_fps: float = 0.0


@dataclass
class TestSummary:
    """Summary of all test results."""
    total_tests: int
    successful_tests: int
    failed_tests: int
    total_execution_time_s: float
    avg_execution_time_s: float
    total_blinks_detected: int
    total_breaths_detected: int
    results: List[TestResult]


class PipelineTester:
    """
    Test framework for the WIZ Intelligence Pipeline.
    
    Provides functionality testing, accuracy validation, and performance benchmarking.
    """
    
    def __init__(self, test_videos_dir: str = "test_videos") -> None:
        """
        Initialize the pipeline tester.
        
        Args:
            test_videos_dir: Directory containing test video files
        """
        self.test_videos_dir = Path(test_videos_dir)
        self.results: List[TestResult] = []
        
        # Create test videos directory if it doesn't exist
        self.test_videos_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = Pipeline()
        
    def discover_test_videos(self) -> List[Path]:
        """
        Discover all test video files in the test directory.
        
        Returns:
            List of video file paths
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        video_files = []
        if self.test_videos_dir.exists():
            for file_path in self.test_videos_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                    video_files.append(file_path)
        
        return sorted(video_files)
    
    def run_basic_functionality_test(self, video_path: Path) -> TestResult:
        """
        Test basic pipeline functionality on a video.
        
        Args:
            video_path: Path to test video
            
        Returns:
            TestResult with test outcomes
        """
        test_name = f"basic_functionality_{video_path.stem}"
        
        print(f"  Running basic functionality test on {video_path.name}...")
        
        start_time = time.time()
        
        try:
            # Run the pipeline
            context = self.pipeline.run(str(video_path))
            
            execution_time = time.time() - start_time
            
            # Calculate metrics
            blink_count = len(context.blink_events)
            breath_count = len(context.breath_events)
            
            avg_blink_conf = (
                np.mean([e.confidence for e in context.blink_events]) 
                if context.blink_events else 0.0
            )
            avg_breath_conf = (
                np.mean([e.confidence for e in context.breath_events]) 
                if context.breath_events else 0.0
            )
            
            video_duration = context.video_metadata.duration_seconds if context.video_metadata else 0.0
            processing_fps = (
                context.video_metadata.total_frames / execution_time 
                if context.video_metadata and execution_time > 0 else 0.0
            )
            
            result = TestResult(
                video_path=str(video_path),
                test_name=test_name,
                success=True,
                execution_time_s=execution_time,
                blink_count=blink_count,
                breath_count=breath_count,
                avg_blink_confidence=float(avg_blink_conf),
                avg_breath_confidence=float(avg_breath_conf),
                video_duration_s=video_duration,
                processing_fps=processing_fps
            )
            
            print(f"    ✓ Success: {blink_count} blinks, {breath_count} breaths in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            result = TestResult(
                video_path=str(video_path),
                test_name=test_name,
                success=False,
                execution_time_s=execution_time,
                blink_count=0,
                breath_count=0,
                avg_blink_confidence=0.0,
                avg_breath_confidence=0.0,
                error_message=error_msg
            )
            
            print(f"    ✗ Failed: {error_msg}")
        
        return result
    
    def run_accuracy_validation_test(self, video_path: Path) -> TestResult:
        """
        Run accuracy validation tests with sanity checks.
        
        Args:
            video_path: Path to test video
            
        Returns:
            TestResult with validation outcomes
        """
        test_name = f"accuracy_validation_{video_path.stem}"
        
        print(f"  Running accuracy validation on {video_path.name}...")
        
        start_time = time.time()
        
        try:
            # Run the pipeline
            context = self.pipeline.run(str(video_path))
            
            execution_time = time.time() - start_time
            
            # Validation checks
            validation_errors = []
            
            # Check blink events
            for i, blink in enumerate(context.blink_events):
                if blink.start_frame >= blink.end_frame:
                    validation_errors.append(f"Blink {i}: start_frame >= end_frame")
                if blink.duration_ms <= 0:
                    validation_errors.append(f"Blink {i}: duration_ms <= 0")
                if not 0 <= blink.confidence <= 1:
                    validation_errors.append(f"Blink {i}: confidence out of range [0,1]")
                if blink.duration_ms > 2000:  # Blinks shouldn't be > 2 seconds
                    validation_errors.append(f"Blink {i}: duration too long ({blink.duration_ms}ms)")
            
            # Check breath events
            for i, breath in enumerate(context.breath_events):
                if breath.start_time >= breath.end_time:
                    validation_errors.append(f"Breath {i}: start_time >= end_time")
                if breath.duration_ms <= 0:
                    validation_errors.append(f"Breath {i}: duration_ms <= 0")
                if not 0 <= breath.confidence <= 1:
                    validation_errors.append(f"Breath {i}: confidence out of range [0,1]")
                if breath.duration_ms > 5000:  # Breaths shouldn't be > 5 seconds
                    validation_errors.append(f"Breath {i}: duration too long ({breath.duration_ms}ms)")
            
            # Check for reasonable detection rates
            video_duration = context.video_metadata.duration_seconds if context.video_metadata else 1.0
            blink_rate_per_minute = len(context.blink_events) / (video_duration / 60)
            breath_rate_per_minute = len(context.breath_events) / (video_duration / 60)
            
            if blink_rate_per_minute > 60:  # More than 1 blink per second seems excessive
                validation_errors.append(f"Blink rate too high: {blink_rate_per_minute:.1f} per minute")
            
            if breath_rate_per_minute > 120:  # More than 2 breaths per second seems excessive
                validation_errors.append(f"Breath rate too high: {breath_rate_per_minute:.1f} per minute")
            
            # Determine success
            success = len(validation_errors) == 0
            
            blink_count = len(context.blink_events)
            breath_count = len(context.breath_events)
            
            avg_blink_conf = (
                np.mean([e.confidence for e in context.blink_events]) 
                if context.blink_events else 0.0
            )
            avg_breath_conf = (
                np.mean([e.confidence for e in context.breath_events]) 
                if context.breath_events else 0.0
            )
            
            result = TestResult(
                video_path=str(video_path),
                test_name=test_name,
                success=success,
                execution_time_s=execution_time,
                blink_count=blink_count,
                breath_count=breath_count,
                avg_blink_confidence=float(avg_blink_conf),
                avg_breath_confidence=float(avg_breath_conf),
                error_message="; ".join(validation_errors) if validation_errors else None,
                video_duration_s=video_duration,
                processing_fps=context.video_metadata.total_frames / execution_time if context.video_metadata and execution_time > 0 else 0.0
            )
            
            if success:
                print(f"    ✓ Validation passed: {blink_count} blinks, {breath_count} breaths")
            else:
                print(f"    ✗ Validation failed: {len(validation_errors)} issues found")
                for error in validation_errors[:3]:  # Show first 3 errors
                    print(f"      - {error}")
                if len(validation_errors) > 3:
                    print(f"      ... and {len(validation_errors) - 3} more")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            result = TestResult(
                video_path=str(video_path),
                test_name=test_name,
                success=False,
                execution_time_s=execution_time,
                blink_count=0,
                breath_count=0,
                avg_blink_confidence=0.0,
                avg_breath_confidence=0.0,
                error_message=error_msg
            )
            
            print(f"    ✗ Test failed: {error_msg}")
        
        return result
    
    def run_performance_benchmark(self, video_path: Path) -> TestResult:
        """
        Run performance benchmarking test.
        
        Args:
            video_path: Path to test video
            
        Returns:
            TestResult with performance metrics
        """
        test_name = f"performance_benchmark_{video_path.stem}"
        
        print(f"  Running performance benchmark on {video_path.name}...")
        
        # Run multiple times and take average
        num_runs = 3
        execution_times = []
        last_context = None
        
        try:
            for run in range(num_runs):
                start_time = time.time()
                context = self.pipeline.run(str(video_path))
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                last_context = context
                print(f"    Run {run + 1}/{num_runs}: {execution_time:.2f}s")
            
            avg_execution_time = np.mean(execution_times)
            min_execution_time = np.min(execution_times)
            max_execution_time = np.max(execution_times)
            
            blink_count = len(last_context.blink_events)
            breath_count = len(last_context.breath_events)
            
            avg_blink_conf = (
                np.mean([e.confidence for e in last_context.blink_events]) 
                if last_context.blink_events else 0.0
            )
            avg_breath_conf = (
                np.mean([e.confidence for e in last_context.breath_events]) 
                if last_context.breath_events else 0.0
            )
            
            video_duration = last_context.video_metadata.duration_seconds if last_context.video_metadata else 0.0
            processing_fps = (
                last_context.video_metadata.total_frames / avg_execution_time 
                if last_context.video_metadata and avg_execution_time > 0 else 0.0
            )
            
            result = TestResult(
                video_path=str(video_path),
                test_name=test_name,
                success=True,
                execution_time_s=avg_execution_time,
                blink_count=blink_count,
                breath_count=breath_count,
                avg_blink_confidence=float(avg_blink_conf),
                avg_breath_confidence=float(avg_breath_conf),
                video_duration_s=video_duration,
                processing_fps=processing_fps
            )
            
            print(f"    ✓ Avg: {avg_execution_time:.2f}s, Min: {min_execution_time:.2f}s, Max: {max_execution_time:.2f}s")
            print(f"    Processing rate: {processing_fps:.1f} fps")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            result = TestResult(
                video_path=str(video_path),
                test_name=test_name,
                success=False,
                execution_time_s=0.0,
                blink_count=0,
                breath_count=0,
                avg_blink_confidence=0.0,
                avg_breath_confidence=0.0,
                error_message=error_msg
            )
            
            print(f"    ✗ Benchmark failed: {error_msg}")
        
        return result
    
    def run_all_tests(self) -> TestSummary:
        """
        Run all tests on all available test videos.
        
        Returns:
            TestSummary with complete results
        """
        print("🧪 WIZ Intelligence Pipeline Test Suite")
        print("=" * 50)
        
        # Discover test videos
        video_files = self.discover_test_videos()
        
        if not video_files:
            print(f"⚠️  No test videos found in {self.test_videos_dir}")
            print(f"   Please add your video files (.mp4, .avi, .mov, etc.) to this directory")
            print(f"   Then run this test script again.")
            print()
            print(f"   Example:")
            print(f"     cp /path/to/your/videos/*.mp4 {self.test_videos_dir}/")
            print(f"     python test_pipeline.py")
            return TestSummary(
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                total_execution_time_s=0.0,
                avg_execution_time_s=0.0,
                total_blinks_detected=0,
                total_breaths_detected=0,
                results=[]
            )
        
        print(f"📹 Found {len(video_files)} test videos:")
        for video_file in video_files:
            print(f"   - {video_file.name}")
        print()
        
        # Run tests
        all_results = []
        start_time = time.time()
        
        for video_file in video_files:
            print(f"Testing: {video_file.name}")
            
            # Run basic functionality test
            result1 = self.run_basic_functionality_test(video_file)
            all_results.append(result1)
            
            # Only run other tests if basic functionality passes
            if result1.success:
                # Run accuracy validation
                result2 = self.run_accuracy_validation_test(video_file)
                all_results.append(result2)
                
                # Run performance benchmark
                result3 = self.run_performance_benchmark(video_file)
                all_results.append(result3)
            else:
                print("  ⚠️  Skipping additional tests due to basic functionality failure")
            
            print()
        
        total_execution_time = time.time() - start_time
        
        # Calculate summary
        successful_tests = sum(1 for r in all_results if r.success)
        failed_tests = len(all_results) - successful_tests
        avg_execution_time = np.mean([r.execution_time_s for r in all_results]) if all_results else 0.0
        total_blinks = sum(r.blink_count for r in all_results)
        total_breaths = sum(r.breath_count for r in all_results)
        
        summary = TestSummary(
            total_tests=len(all_results),
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            total_execution_time_s=total_execution_time,
            avg_execution_time_s=avg_execution_time,
            total_blinks_detected=total_blinks,
            total_breaths_detected=total_breaths,
            results=all_results
        )
        
        self.results = all_results
        return summary
    
    def print_summary(self, summary: TestSummary) -> None:
        """
        Print a formatted summary of test results.
        
        Args:
            summary: TestSummary to print
        """
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        if summary.total_tests == 0:
            print("No tests were run.")
            return
        
        # Overall results
        success_rate = (summary.successful_tests / summary.total_tests) * 100
        print(f"Tests Run: {summary.total_tests}")
        print(f"Successful: {summary.successful_tests} ({success_rate:.1f}%)")
        print(f"Failed: {summary.failed_tests}")
        print(f"Total Execution Time: {summary.total_execution_time_s:.2f}s")
        print(f"Average Test Time: {summary.avg_execution_time_s:.2f}s")
        print()
        
        # Detection results
        print("🔍 DETECTION RESULTS")
        print(f"Total Blinks Detected: {summary.total_blinks_detected}")
        print(f"Total Breaths Detected: {summary.total_breaths_detected}")
        print()
        
        # Per-test breakdown
        print("📋 DETAILED RESULTS")
        for result in summary.results:
            status = "✓" if result.success else "✗"
            video_name = Path(result.video_path).name
            print(f"{status} {result.test_name}")
            print(f"   Video: {video_name}")
            print(f"   Time: {result.execution_time_s:.2f}s")
            print(f"   Blinks: {result.blink_count} (avg conf: {result.avg_blink_confidence:.3f})")
            print(f"   Breaths: {result.breath_count} (avg conf: {result.avg_breath_confidence:.3f})")
            if result.processing_fps > 0:
                print(f"   Processing: {result.processing_fps:.1f} fps")
            if not result.success and result.error_message:
                print(f"   Error: {result.error_message}")
            print()
    
    def save_results(self, summary: TestSummary, output_file: str = "test_results.json") -> None:
        """
        Save test results to JSON file.
        
        Args:
            summary: TestSummary to save
            output_file: Output file path
        """
        results_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": asdict(summary),
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to {output_file}")


def main():
    """Main test function."""
    # Check if test videos directory exists and has videos
    test_videos_dir = Path("test_videos")
    
    if not test_videos_dir.exists():
        print("📁 Creating test_videos directory...")
        test_videos_dir.mkdir()
        print(f"✨ Created {test_videos_dir.absolute()}")
        print()
        print("📹 Please add your test video files to this directory:")
        print(f"   cp /path/to/your/videos/*.mp4 {test_videos_dir.absolute()}/")
        print()
        print("Then run this script again:")
        print("   python test_pipeline.py")
        return
    
    # Initialize tester and run all tests
    tester = PipelineTester()
    summary = tester.run_all_tests()
    
    # Print and save results
    tester.print_summary(summary)
    tester.save_results(summary)
    
    # Exit with appropriate code
    if summary.failed_tests > 0:
        print("🚨 Some tests failed. Check the results above.")
        sys.exit(1)
    else:
        print("🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()