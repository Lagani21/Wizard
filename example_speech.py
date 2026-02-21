#!/usr/bin/env python3
"""
Example usage of the WIZ Intelligence Pipeline with speech processing.

This script demonstrates how to use the pipeline for speech transcription
and speaker diarization along with blink and breath detection.
"""

import sys
import json
from pathlib import Path
from core.pipeline import Pipeline


def main():
    """Main example function for speech processing."""
    if len(sys.argv) != 2:
        print("Usage: python example_speech.py <video_path>")
        print("Example: python example_speech.py input_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Validate input file exists
    if not Path(video_path).exists():
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)
    
    print("=" * 60)
    print("WIZ Intelligence Pipeline - Speech Processing Example")
    print("Blink + Breath + Transcription + Speaker Diarization")
    print("=" * 60)
    
    try:
        # Create pipeline with speech processing enabled
        print("Initializing pipeline with speech processing...")
        pipeline = Pipeline(enable_speech_processing=True)
        
        # Run the pipeline
        print(f"Processing video: {video_path}")
        print("This may take several minutes for speech processing...")
        
        context = pipeline.run(video_path)
        
        # Get results summary
        results = pipeline.get_results_summary(context)
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        # Video information
        video_info = results['video_metadata']
        print(f"Video: {Path(video_info['path']).name}")
        print(f"Duration: {video_info['duration_s']:.2f} seconds")
        print(f"Resolution: {video_info['resolution']}")
        print(f"FPS: {video_info['fps']:.2f}")
        
        # Blink detection results
        blink_info = results['blink_detection']
        print(f"\n👁️  Blink Events: {blink_info['total_events']}")
        
        # Breath detection results
        breath_info = results['breath_detection']
        print(f"🫁 Breath Events: {breath_info['total_events']}")
        
        # Speech processing results
        if 'speech_processing' in results:
            speech_info = results['speech_processing']
            
            # Transcription results
            transcription = speech_info['transcription']
            print(f"\n🎤 Transcription:")
            print(f"   Words: {transcription['total_words']}")
            print(f"   Segments: {transcription['total_segments']}")
            
            # Show sample transcript
            if transcription['segments']:
                sample_text = transcription['segments'][0]['text']
                if len(sample_text) > 100:
                    sample_text = sample_text[:100] + "..."
                print(f"   Sample: \"{sample_text}\"")
            
            # Diarization results
            diarization = speech_info['diarization']
            print(f"\n👥 Speaker Diarization:")
            print(f"   Speakers: {diarization['num_speakers']}")
            print(f"   Segments: {diarization['total_segments']}")
            
            # Alignment results
            alignment = speech_info['alignment']
            print(f"\n🔗 Speaker-Transcript Alignment:")
            print(f"   Aligned segments: {alignment['total_segments']}")
            
            # Show speaker breakdown
            if alignment['segments']:
                print("\n   Speaker breakdown:")
                speaker_totals = {}
                for segment in alignment['segments']:
                    speaker_id = segment['speaker_id']
                    duration = segment['end_time'] - segment['start_time']
                    word_count = segment['word_count']
                    
                    if speaker_id not in speaker_totals:
                        speaker_totals[speaker_id] = {'duration': 0.0, 'words': 0}
                    speaker_totals[speaker_id]['duration'] += duration
                    speaker_totals[speaker_id]['words'] += word_count
                
                total_duration = sum(info['duration'] for info in speaker_totals.values())
                
                for speaker_id, info in speaker_totals.items():
                    percentage = (info['duration'] / total_duration * 100) if total_duration > 0 else 0
                    print(f"     {speaker_id}: {info['duration']:.1f}s ({percentage:.1f}%), {info['words']} words")
            
            # Show sample aligned segments
            if alignment['segments']:
                print("\n   Sample aligned segments:")
                for i, segment in enumerate(alignment['segments'][:3]):
                    text = segment['text']
                    if len(text) > 80:
                        text = text[:80] + "..."
                    print(f"     {segment['speaker_id']}: \"{text}\"")
                if len(alignment['segments']) > 3:
                    print(f"     ... and {len(alignment['segments']) - 3} more segments")
        
        # Save results to JSON file
        output_file = f"speech_results_{Path(video_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        print("\n" + "=" * 60)
        print("Speech processing pipeline completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {str(e)}")
        print("\nTo enable speech processing, install:")
        print("  pip install openai-whisper pyannote.audio torch")
        print("\nNote: You may need to accept Hugging Face model terms at:")
        print("  https://hf.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Speech processing pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()