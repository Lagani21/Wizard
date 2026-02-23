#!/usr/bin/env python3
"""
Offline setup script for WIZ Video Intelligence
Downloads and caches all required models for offline operation
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import json
import hashlib

class OfflineModelManager:
    """Manages model downloads and caching for offline operation"""
    
    def __init__(self):
        self.models_dir = Path.home() / '.wiz_models'
        self.models_dir.mkdir(exist_ok=True)
        
        # Define required models
        self.required_models = {
            'whisper': {
                'sizes': ['tiny', 'base', 'small'],  # Start with lighter models
                'description': 'Speech transcription models'
            },
            'mediaface': {
                'description': 'Face detection models (MediaPipe)'
            }
        }
    
    def setup_offline_environment(self):
        """Setup complete offline environment"""
        print("🧙‍♂️ WIZ Intelligence - Offline Setup")
        print("=" * 50)
        
        success = True
        
        # Check Python environment
        if not self.check_python_environment():
            success = False
        
        # Setup model cache directories
        if not self.setup_model_directories():
            success = False
        
        # Pre-download Whisper models
        if not self.setup_whisper_models():
            success = False
            
        # Setup MediaPipe cache
        if not self.setup_mediaface_cache():
            success = False
        
        # Create offline configuration
        if not self.create_offline_config():
            success = False
        
        if success:
            print("\n✅ Offline setup completed successfully!")
            print(f"📁 Models cached in: {self.models_dir}")
            print("🚀 You can now run the system completely offline")
        else:
            print("\n❌ Setup encountered errors")
            
        return success
    
    def check_python_environment(self):
        """Check if all required packages are available"""
        print("\n📦 Checking Python environment...")
        
        required_packages = [
            'torch',
            'whisper', 
            'mediapipe',
            'opencv-python',
            'numpy',
            'flask',
            'librosa'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✓ {package}")
            except ImportError:
                print(f"✗ {package}")
                missing.append(package)
        
        if missing:
            print(f"\n❌ Missing packages: {', '.join(missing)}")
            print("Install with: pip install " + " ".join(missing))
            return False
        
        return True
    
    def setup_model_directories(self):
        """Create model cache directories"""
        print("\n📂 Setting up model directories...")
        
        dirs = ['whisper', 'mediaface', 'llm', 'pyannote']
        
        for dir_name in dirs:
            dir_path = self.models_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"✓ {dir_path}")
        
        return True
    
    def setup_whisper_models(self):
        """Pre-download Whisper models for offline use"""
        print("\n🎤 Setting up Whisper models...")
        
        try:
            import whisper
            
            # Download smaller models first (better for offline use)
            models_to_download = ['tiny', 'base']  # Start conservative
            
            for model_name in models_to_download:
                print(f"📥 Downloading Whisper {model_name}...")
                try:
                    model = whisper.load_model(model_name, download_root=str(self.models_dir / 'whisper'))
                    print(f"✓ Whisper {model_name} cached")
                    del model  # Free memory
                except Exception as e:
                    print(f"⚠️  Could not cache {model_name}: {e}")
            
            return True
            
        except ImportError:
            print("❌ Whisper not installed")
            return False
        except Exception as e:
            print(f"❌ Error setting up Whisper: {e}")
            return False
    
    def setup_mediaface_cache(self):
        """Setup MediaPipe model cache"""
        print("\n👁️  Setting up MediaPipe models...")
        
        try:
            import mediapipe as mp
            
            # Initialize face mesh to download models
            print("📥 Initializing MediaPipe Face Mesh...")
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            print("✓ MediaPipe models cached")
            face_mesh.close()
            
            return True
            
        except ImportError:
            print("❌ MediaPipe not installed")
            return False
        except Exception as e:
            print(f"⚠️  MediaPipe setup warning: {e}")
            return True  # Non-critical error
    
    def create_offline_config(self):
        """Create configuration for offline operation"""
        print("\n⚙️  Creating offline configuration...")
        
        config = {
            'offline_mode': True,
            'models': {
                'whisper_model_path': str(self.models_dir / 'whisper'),
                'preferred_whisper_model': 'base',
                'enable_speech_processing': True,
                'enable_tone_detection': True,
                'enable_context_summary': True,
                'llm_backend': 'mock'  # Use mock LLM for offline operation
            },
            'cache_paths': {
                'models_dir': str(self.models_dir),
                'whisper_cache': str(self.models_dir / 'whisper'),
                'mediaface_cache': str(self.models_dir / 'mediaface')
            },
            'created_at': str(os.path.getctime)
        }
        
        config_file = self.models_dir / 'offline_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved to {config_file}")
        return True
    
    def get_offline_config(self):
        """Load offline configuration"""
        config_file = self.models_dir / 'offline_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return None
    
    def verify_offline_setup(self):
        """Verify that offline setup is complete"""
        print("\n🔍 Verifying offline setup...")
        
        config = self.get_offline_config()
        if not config:
            print("❌ No offline configuration found")
            return False
        
        # Check if Whisper models exist
        whisper_path = Path(config['cache_paths']['whisper_cache'])
        if not any(whisper_path.glob('*.pt')):
            print("❌ No Whisper models found")
            return False
        
        print("✅ Offline setup verified")
        return True


class DesktopLauncher:
    """Alternative desktop app approach for offline use"""
    
    def __init__(self):
        self.app_dir = Path(__file__).parent
    
    def create_desktop_app(self):
        """Create a simple desktop application launcher"""
        print("\n🖥️  Creating desktop application...")
        
        # Create a simple tkinter-based file dialog launcher
        desktop_app_code = '''#!/usr/bin/env python3
"""
Desktop launcher for WIZ Video Intelligence
Simple drag-and-drop interface for offline video analysis
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sys
import os
from pathlib import Path

# Add parent directory to path to import pipeline
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.pipeline import Pipeline
except ImportError:
    print("Error: Cannot import WIZ Intelligence Pipeline")
    sys.exit(1)

class WizDesktopApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WIZ Video Intelligence")
        self.root.geometry("600x500")
        self.root.configure(bg='#2c3e50')
        
        self.pipeline = None
        self.current_video = None
        self.results = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="🧙‍♂️ WIZ Video Intelligence", 
                         font=("Arial", 24, "bold"), 
                         bg='#2c3e50', fg='white')
        header.pack(pady=20)
        
        # Subtitle
        subtitle = tk.Label(self.root, text="Offline AI Video Analysis", 
                           font=("Arial", 12), 
                           bg='#2c3e50', fg='#ecf0f1')
        subtitle.pack(pady=(0, 30))
        
        # File selection frame
        file_frame = tk.Frame(self.root, bg='#2c3e50')
        file_frame.pack(pady=20, padx=40, fill='x')
        
        self.file_label = tk.Label(file_frame, text="No video selected", 
                                  font=("Arial", 11), 
                                  bg='#34495e', fg='white', 
                                  relief='sunken', padx=20, pady=10)
        self.file_label.pack(fill='x', pady=(0, 10))
        
        select_btn = tk.Button(file_frame, text="📁 Select Video File", 
                              command=self.select_video,
                              font=("Arial", 12, "bold"),
                              bg='#3498db', fg='white', 
                              relief='flat', padx=20, pady=10)
        select_btn.pack()
        
        # Analysis controls
        controls_frame = tk.Frame(self.root, bg='#2c3e50')
        controls_frame.pack(pady=20, padx=40, fill='x')
        
        self.analyze_btn = tk.Button(controls_frame, text="🚀 Analyze Video", 
                                    command=self.start_analysis,
                                    font=("Arial", 12, "bold"),
                                    bg='#27ae60', fg='white',
                                    relief='flat', padx=20, pady=15,
                                    state='disabled')
        self.analyze_btn.pack()
        
        # Progress bar
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready to analyze")
        
        progress_label = tk.Label(self.root, textvariable=self.progress_var,
                                 font=("Arial", 10),
                                 bg='#2c3e50', fg='#ecf0f1')
        progress_label.pack(pady=(20, 5))
        
        self.progress_bar = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress_bar.pack(pady=10, padx=40, fill='x')
        
        # Results area
        results_frame = tk.Frame(self.root, bg='#2c3e50')
        results_frame.pack(pady=20, padx=40, fill='both', expand=True)
        
        results_label = tk.Label(results_frame, text="Analysis Results",
                               font=("Arial", 12, "bold"),
                               bg='#2c3e50', fg='white')
        results_label.pack(anchor='w')
        
        self.results_text = tk.Text(results_frame, height=10, width=60,
                                   font=("Courier", 9),
                                   bg='#34495e', fg='#ecf0f1',
                                   relief='sunken', padx=10, pady=10)
        self.results_text.pack(fill='both', expand=True, pady=(10, 0))
        
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
    
    def select_video(self):
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.webm'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Select Video File',
            filetypes=filetypes
        )
        
        if filename:
            self.current_video = filename
            self.file_label.config(text=f"Selected: {Path(filename).name}")
            self.analyze_btn.config(state='normal')
    
    def start_analysis(self):
        if not self.current_video:
            return
        
        self.analyze_btn.config(state='disabled')
        self.progress_bar.start()
        self.progress_var.set("Analyzing video...")
        
        # Run analysis in background thread
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        try:
            # Create pipeline with offline-optimized settings
            self.pipeline = Pipeline(
                enable_speech_processing=True,
                enable_tone_detection=True,
                enable_context_summary=True
            )
            
            # Run analysis
            context = self.pipeline.run(self.current_video)
            
            # Format results
            self.format_results(context)
            
        except Exception as e:
            self.show_error(f"Analysis failed: {str(e)}")
        finally:
            self.root.after(0, self.analysis_complete)
    
    def analysis_complete(self):
        self.progress_bar.stop()
        self.progress_var.set("Analysis complete!")
        self.analyze_btn.config(state='normal')
    
    def format_results(self, context):
        results_text = "📊 WIZ VIDEO ANALYSIS RESULTS\\n"
        results_text += "=" * 50 + "\\n\\n"
        
        # Basic stats
        blink_count = len(getattr(context, 'blink_events', []))
        breath_count = len(getattr(context, 'breath_events', []))
        
        results_text += f"👁️  Blinks detected: {blink_count}\\n"
        results_text += f"🫁 Breaths detected: {breath_count}\\n"
        
        # Speech processing results
        if hasattr(context, 'aligned_segments') and context.aligned_segments:
            speakers = set(seg.speaker_id for seg in context.aligned_segments)
            results_text += f"🎤 Speakers detected: {len(speakers)}\\n"
            results_text += f"📝 Transcript segments: {len(context.aligned_segments)}\\n"
            
            results_text += "\\n🗣️  TRANSCRIPT:\\n"
            results_text += "-" * 30 + "\\n"
            
            for i, segment in enumerate(context.aligned_segments[:5]):  # Show first 5
                results_text += f"{segment.speaker_id}: {segment.text[:100]}...\\n"
            
            if len(context.aligned_segments) > 5:
                results_text += f"... and {len(context.aligned_segments) - 5} more segments\\n"
        
        # Tone detection results
        if hasattr(context, 'tone_events') and context.tone_events:
            results_text += "\\n😊 EMOTIONAL TONE ANALYSIS:\\n"
            results_text += "-" * 30 + "\\n"
            
            tone_counts = {}
            for event in context.tone_events:
                tone_counts[event.tone_label] = tone_counts.get(event.tone_label, 0) + 1
            
            for tone, count in sorted(tone_counts.items()):
                results_text += f"{tone.capitalize()}: {count} segments\\n"
        
        # Scene summaries
        if hasattr(context, 'scene_summaries') and context.scene_summaries:
            results_text += "\\n📽️  SCENE SUMMARIES:\\n"
            results_text += "-" * 30 + "\\n"
            
            for summary in context.scene_summaries[:3]:  # Show first 3
                results_text += f"Scene {summary.scene_id} ({summary.tone_label}): {summary.summary_text[:100]}...\\n\\n"
        
        # Update UI from main thread
        self.root.after(0, lambda: self.update_results_text(results_text))
    
    def update_results_text(self, text):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
    
    def show_error(self, message):
        self.root.after(0, lambda: messagebox.showerror("Error", message))
    
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    app = WizDesktopApp()
    app.run()
'''
        
        desktop_file = self.app_dir / 'desktop_app.py'
        with open(desktop_file, 'w') as f:
            f.write(desktop_app_code)
        
        # Make executable
        os.chmod(desktop_file, 0o755)
        
        print(f"✓ Desktop app created: {desktop_file}")
        return True


def main():
    """Main setup function"""
    print("🧙‍♂️ WIZ Video Intelligence - Offline Setup")
    print("Choose your preferred offline setup:")
    print("1. Web Interface (Flask server)")
    print("2. Desktop Application (Tkinter GUI)")
    print("3. Both")
    print("4. Just setup models")
    
    choice = input("Enter choice (1-4): ").strip()
    
    # Setup model manager
    model_manager = OfflineModelManager()
    
    if choice in ['1', '2', '3', '4']:
        print(f"\\n📦 Setting up offline models...")
        if not model_manager.setup_offline_environment():
            print("❌ Model setup failed")
            return False
    
    if choice in ['2', '3']:
        launcher = DesktopLauncher()
        if not launcher.create_desktop_app():
            print("❌ Desktop app setup failed")
            return False
    
    print("\\n🎉 Setup completed!")
    
    if choice == '1':
        print("🌐 Run: python web/run.py")
    elif choice == '2':
        print("🖥️  Run: python web/desktop_app.py")
    elif choice == '3':
        print("🌐 Web: python web/run.py")
        print("🖥️  Desktop: python web/desktop_app.py")
    
    print("\\n💡 All processing happens locally - no internet required!")
    return True

if __name__ == '__main__':
    main()