#!/usr/bin/env python3
"""
Simple startup script for the WIZ Video Intelligence Web Interface
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import numpy
        print("✓ Required packages found")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install with: pip install flask numpy")
        return False

def check_pipeline():
    """Check if the main pipeline is accessible"""
    try:
        sys.path.append('..')
        from core.pipeline import Pipeline
        print("✓ WIZ Intelligence Pipeline found")
        return True
    except ImportError as e:
        print(f"✗ Cannot import pipeline: {e}")
        print("Make sure you're running from the web directory and the pipeline is properly installed")
        return False

def start_server():
    """Start the Flask development server"""
    print("\n" + "="*50)
    print("🧙‍♂️ WIZ VIDEO INTELLIGENCE WEB INTERFACE")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        return False
        
    if not check_pipeline():
        return False
    
    # Create required directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("✓ Directories created")
    
    print("\n📊 Starting web server...")
    print("🌐 Interface will be available at: http://localhost:5000")
    print("📁 Upload directory: ./uploads/")
    print("📋 Results directory: ./results/")
    print("\n💡 Upload a video file to analyze it with AI!")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start Flask app
    try:
        # Import and run the app
        from app import app
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(1.5)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        return True
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False

if __name__ == '__main__':
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Please run this script from the web directory")
        print("   cd web && python run.py")
        sys.exit(1)
    
    success = start_server()
    if success:
        print("✅ Server stopped successfully")
    else:
        print("❌ Server encountered an error")
        sys.exit(1)