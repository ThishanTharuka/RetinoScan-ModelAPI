#!/usr/bin/env python3
"""
Startup script for RetinoScan Model API
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main startup function"""
    print("🔥 Starting RetinoScan Model API...")
    print("="*60)
    
    # Get the current directory
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    # Check if virtual environment exists
    venv_path = current_dir / "venv"
    if not venv_path.exists():
        print("📦 Virtual environment not found. Creating one...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully!")
    
    # Determine the correct python executable
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # Install/upgrade dependencies
    print("📦 Installing/updating dependencies...")
    
    # Try to upgrade pip (non-critical if it fails)
    try:
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=False)
        print("✅ Pip upgrade completed!")
    except subprocess.CalledProcessError:
        print("⚠️  Pip upgrade failed, but continuing with existing version...")
    except Exception as e:
        print(f"⚠️  Pip upgrade skipped: {e}")
    
    # Install project dependencies
    print("📦 Installing project dependencies...")
    
    # Try with regular requirements first
    try:
        subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to install from requirements.txt: {e}")
        print("📦 Trying simplified installation...")
        
        # Try with simplified requirements
        if (current_dir / "requirements-simple.txt").exists():
            try:
                subprocess.run([str(pip_exe), "install", "-r", "requirements-simple.txt"], check=True)
                print("✅ Simplified dependencies installed successfully!")
            except subprocess.CalledProcessError:
                print("❌ Failed to install simplified dependencies")
                print("📦 Trying individual package installation...")
                
                # Try installing core packages individually
                core_packages = [
                    "fastapi", "uvicorn[standard]", "python-multipart", 
                    "python-dotenv", "pydantic", "pillow", "numpy"
                ]
                
                for package in core_packages:
                    try:
                        subprocess.run([str(pip_exe), "install", package], check=True)
                        print(f"✅ Installed {package}")
                    except subprocess.CalledProcessError:
                        print(f"⚠️  Failed to install {package}")
                
                print("⚠️  Some packages may be missing. The API might not work fully.")
        else:
            print("❌ No alternative requirements file found")
    
    # Check if model file exists
    model_path = current_dir / "models" / "best_model.pth"
    if model_path.exists():
        print(f"🤖 Model file found: {model_path}")
    else:
        print(f"⚠️  Model file not found at: {model_path}")
        print("   The API will run with a dummy model for development.")
        print("   Place your trained 'best_model.pth' in the models/ folder.")
    
    # Check if .env file exists
    env_path = current_dir / ".env"
    if not env_path.exists():
        print("📝 Creating .env file from example...")
        example_path = current_dir / ".env.example"
        if example_path.exists():
            with open(example_path, 'r') as src, open(env_path, 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created. You can modify it as needed.")
        else:
            print("⚠️  .env.example not found. Creating basic .env...")
            with open(env_path, 'w') as f:
                f.write("""# RetinoScan Model API Configuration
API_HOST=0.0.0.0
API_PORT=8001
DEBUG=True
MODEL_PATH=./models/best_model.pth
MODEL_INPUT_SIZE=300
CONFIDENCE_THRESHOLD=0.5
LOG_LEVEL=INFO
""")
    
    print("\n🚀 Starting the FastAPI server...")
    print("   API will be available at: http://localhost:8001")
    print("   API documentation: http://localhost:8001/docs")
    print("   Press Ctrl+C to stop the server")
    print("="*60)
    
    # Start the server
    try:
        subprocess.run([
            str(python_exe), "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8001", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())