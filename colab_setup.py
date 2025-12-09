"""
Google Colab Setup Script for Deepfake Detection Project
Run this at the beginning of your Colab notebook to set up the environment.
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_colab():
    """Setup environment for Google Colab"""
    
    print("=" * 60)
    print("Setting up Google Colab Environment")
    print("=" * 60)
    
    # 1. Check if running in Colab
    try:
        from google.colab import drive
        IN_COLAB = True
        print("✅ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("⚠️ Not running in Google Colab")
    
    # 2. Mount Google Drive (if in Colab)
    if IN_COLAB:
        print("\n📁 Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        project_path = '/content/drive/MyDrive/Final-year-project'
        print(f"Project path: {project_path}")
    else:
        project_path = os.getcwd()
    
    # 3. Clone repository (if not already present)
    if not os.path.exists('Final-year-project'):
        print("\n📥 Cloning repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/Snaju003/Final-year-project.git'
        ], check=True)
        project_path = 'Final-year-project'
    
    # 4. Install requirements
    print("\n📦 Installing dependencies...")
    requirements_path = os.path.join(project_path, 'requirements.txt')
    
    if os.path.exists(requirements_path):
        # Install from requirements.txt
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            '-q', '-r', requirements_path
        ], check=True)
    else:
        # Fallback: install key packages
        packages = [
            'torch==2.0.1',
            'torchvision==0.15.2',
            'timm==0.9.12',
            'facenet-pytorch==2.5.3',
            'opencv-python==4.8.1.78',
            'numpy==1.26.4',
            'tqdm==4.65.0',
            'pytorch-gradcam==0.2.1',
            'scikit-image==0.21.0',
            'scikit-learn==1.3.2',
            'Pillow==10.0.1',
        ]
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q'
        ] + packages, check=True)
    
    # 5. Verify GPU availability
    print("\n🖥️  Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️ GPU not available. Using CPU.")
    except Exception as e:
        print(f"⚠️ Could not check GPU: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    
    return project_path, IN_COLAB

if __name__ == "__main__":
    setup_colab()
