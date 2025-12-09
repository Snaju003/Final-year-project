"""
Colab Diagnostic and Setup Helper
Run this to diagnose and fix common Colab setup issues
"""

import os
import sys
import subprocess
from pathlib import Path


class ColabSetupHelper:
    """Helper class for Colab setup and diagnostics"""
    
    @staticmethod
    def check_environment():
        """Check current Colab environment"""
        print("=" * 70)
        print("COLAB ENVIRONMENT DIAGNOSTICS")
        print("=" * 70)
        
        # 1. Check if in Colab
        try:
            from google.colab import drive
            print("✅ Running in Google Colab")
            in_colab = True
        except ImportError:
            print("❌ Not running in Google Colab")
            in_colab = False
        
        # 2. Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✅ GPU Available: {gpu_name} ({gpu_mem:.1f} GB)")
            else:
                print("❌ GPU not available")
        except Exception as e:
            print(f"⚠️ Could not check GPU: {e}")
        
        # 3. Check current directory
        current_dir = os.getcwd()
        print(f"📁 Current directory: {current_dir}")
        
        # 4. Check if project is available
        if os.path.exists('requirements.txt') and os.path.exists('src'):
            print("✅ Project found in current directory")
        else:
            print("❌ Project NOT found in current directory")
        
        # 5. Check internet connectivity
        print("\n🌐 Network Status:")
        try:
            # Use a simple DNS lookup instead of ping (more reliable in Colab)
            import socket
            socket.gethostbyname('github.com')
            print("✅ Can reach GitHub")
        except:
            print("❌ Cannot reach GitHub (may be blocked/offline)")
        
        return in_colab
    
    @staticmethod
    def setup_from_github():
        """Clone from GitHub"""
        print("\n" + "=" * 70)
        print("SETTING UP FROM GITHUB")
        print("=" * 70)
        
        try:
            subprocess.run(
                ['git', 'clone', 'https://github.com/Snaju003/Final-year-project.git'],
                check=True,
                capture_output=True
            )
            os.chdir('Final-year-project')
            print("✅ Repository cloned successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone from GitHub: {e}")
            print("→ Either repo is private or network is blocked")
            return False
    
    @staticmethod
    def setup_from_drive():
        """Setup from Google Drive"""
        print("\n" + "=" * 70)
        print("SETTING UP FROM GOOGLE DRIVE")
        print("=" * 70)
        
        try:
            from google.colab import drive
            import shutil
            import zipfile
            
            # Mount Drive
            print("📁 Mounting Google Drive...")
            drive.mount('/content/drive', force_remount=True)
            
            drive_path = '/content/drive/MyDrive'
            
            # Check for ZIP file
            zip_files = list(Path(drive_path).glob('*.zip'))
            if zip_files:
                for zf in zip_files:
                    if 'Final-year-project' in str(zf) or 'fyp' in str(zf).lower():
                        print(f"📦 Found ZIP: {zf.name}")
                        print(f"📥 Extracting...")
                        
                        with zipfile.ZipFile(zf, 'r') as z:
                            z.extractall('/content')
                        
                        # Navigate to extracted folder
                        for folder in Path('/content').iterdir():
                            if 'Final-year-project' in str(folder):
                                os.chdir(folder)
                                print(f"✅ Extracted to {folder}")
                                return True
            
            # Check for folder
            drive_project = Path(drive_path) / 'Final-year-project'
            if drive_project.exists():
                print(f"📁 Found folder: {drive_project}")
                print("📋 Copying from Drive...")
                shutil.copytree(
                    drive_project,
                    '/content/Final-year-project'
                )
                os.chdir('/content/Final-year-project')
                print("✅ Project copied from Drive")
                return True
            
            print("❌ No project ZIP or folder found in Drive/MyDrive/")
            print("   Expected: ~/Final-year-project.zip or ~/Final-year-project/")
            return False
        
        except Exception as e:
            print(f"❌ Failed to setup from Drive: {e}")
            return False
    
    @staticmethod
    def install_dependencies():
        """Install project dependencies"""
        print("\n" + "=" * 70)
        print("INSTALLING DEPENDENCIES")
        print("=" * 70)
        
        if not os.path.exists('requirements.txt'):
            print("⚠️ requirements.txt not found")
            print("Installing essential packages...")
            
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
                'matplotlib==3.10.7',
            ]
            
            for pkg in packages:
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-q', pkg],
                    capture_output=True
                )
            print("✅ Essential packages installed")
        else:
            print("📦 Installing from requirements.txt...")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],
                capture_output=True
            )
            print("✅ Dependencies installed")
        
        return True
    
    @staticmethod
    def verify_setup():
        """Verify the setup is complete"""
        print("\n" + "=" * 70)
        print("VERIFYING SETUP")
        print("=" * 70)
        
        checks = {
            "requirements.txt": os.path.exists('requirements.txt'),
            "src/ directory": os.path.exists('src'),
            "models/ directory": os.path.exists('models'),
            "src/inference.py": os.path.exists('src/inference.py'),
            "src/models.py": os.path.exists('src/models.py'),
        }
        
        all_ok = True
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"{status} {check}")
            if not result:
                all_ok = False
        
        if all_ok:
            print("\n" + "🎉 " * 20)
            print("SETUP COMPLETE! Ready to run inference!")
            print("🎉 " * 20)
        
        return all_ok
    
    @staticmethod
    def auto_setup():
        """Automatically setup based on environment"""
        print("\n🚀 STARTING AUTOMATIC SETUP\n")
        
        # Check environment
        in_colab = ColabSetupHelper.check_environment()
        
        # Try GitHub first
        if ColabSetupHelper.setup_from_github():
            pass  # Success
        elif in_colab:
            # Fallback to Drive
            if not ColabSetupHelper.setup_from_drive():
                print("\n❌ SETUP FAILED")
                print("Please upload your project to Google Drive/MyDrive/")
                return False
        else:
            print("\n❌ SETUP FAILED - Not in Colab and no local project")
            return False
        
        # Install dependencies
        ColabSetupHelper.install_dependencies()
        
        # Verify
        return ColabSetupHelper.verify_setup()


if __name__ == "__main__":
    # Run automatic setup
    success = ColabSetupHelper.auto_setup()
    sys.exit(0 if success else 1)
