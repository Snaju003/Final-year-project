"""
Dataset Cleaning Script for Deepfake Detection - OPTIMIZED
Removes noisy images from Dataset 1:
1. Images without proper faces (hands, objects, etc.)
2. Low quality/corrupt images
3. Potential misclassified images

Uses face detection to validate that images contain actual faces.
Optimized with batch processing and parallel execution.
"""

import os
import sys
from pathlib import Path
import shutil
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

# Fix Unicode encoding for Windows PowerShell
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

# Try to import face detection libraries
try:
    from facenet_pytorch import MTCNN
    USE_MTCNN = True
except ImportError:
    USE_MTCNN = False
    print("⚠️ facenet-pytorch not installed. Using OpenCV face detection.")

# ============ CONFIGURATION ============
DATA_ROOT = Path(r"X:\Final-year-project-data\data\dataset 1")
TRASH_DIR = Path(r"X:\Final-year-project-data\data\dataset 1_trash")  # Where rejected images go

# Detection settings
MIN_FACE_SIZE = 40  # Minimum face size in pixels
FACE_CONFIDENCE_THRESHOLD = 0.9  # MTCNN confidence threshold
MIN_IMAGE_SIZE = 64  # Minimum image dimension
MAX_IMAGE_SIZE = 4096  # Maximum image dimension (likely corrupt if larger)

# Quality thresholds
MIN_BRIGHTNESS = 20  # Images darker than this are too dark
MAX_BRIGHTNESS = 240  # Images brighter than this are overexposed
MIN_VARIANCE = 100  # Images with variance below this are likely blank/solid color
BLUR_THRESHOLD = 50  # Laplacian variance below this = too blurry

# Processing settings - OPTIMIZED
BATCH_SIZE = 64  # Batch size for MTCNN
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)  # Leave 2 cores free
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# What to do with bad images
MODE = 'move'  # 'move' = move to trash, 'delete' = permanently delete, 'report' = just report


class DatasetCleaner:
    """Clean dataset by removing noisy/invalid images - OPTIMIZED."""
    
    def __init__(self, use_gpu=True):
        self.device = DEVICE if use_gpu else 'cpu'
        self.stats = defaultdict(int)
        
        # Initialize face detector
        if USE_MTCNN:
            print(f"🔧 Initializing MTCNN face detector on {self.device}...")
            self.face_detector = MTCNN(
                image_size=160,
                margin=20,
                min_face_size=MIN_FACE_SIZE,
                thresholds=[0.6, 0.7, 0.7],  # Detection thresholds
                factor=0.709,
                post_process=False,
                device=self.device,
                keep_all=True  # Detect all faces
            )
        else:
            # Fallback to OpenCV Haar Cascade
            print("🔧 Initializing OpenCV Haar Cascade face detector...")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
    
    def check_image_fast(self, img_path):
        """Fast combined check: loadable + quality. Returns (is_valid, reason, cv_img or None)."""
        try:
            # Use cv2 directly - faster than PIL for validation
            img = cv2.imread(str(img_path))
            if img is None:
                return False, "corrupt", None
            
            h, w = img.shape[:2]
            if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                return False, "too_small", None
            if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
                return False, "too_large", None
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < MIN_BRIGHTNESS:
                return False, "too_dark", None
            if mean_brightness > MAX_BRIGHTNESS:
                return False, "too_bright", None
            
            # Check variance (detect blank/solid images)
            variance = np.var(gray)
            if variance < MIN_VARIANCE:
                return False, "low_variance", None
            
            # Check blur using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < BLUR_THRESHOLD:
                return False, "too_blurry", None
            
            return True, "ok", img
        except Exception as e:
            return False, "corrupt", None
    
    def detect_faces_batch_mtcnn(self, images_pil):
        """Detect faces in a batch of PIL images using MTCNN."""
        try:
            # Batch detection
            boxes_batch, probs_batch = self.face_detector.detect(images_pil)
            
            results = []
            for boxes, probs in zip(boxes_batch, probs_batch):
                if boxes is None or len(boxes) == 0:
                    results.append((False, "no_face"))
                else:
                    max_conf = max(probs) if probs is not None else 0
                    if max_conf < FACE_CONFIDENCE_THRESHOLD:
                        results.append((False, "low_confidence"))
                    else:
                        results.append((True, "ok"))
            return results
        except Exception as e:
            # Fallback to individual detection on error
            return [(False, f"error: {str(e)}")] * len(images_pil)
    
    def detect_face_opencv(self, img_cv):
        """Detect face using OpenCV Haar Cascade (fallback) - takes cv2 image."""
        try:
            if img_cv is None:
                return False, "unreadable"
            
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )
            
            if len(faces) == 0:
                return False, "no_face"
            
            return True, "ok"
        except Exception as e:
            return False, f"error: {str(e)}"
    
    def handle_bad_image(self, img_path, reason, category):
        """Handle a bad image based on MODE setting."""
        if MODE == 'report':
            return  # Just counting stats, don't move/delete
        
        elif MODE == 'move':
            # Move to trash folder organized by reason
            trash_category_dir = TRASH_DIR / category / reason
            trash_category_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = trash_category_dir / img_path.name
            
            # Handle duplicate names
            if dest_path.exists():
                base = dest_path.stem
                ext = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = trash_category_dir / f"{base}_{counter}{ext}"
                    counter += 1
            
            shutil.move(str(img_path), str(dest_path))
        
        elif MODE == 'delete':
            os.remove(str(img_path))
    
    def process_batch_quality(self, file_batch):
        """Process a batch of files for quality checks in parallel."""
        results = []
        for img_path in file_batch:
            is_valid, reason, img_cv = self.check_image_fast(img_path)
            results.append((img_path, is_valid, reason, img_cv))
        return results
    
    def clean_folder(self, folder_path, category):
        """Clean a folder (Real or Fake) of noisy images - OPTIMIZED."""
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_path}")
            return
        
        # Get all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        for ext in extensions:
            files.extend(folder_path.glob(ext))
        
        files = list(files)  # Convert to list for indexing
        
        if not files:
            print(f"⚠️ No images found in {folder_path}")
            return
        
        print(f"\n📂 Processing {category}: {len(files):,} images")
        print(f"   Using {NUM_WORKERS} workers, batch size {BATCH_SIZE}")
        
        kept = 0
        removed = 0
        reasons = defaultdict(int)
        
        # Process in batches
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
        pbar = tqdm(total=len(files), desc=f"Cleaning {category}", ncols=100)
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(files))
            batch_files = files[start_idx:end_idx]
            
            # Step 1: Parallel quality checks using ThreadPoolExecutor
            quality_results = []
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Split batch into chunks for parallel processing
                chunk_size = max(1, len(batch_files) // NUM_WORKERS)
                chunks = [batch_files[i:i+chunk_size] for i in range(0, len(batch_files), chunk_size)]
                futures = [executor.submit(self.process_batch_quality, chunk) for chunk in chunks]
                
                for future in as_completed(futures):
                    quality_results.extend(future.result())
            
            # Separate valid and invalid from quality checks
            valid_for_face = []  # (img_path, img_cv) tuples
            
            for img_path, is_valid, reason, img_cv in quality_results:
                if not is_valid:
                    removed += 1
                    reasons[reason] += 1
                    self.stats[f'{category}_removed'] += 1
                    self.stats[f'{category}_{reason}'] += 1
                    self.handle_bad_image(img_path, reason, category)
                else:
                    valid_for_face.append((img_path, img_cv))
            
            # Step 2: Face detection on valid images
            if valid_for_face:
                if USE_MTCNN:
                    # Batch face detection with MTCNN
                    pil_images = []
                    for img_path, img_cv in valid_for_face:
                        try:
                            # Convert BGR to RGB and then to PIL
                            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                            pil_images.append(Image.fromarray(img_rgb))
                        except:
                            pil_images.append(None)
                    
                    # Filter out None images
                    valid_indices = [i for i, img in enumerate(pil_images) if img is not None]
                    valid_pil = [pil_images[i] for i in valid_indices]
                    
                    if valid_pil:
                        face_results = self.detect_faces_batch_mtcnn(valid_pil)
                        
                        # Map results back
                        result_idx = 0
                        for i, (img_path, img_cv) in enumerate(valid_for_face):
                            if i in valid_indices:
                                has_face, reason = face_results[result_idx]
                                result_idx += 1
                            else:
                                has_face, reason = False, "corrupt"
                            
                            if has_face:
                                kept += 1
                                self.stats[f'{category}_kept'] += 1
                            else:
                                removed += 1
                                reasons[reason] += 1
                                self.stats[f'{category}_removed'] += 1
                                self.stats[f'{category}_{reason}'] += 1
                                self.handle_bad_image(img_path, reason, category)
                    else:
                        # All images failed to convert
                        for img_path, _ in valid_for_face:
                            removed += 1
                            reasons["corrupt"] += 1
                            self.stats[f'{category}_removed'] += 1
                            self.stats[f'{category}_corrupt'] += 1
                            self.handle_bad_image(img_path, "corrupt", category)
                else:
                    # OpenCV face detection (not batched)
                    for img_path, img_cv in valid_for_face:
                        has_face, reason = self.detect_face_opencv(img_cv)
                        
                        if has_face:
                            kept += 1
                            self.stats[f'{category}_kept'] += 1
                        else:
                            removed += 1
                            reasons[reason] += 1
                            self.stats[f'{category}_removed'] += 1
                            self.stats[f'{category}_{reason}'] += 1
                            self.handle_bad_image(img_path, reason, category)
            
            # Update progress bar
            pbar.update(len(batch_files))
            pbar.set_postfix({
                'kept': kept,
                'removed': removed
            })
        
        pbar.close()
        
        # Print summary for this folder
        print(f"\n   ✅ Kept: {kept:,} images")
        print(f"   ❌ Removed: {removed:,} images")
        if reasons:
            print(f"   📊 Removal reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"      - {reason}: {count:,}")
    
    def clean_dataset(self):
        """Clean the entire dataset."""
        print("=" * 60)
        print("🧹 DATASET CLEANING TOOL (OPTIMIZED)")
        print("=" * 60)
        print(f"\n📁 Dataset: {DATA_ROOT}")
        print(f"📁 Trash:   {TRASH_DIR}")
        print(f"🔧 Mode:    {MODE}")
        print(f"🖥️ Device:  {self.device}")
        print(f"👤 Face detector: {'MTCNN (batch)' if USE_MTCNN else 'OpenCV Haar Cascade'}")
        
        print(f"\n⚙️ Settings:")
        print(f"   Min face size: {MIN_FACE_SIZE}px")
        print(f"   Face confidence: {FACE_CONFIDENCE_THRESHOLD}")
        print(f"   Min image size: {MIN_IMAGE_SIZE}px")
        print(f"   Blur threshold: {BLUR_THRESHOLD}")
        print(f"\n⚡ Optimization:")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Workers: {NUM_WORKERS}")
        
        # Create trash directory if moving
        if MODE == 'move':
            TRASH_DIR.mkdir(parents=True, exist_ok=True)
        
        # Clean Real folder
        real_folder = DATA_ROOT / 'Real'
        self.clean_folder(real_folder, 'Real')
        
        # Clean Fake folder
        fake_folder = DATA_ROOT / 'Fake'
        self.clean_folder(fake_folder, 'Fake')
        
        # Print final summary
        self.print_summary()
    
    def print_summary(self):
        """Print final cleaning summary."""
        print("\n" + "=" * 60)
        print("📊 CLEANING SUMMARY")
        print("=" * 60)
        
        real_kept = self.stats.get('Real_kept', 0)
        real_removed = self.stats.get('Real_removed', 0)
        fake_kept = self.stats.get('Fake_kept', 0)
        fake_removed = self.stats.get('Fake_removed', 0)
        
        total_kept = real_kept + fake_kept
        total_removed = real_removed + fake_removed
        total = total_kept + total_removed
        
        print(f"\n{'Category':<10} {'Kept':>10} {'Removed':>10} {'Total':>10} {'% Removed':>12}")
        print("-" * 55)
        print(f"{'Real':<10} {real_kept:>10,} {real_removed:>10,} {real_kept + real_removed:>10,} {real_removed/(real_kept + real_removed)*100 if (real_kept + real_removed) > 0 else 0:>11.1f}%")
        print(f"{'Fake':<10} {fake_kept:>10,} {fake_removed:>10,} {fake_kept + fake_removed:>10,} {fake_removed/(fake_kept + fake_removed)*100 if (fake_kept + fake_removed) > 0 else 0:>11.1f}%")
        print("-" * 55)
        print(f"{'TOTAL':<10} {total_kept:>10,} {total_removed:>10,} {total:>10,} {total_removed/total*100 if total > 0 else 0:>11.1f}%")
        
        # Print removal breakdown
        print("\n📋 Removal Breakdown:")
        removal_reasons = ['no_face', 'low_confidence', 'corrupt', 'too_small', 'too_large',
                          'too_dark', 'too_bright', 'low_variance', 'too_blurry', 'unreadable']
        
        for reason in removal_reasons:
            real_count = self.stats.get(f'Real_{reason}', 0)
            fake_count = self.stats.get(f'Fake_{reason}', 0)
            if real_count > 0 or fake_count > 0:
                print(f"   {reason:<15}: Real={real_count:,}, Fake={fake_count:,}")
        
        if MODE == 'move':
            print(f"\n💾 Removed images moved to: {TRASH_DIR}")
        elif MODE == 'delete':
            print(f"\n🗑️ Removed images permanently deleted")
        else:
            print(f"\n📝 Report mode - no files were modified")


def analyze_sample(sample_size=100):
    """Analyze a random sample of images to tune thresholds."""
    print("=" * 60)
    print("🔍 SAMPLE ANALYSIS MODE")
    print("=" * 60)
    
    cleaner = DatasetCleaner()
    
    import random
    
    for category in ['Real', 'Fake']:
        folder = DATA_ROOT / category
        if not folder.exists():
            continue
        
        files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
        sample = random.sample(files, min(sample_size, len(files)))
        
        print(f"\n📂 Analyzing {category} (sample of {len(sample)}):")
        
        results = defaultdict(list)
        
        for img_path in tqdm(sample, desc=f"Analyzing {category}"):
            # Collect metrics
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                results['brightness'].append(np.mean(gray))
                results['variance'].append(np.var(gray))
                results['blur'].append(cv2.Laplacian(gray, cv2.CV_64F).var())
                
                # Face detection
                has_face, _ = cleaner.detect_face(img_path)
                results['has_face'].append(1 if has_face else 0)
                
            except Exception as e:
                continue
        
        # Print statistics
        print(f"\n   Brightness: min={min(results['brightness']):.1f}, max={max(results['brightness']):.1f}, mean={np.mean(results['brightness']):.1f}")
        print(f"   Variance:   min={min(results['variance']):.1f}, max={max(results['variance']):.1f}, mean={np.mean(results['variance']):.1f}")
        print(f"   Blur score: min={min(results['blur']):.1f}, max={max(results['blur']):.1f}, mean={np.mean(results['blur']):.1f}")
        print(f"   Face detected: {sum(results['has_face'])}/{len(results['has_face'])} ({sum(results['has_face'])/len(results['has_face'])*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean deepfake dataset')
    parser.add_argument('--mode', choices=['move', 'delete', 'report'], default='report',
                       help='What to do with bad images (default: report)')
    parser.add_argument('--analyze', action='store_true',
                       help='Run sample analysis to tune thresholds')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Sample size for analysis mode')
    parser.add_argument('--min-face-size', type=int, default=MIN_FACE_SIZE,
                       help='Minimum face size in pixels')
    parser.add_argument('--face-confidence', type=float, default=FACE_CONFIDENCE_THRESHOLD,
                       help='MTCNN face detection confidence threshold')
    parser.add_argument('--blur-threshold', type=float, default=BLUR_THRESHOLD,
                       help='Blur detection threshold (lower = stricter)')
    
    args = parser.parse_args()
    
    # Update global settings from args
    MODE = args.mode
    MIN_FACE_SIZE = args.min_face_size
    FACE_CONFIDENCE_THRESHOLD = args.face_confidence
    BLUR_THRESHOLD = args.blur_threshold
    
    if args.analyze:
        analyze_sample(args.sample_size)
    else:
        cleaner = DatasetCleaner()
        cleaner.clean_dataset()
