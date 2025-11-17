"""
Ultra-Aggressive Face Extractor - Maximum CPU + GPU Utilization
Parallel decode + GPU processing with non-blocking queues
"""

import os
from pathlib import Path
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
from queue import Queue, Empty
import time

# ---------------- PATHS ----------------
DATASET_ROOT = Path(r"X:\Final-year-project\data\dataset")
OUTPUT_ROOT = Path(r"X:\Final-year-project\data\faces")
REAL_DIR = OUTPUT_ROOT / "real"
FAKE_DIR = OUTPUT_ROOT / "fake"

# ---------------- AGGRESSIVE SETTINGS ----------------
FRAME_STEP = 5
MIN_FACE_SIZE = 20
MAX_FACES_PER_VIDEO = None
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360

# MAXIMIZED settings
GPU_BATCH_SIZE = 128              # Larger GPU batches
NUM_DECODE_THREADS = 14           # 14 parallel decoders (more aggressive)
NUM_GPU_WORKERS = 1               # Single GPU thread (avoid conflicts)
NUM_WRITE_THREADS = 8             # 8 write threads
BATCH_QUEUE_SIZE = 20             # Larger queue to keep GPU fed
FACE_QUEUE_SIZE = 1024            # Large face queue

# OpenCV optimization
cv2.setNumThreads(1)              # 1 thread per decoder = 14 total
cv2.ocl.setUseOpenCL(True)

# ---------------- PREP FOLDERS ----------------
REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

print("🧹 Clearing old faces...")
for f in REAL_DIR.glob("*.jpg"):
    f.unlink()
for f in FAKE_DIR.glob("*.jpg"):
    f.unlink()

# ---------------- DEVICE ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

print(f"🚀 Device: {device}")
print(f"⚡ GPU Batch: {GPU_BATCH_SIZE}")
print(f"🧵 Decode Threads: {NUM_DECODE_THREADS}")
print(f"💾 Write Threads: {NUM_WRITE_THREADS}")

# Global MTCNN
mtcnn = MTCNN(
    keep_all=True, 
    device=device, 
    post_process=False,
    thresholds=[0.6, 0.7, 0.7],
    min_face_size=MIN_FACE_SIZE
)

# Shared queues
batch_queue = Queue(maxsize=BATCH_QUEUE_SIZE)
face_queue = Queue(maxsize=FACE_QUEUE_SIZE)

# Thread-safe counters
counter_lock = Lock()
global_counters = {'real': 0, 'fake': 0}
stats_lock = Lock()
stats = {
    'videos_decoded': 0,
    'batches_queued': 0,
    'batches_processed': 0,
    'faces_detected': 0,
    'faces_written': 0,
    'decode_active': 0
}


# ============ DECODE WORKER ============
def decode_video_fast(video_path, video_type):
    """Fast video decoder - sends batches immediately."""
    with stats_lock:
        stats['decode_active'] += 1
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return video_path.stem, 0, 0
        
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        
        frame_idx = 0
        batch_resized = []
        batch_original = []
        batches_sent = 0
        frames_read = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Send final batch
                if batch_resized:
                    batch_queue.put((batch_resized, batch_original, video_type))
                    batches_sent += 1
                    with stats_lock:
                        stats['batches_queued'] += 1
                break
            
            frames_read += 1
            
            if frame_idx % FRAME_STEP == 0:
                frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT),
                                          interpolation=cv2.INTER_LINEAR)
                batch_resized.append(frame_resized)
                batch_original.append(frame.copy())
                
                # Send full batch
                if len(batch_resized) >= GPU_BATCH_SIZE:
                    batch_queue.put((batch_resized, batch_original, video_type))
                    batches_sent += 1
                    with stats_lock:
                        stats['batches_queued'] += 1
                    batch_resized = []
                    batch_original = []
            
            frame_idx += 1
        
        cap.release()
        
        with stats_lock:
            stats['videos_decoded'] += 1
            stats['decode_active'] -= 1
        
        return video_path.stem, batches_sent, frames_read
        
    except Exception as e:
        print(f"❌ {video_path.stem}: {e}")
        with stats_lock:
            stats['decode_active'] -= 1
        return video_path.stem, 0, 0


# ============ GPU PROCESSOR ============
def gpu_processor_worker():
    """Continuously process batches from queue."""
    print("🚀 GPU worker started")
    processed = 0
    
    while True:
        try:
            batch_data = batch_queue.get(timeout=2)
            
            frames_resized, frames_original, video_type = batch_data
            
            # RGB conversion
            rgb_batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_resized]
            
            # GPU detection
            with torch.no_grad():
                boxes_batch, _ = mtcnn.detect(rgb_batch)
            
            # Extract faces
            faces_found = 0
            for i, boxes in enumerate(boxes_batch):
                if boxes is not None and len(boxes) > 0:
                    orig = frames_original[i]
                    h, w = orig.shape[:2]
                    
                    sx = w / RESIZE_WIDTH
                    sy = h / RESIZE_HEIGHT
                    
                    boxes_scaled = boxes * np.array([sx, sy, sx, sy])
                    boxes_scaled = boxes_scaled.astype(np.int32)
                    
                    for box in boxes_scaled:
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                            face = orig[y1:y2, x1:x2]
                            if face.size > 0:
                                face_queue.put((face.copy(), video_type))
                                faces_found += 1
            
            processed += 1
            with stats_lock:
                stats['batches_processed'] += 1
                stats['faces_detected'] += faces_found
            
            del frames_resized, frames_original, rgb_batch
            
        except Empty:
            # Check if decoding is done
            with stats_lock:
                if stats['decode_active'] == 0 and batch_queue.empty():
                    break
            continue
        except Exception as e:
            print(f"❌ GPU error: {e}")
    
    # Signal writers
    for _ in range(NUM_WRITE_THREADS):
        face_queue.put(None)
    
    print(f"🚀 GPU worker done: {processed} batches")


# ============ WRITER WORKER ============
def writer_worker(worker_id):
    """Write faces to disk."""
    written = 0
    
    while True:
        try:
            item = face_queue.get(timeout=3)
            
            if item is None:
                break
            
            face, video_type = item
            out_dir = REAL_DIR if video_type == "real" else FAKE_DIR
            
            with counter_lock:
                idx = global_counters[video_type]
                global_counters[video_type] += 1
            
            fname = out_dir / f"{idx:07d}.jpg"
            cv2.imwrite(str(fname), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            written += 1
            with stats_lock:
                stats['faces_written'] += 1
            
        except Empty:
            continue
        except Exception as e:
            print(f"❌ Writer {worker_id}: {e}")
    
    print(f"💾 Writer {worker_id} done: {written} faces")


# ============ PROGRESS MONITOR ============
def progress_monitor(total_videos, start_time):
    """Monitor and display progress."""
    pbar = tqdm(total=total_videos, desc="📹 Videos", ncols=100, position=0)
    last_decoded = 0
    
    while True:
        time.sleep(1)
        
        with stats_lock:
            decoded = stats['videos_decoded']
            batches_q = stats['batches_queued']
            batches_p = stats['batches_processed']
            faces_d = stats['faces_detected']
            faces_w = stats['faces_written']
            active = stats['decode_active']
        
        # Update progress
        if decoded > last_decoded:
            pbar.update(decoded - last_decoded)
            last_decoded = decoded
        
        elapsed = time.time() - start_time
        fps = faces_w / elapsed if elapsed > 0 else 0
        
        pbar.set_postfix({
            'faces': faces_w,
            'f/s': f'{fps:.1f}',
            'batches': f'{batches_p}/{batches_q}',
            'active': active
        })
        
        # Check if done
        if decoded >= total_videos and active == 0:
            pbar.close()
            break


# ============ MAIN ============
def main():
    """Main execution."""
    # Gather videos
    real_videos = sorted((DATASET_ROOT / "DFD_original").glob("*.mp4"))
    fake_videos = sorted((DATASET_ROOT / "DFD_manipulated").glob("*.mp4"))
    all_videos = [("real", v) for v in real_videos] + [("fake", v) for v in fake_videos]
    
    if not all_videos:
        print("❌ No videos found!")
        return
    
    total_videos = len(all_videos)
    
    print(f"\n{'='*60}")
    print(f"📊 CONFIGURATION")
    print(f"{'='*60}")
    print(f"Videos: {len(real_videos)} real + {len(fake_videos)} fake = {total_videos}")
    print(f"Decode threads: {NUM_DECODE_THREADS}")
    print(f"Write threads: {NUM_WRITE_THREADS}")
    print(f"Batch size: {GPU_BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Start GPU worker
    gpu_thread = Thread(target=gpu_processor_worker, daemon=False)
    gpu_thread.start()
    
    # Start writers
    writer_threads = []
    for i in range(NUM_WRITE_THREADS):
        t = Thread(target=writer_worker, args=(i,), daemon=False)
        t.start()
        writer_threads.append(t)
    
    # Start progress monitor
    monitor_thread = Thread(target=progress_monitor, args=(total_videos, start_time), daemon=True)
    monitor_thread.start()
    
    time.sleep(0.5)
    
    # Decode all videos in parallel
    print("🎬 Decoding videos...\n")
    
    with ThreadPoolExecutor(max_workers=NUM_DECODE_THREADS) as executor:
        futures = {
            executor.submit(decode_video_fast, vpath, vtype): (vtype, vpath)
            for vtype, vpath in all_videos
        }
        
        # Wait for completion
        for future in as_completed(futures):
            vtype, vpath = futures[future]
            try:
                name, batches, frames = future.result()
            except Exception as e:
                print(f"❌ {vpath.stem} failed: {e}")
    
    print("\n✅ All videos decoded, waiting for GPU...")
    gpu_thread.join(timeout=300)
    
    print("⏳ Waiting for writers...")
    for t in writer_threads:
        t.join(timeout=60)
    
    elapsed = time.time() - start_time
    
    # Count saved faces
    real_saved = len(list(REAL_DIR.glob("*.jpg")))
    fake_saved = len(list(FAKE_DIR.glob("*.jpg")))
    total_saved = real_saved + fake_saved
    
    print(f"\n{'='*60}")
    print(f"🎉 COMPLETE!")
    print(f"{'='*60}")
    print(f"✨ Real: {real_saved:,} faces")
    print(f"✨ Fake: {fake_saved:,} faces")
    print(f"✨ Total: {total_saved:,} faces")
    print(f"⏱️  Time: {elapsed/60:.2f} min")
    print(f"⚡ Speed: {total_saved/elapsed:.1f} faces/sec")
    print(f"📊 Batches: {stats['batches_processed']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()