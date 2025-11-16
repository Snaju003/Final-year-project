"""
Optimized GPU-batched Face Extractor for all faces in a dataset.
Keeps all detected faces per frame.
Saves faces into:
    data/faces/real/<sequential>.jpg
    data/faces/fake/<sequential>.jpg
"""

import os
from pathlib import Path
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# ---------------- PATHS ----------------
DATASET_ROOT = Path(r"E:\deepfake_detection\data\dataset")
OUTPUT_ROOT = Path(r"E:\deepfake_detection\data\faces")
REAL_DIR = OUTPUT_ROOT / "real"
FAKE_DIR = OUTPUT_ROOT / "fake"

# ---------------- SETTINGS ----------------
FRAME_STEP = 5           # Process every Nth frame
MIN_FACE_SIZE = 20       # Minimum face size (px)
MAX_FACES_PER_VIDEO = None
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360
BATCH_SIZE = 8           # Number of frames per GPU batch

# ---------------- PREP FOLDERS ----------------
REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- DEVICE ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
mtcnn = MTCNN(keep_all=True, device=device)

# ---------------- FACE EXTRACTION ----------------
def extract_faces(video_path, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return 0

    saved = 0
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.stem}", unit="f")

    batch_frames = []
    batch_coords = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                batch_frames.append(frame_resized)
                batch_coords.append(frame)  # keep original frame for saving

            # Process batch
            if len(batch_frames) == BATCH_SIZE or (frame_idx % FRAME_STEP != 0 and len(batch_frames) > 0 and ret == False):
                # Convert frames to RGB
                rgb_batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_frames]
                # Detect faces on batch
                boxes_batch, probs_batch = mtcnn.detect(rgb_batch)

                for idx, boxes in enumerate(boxes_batch):
                    orig_frame = batch_coords[idx]
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = box.astype(int).tolist()
                            w, h = x2 - x1, y2 - y1
                            if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                                scale_x = orig_frame.shape[1] / RESIZE_WIDTH
                                scale_y = orig_frame.shape[0] / RESIZE_HEIGHT
                                x1_o = int(x1 * scale_x)
                                x2_o = int(x2 * scale_x)
                                y1_o = int(y1 * scale_y)
                                y2_o = int(y2 * scale_y)

                                face = orig_frame[y1_o:y2_o, x1_o:x2_o]
                                if face.size != 0:
                                    fname = out_dir / f"{saved:06d}.jpg"
                                    cv2.imwrite(str(fname), face)
                                    saved += 1
                                    if MAX_FACES_PER_VIDEO is not None and saved >= MAX_FACES_PER_VIDEO:
                                        break

                batch_frames = []
                batch_coords = []

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()

    return saved

# ---------------- LOOP DATASET ----------------
def main():
    real_videos = list((DATASET_ROOT / "DFD_original").glob("*.mp4"))
    fake_videos = list((DATASET_ROOT / "DFD_manipulated").glob("*.mp4"))

    print("=== START: REAL VIDEOS ===")
    total_real = 0
    for vid in real_videos:
        total_real += extract_faces(vid, REAL_DIR)
    print(f"✓ Done extracting {total_real} real faces\n")

    print("=== START: MANIPULATED VIDEOS ===")
    total_fake = 0
    for vid in fake_videos:
        total_fake += extract_faces(vid, FAKE_DIR)
    print(f"✓ Done extracting {total_fake} fake faces\n")

    print(f"🎉 ALL DONE — Faces extracted from all videos!")

if __name__ == "__main__":
    main()
