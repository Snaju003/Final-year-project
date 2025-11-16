# src/face_extractor.py
"""
Extract ALL faces from a video using MTCNN (facenet-pytorch).
Saves every detected face into:  data/faces/<video_name>/000000.jpg ...
"""

import os
import argparse
from pathlib import Path
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm


def extract_faces_from_video(video_path, out_dir, frame_step=5,
                             max_faces_per_video=None, min_face_size=20):
    """
    Extract faces from the video and save them as JPGs.

    Args:
        video_path (str): input video path
        out_dir (str): output directory root
        frame_step (int): process every Nth frame
        max_faces_per_video (int or None): limit saved faces
        min_face_size (int): minimum face width/height to accept
    """

    video_path = Path(video_path)
    out_dir = Path(out_dir) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-select GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # keep_all=True → detect ALL faces per frame
    mtcnn = MTCNN(keep_all=True, device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.stem}", unit="f")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only analyze some frames (performance)
            if frame_idx % frame_step == 0:

                # Convert BGR → RGB for MTCNN
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Face detection
                try:
                    boxes, probs = mtcnn.detect(rgb)
                except Exception:
                    boxes = None

                # Save ALL detected faces
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.astype(int).tolist()

                        w = x2 - x1
                        h = y2 - y1

                        # Check size threshold
                        if w >= min_face_size and h >= min_face_size:

                            face = frame[
                                max(y1, 0):max(y2, 0),
                                max(x1, 0):max(x2, 0)
                            ]

                            if face.size != 0:
                                fname = out_dir / f"{saved:06d}.jpg"
                                cv2.imwrite(str(fname), face)
                                saved += 1

                                if max_faces_per_video is not None and saved >= max_faces_per_video:
                                    break

            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        mtcnn = None

    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract faces from a video using MTCNN.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--out", default="data/faces", help="Output root directory.")
    parser.add_argument("--frame_step", type=int, default=5, help="Take every Nth frame.")
    parser.add_argument("--max_faces", type=int, default=None, help="Maximum faces to save.")
    parser.add_argument("--min_face_size", type=int, default=20, help="Minimum face size (px).")

    args = parser.parse_args()

    print("\n===== FACE EXTRACTION SETTINGS =====")
    print("Video:", args.video)
    print("Output folder root:", args.out)
    print("Frame step:", args.frame_step)
    if args.max_faces:
        print("Max faces:", args.max_faces)
    print("====================================\n")

    saved = extract_faces_from_video(
        video_path=args.video,
        out_dir=args.out,
        frame_step=max(1, args.frame_step),
        max_faces_per_video=args.max_faces,
        min_face_size=args.min_face_size,
    )

    print(f"\nFinished. Saved {saved} face images to {Path(args.out) / Path(args.video).stem}\n")


if __name__ == "__main__":
    main()
