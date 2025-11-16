# src/train.py
"""
Two-phase training script for the ensemble deepfake detector (option B).

PHASE 1:
    - Fine-tune each backbone separately on frame images (Xception, EfficientNet-B0, MesoNet).
PHASE 2:
    - Freeze backbones. For each image, compute logits of each backbone and train the small fusion head
      (EnsembleFusion) on concatenated logits.

Usage example:
    python src/train.py --data_root data/dataset --out_dir checkpoints --epochs_backbone 4 --epochs_fusion 4 --batch_size 32
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models import build_all_models  # uses your models.py

# ---------------------------
# Utility helpers
# ---------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    torch.save(state, filename)

# ---------------------------
# Training helpers
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    preds = []
    labels_all = []
    for imgs, labels in tqdm(loader, desc="Train batch"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            out = model(imgs)
            loss = criterion(out, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        # predictions for metrics
        if out.dim() == 1:
            # single-output (not expected here)
            prob = torch.sigmoid(out).detach().cpu().numpy()
            preds.extend(prob.tolist())
        else:
            prob = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            preds.extend(prob.tolist())

        labels_all.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(labels_all, preds)
    except Exception:
        auc = 0.0
    acc = accuracy_score(labels_all, [1 if p>=0.5 else 0 for p in preds])
    return epoch_loss, auc, acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    labels_all = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val batch"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * imgs.size(0)

            prob = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            preds.extend(prob.tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(labels_all, preds)
    except Exception:
        auc = 0.0
    acc = accuracy_score(labels_all, [1 if p>=0.5 else 0 for p in preds])
    return epoch_loss, auc, acc

# ---------------------------
# Create dataloaders
# ---------------------------
def create_dataloaders(data_root, image_size=224, batch_size=32, val_split=0.2, num_workers=4):
    data_root = Path(data_root)

    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Use ImageFolder: expects data_root/{class}/{images}
    full_dataset = datasets.ImageFolder(root=str(data_root), transform=transform_train)
    # split indices (video-level leakage avoidance: we assume folders grouped reasonably)
    n = len(full_dataset)
    val_n = int(n * val_split)
    train_n = n - val_n
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_n, val_n])

    # Replace val transform
    val_set.dataset.transform = transform_val

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# ---------------------------
# Phase 1: Fine-tune a backbone
# ---------------------------
def fine_tune_backbone(model, model_name, train_loader, val_loader, device, epochs=4, lr=2e-4, out_dir="checkpoints", use_amp=True):
    print(f"\n--- Fine-tuning backbone: {model_name} ---")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    best_auc = 0.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_auc, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_auc, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            fp = out_dir / f"{model_name}_best.pth"
            save_checkpoint({'model_state_dict': model.state_dict(), 'auc': best_auc}, str(fp))
            print(f"Saved best {model_name} checkpoint to {fp}")

    return out_dir / f"{model_name}_best.pth"

# ---------------------------
# Phase 2: Train fusion head
# ---------------------------
def train_fusion(backbones, fusion_model, train_loader, val_loader, device, epochs=4, lr=1e-3, out_dir="checkpoints", use_amp=True):
    """
    backbones: dict of name->model (already loaded and set to eval and moved to device)
    fusion_model: nn.Module fusion head (to train)
    For each batch, compute backbone logits (no grad) and train fusion on them.
    """

    print("\n--- Training fusion head (phase 2) ---")
    fusion_model = fusion_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    best_auc = 0.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # helper to produce logits from backbones
    def backbone_forward_batch(imgs):
        logits_list = []
        with torch.no_grad():
            for m in backbones.values():
                out = m(imgs)           # shape (B,2)
                # Use raw logits (no softmax) - fusion will learn
                logits_list.append(out.detach())
        # Concatenate along last dim -> (B, 6)
        return torch.cat(logits_list, dim=1)

    for epoch in range(epochs):
        fusion_model.train()
        running_loss = 0.0
        preds = []
        labels_all = []
        for imgs, labels in tqdm(train_loader, desc="Fusion train batch"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # compute backbone logits (no grad)
            logits = backbone_forward_batch(imgs).to(device)  # (B,6)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                out = fusion_model(logits)
                loss = criterion(out, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            prob = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            preds.extend(prob.tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        try:
            train_auc = roc_auc_score(labels_all, preds)
        except Exception:
            train_auc = 0.0
        train_acc = accuracy_score(labels_all, [1 if p>=0.5 else 0 for p in preds])

        # Validation
        fusion_model.eval()
        running_loss = 0.0
        preds = []
        labels_all = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Fusion val batch"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = backbone_forward_batch(imgs).to(device)
                out = fusion_model(logits)
                loss = criterion(out, labels)
                running_loss += loss.item() * imgs.size(0)
                prob = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
                preds.extend(prob.tolist())
                labels_all.extend(labels.detach().cpu().numpy().tolist())

        val_loss = running_loss / len(val_loader.dataset)
        try:
            val_auc = roc_auc_score(labels_all, preds)
        except Exception:
            val_auc = 0.0
        val_acc = accuracy_score(labels_all, [1 if p>=0.5 else 0 for p in preds])

        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Train Acc: {train_acc:.4f}")
        print(f"                 Val loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f} | Val Acc:   {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            fp = out_dir / f"fusion_best.pth"
            save_checkpoint({'fusion_state_dict': fusion_model.state_dict(), 'auc': best_auc}, str(fp))
            print(f"Saved best fusion checkpoint to {fp}")

    return out_dir / "fusion_best.pth"

# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root folder with class subfolders (data/dataset)")
    parser.add_argument("--out_dir", default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_backbone", type=int, default=4)
    parser.add_argument("--epochs_fusion", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # dataloaders
    train_loader, val_loader = create_dataloaders(args.data_root, image_size=args.image_size,
                                                  batch_size=args.batch_size, val_split=args.val_split,
                                                  num_workers=args.num_workers)

    # load models
    models = build_all_models(device=device)  # expects your build_all_models in models.py
    xcep = models["xception"]
    eff = models["efficientnet"]
    meso = models["mesonet"]
    fusion = models["fusion"]

    # -------------------
    # Phase 1: Fine-tune each backbone alone
    # -------------------
    # Note: we will fine-tune for a small number of epochs to adapt them to your dataset
    ckpts = {}
    for name, m in [("xception", xcep), ("efficientnet", eff), ("mesonet", meso)]:
        ckpt = fine_tune_backbone(m, model_name=name, train_loader=train_loader, val_loader=val_loader,
                                  device=device, epochs=args.epochs_backbone, out_dir=args.out_dir)
        ckpts[name] = ckpt

    # -------------------
    # Phase 2: Train fusion head (freeze backbones)
    # -------------------
    print("\nFreezing backbone weights and preparing for fusion training...")
    # reload best weights and set eval, requires_grad False
    for name, m in [("xception", xcep), ("efficientnet", eff), ("mesonet", meso)]:
        ckpt = torch.load(str(ckpts[name]), map_location=device)
        try:
            m.load_state_dict(ckpt['model_state_dict'])
        except Exception:
            # try partial load if shapes mismatch
            m.load_state_dict(ckpt['model_state_dict'], strict=False)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # Fusion model: expects concatenated logits from 3 models (2+2+2 => 6)
    fusion_model = fusion  # from models.build_all_models()

    # Train fusion head
    fusion_ckpt = train_fusion({"x": xcep, "e": eff, "m": meso}, fusion_model,
                               train_loader=train_loader, val_loader=val_loader,
                               device=device, epochs=args.epochs_fusion, out_dir=args.out_dir)

    print("\nTraining finished. Best checkpoints saved in:", args.out_dir)
    print("Backbone ckpts:", ckpts)
    print("Fusion ckpt:", fusion_ckpt)

if __name__ == "__main__":
    main()
