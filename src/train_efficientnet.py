"""
train_efficientnet.py
EfficientNet-B4 training pipeline for diabetic retinopathy grading.

Usage (run from project root OR src/):
    python src/train_efficientnet.py --test              # 100 images, 2 epochs
    python src/train_efficientnet.py                     # all images, 20 epochs
    python src/train_efficientnet.py --subset 200 --epochs 5
"""

import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # works on servers / Colab / no-display envs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from dataset import DRDataset
from model import build_model

# ── Default paths (IDRiD layout) ─────────────────────────────────────────────
DATA_DIR    = ROOT / 'data/raw/idrid/B. Disease Grading/1. Original Images/a. Training Set'
CSV_FILE    = ROOT / 'data/raw/idrid/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'
CKPT_DIR    = ROOT / 'models/checkpoints'
METRICS_DIR = ROOT / 'results/metrics'
FIGURES_DIR = ROOT / 'results/figures'

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES = 5
TARGET_SIZE = 512
VAL_SPLIT   = 0.2
LR          = 1e-4
BATCH_SIZE  = 16
EPOCHS      = 20
SEED        = 42

# ── Test-mode overrides ───────────────────────────────────────────────────────
TEST_SUBSET = 100
TEST_EPOCHS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_labels_fast(dataset):
    """
    Extract all labels without loading images.
    Works with DRDataset directly or a Subset wrapping it.
    Falls back to iterating (slow) only when labels_df is absent.
    """
    # Unwrap Subset to access DRDataset internals
    if isinstance(dataset, Subset):
        base    = dataset.dataset
        indices = dataset.indices
    else:
        base    = dataset
        indices = list(range(len(dataset)))

    if base.labels_df is not None:
        labels = []
        for i in indices:
            stem = base.image_paths[i].stem
            try:
                labels.append(int(base.labels_df.loc[stem, 'Retinopathy grade']))
            except KeyError:
                labels.append(0)
        return np.array(labels)

    # No CSV: return zeros (uniform weights will be used)
    return np.zeros(len(indices), dtype=int)


def make_dataloaders(subset_size=None, batch_size=BATCH_SIZE):
    """Build train/val DataLoaders from IDRiD data."""
    full_ds = DRDataset(
        image_dir=DATA_DIR,
        csv_file=CSV_FILE if CSV_FILE.exists() else None,
        target_size=TARGET_SIZE,
    )

    # Optional subset (first N images by sorted filename)
    if subset_size is not None:
        n = min(subset_size, len(full_ds))
        full_ds = Subset(full_ds, list(range(n)))

    n_total = len(full_ds)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"  Train samples : {n_train}")
    print(f"  Val   samples : {n_val}")
    return train_loader, val_loader, full_ds


def build_class_weights(dataset, num_classes, device):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    labels = get_labels_fast(dataset)
    present = np.unique(labels)

    if len(present) < 2:
        print("  Class weights : uniform (only one class found)")
        return torch.ones(num_classes, dtype=torch.float32, device=device)

    # Compute weights only for classes that actually appear in this split
    partial = compute_class_weight('balanced', classes=present, y=labels)

    # Fill all-classes array; missing classes default to 1.0
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(present, partial):
        weights[cls] = w

    print(f"  Class weights : {np.round(weights, 3)}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Training / Validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0

    for images, labels in tqdm(loader, desc='  [train]', leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc='  [val]  ', leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    all_probs  = np.concatenate(all_probs,  axis=0)   # (N, C)
    all_labels = np.concatenate(all_labels, axis=0)   # (N,)

    try:
        auroc = roc_auc_score(all_labels, all_probs,
                              multi_class='ovr', average='macro',
                              labels=list(range(num_classes)))
    except ValueError:
        auroc = float('nan')

    return total_loss / total, correct / total, auroc


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='Train')
    axes[1].plot(epochs, history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()

    axes[2].plot(epochs, history['val_auroc'], color='green', label='Val AUROC')
    axes[2].set_title('Val AUROC'); axes[2].set_xlabel('Epoch'); axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Curves saved  : {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(subset_size=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    print(f"\n{'='*60}")
    print(f"EfficientNet-B4 DR Training")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  LR            : {LR}")
    if subset_size:
        print(f"  Subset        : {subset_size} images")
    print()

    for d in [CKPT_DIR, METRICS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    train_loader, val_loader, full_ds = make_dataloaders(subset_size, batch_size)

    # ── Class weights ─────────────────────────────────────────────────────────
    print("Computing class weights...")
    class_weights = build_class_weights(full_ds, NUM_CLASSES, device)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building model (EfficientNet-B4, pretrained)...")
    model     = build_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params  : {total_params:,}")
    print(f"  Trainable     : {trainable_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auroc']}
    best_auroc = -1.0
    ckpt_path  = CKPT_DIR / 'efficientnet_b4_best.pth'

    print(f"\nStarting training...\n{'─'*60}")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        train_loss, train_acc            = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc,   val_auroc = validate(model, val_loader, criterion, device, NUM_CLASSES)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc if not np.isnan(val_auroc) else 0.0)

        auroc_str = f"{val_auroc:.4f}" if not np.isnan(val_auroc) else "N/A (too few classes in val)"
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.3f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.3f}  AUROC={auroc_str}")

        # Best model checkpoint
        if not np.isnan(val_auroc) and val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'val_acc':   val_acc,
            }, ckpt_path)
            print(f"  ✓ Best checkpoint saved  (AUROC={best_auroc:.4f})")

        print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path = METRICS_DIR / 'training_history.csv'
    pd.DataFrame(history).to_csv(csv_path, index=False)
    print(f"  History saved : {csv_path}")

    plot_curves(history, FIGURES_DIR / 'training_curves.png')

    print(f"\n{'='*60}")
    print(f"Training complete")
    print(f"  Best Val AUROC : {best_auroc:.4f}")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EfficientNet-B4 for DR grading')
    parser.add_argument('--test',       action='store_true',
                        help=f'Quick test mode: {TEST_SUBSET} images, {TEST_EPOCHS} epochs')
    parser.add_argument('--subset',     type=int,   default=None,
                        help='Number of images to use (default: all)')
    parser.add_argument('--epochs',     type=int,   default=None,
                        help=f'Epochs (default: {EPOCHS}, or {TEST_EPOCHS} in --test)')
    parser.add_argument('--batch_size', type=int,   default=BATCH_SIZE)
    args = parser.parse_args()

    if args.test:
        train(
            subset_size=args.subset or TEST_SUBSET,
            epochs=args.epochs or TEST_EPOCHS,
            batch_size=args.batch_size,
        )
    else:
        train(
            subset_size=args.subset,
            epochs=args.epochs or EPOCHS,
            batch_size=args.batch_size,
        )
