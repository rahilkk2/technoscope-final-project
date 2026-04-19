"""
train.py — Training Script for AIDetectorCNN
=============================================
Trains the CNN defined in classifier.py on a 3-class dataset:
    Real        → data/real/
    AI          → data/ai/
    Screenshot  → data/screenshot/

Usage:
    python train.py                          # full training run
    python train.py --data /path/to/data     # custom data root
    python train.py --epochs 50 --batch 32   # custom hyperparams
    python train.py --eval /path/to/image    # quick eval on one image

Output:
    backend/models/classifier.pt             # trained model weights
    training_log.csv                         # per-epoch loss & accuracy

Requirements:
    pip install torch torchvision Pillow scikit-learn tqdm
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, 'data')
MODEL_OUT   = os.path.join(SCRIPT_DIR, 'backend', 'models', 'classifier.pt')
LOG_OUT     = os.path.join(SCRIPT_DIR, 'training_log.csv')

CLASS_NAMES = ['Real', 'AI', 'Screenshot']   # must match folder names exactly
#   data/
#   ├── Real/           ← real camera photos
#   ├── AI/             ← AI-generated images
#   └── Screenshot/     ← screen captures


# ── Model (copy of AIDetectorCNN from classifier.py) ─────────────────────────

class AIDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress(iterable, desc='', total=None):
    """Wrap with tqdm if available, otherwise bare iteration."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


def _check_data_structure(data_root):
    """Verify the data folder has all three class sub-folders."""
    missing = []
    for cls in CLASS_NAMES:
        p = os.path.join(data_root, cls)
        if not os.path.isdir(p):
            missing.append(p)
    if missing:
        print('\n[ERROR] Missing class folders:')
        for m in missing:
            print(f'  {m}')
        print('\nExpected layout:')
        print(f'  {data_root}/')
        for cls in CLASS_NAMES:
            print(f'    {cls}/   ← put images here')
        print('\nMinimum recommended: 1 000 images per class (3 000 total).')
        sys.exit(1)

    counts = {}
    for cls in CLASS_NAMES:
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        p = os.path.join(data_root, cls)
        n = sum(1 for f in os.listdir(p) if os.path.splitext(f)[1].lower() in exts)
        counts[cls] = n
    return counts


def _accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    data_root = args.data
    n_epochs  = args.epochs
    batch     = args.batch
    lr        = args.lr
    val_split = args.val_split
    seed      = args.seed
    out_path  = args.output

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────────
    counts = _check_data_structure(data_root)
    print('[INFO] Dataset class counts:')
    for cls, n in counts.items():
        warn = '  ⚠  (recommend ≥1 000)' if n < 1000 else ''
        print(f'  {cls:12s}: {n:>5d} images{warn}')

    full_dataset = datasets.ImageFolder(data_root, transform=TRAIN_TRANSFORM)

    # Map ImageFolder's auto-sorted class_to_idx to our CLASS_NAMES order
    idx_map = full_dataset.class_to_idx            # e.g. {'AI':0,'Real':1,'Screenshot':2}
    print(f'[INFO] Folder→index mapping: {idx_map}')

    # Validate all three classes found
    found = set(idx_map.keys())
    required = set(CLASS_NAMES)
    if found != required:
        print(f'[ERROR] Expected classes {required}, found {found}')
        sys.exit(1)

    # Train / val split
    n_val   = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    # Override val transform (no augmentation)
    val_ds.dataset.transform = VAL_TRANSFORM   # shared dataset object — set after split

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True)

    print(f'[INFO] Train: {n_train} | Val: {n_val} | Batch: {batch} | Epochs: {n_epochs}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AIDetectorCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Parameters: {total_params:,}')

    # Class-weighted loss to handle imbalanced datasets
    class_counts = [counts.get(cls, 1) for cls in
                    sorted(idx_map, key=idx_map.get)]   # order by index
    weights = torch.tensor([1.0 / (c + 1e-9) for c in class_counts], dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # ── Log file ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(LOG_OUT)), exist_ok=True)
    log_path = args.log if args.log else LOG_OUT
    log_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    print(f'[INFO] Logging to {log_path}')

    # ── Epoch loop ────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # — Train —
        model.train()
        tr_loss, tr_acc, tr_n = 0.0, 0.0, 0
        for imgs, labels in _progress(train_loader, desc=f'Epoch {epoch}/{n_epochs} train'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bs = imgs.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += _accuracy(logits, labels) * bs
            tr_n    += bs

        tr_loss /= tr_n
        tr_acc  /= tr_n

        # — Validate —
        model.eval()
        va_loss, va_acc, va_n = 0.0, 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in _progress(val_loader, desc=f'Epoch {epoch}/{n_epochs} val'):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                bs = imgs.size(0)
                va_loss += loss.item() * bs
                va_acc  += _accuracy(logits, labels) * bs
                va_n    += bs
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        va_loss /= va_n
        va_acc  /= va_n
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        elapsed = time.time() - t0
        print(f'Epoch {epoch:>3}/{n_epochs} | '
              f'train loss {tr_loss:.4f} acc {tr_acc:.3f} | '
              f'val loss {va_loss:.4f} acc {va_acc:.3f} | '
              f'lr {current_lr:.2e} | {elapsed:.1f}s')

        csv_writer.writerow([epoch, f'{tr_loss:.6f}', f'{tr_acc:.6f}',
                             f'{va_loss:.6f}', f'{va_acc:.6f}', f'{current_lr:.2e}'])
        log_file.flush()

        # — Save best —
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  ↑ New best val acc: {best_val_acc:.3f} — checkpoint saved')

    log_file.close()

    # ── Save model ────────────────────────────────────────────────────────────
    output_path = args.output if args.output else MODEL_OUT
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({'state_dict': best_state, 'val_acc': best_val_acc,
                'class_to_idx': idx_map, 'epochs': n_epochs},
               output_path)
    print(f'\n[DONE] Best val acc: {best_val_acc:.3f}')
    print(f'[DONE] Model saved → {output_path}')

    # ── Per-class report ──────────────────────────────────────────────────────
    if HAS_SKLEARN and all_labels:
        idx_to_name = {v: k for k, v in idx_map.items()}
        target_names = [idx_to_name.get(i, str(i)) for i in sorted(idx_to_name)]
        print('\n[INFO] Final validation classification report:')
        print(classification_report(all_labels, all_preds, target_names=target_names))
        cm = confusion_matrix(all_labels, all_preds)
        print('[INFO] Confusion matrix (rows=actual, cols=predicted):')
        header = '          ' + '  '.join(f'{n:>10}' for n in target_names)
        print(header)
        for i, row in enumerate(cm):
            name = target_names[i] if i < len(target_names) else str(i)
            print(f'{name:>10}' + '  '.join(f'{v:>10}' for v in row))


# ── Quick eval on a single image ─────────────────────────────────────────────

def eval_single(image_path, model_path):
    from PIL import Image as PILImage
    device = torch.device('cpu')
    model  = AIDetectorCNN()
    ckpt   = torch.load(model_path, map_location=device)
    state  = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    img    = PILImage.open(image_path).convert('RGB')
    tensor = VAL_TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).tolist()

    idx_map = ckpt.get('class_to_idx', {'Real': 0, 'AI': 1, 'Screenshot': 2}) \
              if isinstance(ckpt, dict) else {'Real': 0, 'AI': 1, 'Screenshot': 2}
    idx_to_name = {v: k for k, v in idx_map.items()}
    print(f'\n[EVAL] {image_path}')
    for i, p in enumerate(probs):
        name = idx_to_name.get(i, str(i))
        bar  = '█' * int(p * 40)
        print(f'  {name:>12}: {p:.4f}  {bar}')
    winner = idx_to_name.get(int(torch.tensor(probs).argmax()), '?')
    print(f'  → Verdict: {winner}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Train AIDetectorCNN')
    p.add_argument('--data',       default=DEFAULT_DATA,
                   help='Root data folder with Real/, AI/, Screenshot/ sub-folders')
    p.add_argument('--output',     default=MODEL_OUT,
                   help='Where to save classifier.pt')
    p.add_argument('--log',        default=LOG_OUT,
                   help='Where to save training_log.csv')
    p.add_argument('--epochs',     type=int,   default=30,
                   help='Number of training epochs (default: 30)')
    p.add_argument('--batch',      type=int,   default=32,
                   help='Batch size (default: 32; lower to 16 if OOM)')
    p.add_argument('--lr',         type=float, default=3e-4,
                   help='Initial learning rate (default: 3e-4)')
    p.add_argument('--val-split',  type=float, default=0.15,
                   help='Fraction of data held out for validation (default: 0.15)')
    p.add_argument('--seed',       type=int,   default=42,
                   help='Random seed for reproducibility (default: 42)')
    p.add_argument('--eval',       metavar='IMAGE_PATH', default=None,
                   help='Skip training — evaluate a single image with the saved model')
    args = p.parse_args()

    if args.eval:
        model_path = args.output if os.path.isfile(args.output) else MODEL_OUT
        if not os.path.isfile(model_path):
            print(f'[ERROR] No model found at {model_path}. Train first.')
            sys.exit(1)
        eval_single(args.eval, model_path)
    else:
        train(args)


if __name__ == '__main__':
    main()
