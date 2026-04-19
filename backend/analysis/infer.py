"""
infer.py — Dual-mode CNN inference (PyTorch / ONNX)
====================================================
Standalone script that classifies a single image as Real / AI / Screenshot.
Replaces the CNN portion of classifier.py for direct invocation.

Usage:
    python infer.py <image_path>                        # default: pytorch mode
    python infer.py <image_path> --mode pytorch         # explicit pytorch
    python infer.py <image_path> --mode onnx            # onnxruntime (faster)
    python infer.py <image_path> --mode onnx --onnx-path /path/to/model.onnx

Output: JSON to stdout (same schema as classifier.py)

Both modes produce identical JSON:
{
    "model_loaded": true,
    "mode": "pytorch" | "onnx",
    "predicted_class": "AI",
    "confidence": 0.87,
    "probabilities": { "Real": 0.08, "AI": 0.87, "Screenshot": 0.05 }
}
"""

import argparse
import json
import os
import sys

import numpy as np

# ── Dependency handling ───────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print(json.dumps({"error": "Missing cv2. Run: pip install opencv-python"}))
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print(json.dumps({"error": "Missing Pillow. Run: pip install Pillow"}))
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import onnxruntime as ort
    ORT_OK = True
except ImportError:
    ORT_OK = False


CLASS_NAMES = ['Real', 'AI', 'Screenshot']

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PT   = os.path.join(SCRIPT_DIR, '..', 'models', 'classifier.pt')
DEFAULT_ONNX = os.path.join(SCRIPT_DIR, '..', 'models', 'classifier.onnx')


# ── Model architecture (must match train.py / classifier.py) ─────────────────

if TORCH_OK:
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

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Preprocessing (shared by both modes) ─────────────────────────────────────

def preprocess_numpy(image_path):
    """Load image → resize 224×224 → normalize → NCHW float32 numpy array."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std

    # HWC → NCHW
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr


def softmax(x):
    """Numerically stable softmax over last axis."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


# ── PyTorch inference ─────────────────────────────────────────────────────────

def infer_pytorch(image_path, pt_path):
    if not TORCH_OK:
        return {"error": "PyTorch not installed. Run: pip install torch torchvision"}

    if not os.path.isfile(pt_path):
        return {"error": f"Weights not found: {pt_path}. Train first with train.py"}

    device = torch.device('cpu')
    model  = AIDetectorCNN()

    ckpt  = torch.load(pt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    pil_img = Image.open(image_path).convert('RGB')
    tensor  = TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).tolist()

    return build_result(probs, 'pytorch')


# ── ONNX inference ────────────────────────────────────────────────────────────

def infer_onnx(image_path, onnx_path):
    if not ORT_OK:
        return {"error": "onnxruntime not installed. Run: pip install onnxruntime"}

    if not os.path.isfile(onnx_path):
        return {"error": f"ONNX model not found: {onnx_path}. Export first with export_onnx.py"}

    session   = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_arr = preprocess_numpy(image_path)

    input_name = session.get_inputs()[0].name
    logits     = session.run(None, {input_name: input_arr})[0]
    probs      = softmax(logits)[0].tolist()

    return build_result(probs, 'onnx')


# ── Build output ─────────────────────────────────────────────────────────────

def build_result(probs, mode):
    prob_dict     = {CLASS_NAMES[i]: round(probs[i], 6) for i in range(len(CLASS_NAMES))}
    predicted_idx = int(np.argmax(probs))

    return {
        "model_loaded":    True,
        "mode":            mode,
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence":      round(max(probs), 6),
        "probabilities":   prob_dict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='TechnoScope CNN Inference (dual-mode)')
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('--mode', choices=['pytorch', 'onnx'], default='pytorch',
                        help='Inference backend (default: pytorch)')
    parser.add_argument('--pt-path',   default=DEFAULT_PT,   help='Path to .pt weights')
    parser.add_argument('--onnx-path', default=DEFAULT_ONNX,  help='Path to .onnx model')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(json.dumps({"error": f"File not found: {args.image}"}))
        sys.exit(1)

    if args.mode == 'pytorch':
        result = infer_pytorch(args.image, args.pt_path)
    else:
        result = infer_onnx(args.image, args.onnx_path)

    print(json.dumps(result))

    if 'error' in result:
        sys.exit(1)


if __name__ == '__main__':
    main()
