"""
export_onnx.py — Export AIDetectorCNN from .pt → .onnx
=====================================================
Loads weights from backend/models/classifier.pt,
exports to backend/models/classifier.onnx with dynamic batch + spatial axes.

Usage:
    python backend/analysis/export_onnx.py
    python backend/analysis/export_onnx.py --pt /path/to/classifier.pt --onnx /path/to/output.onnx

Requirements:
    pip install torch torchvision onnx onnxruntime
"""

import argparse
import os
import sys
import json

import torch
import torch.nn as nn
import numpy as np


# ── Model definition (must match classifier.py / train.py exactly) ────────────

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


CLASS_NAMES = ['Real', 'AI', 'Screenshot']


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_pt   = os.path.join(script_dir, '..', 'models', 'classifier.pt')
    default_onnx = os.path.join(script_dir, '..', 'models', 'classifier.onnx')

    parser = argparse.ArgumentParser(description='Export AIDetectorCNN → ONNX')
    parser.add_argument('--pt',   default=default_pt,   help='Input .pt weights path')
    parser.add_argument('--onnx', default=default_onnx,  help='Output .onnx path')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version (default: 17)')
    args = parser.parse_args()

    pt_path   = os.path.abspath(args.pt)
    onnx_path = os.path.abspath(args.onnx)

    # ── Load PyTorch weights ──────────────────────────────────────────────────
    if not os.path.isfile(pt_path):
        print(f'[ERROR] Weights not found: {pt_path}')
        print('        Train first with: python train.py')
        sys.exit(1)

    print(f'[1/4] Loading weights from {pt_path}')
    device = torch.device('cpu')
    model  = AIDetectorCNN()

    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('features.') for k in ckpt):
        state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state)
    model.eval()
    print(f'       Model loaded — {sum(p.numel() for p in model.parameters()):,} params')

    # ── Export to ONNX ────────────────────────────────────────────────────────
    print(f'[2/4] Exporting to ONNX (opset {args.opset})')
    dummy = torch.randn(1, 3, 224, 224)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=args.opset,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image':  {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        },
    )
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f'       Exported → {onnx_path} ({onnx_size_mb:.1f} MB)')

    # ── Validate ONNX ────────────────────────────────────────────────────────
    print('[3/4] Validating ONNX model')
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print('       ✓ ONNX model is valid')
    except ImportError:
        print('       ⚠ onnx package not installed — skipping validation')
    except Exception as e:
        print(f'       ✗ Validation failed: {e}')

    # ── Self-test: compare PyTorch vs ONNX outputs ────────────────────────────
    print('[4/4] Self-test: dummy inference comparison')
    try:
        import onnxruntime as ort

        # PyTorch inference
        with torch.no_grad():
            pt_logits = model(dummy).numpy()

        # ONNX inference
        session   = ort.InferenceSession(onnx_path)
        ort_input = {session.get_inputs()[0].name: dummy.numpy()}
        ort_logits = session.run(None, ort_input)[0]

        # Compare
        max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
        print(f'       PyTorch output: {pt_logits[0].tolist()}')
        print(f'       ONNX output:    {ort_logits[0].tolist()}')
        print(f'       Max abs diff:   {max_diff:.8f}')

        if max_diff < 1e-4:
            print('       ✓ Outputs match (diff < 1e-4)')
        else:
            print(f'       ⚠ Outputs differ by {max_diff:.6f} — check numerics')

        # Output shape
        print(f'       Output shape:   {ort_logits.shape}  (batch, {len(CLASS_NAMES)} classes)')

    except ImportError:
        print('       ⚠ onnxruntime not installed — skipping runtime test')

    print('\n[DONE] Export complete.')
    print(f'       .pt:   {pt_path}')
    print(f'       .onnx: {onnx_path}')


if __name__ == '__main__':
    main()
