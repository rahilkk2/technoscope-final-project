"""
classifier.py — CNN + Fixed-Threshold Weighted Heuristic (8 signals)
Corrected thresholds to avoid false positives on real phone photos.
Usage: python classifier.py <image_path>
Output: JSON to stdout
"""

import sys
import json
import os
import math

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print(json.dumps({"error": "Missing: cv2. Run: pip install opencv-python"}))
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print(json.dumps({"error": "Missing: numpy. Run: pip install numpy"}))
    sys.exit(1)
try:
    from PIL import Image
except ImportError:
    print(json.dumps({"error": "Missing: Pillow. Run: pip install Pillow"}))
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
    import pywt
    PYWT_OK = True
except ImportError:
    PYWT_OK = False

try:
    import exifread
    EXIFREAD_OK = True
except ImportError:
    EXIFREAD_OK = False


# ── CNN ───────────────────────────────────────────────────────────────────────

if TORCH_OK:
    class AIDetectorCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2,2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2,2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d((4,4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*4*4, 512), nn.ReLU(True), nn.Dropout(0.4),
                nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.3),
                nn.Linear(128, 3),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

CLASS_NAMES = ['Real', 'AI', 'Screenshot']


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED WEIGHTED HEURISTIC SCORER — 8 signals, corrected thresholds
# Core rule: no single signal pushes confidence > 60%. Multiple must agree.
# ══════════════════════════════════════════════════════════════════════════════

def heuristic_classify(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return [0.34, 0.33, 0.33]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Signal 1: Laplacian Variance  (weight 0.20)
    # FIXED: phone JPEGs compress to lap_var 15–80 — don't flag as AI
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 12:
        # Only flag extreme smoothness (true AI like Midjourney at low res)
        sig1 = {'AI': 0.65, 'Real': 0.25, 'Screenshot': 0.10}
    elif lap_var < 50:
        # Weak signal — compressed JPEG; lean slightly AI but uncertain
        sig1 = {'AI': 0.42, 'Real': 0.42, 'Screenshot': 0.16}
    elif lap_var < 100:
        # Moderate — neutral
        sig1 = {'AI': 0.33, 'Real': 0.50, 'Screenshot': 0.17}
    elif lap_var > 300:
        # Strong real signal — natural sensor noise
        sig1 = {'AI': 0.08, 'Real': 0.82, 'Screenshot': 0.10}
    else:
        # Moderate real signal (100 <= lap_var <= 300)
        sig1 = {'AI': 0.18, 'Real': 0.68, 'Screenshot': 0.14}

    # ── Signal 2: Local Noise Uniformity  (weight 0.15)
    bs = 16
    stds = []
    for y in range(0, h - bs, bs):
        for x in range(0, w - bs, bs):
            stds.append(float(np.std(gray[y:y+bs, x:x+bs])))
    if stds:
        mean_std  = float(np.mean(stds))
        std_std   = float(np.std(stds))
        uniformity = std_std / (mean_std + 1e-6)
    else:
        uniformity = 0.5

    if uniformity < 0.12:
        sig2 = {'AI': 0.72, 'Real': 0.18, 'Screenshot': 0.10}
    elif uniformity < 0.25:
        sig2 = {'AI': 0.50, 'Real': 0.35, 'Screenshot': 0.15}
    elif uniformity > 0.65:
        sig2 = {'AI': 0.12, 'Real': 0.76, 'Screenshot': 0.12}
    else:
        sig2 = {'AI': 0.30, 'Real': 0.52, 'Screenshot': 0.18}

    # ── Signal 3: FFT High-Frequency Ratio  (weight 0.15)
    f      = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag    = np.log1p(np.abs(fshift))
    cy2, cx2 = h//2, w//2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy2)**2 + (x_idx - cx2)**2)
    r_low = min(h, w) // 8
    low_e  = float(np.sum(mag[dist <= r_low]))
    high_e = float(np.sum(mag[dist >  r_low]))
    hf_ratio = high_e / (low_e + high_e + 1e-9)

    if hf_ratio < 0.22:
        sig3 = {'AI': 0.70, 'Real': 0.18, 'Screenshot': 0.12}
    elif hf_ratio < 0.38:
        sig3 = {'AI': 0.42, 'Real': 0.40, 'Screenshot': 0.18}
    elif hf_ratio > 0.58:
        sig3 = {'AI': 0.10, 'Real': 0.75, 'Screenshot': 0.15}
    else:
        sig3 = {'AI': 0.25, 'Real': 0.57, 'Screenshot': 0.18}

    # ── Signal 4: Edge Density  (weight 0.15)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / (h * w + 1e-9)
    if edge_density > 0.12:
        sig4 = {'AI': 0.04, 'Real': 0.10, 'Screenshot': 0.86}
    elif edge_density > 0.06:
        sig4 = {'AI': 0.15, 'Real': 0.28, 'Screenshot': 0.57}
    elif edge_density < 0.015:
        sig4 = {'AI': 0.55, 'Real': 0.36, 'Screenshot': 0.09}
    else:
        sig4 = {'AI': 0.24, 'Real': 0.58, 'Screenshot': 0.18}

    # ── Signal 5: Saturation Analysis  (weight 0.10)
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:,:,1]))
    sat_std  = float(np.std(hsv[:,:,1]))
    if sat_mean > 160 or sat_mean < 15:
        sig5 = {'AI': 0.60, 'Real': 0.25, 'Screenshot': 0.15}
    elif 40 < sat_mean < 130 and sat_std > 25:
        sig5 = {'AI': 0.14, 'Real': 0.76, 'Screenshot': 0.10}
    else:
        sig5 = {'AI': 0.30, 'Real': 0.48, 'Screenshot': 0.22}

    # ── Signal 6: Corner Uniformity  (weight 0.10)
    cs = max(10, min(h, w) // 20)
    corners = [
        img[0:cs, 0:cs], img[0:cs, w-cs:w],
        img[h-cs:h, 0:cs], img[h-cs:h, w-cs:w],
    ]
    uniform_count = sum(1 for c in corners if float(np.var(c)) < 50)
    if uniform_count >= 3:
        sig6 = {'AI': 0.04, 'Real': 0.10, 'Screenshot': 0.86}
    elif uniform_count == 2:
        sig6 = {'AI': 0.14, 'Real': 0.28, 'Screenshot': 0.58}
    else:
        sig6 = {'AI': 0.33, 'Real': 0.52, 'Screenshot': 0.15}

    # ── Signal 7: Wavelet Diagonal Energy  (weight 0.10)
    # FIXED: JPEG compression kills wavelet energy in real photos too
    # Only flag EXTREMELY low values, not just "low"
    if PYWT_OK:
        _, (_, _, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
        diag_e = float(np.sum(cD**2)) / (cD.size + 1e-9)
    else:
        diag_e = 50.0  # neutral fallback

    if diag_e < 0.04:
        # Truly extreme AI-smooth
        sig7 = {'AI': 0.72, 'Real': 0.18, 'Screenshot': 0.10}
    elif diag_e < 0.10:
        # Weak AI signal — could be compressed JPEG
        sig7 = {'AI': 0.42, 'Real': 0.42, 'Screenshot': 0.16}
    elif diag_e > 1.5:
        sig7 = {'AI': 0.10, 'Real': 0.78, 'Screenshot': 0.12}
    else:
        sig7 = {'AI': 0.28, 'Real': 0.56, 'Screenshot': 0.16}

    # ── Signal 8: EXIF Presence  (weight 0.05)
    # FIXED: Social media (WhatsApp/Instagram/Telegram) strips EXIF from REAL photos
    # So missing EXIF is now a VERY weak signal — only software tag counts strongly
    has_exif        = False
    ai_sw_detected  = False
    if EXIFREAD_OK:
        try:
            with open(image_path, 'rb') as ef:
                tags = exifread.process_file(ef, details=False)
            has_exif = len(tags) > 0
            ai_keywords = ['midjourney','stable diffusion','dall-e','firefly',
                           'runway','novelai','comfyui','automatic1111','diffusers']
            sw_str = str(tags.get('Image Software', '')).lower()
            ai_sw_detected = any(kw in sw_str for kw in ai_keywords)
        except Exception:
            has_exif = False

    if ai_sw_detected:
        # Strong signal — confirmed AI software
        sig8 = {'AI': 0.85, 'Real': 0.10, 'Screenshot': 0.05}
    elif not has_exif:
        # WEAK signal — social media strips EXIF, don't over-penalize
        sig8 = {'AI': 0.38, 'Real': 0.47, 'Screenshot': 0.15}
    else:
        # Has EXIF — moderate real signal
        sig8 = {'AI': 0.14, 'Real': 0.76, 'Screenshot': 0.10}

    # ── Weighted combination ──────────────────────────────────────────────────
    weights = [0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05]
    signals = [sig1, sig2, sig3, sig4, sig5, sig6, sig7, sig8]

    result = {'AI': 0.0, 'Real': 0.0, 'Screenshot': 0.0}
    for w_val, sig in zip(weights, signals):
        for cls in result:
            result[cls] += w_val * sig[cls]

    # Normalize
    total = sum(result.values()) + 1e-9

    # CRITICAL RULE: cap any class at 60% so no single signal dominates.
    # Iterative: redistribute excess from capped classes to uncapped ones.
    # This ensures stable, deterministic output — no single signal can push
    # confidence above 60% without multi-signal consensus.
    cap = 0.60
    probs_dict = {k: result[k] / total for k in result}
    for _ in range(10):                          # converges in ≤3 passes
        over  = {k: v for k, v in probs_dict.items() if v > cap}
        if not over:
            break
        excess      = sum(v - cap for v in over.values())
        under_total = sum(v for k, v in probs_dict.items() if v <= cap) + 1e-9
        for k in over:
            probs_dict[k] = cap
        for k in probs_dict:
            if probs_dict[k] < cap:
                probs_dict[k] += excess * (probs_dict[k] / under_total)
    renorm = sum(probs_dict.values()) + 1e-9
    final = [probs_dict['Real'] / renorm,
             probs_dict['AI']   / renorm,
             probs_dict['Screenshot'] / renorm]

    # Sanity check: ensure probabilities are valid floats
    for i in range(3):
        if not math.isfinite(final[i]):
            final[i] = 0.34
    s = sum(final) + 1e-9
    return [f / s for f in final]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path provided'}))
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(json.dumps({'error': f'File not found: {image_path}'}))
        sys.exit(1)

    model_path   = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pt')
    )
    probabilities = None
    model_loaded  = False

    # CNN path: attempt to load trained model, fail gracefully to heuristic
    if TORCH_OK and os.path.isfile(model_path):
        try:
            device = torch.device('cpu')
            model  = AIDetectorCNN()
            state  = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            model.load_state_dict(state)
            model.eval()
            pil_img = Image.open(image_path).convert('RGB')
            tensor  = TRANSFORM(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1).squeeze(0).tolist()
            # Validate CNN output before accepting
            if len(probs) == 3 and all(math.isfinite(p) for p in probs):
                probabilities = probs
                model_loaded  = True
            else:
                probabilities = None
        except Exception as e:
            # Always fall through to heuristic — never crash
            probabilities = None

    # Heuristic fallback: always runs if CNN failed or gave invalid output
    if not model_loaded:
        try:
            probabilities = heuristic_classify(image_path)
        except Exception as e:
            # Ultimate fallback: uniform distribution
            probabilities = [0.34, 0.33, 0.33]

    result = {
        'model_loaded': model_loaded,
        'probabilities': {
            CLASS_NAMES[i]: round(probabilities[i], 6)
            for i in range(len(CLASS_NAMES))
        },
    }
    predicted_idx = probabilities.index(max(probabilities))
    result['predicted_class'] = CLASS_NAMES[predicted_idx]
    result['confidence'] = round(max(probabilities), 6)
    print(json.dumps(result))


if __name__ == '__main__':
    main()