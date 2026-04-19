"""
metadata.py — EXIF and screenshot metadata extraction.
Usage: python metadata.py <image_path>
Output: JSON to stdout
"""

import sys
import json
import os

# ── Dependency checks (hard requirements) ─────────────────────────────────────
try:
    import cv2
except ImportError:
    print(json.dumps({"error": "Missing library: cv2. Run: pip install opencv-python"}))
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print(json.dumps({"error": "Missing library: numpy. Run: pip install numpy"}))
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print(json.dumps({"error": "Missing library: Pillow. Run: pip install Pillow"}))
    sys.exit(1)

# ── exifread is OPTIONAL — gracefully disabled if missing ─────────────────────
try:
    import exifread
    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False


# ── EXIF Extraction ───────────────────────────────────────────────────────────

def extract_exif(image_path):
    """Read EXIF tags. Returns empty dict if exifread not installed or fails."""
    if not EXIFREAD_AVAILABLE:
        return {}
    exif_data = {}
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        for key, value in tags.items():
            exif_data[key] = str(value)
    except Exception:
        pass
    return exif_data


def analyze_exif_signals(exif_data):
    signals = {
        'has_exif':         len(exif_data) > 0,
        'has_gps':          any('GPS' in k for k in exif_data),
        'has_camera_make':  'Image Make'     in exif_data,
        'has_camera_model': 'Image Model'    in exif_data,
        'has_datetime':     'Image DateTime' in exif_data,
        'has_software_tag': 'Image Software' in exif_data,
        'software_tag':      exif_data.get('Image Software', ''),
        'exifread_available': EXIFREAD_AVAILABLE,
    }
    signals['exif_stripped'] = not signals['has_exif']

    ai_keywords = [
        'midjourney', 'stable diffusion', 'dall-e', 'firefly',
        'runway', 'imagen', 'novelai', 'automatic1111', 'comfyui',
        'controlnet', 'lora', 'diffusers',
    ]
    sw = signals['software_tag'].lower()
    signals['ai_software_detected'] = any(kw in sw for kw in ai_keywords)
    return signals


# ── File Metadata ─────────────────────────────────────────────────────────────

def estimate_file_metadata(image_path):
    stat = os.stat(image_path)
    return {
        'file_size_bytes': stat.st_size,
        'extension':       os.path.splitext(image_path)[1].lower(),
    }


# ── Screenshot Signals ────────────────────────────────────────────────────────

def detect_screenshot_signals(image):
    h, w = image.shape[:2]
    corner_size = max(10, min(h, w) // 20)
    corners = [
        image[0:corner_size, 0:corner_size],
        image[0:corner_size, w - corner_size:w],
        image[h - corner_size:h, 0:corner_size],
        image[h - corner_size:h, w - corner_size:w],
    ]
    uniform_corners = sum(1 for c in corners if float(np.var(c)) < 50)
    round_dims      = (w % 8 == 0) and (h % 8 == 0)
    return {
        'uniform_corner_count': uniform_corners,
        'round_dimensions':     round_dims,
        'image_width':          w,
        'image_height':         h,
        'aspect_ratio':         round(w / h, 4) if h > 0 else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path provided'}))
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(json.dumps({'error': f'File not found: {image_path}'}))
        sys.exit(1)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(json.dumps({'error': 'cv2 could not read image'}))
        sys.exit(1)

    exif_data       = extract_exif(image_path)
    exif_signals    = analyze_exif_signals(exif_data)
    file_meta       = estimate_file_metadata(image_path)
    screenshot_sigs = detect_screenshot_signals(img_bgr)

    result = {
        'exif':       exif_signals,
        'file':       file_meta,
        'screenshot': screenshot_sigs,
    }

    print(json.dumps(result))


if __name__ == '__main__':
    main()
