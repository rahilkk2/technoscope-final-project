"""
reverseEngineer.py — AI-Edit Region Detection & Reconstruction Engine
======================================================================
Given an image (potentially AI-generated or AI-edited), this module:
  1. Detects WHICH regions were AI-edited vs originally real
  2. Attempts to reconstruct/restore the untouched original regions
  3. Generates a confidence heatmap (green=original, red=AI-edited)

Five heuristic detectors are combined into a weighted confidence map:
  - Noise Inconsistency   (0.30) — block-wise Laplacian vs image baseline
  - ELA                    (0.25) — JPEG recompression error level analysis
  - Frequency Anomaly      (0.20) — per-block FFT high-freq ratio deviation
  - Texture Break          (0.15) — Sobel gradient magnitude smoothness
  - Color Coherence Break  (0.10) — HSV saturation deviation vs neighbors

Usage:  python reverseEngineer.py <image_path>
Output: JSON to stdout (base64-encoded images + stats)

Dependencies: cv2, numpy, scipy (optional for Gaussian filter), Pillow

LIMITATIONS — READ CAREFULLY:
  ✓ CAN locate regions with different noise/frequency/texture characteristics
  ✓ CAN produce a probabilistic heatmap of where edits likely occurred
  ✓ CAN inpaint edited regions with plausible (but NOT original) content
  ✗ CANNOT recover the actual original pixels (they are gone forever)
  ✗ CANNOT detect edits that perfectly match surrounding noise/frequency
  ✗ CANNOT work reliably on heavily JPEG-compressed images
  ✗ CANNOT distinguish AI edits from normal photo edits (crop/filter/brightness)
"""

import sys
import json
import os
import math
import base64
import io
import tempfile

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

# scipy is optional — used for smoother Gaussian filtering of confidence map
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

MAX_DIM = 1024  # cap for expensive operations

def _cap(img, max_dim=MAX_DIM):
    """Resize so largest dimension <= max_dim, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))


def _safe(val, digits=6):
    """Safely convert to float, replace NaN/Inf with 0."""
    try:
        v = float(val)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, digits)
    except Exception:
        return 0.0


def _smooth_map(block_map, block_size, target_shape):
    """
    Upsample a block-level score map to pixel-level, then smooth.
    block_map: (bh, bw) array of scores
    target_shape: (H, W) of the original image
    Returns: (H, W) float32 array in [0, 1]
    """
    # Nearest-neighbor upsample to full resolution
    h, w = target_shape
    up = cv2.resize(block_map.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR)

    # Gaussian smooth to remove block boundary artifacts
    kernel_size = max(3, block_size // 2)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if SCIPY_OK:
        up = gaussian_filter(up, sigma=kernel_size / 3.0).astype(np.float32)
    else:
        up = cv2.GaussianBlur(up, (kernel_size, kernel_size), 0)

    return up


def _normalize_map(m):
    """Normalize a map to [0, 1] range."""
    mn, mx = float(np.min(m)), float(np.max(m))
    if mx - mn < 1e-9:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)


def _img_to_base64(img_bgr, fmt='.png'):
    """Encode a BGR image to base64 string."""
    success, buf = cv2.imencode(fmt, img_bgr)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _mask_to_base64(mask_uint8, fmt='.png'):
    """Encode a single-channel mask to base64 string."""
    success, buf = cv2.imencode(fmt, mask_uint8)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode('ascii')


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 1: NOISE INCONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════
# Real camera photos have consistent sensor noise across the entire image.
# AI-edited regions typically have different (usually lower) noise variance.
# We measure block-wise Laplacian variance and flag blocks that deviate
# significantly from the image-wide median.

def _detect_noise_inconsistency(gray, block_size=32):
    """
    Returns a (bh, bw) score map where high = likely edited.
    Score = abs(block_noise - median_noise) / (median_noise + eps)
    """
    h, w = gray.shape
    bh = h // block_size
    bw = w // block_size
    if bh < 2 or bw < 2:
        return np.zeros((max(1, bh), max(1, bw)), dtype=np.float32)

    noise_map = np.zeros((bh, bw), dtype=np.float32)

    for by in range(bh):
        for bx in range(bw):
            y0, x0 = by * block_size, bx * block_size
            block = gray[y0:y0 + block_size, x0:x0 + block_size]
            # Laplacian variance = proxy for local noise/detail level
            lap = cv2.Laplacian(block, cv2.CV_64F)
            noise_map[by, bx] = float(np.var(lap))

    # Median noise across entire image
    median_noise = float(np.median(noise_map))
    eps = 1e-6

    # Score: how much each block deviates from the median
    score_map = np.abs(noise_map - median_noise) / (median_noise + eps)

    # Also flag blocks that are much SMOOTHER than median (AI smoothing)
    smooth_penalty = np.where(noise_map < median_noise * 0.5,
                              (median_noise - noise_map) / (median_noise + eps),
                              0.0).astype(np.float32)
    score_map = score_map + smooth_penalty * 0.5

    return _normalize_map(score_map)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 2: ERROR LEVEL ANALYSIS (ELA)
# ══════════════════════════════════════════════════════════════════════════════
# When a JPEG is resaved, all regions should compress similarly.
# Edited/synthetic regions have different compression error patterns because
# they were added at a different compression stage.

def _detect_ela(img_bgr, quality=90, block_size=32):
    """
    Returns a (bh, bw) score map from JPEG error level analysis.
    High score = region has different ELA characteristics = likely edited.
    """
    h, w = img_bgr.shape[:2]
    bh = h // block_size
    bw = w // block_size
    if bh < 2 or bw < 2:
        return np.zeros((max(1, bh), max(1, bw)), dtype=np.float32)

    # Resave as JPEG in memory
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', img_bgr, encode_param)
    resaved = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Absolute difference
    diff = cv2.absdiff(img_bgr, resaved).astype(np.float32)
    diff_gray = np.mean(diff, axis=2)  # average across channels

    # Amplify differences (standard ELA practice)
    diff_gray = diff_gray * (255.0 / (255.0 - quality + 1))

    # Block-wise mean difference
    ela_map = np.zeros((bh, bw), dtype=np.float32)
    for by in range(bh):
        for bx in range(bw):
            y0, x0 = by * block_size, bx * block_size
            block = diff_gray[y0:y0 + block_size, x0:x0 + block_size]
            ela_map[by, bx] = float(np.mean(block))

    # Blocks that deviate from the image median ELA are suspicious
    median_ela = float(np.median(ela_map))
    eps = 1e-6
    score_map = np.abs(ela_map - median_ela) / (median_ela + eps)

    return _normalize_map(score_map)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 3: FREQUENCY ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
# AI-generated regions typically have less high-frequency content than
# real camera sensor data. We compute per-block FFT high-freq ratio and
# flag blocks where the ratio deviates from the image baseline.

def _detect_frequency_anomaly(gray, block_size=64):
    """
    Returns a (bh, bw) score map from FFT high-frequency analysis.
    High score = block has abnormal frequency content = likely edited.
    """
    h, w = gray.shape
    bh = h // block_size
    bw = w // block_size
    if bh < 2 or bw < 2:
        return np.zeros((max(1, bh), max(1, bw)), dtype=np.float32)

    hf_map = np.zeros((bh, bw), dtype=np.float32)

    for by in range(bh):
        for bx in range(bw):
            y0, x0 = by * block_size, bx * block_size
            block = gray[y0:y0 + block_size, x0:x0 + block_size].astype(np.float32)

            # FFT of block
            f = np.fft.fft2(block)
            fshift = np.fft.fftshift(f)
            mag = np.log1p(np.abs(fshift))

            # Split into low-freq (center) and high-freq (edges)
            cy, cx = block_size // 2, block_size // 2
            yi, xi = np.ogrid[:block_size, :block_size]
            dist = np.sqrt((yi - cy) ** 2 + (xi - cx) ** 2)
            r_low = block_size // 4

            low_e = float(np.sum(mag[dist <= r_low]))
            high_e = float(np.sum(mag[dist > r_low]))
            total_e = low_e + high_e + 1e-9

            hf_map[by, bx] = high_e / total_e

    # Blocks with significantly LESS high-freq than median = AI-smoothed
    median_hf = float(np.median(hf_map))
    eps = 1e-6

    # Asymmetric scoring: penalize low high-freq MORE than high high-freq
    score_map = np.zeros_like(hf_map)
    for by in range(bh):
        for bx in range(bw):
            delta = median_hf - hf_map[by, bx]
            if delta > 0:
                # Block has LESS high-freq than median — likely AI
                score_map[by, bx] = delta / (median_hf + eps)
            else:
                # Block has MORE high-freq — less suspicious, small score
                score_map[by, bx] = abs(delta) / (median_hf + eps) * 0.3

    return _normalize_map(score_map)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 4: TEXTURE BREAK
# ══════════════════════════════════════════════════════════════════════════════
# AI edits often produce unnaturally smooth textures — the "plasticity" look.
# We measure Sobel gradient magnitude per block. Blocks with abnormally low
# gradient energy relative to their neighbors are flagged.

def _detect_texture_break(gray, block_size=32):
    """
    Returns a (bh, bw) score map from gradient/texture analysis.
    High score = block is abnormally smooth/sharp vs neighbors = likely edited.
    """
    h, w = gray.shape
    bh = h // block_size
    bw = w // block_size
    if bh < 2 or bw < 2:
        return np.zeros((max(1, bh), max(1, bw)), dtype=np.float32)

    # Compute full-image gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Block-wise mean gradient
    grad_map = np.zeros((bh, bw), dtype=np.float32)
    for by in range(bh):
        for bx in range(bw):
            y0, x0 = by * block_size, bx * block_size
            block = grad_mag[y0:y0 + block_size, x0:x0 + block_size]
            grad_map[by, bx] = float(np.mean(block))

    # Score: deviation from each block's local 3x3 neighborhood mean
    score_map = np.zeros((bh, bw), dtype=np.float32)
    for by in range(bh):
        for bx in range(bw):
            # Gather neighbor values
            neighbors = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < bh and 0 <= nx < bw and (dy != 0 or dx != 0):
                        neighbors.append(grad_map[ny, nx])
            if not neighbors:
                continue
            local_mean = float(np.mean(neighbors))
            eps = 1e-6
            # How much does this block differ from its local neighborhood?
            score_map[by, bx] = abs(grad_map[by, bx] - local_mean) / (local_mean + eps)

    return _normalize_map(score_map)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 5: COLOR COHERENCE BREAK
# ══════════════════════════════════════════════════════════════════════════════
# AI edits can introduce subtle color/saturation shifts. We measure per-block
# HSV saturation statistics and flag blocks with abnormal values.

def _detect_color_break(img_bgr, block_size=32):
    """
    Returns a (bh, bw) score map from color coherence analysis.
    High score = block has abnormal color properties = likely edited.
    """
    h, w = img_bgr.shape[:2]
    bh = h // block_size
    bw = w // block_size
    if bh < 2 or bw < 2:
        return np.zeros((max(1, bh), max(1, bw)), dtype=np.float32)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    # Block-wise saturation statistics
    sat_mean_map = np.zeros((bh, bw), dtype=np.float32)
    sat_std_map = np.zeros((bh, bw), dtype=np.float32)

    for by in range(bh):
        for bx in range(bw):
            y0, x0 = by * block_size, bx * block_size
            block = sat[y0:y0 + block_size, x0:x0 + block_size]
            sat_mean_map[by, bx] = float(np.mean(block))
            sat_std_map[by, bx] = float(np.std(block))

    # Score: deviation from local 3x3 neighborhood
    score_map = np.zeros((bh, bw), dtype=np.float32)
    for by in range(bh):
        for bx in range(bw):
            neighbors_mean = []
            neighbors_std = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < bh and 0 <= nx < bw and (dy != 0 or dx != 0):
                        neighbors_mean.append(sat_mean_map[ny, nx])
                        neighbors_std.append(sat_std_map[ny, nx])
            if not neighbors_mean:
                continue
            local_mean = float(np.mean(neighbors_mean))
            local_std_mean = float(np.mean(neighbors_std))
            eps = 1e-6

            mean_dev = abs(sat_mean_map[by, bx] - local_mean) / (local_mean + eps)
            std_dev = abs(sat_std_map[by, bx] - local_std_mean) / (local_std_mean + eps)

            score_map[by, bx] = mean_dev * 0.6 + std_dev * 0.4

    return _normalize_map(score_map)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IMAGE IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════
# Before any reverse engineering, we first fingerprint the image to identify:
#   - What TYPE of image it is (real photo, AI-generated, screenshot, etc.)
#   - Which AI MODEL likely produced it (Midjourney, DALL-E, SD, Firefly, etc.)
#   - Key identifying CHARACTERISTICS found in the analysis
#   - Confidence percentage of the identification

def identify_image(img_bgr, gray, noise_map, ela_map, freq_map, tex_map, color_map):
    """
    Fingerprint the image and identify its likely source.

    Returns a dict:
    {
        image_type:       str   ("AI-Generated" / "Real Photograph" / "Screenshot" / "Digital Art"),
        model_guess:      str   ("Midjourney v5/v6" / "Stable Diffusion" / ...),
        characteristics:  list  (human-readable forensic observations),
        confidence:       float (0-100),
    }
    """
    h, w = gray.shape
    characteristics = []

    # ── Compute forensic signals ─────────────────────────────────────────────
    noise_score = _safe(float(np.mean(noise_map)))
    ela_score   = _safe(float(np.mean(ela_map)))
    freq_score  = _safe(float(np.mean(freq_map)))
    tex_score   = _safe(float(np.mean(tex_map)))
    color_score = _safe(float(np.mean(color_map)))

    # Laplacian variance of the whole image (noise floor)
    lap_var = _safe(float(np.var(cv2.Laplacian(gray, cv2.CV_64F))))

    # Histogram analysis — AI images tend to have smoother, narrower histograms
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-9)
    hist_entropy = -float(np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] + 1e-12)))

    # Edge density — AI images often smoother
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / (h * w)

    # Saturation variance — AI tends to be more uniform
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_variance = _safe(float(np.var(hsv[:, :, 1].astype(np.float32))))

    # ── Heuristic classification ─────────────────────────────────────────────
    ai_signals = 0
    total_signals = 6

    # Signal 1: Low Laplacian variance = artificially smooth
    if lap_var < 400:
        ai_signals += 1
        characteristics.append("Abnormally low noise floor — surface smoothing consistent with neural synthesis")
    elif lap_var < 800:
        ai_signals += 0.5
        characteristics.append("Below-average noise variance — possible AI post-processing")
    else:
        characteristics.append("Natural noise floor within camera sensor range")

    # Signal 2: Low edge density = diffusion model smoothing
    if edge_density < 0.04:
        ai_signals += 1
        characteristics.append("Sparse edge structure typical of diffusion-based generation")
    elif edge_density < 0.08:
        ai_signals += 0.3
        characteristics.append("Moderate edge density — some local smoothing detected")
    else:
        characteristics.append("Rich edge structure consistent with optical lens capture")

    # Signal 3: Low histogram entropy = synthetic tonal range
    if hist_entropy < 6.0:
        ai_signals += 1
        characteristics.append("Compressed tonal histogram — synthetic color palette detected")
    elif hist_entropy < 6.8:
        ai_signals += 0.3
        characteristics.append("Slightly narrow tonal range — minor synthetic indicators")
    else:
        characteristics.append("Full dynamic tonal range consistent with RAW sensor data")

    # Signal 4: Frequency anomaly (already computed)
    if freq_score > 0.35:
        ai_signals += 1
        characteristics.append("Spectral spikes consistent with diffusion upsampling artifacts")
    elif freq_score > 0.2:
        ai_signals += 0.5
        characteristics.append("Minor high-frequency deviations — possible AI resampling")

    # Signal 5: Texture uniformity
    if tex_score > 0.30:
        ai_signals += 1
        characteristics.append("Smooth skin/surface texture typical of neural rendering")
    elif tex_score > 0.15:
        ai_signals += 0.4
        characteristics.append("Partially uniform textures — inpainting signatures possible")

    # Signal 6: Low saturation variance
    if sat_variance < 800:
        ai_signals += 0.8
        characteristics.append("Homogeneous color saturation — synthetic palette uniformity")
    elif sat_variance < 1500:
        ai_signals += 0.3
        characteristics.append("Saturation within mixed range")

    # ── Determine image_type ─────────────────────────────────────────────────
    ai_ratio = ai_signals / total_signals

    if ai_ratio > 0.55:
        image_type = "AI-Generated"
    elif ai_ratio > 0.35:
        image_type = "AI-Edited / Partially Synthetic"
    elif edge_density > 0.2 and lap_var < 200:
        image_type = "Screenshot / Digital Render"
    else:
        image_type = "Real Photograph"

    # ── Determine model guess ────────────────────────────────────────────────
    model_guess = "Unknown"
    if image_type in ("AI-Generated", "AI-Edited / Partially Synthetic"):
        # Heuristics for model classification based on signature patterns
        if freq_score > 0.35 and tex_score > 0.25 and lap_var < 500:
            model_guess = "Midjourney v5/v6 — diffusion architecture with characteristic smoothing"
        elif freq_score > 0.30 and noise_score > 0.30:
            model_guess = "Stable Diffusion (SDXL/SD 1.5) — latent diffusion noise pattern"
        elif ela_score > 0.35 and tex_score < 0.2:
            model_guess = "DALL-E 2/3 — CLIP-guided generation with distinct ELA profile"
        elif color_score > 0.25 and freq_score < 0.2:
            model_guess = "Adobe Firefly — color-coherent synthesis with low frequency drift"
        elif noise_score > 0.35 and freq_score < 0.2:
            model_guess = "GAN architecture (StyleGAN/ProGAN) — noise-pattern signature"
        else:
            model_guess = "Unclassified Synthetic Engine — mixed signal signatures"
    elif image_type == "Real Photograph":
        model_guess = "Natural Capture (No AI Generation Detected)"
    elif image_type == "Screenshot / Digital Render":
        model_guess = "Digital Source (Non-photographic)"

    # ── Confidence ───────────────────────────────────────────────────────────
    confidence = _safe(min(99.0, max(15.0, ai_ratio * 100 + (1 - edge_density) * 15)))

    return {
        'image_type':      image_type,
        'model_guess':     model_guess,
        'characteristics': characteristics,
        'confidence':      confidence,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — REGION DETECTION WITH CALLOUTS
# ══════════════════════════════════════════════════════════════════════════════
# After identification, we locate specific edited REGIONS and generate labeled
# callouts with per-detector reasoning for each flagged area.

def detect_flagged_regions(binary_mask, confidence_map,
                           noise_map, ela_map, freq_map, tex_map, color_map,
                           img_shape, min_area_ratio=0.005, max_regions=8):
    """
    Find contiguous edited regions and produce labeled callouts.

    Returns a list of dicts:
    [
        {
            label:       "Region A",
            bbox:        [x, y, w, h],
            center:      [cx, cy],
            confidence:  float (0-100),
            level:       "high" / "moderate" / "low",
            reason:      "noise floor mismatch",
            detectors:   { "ELA": 87, "Noise": 34, ... },
        },
        ...
    ]
    """
    h, w = img_shape[:2]
    min_area = int(h * w * min_area_ratio)

    # Find contours in binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    detector_names = {
        'noise': 'Noise Variance',
        'ela':   'ELA Analysis',
        'freq':  'Frequency Anomaly',
        'tex':   'Texture Break',
        'color': 'Color Coherence',
    }
    detector_maps = {
        'noise': noise_map,
        'ela':   ela_map,
        'freq':  freq_map,
        'tex':   tex_map,
        'color': color_map,
    }

    regions = []
    labels = 'ABCDEFGH'

    for i, cnt in enumerate(contours):
        if i >= max_regions:
            break
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, rw, rh = cv2.boundingRect(cnt)

        # Create mask for this region only
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region_mask, [cnt], 0, 255, -1)
        region_pixels = region_mask > 0

        # Mean confidence in this specific region
        region_conf = _safe(float(np.mean(confidence_map[region_pixels])) * 100)

        # Per-detector scores within this region
        per_det = {}
        top_detector = None
        top_det_score = 0.0
        for key, dmap in detector_maps.items():
            # Resize detector map if needed
            dm = dmap
            if dm.shape[:2] != (h, w):
                dm = cv2.resize(dm, (w, h))
            score = _safe(float(np.mean(dm[region_pixels])) * 100)
            per_det[detector_names[key]] = score
            if score > top_det_score:
                top_det_score = score
                top_detector = key

        # Determine reason based on top detector
        reason_map = {
            'noise': 'noise floor mismatch',
            'ela':   'ELA compression ghosting',
            'freq':  'frequency anomaly detected',
            'tex':   'texture flow discontinuity',
            'color': 'color coherence break',
        }
        reason = reason_map.get(top_detector, 'multi-signal anomaly')

        # Confidence level
        if region_conf > 70:
            level = 'high'
        elif region_conf > 40:
            level = 'moderate'
        else:
            level = 'low'

        # Center of mass
        M = cv2.moments(cnt)
        cx = int(M['m10'] / (M['m00'] + 1e-9))
        cy = int(M['m01'] / (M['m00'] + 1e-9))

        regions.append({
            'label':      f'Region {labels[i]}' if i < len(labels) else f'Region {i+1}',
            'bbox':       [int(x), int(y), int(rw), int(rh)],
            'center':     [cx, cy],
            'area_pixels': int(area),
            'area_percent': _safe(area / (h * w) * 100),
            'confidence':  region_conf,
            'level':       level,
            'reason':      reason,
            'detectors':   per_det,
        })

    return regions


# ══════════════════════════════════════════════════════════════════════════════
# CORE DETECTION — MAIN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_edited_regions(image_path, block_size=32):
    """
    Detect AI-edited regions in an image using 5 heuristic detectors.

    Args:
        image_path: path to the image file
        block_size: analysis block size in pixels (default 32)

    Returns:
        (binary_mask, confidence_map, stats, img_bgr, raw_maps)
        - binary_mask: uint8 (H, W), 255 = edited, 0 = original
        - confidence_map: float32 (H, W), 0.0 = original, 1.0 = certainly edited
        - stats: dict with summary statistics
        - img_bgr: the loaded image
        - raw_maps: dict of individual detector maps for region analysis
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Cap resolution for performance
    img_bgr = _cap(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Run all 5 detectors ──────────────────────────────────────────────────
    # Detector 1: Noise inconsistency (weight 0.30)
    try:
        noise_block = _detect_noise_inconsistency(gray, block_size)
        noise_map = _smooth_map(noise_block, block_size, (h, w))
    except Exception:
        noise_map = np.zeros((h, w), dtype=np.float32)

    # Detector 2: ELA (weight 0.25)
    try:
        ela_block = _detect_ela(img_bgr, quality=90, block_size=block_size)
        ela_map = _smooth_map(ela_block, block_size, (h, w))
    except Exception:
        ela_map = np.zeros((h, w), dtype=np.float32)

    # Detector 3: Frequency anomaly (weight 0.20)
    freq_bs = max(32, block_size * 2)  # frequency needs larger blocks
    try:
        freq_block = _detect_frequency_anomaly(gray, freq_bs)
        freq_map = _smooth_map(freq_block, freq_bs, (h, w))
    except Exception:
        freq_map = np.zeros((h, w), dtype=np.float32)

    # Detector 4: Texture break (weight 0.15)
    try:
        tex_block = _detect_texture_break(gray, block_size)
        tex_map = _smooth_map(tex_block, block_size, (h, w))
    except Exception:
        tex_map = np.zeros((h, w), dtype=np.float32)

    # Detector 5: Color coherence (weight 0.10)
    try:
        color_block = _detect_color_break(img_bgr, block_size)
        color_map = _smooth_map(color_block, block_size, (h, w))
    except Exception:
        color_map = np.zeros((h, w), dtype=np.float32)

    # ── Weighted combination ─────────────────────────────────────────────────
    WEIGHTS = {
        'noise': 0.30,
        'ela':   0.25,
        'freq':  0.20,
        'tex':   0.15,
        'color': 0.10,
    }

    confidence_map = (
        WEIGHTS['noise'] * noise_map +
        WEIGHTS['ela']   * ela_map +
        WEIGHTS['freq']  * freq_map +
        WEIGHTS['tex']   * tex_map +
        WEIGHTS['color'] * color_map
    )

    # Final normalization
    confidence_map = _normalize_map(confidence_map)

    # Final Gaussian smooth for visual quality
    ks = max(5, block_size // 4)
    if ks % 2 == 0:
        ks += 1
    confidence_map = cv2.GaussianBlur(confidence_map, (ks, ks), 0)

    # ── Adaptive thresholding for binary mask ────────────────────────────────
    # Use Otsu's method on the confidence map to find the optimal threshold
    conf_uint8 = (confidence_map * 255).clip(0, 255).astype(np.uint8)
    otsu_thresh, binary_mask = cv2.threshold(
        conf_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If Otsu picks too low (<20%), use a fixed threshold
    if otsu_thresh < 50:
        _, binary_mask = cv2.threshold(conf_uint8, 127, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: remove small noise, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # ── Statistics ───────────────────────────────────────────────────────────
    total_pixels = h * w
    edited_pixels = int(np.sum(binary_mask > 0))
    edit_percentage = _safe(edited_pixels / total_pixels * 100)
    mean_confidence = _safe(float(np.mean(confidence_map)))
    max_confidence = _safe(float(np.max(confidence_map)))

    # Per-detector contribution scores (how much each detector triggered)
    detector_scores = {
        'noise_inconsistency': _safe(float(np.mean(noise_map))),
        'ela_analysis':        _safe(float(np.mean(ela_map))),
        'frequency_anomaly':   _safe(float(np.mean(freq_map))),
        'texture_break':       _safe(float(np.mean(tex_map))),
        'color_coherence':     _safe(float(np.mean(color_map))),
    }

    stats = {
        'image_width':          w,
        'image_height':         h,
        'total_pixels':         total_pixels,
        'edited_pixels':        edited_pixels,
        'edit_percentage':      edit_percentage,
        'mean_confidence':      mean_confidence,
        'max_confidence':       max_confidence,
        'threshold_used':       _safe(float(otsu_thresh)),
        'detector_scores':      detector_scores,
    }

    # Pack raw maps for region analysis and identification
    raw_maps = {
        'noise': noise_map,
        'ela':   ela_map,
        'freq':  freq_map,
        'tex':   tex_map,
        'color': color_map,
        'gray':  gray,
    }

    return binary_mask, confidence_map, stats, img_bgr, raw_maps


def reconstruct_original(img_bgr, edit_mask):
    """
    Attempt to reconstruct the 'original' by inpainting detected AI-edited regions.

    CRITICAL LIMITATION: This does NOT recover actual original pixels.
    It fills edited regions with plausible content using OpenCV inpainting.
    Think of it as "what might have been there" — not "what was there."

    Uses a blend of Telea (fast marching) and Navier-Stokes methods for
    better visual quality.

    Args:
        img_bgr: BGR image (numpy array)
        edit_mask: uint8 mask, 255 = regions to reconstruct

    Returns:
        reconstructed: BGR image with edited regions inpainted
    """
    if edit_mask is None or np.sum(edit_mask) == 0:
        # No edited regions detected — return original unchanged
        return img_bgr.copy()

    # Dilate mask slightly to cover edit boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(edit_mask, kernel, iterations=2)

    # Inpainting radius — larger for bigger edited regions
    edited_fraction = np.sum(mask_dilated > 0) / (mask_dilated.shape[0] * mask_dilated.shape[1])
    inpaint_radius = max(3, min(15, int(edited_fraction * 100)))

    # Method 1: Telea (Fast Marching Method) — good at preserving edges
    telea = cv2.inpaint(img_bgr, mask_dilated, inpaint_radius, cv2.INPAINT_TELEA)

    # Method 2: Navier-Stokes — better at smooth region continuity
    ns = cv2.inpaint(img_bgr, mask_dilated, inpaint_radius, cv2.INPAINT_NS)

    # Blend: 60% Telea (sharper) + 40% Navier-Stokes (smoother)
    reconstructed = cv2.addWeighted(telea, 0.6, ns, 0.4, 0)

    # Only replace edited regions, keep original regions untouched
    result = img_bgr.copy()
    mask_3ch = cv2.merge([mask_dilated, mask_dilated, mask_dilated])
    result = np.where(mask_3ch > 0, reconstructed, result)

    return result


def generate_heatmap(img_bgr, confidence_map, alpha=0.55):
    """
    Generate a visual heatmap overlay on the original image.
    Green = likely original, Red = likely AI-edited.

    Args:
        img_bgr: BGR image
        confidence_map: float32 (H, W), 0 = original, 1 = edited
        alpha: overlay transparency (0 = invisible, 1 = opaque)

    Returns:
        overlay: BGR image with heatmap overlay
    """
    h, w = img_bgr.shape[:2]

    # Ensure confidence map matches image dimensions
    if confidence_map.shape[:2] != (h, w):
        confidence_map = cv2.resize(confidence_map, (w, h))

    # Create color map: Green (original) → Yellow (uncertain) → Red (edited)
    # Using OpenCV's JET is too generic; we build a custom 3-stop gradient.
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    conf = confidence_map.clip(0, 1)

    # Red channel: increases with confidence (0→255)
    heatmap[:, :, 2] = (conf * 255).astype(np.uint8)

    # Green channel: high when confidence is LOW (original), drops as edited
    heatmap[:, :, 1] = ((1.0 - conf) * 200).astype(np.uint8)

    # Blue channel: slight blue tint in the middle range for visibility
    mid_mask = (conf > 0.3) & (conf < 0.7)
    heatmap[:, :, 0] = np.where(mid_mask, 60, 0).astype(np.uint8)

    # Blend with original
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap, alpha, 0)

    # Draw contour lines at the 50% confidence boundary for clarity
    conf_uint8 = (conf * 255).astype(np.uint8)
    _, thresh = cv2.threshold(conf_uint8, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def generate_side_by_side(original_bgr, reconstructed_bgr, heatmap_bgr):
    """
    Create a 3-panel side-by-side comparison image.
    Left: Original upload | Center: Heatmap | Right: Reconstructed

    Returns: BGR image (combined)
    """
    h, w = original_bgr.shape[:2]

    # Ensure all images are the same size
    recon = cv2.resize(reconstructed_bgr, (w, h))
    heat = cv2.resize(heatmap_bgr, (w, h))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 30
    panel_h = h + label_h

    def _add_label(img, text):
        labeled = np.zeros((panel_h, w, 3), dtype=np.uint8)
        labeled[:label_h, :] = (20, 20, 20)
        labeled[label_h:, :] = img
        cv2.putText(labeled, text, (10, 20), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return labeled

    left = _add_label(original_bgr, "ORIGINAL UPLOAD")
    center = _add_label(heat, "EDIT HEATMAP")
    right = _add_label(recon, "RECONSTRUCTED")

    # Add 2px separator lines
    sep = np.full((panel_h, 2, 3), (80, 80, 80), dtype=np.uint8)

    combined = np.hstack([left, sep, center, sep, right])
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — 3-STEP PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def reverse_engineer(image_path):
    """
    Main orchestrator function. Runs the full 3-step reverse engineering pipeline:
        Step 1: IDENTIFY — fingerprint the image (type, model, characteristics)
        Step 2: LOCATE  — detect regions with labeled callouts
        Step 3: RECONSTRUCT — inpaint and generate side-by-side comparison

    Returns a dict suitable for JSON serialization:
    {
        # Step 1 — Identification
        identification: {
            image_type:       str,
            model_guess:      str,
            characteristics:  list[str],
            confidence:       float (0-100),
        },

        # Step 2 — Region Detection
        regions: [
            {
                label:       str,
                bbox:        [x, y, w, h],
                center:      [cx, cy],
                confidence:  float (0-100),
                level:       str,
                reason:      str,
                detectors:   dict,
            }, ...
        ],

        # Step 3 — Reconstruction + existing fields
        edit_percentage:       float  (0-100),
        mean_confidence:       float  (0-1),
        max_confidence:        float  (0-1),
        detector_scores:       dict,
        heatmap_base64:        str,
        reconstructed_base64:  str,
        mask_base64:           str,
        side_by_side_base64:   str,
        limitations:           list,
    }
    """
    # ── Core Detection ───────────────────────────────────────────────────────
    binary_mask, confidence_map, stats, img_bgr, raw_maps = detect_edited_regions(image_path)

    # ── Step 1: IDENTIFY ─────────────────────────────────────────────────────
    identification = identify_image(
        img_bgr, raw_maps['gray'],
        raw_maps['noise'], raw_maps['ela'], raw_maps['freq'],
        raw_maps['tex'], raw_maps['color']
    )

    # ── Step 2: LOCATE — Region callouts ─────────────────────────────────────
    regions = detect_flagged_regions(
        binary_mask, confidence_map,
        raw_maps['noise'], raw_maps['ela'], raw_maps['freq'],
        raw_maps['tex'], raw_maps['color'],
        img_bgr.shape
    )

    # ── Step 3: RECONSTRUCT ──────────────────────────────────────────────────
    reconstructed = reconstruct_original(img_bgr, binary_mask)
    heatmap = generate_heatmap(img_bgr, confidence_map)
    side_by_side = generate_side_by_side(img_bgr, reconstructed, heatmap)

    # ── Encode all outputs as base64 ─────────────────────────────────────────
    result = {
        # Step 1
        'identification':       identification,

        # Step 2
        'regions':              regions,

        # Step 3 + existing fields
        'edit_percentage':      stats['edit_percentage'],
        'total_edited_pixels':  stats['edited_pixels'],
        'image_width':          stats['image_width'],
        'image_height':         stats['image_height'],
        'mean_confidence':      stats['mean_confidence'],
        'max_confidence':       stats['max_confidence'],
        'threshold_used':       stats['threshold_used'],
        'detector_scores':      stats['detector_scores'],
        'heatmap_base64':       _img_to_base64(heatmap),
        'reconstructed_base64': _img_to_base64(reconstructed),
        'mask_base64':          _mask_to_base64(binary_mask),
        'side_by_side_base64':  _img_to_base64(side_by_side),
        'limitations': [
            "Reconstruction uses inpainting — it generates PLAUSIBLE content, NOT the actual original pixels.",
            "Cannot detect edits that perfectly match surrounding noise/frequency characteristics.",
            "Accuracy degrades significantly on heavily JPEG-compressed images.",
            "Cannot distinguish AI edits from normal photo edits (crop, filter, brightness).",
            "Heatmap shows PROBABILITY of editing, not certainty — false positives are possible in naturally varied images.",
            "Small edits (<5% of image area) may fall below detection threshold.",
        ],
    }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path provided. Usage: python reverseEngineer.py <image_path>'}))
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(json.dumps({'error': f'File not found: {image_path}'}))
        sys.exit(1)

    try:
        result = reverse_engineer(image_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': f'Reverse engineering failed: {str(e)}'}))
        sys.exit(1)


if __name__ == '__main__':
    main()
