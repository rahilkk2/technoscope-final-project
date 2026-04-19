"""
featureExtractor.py — Full 65-Feature + 15 Advanced = 80 Feature Forensic Engine
Groups: A=Sensor/Noise, B=Texture, C=Color, D=Edge/Geometry,
        E=Frequency, F=File/Metadata, G=Semantic/Structural, ADV=Advanced
Usage: python featureExtractor.py <image_path>
Output: JSON to stdout
CRASH-PROOF: every computation wrapped in try/except BaseException
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
    import pywt
    PYWT_OK = True
except ImportError:
    PYWT_OK = False

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

try:
    from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

MAX_DIM = 512  # max dimension for expensive operations

def cap(img):
    """Resize so largest dim <= MAX_DIM (for expensive ops like corrcoef/wavelet)."""
    h, w = img.shape[:2]
    if max(h, w) <= MAX_DIM:
        return img
    scale = MAX_DIM / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h))

def safe(val, digits=6):
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return round(v, digits)
    except Exception:
        return 0.0

def img_entropy(arr):
    try:
        flat = arr.flatten().astype(np.uint8)
        counts = np.bincount(flat, minlength=256).astype(np.float64)
        p = counts / (counts.sum() + 1e-9)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))
    except Exception:
        return 0.0

def channel_skew(ch):
    try:
        flat = ch.flatten().astype(np.float64)
        if SCIPY_OK:
            return float(scipy_skew(flat))
        mean = np.mean(flat); std = np.std(flat)
        if std < 1e-9: return 0.0
        return float(np.mean(((flat - mean) / std) ** 3))
    except Exception:
        return 0.0

def channel_kurt(ch):
    try:
        flat = ch.flatten().astype(np.float64)
        if SCIPY_OK:
            return float(scipy_kurtosis(flat))
        mean = np.mean(flat); std = np.std(flat)
        if std < 1e-9: return 0.0
        return float(np.mean(((flat - mean) / std) ** 4) - 3.0)
    except Exception:
        return 0.0

def try_feat(fn, *args, default=0.0):
    """Run a single feature function and swallow any error."""
    try:
        return fn(*args)
    except BaseException:
        return default


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A — Sensor & Noise (10)
# ══════════════════════════════════════════════════════════════════════════════

def group_a_sensor_noise(img_bgr, img_rgb, gray):
    r = {}
    h, w = gray.shape

    # Work on capped version for expensive ops
    img_rgb_s = cap(img_rgb)
    gray_s    = cap(gray)

    # A1 — sensor_pattern_noise
    def _spn():
        b = cv2.GaussianBlur(gray_s.astype(np.float32), (5, 5), 0)
        return safe(np.var(gray_s.astype(np.float32) - b))
    r['sensor_pattern_noise'] = try_feat(_spn)

    # A2 — prnu_score
    def _prnu():
        res = []
        for c in range(3):
            ch = img_rgb_s[:, :, c].astype(np.float32)
            bl = cv2.GaussianBlur(ch, (3, 3), 0)
            res.append((ch - bl).flatten())
        rg = float(np.corrcoef(res[0], res[1])[0, 1])
        gb = float(np.corrcoef(res[1], res[2])[0, 1])
        return safe((rg + gb) / 2.0)
    r['prnu_score'] = try_feat(_prnu)

    # A3 — iso_noise_variance
    def _iso():
        k = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
        hp = cv2.filter2D(gray_s.astype(np.float32), -1, k)
        return safe(np.var(hp))
    r['iso_noise_variance'] = try_feat(_iso)

    # A4 — channel noise ratios
    def _ch_noise():
        vs = []
        for c in range(3):
            ch = img_rgb_s[:, :, c].astype(np.float32)
            bl = cv2.GaussianBlur(ch, (3, 3), 0)
            vs.append(float(np.var(ch - bl)))
        rv, gv, bv = vs
        r['channel_noise_ratio_r'] = safe(rv)
        r['channel_noise_ratio_g'] = safe(gv)
        r['channel_noise_ratio_b'] = safe(bv)
        r['green_noise_dominance']  = safe(gv / (rv + bv + 1e-9))
    try_feat(_ch_noise, default=None)

    # A5 — noise_patch_correlation
    def _patch_corr():
        bs = 16
        blurred = cv2.GaussianBlur(gray_s.astype(np.float32), (5, 5), 0)
        res_map  = gray_s.astype(np.float32) - blurred
        patches  = [res_map[y:y+bs, x:x+bs].flatten()
                    for y in range(0, gray_s.shape[0] - bs, bs)
                    for x in range(0, gray_s.shape[1] - bs, bs)]
        sample = patches[:min(30, len(patches))]
        corrs = []
        for i in range(len(sample) - 1):
            c_ = np.corrcoef(sample[i], sample[i+1])[0, 1]
            if not math.isnan(float(c_)):
                corrs.append(float(c_))
        return safe(np.mean(corrs) if corrs else 0.0)
    r['noise_patch_correlation'] = try_feat(_patch_corr)

    # A6 — high_freq_noise_entropy
    def _hf_ent():
        k = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
        hp = cv2.filter2D(gray_s.astype(np.float32), -1, k)
        hp_n = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return safe(img_entropy(hp_n))
    r['high_freq_noise_entropy'] = try_feat(_hf_ent)

    # A7 — demosaicing_artifact_score
    def _demo():
        g = img_rgb_s[:, :, 1].astype(np.float32)
        G = np.abs(np.fft.fftshift(np.fft.fft2(g)))
        cy2, cx2 = G.shape[0]//2, G.shape[1]//2
        hy2, hx2 = G.shape[0]//4, G.shape[1]//4
        qms = [float(np.mean(G[cy2-hy2:cy2, cx2-hx2:cx2])),
               float(np.mean(G[cy2-hy2:cy2, cx2:cx2+hx2])),
               float(np.mean(G[cy2:cy2+hy2, cx2-hx2:cx2])),
               float(np.mean(G[cy2:cy2+hy2, cx2:cx2+hx2]))]
        return safe(np.std(qms) / (np.mean(qms) + 1e-9))
    r['demosaicing_artifact_score'] = try_feat(_demo)

    # A8 — cfa_interpolation_trace
    def _cfa():
        R_ = img_rgb_s[:,:,0].astype(np.float32)
        G_ = img_rgb_s[:,:,1].astype(np.float32)
        B_ = img_rgb_s[:,:,2].astype(np.float32)
        RB = (R_ + B_) / 2.0
        cov = np.mean((G_ - G_.mean()) * (RB - RB.mean()))
        return safe(cov / (np.std(G_) * np.std(RB) + 1e-9))
    r['cfa_interpolation_trace'] = try_feat(_cfa)

    # A9 — shot_noise_curve
    def _shot():
        gf = gray_s.flatten().astype(np.float32)
        bl = cv2.GaussianBlur(gray_s.astype(np.float32), (5, 5), 0)
        rf = np.abs((gray_s.astype(np.float32) - bl).flatten())
        dm = gf < np.percentile(gf, 20)
        lm = gf > np.percentile(gf, 80)
        dn = float(np.mean(rf[dm])) if dm.sum() > 0 else 0.0
        ln = float(np.mean(rf[lm])) if lm.sum() > 0 else 0.0
        r['shot_noise_dark_mean']  = safe(dn)
        r['shot_noise_light_mean'] = safe(ln)
        return safe(dn - ln)
    r['shot_noise_curve'] = try_feat(_shot)

    # A10 — dark_region_noise
    def _drn():
        gf = gray_s.flatten().astype(np.float32)
        bl = cv2.GaussianBlur(gray_s.astype(np.float32), (5, 5), 0)
        rf = np.abs((gray_s.astype(np.float32) - bl).flatten())
        m  = gf < np.percentile(gf, 10)
        return safe(np.std(rf[m]) if m.sum() > 0 else 0.0)
    r['dark_region_noise'] = try_feat(_drn)

    return r


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B — Texture & Microstructure (10)
# ══════════════════════════════════════════════════════════════════════════════

def _lbp_uniform(gray_256):
    """LBP — use skimage if available, else skip."""
    if SKIMAGE_OK:
        return local_binary_pattern(gray_256, P=8, R=1, method='uniform')
    # Minimal pure-numpy fallback (slow but works)
    h, w = gray_256.shape
    out = np.zeros((h, w), dtype=np.uint8)
    padded = np.pad(gray_256, 1, mode='edge')
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for k, (dy, dx) in enumerate(neighbors):
        out += (padded[1+dy:h+1+dy, 1+dx:w+1+dx] >= gray_256).astype(np.uint8) * (1 << k)
    return out % 59

def _glcm_fast(gray_128):
    levels = 32
    q = (gray_128.astype(np.float32) * levels / 256).clip(0, levels-1).astype(np.int32)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    for y in range(q.shape[0]):
        for x in range(q.shape[1] - 1):
            glcm[q[y, x], q[y, x+1]] += 1
    glcm /= (glcm.sum() + 1e-9)
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
    contrast    = float(np.sum((i_idx - j_idx)**2 * glcm))
    homogeneity = float(np.sum(glcm / (1 + np.abs(i_idx - j_idx))))
    return contrast, homogeneity

def group_b_texture(img_bgr, gray):
    r = {}
    h, w = gray.shape
    g256 = cv2.resize(gray, (256, 256))

    # B11 — lbp_entropy
    def _lbp():
        lbp = _lbp_uniform(g256)
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        hp = hist.astype(np.float64) / (hist.sum() + 1e-9)
        return safe(-np.sum(hp * np.log2(hp + 1e-9)))
    r['lbp_entropy'] = try_feat(_lbp)

    # B12, B13 — GLCM
    def _glcm():
        contrast, hom = _glcm_fast(cv2.resize(gray, (128, 128)))
        r['glcm_contrast']    = safe(contrast)
        r['glcm_homogeneity'] = safe(hom)
    try_feat(_glcm, default=None)

    # B14 — micro_texture_randomness
    def _mtr():
        bs = 16
        if SKIMAGE_OK:
            lbp = local_binary_pattern(g256, P=8, R=1, method='uniform')
        else:
            lbp = g256  # fallback
        pvs = [float(np.var(lbp[y:y+bs, x:x+bs])) for y in range(0,256-bs,bs) for x in range(0,256-bs,bs)]
        return safe(np.var(pvs) if pvs else 0.0)
    r['micro_texture_randomness'] = try_feat(_mtr)

    # B15 — texture_repetition_index
    def _tri():
        bs = 32; patches = []
        sg = g256.astype(np.float32)
        for y in range(0,256-bs,bs):
            for x in range(0,256-bs,bs):
                patches.append(sg[y:y+bs,x:x+bs].flatten())
        mses = []
        for i in range(len(patches)):
            for j in range(i+1, min(i+4,len(patches))):
                mses.append(float(np.mean((patches[i]-patches[j])**2)))
        return safe(np.mean(mses) if mses else 0.0)
    r['texture_repetition_index'] = try_feat(_tri)

    # B16 — patch_similarity_rate
    def _psr():
        bs = 32; patches = []
        sg = g256.astype(np.float32)
        for y in range(0,256-bs,bs):
            for x in range(0,256-bs,bs):
                patches.append(sg[y:y+bs,x:x+bs].flatten())
        mses = []
        for i in range(len(patches)):
            for j in range(i+1, min(i+4,len(patches))):
                mses.append(float(np.mean((patches[i]-patches[j])**2)))
        sim = sum(1 for m in mses if m < 0.15 * 255.0**2)
        return safe(sim / (len(mses) + 1e-9))
    r['patch_similarity_rate'] = try_feat(_psr)

    # B17 — plasticity_smoothness_index
    def _plas():
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        return safe(float(np.mean(mag < 5)))
    r['plasticity_smoothness_index'] = try_feat(_plas)

    # B18 — surface_roughness_deviation
    def _srd():
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        bs = 32
        ps = [float(np.std(mag[y:y+bs,x:x+bs])) for y in range(0,h-bs,bs) for x in range(0,w-bs,bs)]
        return safe(np.std(ps) if ps else 0.0)
    r['surface_roughness_deviation'] = try_feat(_srd)

    # B19 — micro_edge_density
    def _med():
        tiny = cv2.resize(gray, (50, 50))
        e = cv2.Canny(tiny, 50, 150)
        return safe(float(np.sum(e > 0)) / 2500.0)
    r['micro_edge_density'] = try_feat(_med)

    # B20 — skin_tone_smoothness
    def _sts():
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = (hsv[:,:,0] <= 30) & (hsv[:,:,1] > 30)
        if mask.sum() < 100:
            return -1.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        return safe(float(np.var(mag[mask])))
    r['skin_tone_smoothness'] = try_feat(_sts)

    return r


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C — Color & Histogram (10)
# ══════════════════════════════════════════════════════════════════════════════

def group_c_color(img_rgb, img_bgr):
    r = {}
    R = img_rgb[:,:,0].astype(np.float32)
    G = img_rgb[:,:,1].astype(np.float32)
    B = img_rgb[:,:,2].astype(np.float32)

    r['rgb_skewness_r'] = try_feat(channel_skew, R)
    r['rgb_skewness_g'] = try_feat(channel_skew, G)
    r['rgb_skewness_b'] = try_feat(channel_skew, B)
    r['rgb_histogram_skewness'] = safe((abs(r['rgb_skewness_r'])+abs(r['rgb_skewness_g'])+abs(r['rgb_skewness_b']))/3)
    r['kurtosis_r'] = try_feat(channel_kurt, R)
    r['kurtosis_g'] = try_feat(channel_kurt, G)
    r['kurtosis_b'] = try_feat(channel_kurt, B)

    def _cent():
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return safe(img_entropy(gray))
    r['color_entropy'] = try_feat(_cent)

    def _sat():
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:,:,1].astype(np.float32)
        r['saturation_mean'] = safe(float(np.mean(s)))
        r['saturation_std']  = safe(float(np.std(s)))
    try_feat(_sat, default=None)

    def _hue():
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,0].astype(np.float32)
        hh = np.histogram(hue, bins=36, range=(0,180))[0].astype(np.float64)
        hh /= (hh.sum() + 1e-9)
        return safe(-np.sum(hh * np.log2(hh + 1e-9)))
    r['hue_clustering_score'] = try_feat(_hue)

    def _wb():
        mr, mg, mb = float(np.mean(R)), float(np.mean(G)), float(np.mean(B))
        r['wb_rg_ratio'] = safe(mr / (mg + 1e-9))
        r['wb_bg_ratio'] = safe(mb / (mg + 1e-9))
        r['white_balance_deviation'] = safe(abs(mr/(mg+1e-9)-1)+abs(mb/(mg+1e-9)-1))
    try_feat(_wb, default=None)

    def _cct():
        th = img_rgb.shape[0] // 3
        means_r = [float(np.mean(img_rgb[i*th:(i+1)*th,:,0])) for i in range(3)]
        means_b = [float(np.mean(img_rgb[i*th:(i+1)*th,:,2])) for i in range(3)]
        ct = [_r/(_b+1e-9) for _r,_b in zip(means_r,means_b)]
        return safe(float(np.std(ct)))
    r['color_temperature_consistency'] = try_feat(_cct)

    # Downsampled corrcoef for speed
    def _ch_corr():
        Rf = cv2.resize(R, (256,256)).flatten()
        Gf = cv2.resize(G, (256,256)).flatten()
        Bf = cv2.resize(B, (256,256)).flatten()
        r['channel_corr_rg'] = safe(float(np.corrcoef(Rf, Gf)[0,1]))
        r['channel_corr_gb'] = safe(float(np.corrcoef(Gf, Bf)[0,1]))
        r['channel_corr_rb'] = safe(float(np.corrcoef(Rf, Bf)[0,1]))
    try_feat(_ch_corr, default=None)

    def _og():
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return safe(float(np.mean(hsv[:,:,1] > 200)))
    r['overgrading_index'] = try_feat(_og)

    def _flat():
        r8 = img_rgb[:,:,0].astype(np.int16)
        g8 = img_rgb[:,:,1].astype(np.int16)
        b8 = img_rgb[:,:,2].astype(np.int16)
        return safe(float(np.mean((np.abs(r8-g8)<=3)&(np.abs(g8-b8)<=3))))
    r['flat_rgb_ratio'] = try_feat(_flat)

    return r


# ══════════════════════════════════════════════════════════════════════════════
# GROUP D — Edge & Geometry (10)
# ══════════════════════════════════════════════════════════════════════════════

def group_d_edges(gray, img_bgr):
    r = {}
    h, w = gray.shape

    def _base():
        edges = cv2.Canny(gray, 50, 150)
        r['canny_edge_density'] = safe(float(np.sum(edges > 0)) / (h*w+1e-9))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        r['mean_gradient'] = safe(float(np.mean(mag)))
        r['std_gradient']  = safe(float(np.std(mag)))
        return edges, mag
    edges, grad_mag = try_feat(_base, default=(
        cv2.Canny(gray, 50, 150),
        np.zeros_like(gray, dtype=np.float32)
    ))

    def _lines():
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        n = len(lines) if lines is not None else 0
        ep = max(1, int(np.sum(edges > 0)))
        r['straight_line_ratio'] = safe(n / (ep/100.0 + 1e-9))
        r['detected_lines'] = int(n)
    try_feat(_lines, default=None)

    def _harris():
        gf = np.float32(gray)
        h_ = cv2.cornerHarris(gf, 2, 3, 0.04)
        hn = cv2.dilate(h_, None)
        cnt = np.sum(hn > 0.01 * hn.max())
        r['corner_frequency'] = safe(cnt / ((h*w/1000.0)+1e-9))
    try_feat(_harris, default=None)

    def _esv():
        ev = grad_mag[edges > 0] if edges.sum() > 0 else np.array([0.0])
        r['edge_sharpness_variance'] = safe(float(np.var(ev)))
    try_feat(_esv, default=None)

    def _blur_map():
        bs = 32
        lvs = [float(cv2.Laplacian(gray[y:y+bs,x:x+bs],cv2.CV_64F).var())
               for y in range(0,h-bs,bs) for x in range(0,w-bs,bs)]
        r['blur_inconsistency_map'] = safe(np.std(lvs) if lvs else 0.0)
    try_feat(_blur_map, default=None)

    def _dof():
        cy2,cx2 = h//2, w//2
        cl = float(cv2.Laplacian(gray[cy2-50:cy2+50,cx2-50:cx2+50],cv2.CV_64F).var()) if h>100 and w>100 else 0.0
        bs2 = min(40,h//8,w//8)
        bl = float((cv2.Laplacian(gray[:bs2,:],cv2.CV_64F).var()+
                    cv2.Laplacian(gray[-bs2:,:],cv2.CV_64F).var())/2.0)
        r['dof_gradient_consistency'] = safe(cl/(bl+1e-9))
    try_feat(_dof, default=None)

    def _misc():
        r['shadow_coherence']     = safe(abs(float(np.mean(gray[:,:w//2]<60)) - float(np.mean(gray[:,w//2:]<60))))
        r['reflection_consistency'] = safe(abs(float(np.mean(gray[:,:w//2]>220)) - float(np.mean(gray[:,w//2:]>220))))
        r['perspective_score']    = 0.0
        r['lens_distortion_presence'] = 0.0
    try_feat(_misc, default=None)

    return r


# ══════════════════════════════════════════════════════════════════════════════
# GROUP E — Frequency Domain (10)
# ══════════════════════════════════════════════════════════════════════════════

def group_e_frequency(gray):
    r = {}
    gs = cap(gray)
    h, w = gs.shape

    def _fft_base():
        f     = np.fft.fft2(gs.astype(np.float32))
        fsh   = np.fft.fftshift(f)
        mag   = np.log1p(np.abs(fsh))
        cy2, cx2 = h//2, w//2
        yi, xi = np.ogrid[:h, :w]
        dist  = np.sqrt((yi-cy2)**2 + (xi-cx2)**2)
        r_max = min(cy2, cx2)

        # spectrum peakedness
        mf = mag.flatten()
        r['fft_spectrum_clustering'] = safe(float(np.std(mf)) / (float(np.mean(mf))+1e-9))

        # radial bands
        bands = {}
        for k in range(1,9):
            lo = r_max*(k-1)/8; hi = r_max*k/8
            mask = (dist>=lo)&(dist<hi)
            bands[f'band_{k}'] = safe(float(np.mean(mag[mask])) if mask.sum()>0 else 0.0)
        r['radial_freq_distribution'] = bands
        bm = [bands[f'band_{k}'] for k in range(1,9)]
        r['high_freq_dropoff_rate'] = safe(bm[0]-bm[-1])

        # periodic artifacts
        mag_ndc = mag.copy()
        mag_ndc[cy2-5:cy2+5, cx2-5:cx2+5] = 0
        mm = float(np.mean(mag_ndc)); ms = float(np.std(mag_ndc))
        r['periodic_artifact_score'] = safe(int(np.sum(mag_ndc > mm+3*ms)) / (h*w/1000.0+1e-9))
        r['repetitive_spectral_spike'] = int(np.sum(mag_ndc > mm+3*ms))

        # GAN fingerprint
        mn = mag_ndc / (mag_ndc.max()+1e-9)
        gs2 = 0.0
        for step in [4,8,16]:
            for dy in range(-step,step+1,step):
                for dx in range(-step,step+1,step):
                    if dy==0 and dx==0: continue
                    yp=cy2+dy; xp=cx2+dx
                    if 0<=yp<h and 0<=xp<w:
                        gs2 += float(mn[yp,xp])
        r['gan_fingerprint_score'] = safe(gs2)

        # spectral entropy
        mp = mf / (mf.sum()+1e-9); mp=mp[mp>0]
        r['spectral_entropy'] = safe(float(-np.sum(mp*np.log2(mp+1e-9))))

        # DCT block
        hn=(h//8)*8; wn=(w//8)*8
        gc = gs[:hn,:wn].astype(np.float32)
        bh = [gc[y,:] for y in range(8,hn,8)]
        bv = [gc[:,x] for x in range(8,wn,8)]
        if bh and bv:
            r['dct_block_artifact_score'] = safe(float(np.std(np.concatenate(bh)))+float(np.std(np.concatenate(bv))))
        else:
            r['dct_block_artifact_score'] = 0.0

        # diffusion residual
        blst = cv2.GaussianBlur(gs.astype(np.float32),(21,21),0)
        res  = gs.astype(np.float32)-blst
        rfft = np.abs(np.fft.fftshift(np.fft.fft2(res)))
        rfft[cy2-3:cy2+3,cx2-3:cx2+3]=0
        r['diffusion_residual_score'] = safe(float(np.std(rfft)/(np.mean(rfft)+1e-9)))

        # legacy fft dict
        rl = dist <= min(h,w)//8; rh = dist > min(h,w)//8
        le = float(np.sum(mag[rl])); he = float(np.sum(mag[rh])); te=le+he+1e-9
        r['fft'] = {
            'low_freq_ratio':  safe(le/te),
            'high_freq_ratio': safe(he/te),
            'mean_magnitude':  safe(float(np.mean(mag))),
            'std_magnitude':   safe(float(np.std(mag))),
        }
    try_feat(_fft_base, default=None)

    # wavelet
    def _wav():
        if not PYWT_OK: return
        cA,(cH,cV,cD) = pywt.dwt2(gs.astype(np.float32),'db1')
        cA2,(cH2,cV2,cD2) = pywt.dwt2(cA,'db1')
        def e(a): return float(np.sum(a**2))/(a.size+1e-9)
        l1 = e(cH)+e(cV)+e(cD); l2 = e(cH2)+e(cV2)+e(cD2)
        r['wavelet_level1_detail'] = safe(l1)
        r['wavelet_level2_detail'] = safe(l2)
        r['wavelet_energy_ratio']  = safe(l1/(l2+1e-9))
        r['level1_diagonal_energy']   = safe(e(cD))
        r['level1_horizontal_energy'] = safe(e(cH))
        r['level1_vertical_energy']   = safe(e(cV))
    try_feat(_wav, default=None)

    if 'wavelet_energy_ratio' not in r:
        r.update({'wavelet_energy_ratio':0.0,'level1_diagonal_energy':0.0,
                  'level1_horizontal_energy':0.0,'level1_vertical_energy':0.0})
    if 'fft' not in r:
        r['fft'] = {'low_freq_ratio':0.0,'high_freq_ratio':0.0,'mean_magnitude':0.0,'std_magnitude':0.0}

    return r


# ══════════════════════════════════════════════════════════════════════════════
# GROUP F — File/Metadata (8)
# ══════════════════════════════════════════════════════════════════════════════

def group_f_file(image_path, img_rgb):
    r = {}
    try:
        stat = os.stat(image_path)
        ext  = os.path.splitext(image_path)[1].lower()
        h_, w_ = img_rgb.shape[:2]
        tp = max(1, h_*w_)

        r['exif_camera_make'] = False
        r['exif_aperture_iso'] = False
        r['exif_gps'] = False
        r['exif_ai_software'] = False
        try:
            import exifread
            with open(image_path,'rb') as ef:
                tags = exifread.process_file(ef, details=False)
            r['exif_camera_make']    = 'Image Make' in tags
            r['exif_aperture_iso']   = ('EXIF FNumber' in tags or 'EXIF ISOSpeedRatings' in tags)
            r['exif_gps']            = any('GPS' in k for k in tags)
            ai_sw = ['midjourney','stable diffusion','dall-e','firefly','runway',
                     'novelai','comfyui','automatic1111','diffusers','controlnet']
            sw = str(tags.get('Image Software','')).lower()
            r['exif_ai_software']    = any(kw in sw for kw in ai_sw)
        except Exception:
            pass

        r['timestamp_consistency']     = bool(stat.st_mtime >= stat.st_ctime)
        r['bytes_per_pixel']           = safe(stat.st_size / tp)
        r['png_flag']                  = (ext == '.png')
        r['filesize_resolution_ratio'] = safe(stat.st_size / (tp+1))
        r['image_width']  = int(w_)
        r['image_height'] = int(h_)
        r['file_extension'] = ext
    except Exception as ex:
        r['_error'] = str(ex)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# GROUP G — Semantic & Structural (7)
# ══════════════════════════════════════════════════════════════════════════════

def group_g_semantic(img_bgr, gray, img_rgb):
    r = {}
    h, w = gray.shape
    gs = cap(gray)

    def _skin():
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:,:,0]<=30)&(hsv[:,:,1]>30)&(hsv[:,:,2]>50)).astype(np.uint8)
        conts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not conts: return -1.0
        big = max(conts, key=cv2.contourArea)
        area = cv2.contourArea(big); perim = cv2.arcLength(big,True)
        return safe(perim**2/(4*math.pi*area+1e-9))
    r['hand_region_complexity'] = try_feat(_skin)

    def _eye():
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        val = hsv[:,:,2]
        um = val[:h//2,:] > 230
        if um.sum() < 10: return -1.0
        lb = float(np.mean(um[:,:w//2])); rb = float(np.mean(um[:,w//2:]))
        return safe(1-abs(lb-rb))
    r['eye_reflection_symmetry'] = try_feat(_eye)

    def _text():
        _,tm = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        tlap = cv2.Laplacian(gray, cv2.CV_64F)
        bnd  = cv2.Canny(tm, 50, 150)
        if bnd.sum() == 0: return 0.0
        return safe(float(np.mean(np.abs(tlap[bnd>0]))))
    r['text_region_sharpness'] = try_feat(_text)

    def _bg():
        bg = cv2.Canny(gray[:h//5,:], 30, 90)
        fg = cv2.Canny(gray[h//2:,:], 30, 90)
        return safe(abs(float(np.sum(bg>0))/(bg.size+1e-9) - float(np.sum(fg>0))/(fg.size+1e-9)))
    r['background_distortion_rate'] = try_feat(_bg)

    def _sym():
        left  = gs[:, :gs.shape[1]//2].astype(np.float32)
        right = np.fliplr(gs[:, gs.shape[1]//2:]).astype(np.float32)
        mw = min(left.shape[1], right.shape[1])
        diff = np.abs(left[:,:mw]-right[:,:mw])
        return safe(1-float(np.mean(diff))/128.0)
    r['symmetry_score'] = try_feat(_sym)

    def _face():
        try:
            fcp = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.isfile(fcp):
                r['face_detected'] = False; return -1.0
            fc = cv2.CascadeClassifier(fcp)
            faces = fc.detectMultiScale(gs, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                fx,fy,fw,fh = faces[0]
                r['face_detected'] = True
                return safe(fw*fh/(gs.shape[0]*gs.shape[1]+1e-9))
            r['face_detected'] = False; return -1.0
        except Exception:
            r['face_detected'] = False; return -1.0
    r['anatomical_proportion_score'] = try_feat(_face)

    def _acc():
        ed = cv2.Canny(gray, 80, 200)
        bs = 16
        ld = [float(np.sum(ed[y:y+bs,x:x+bs]>0))/(bs*bs)
              for y in range(0,h-bs,bs) for x in range(0,w-bs,bs)]
        return safe(float(np.std(ld)) if ld else 0.0)
    r['accessory_mismatch'] = try_feat(_acc)

    return r


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED GROUP — 15 Additional Features (backward compatible)
# ══════════════════════════════════════════════════════════════════════════════

def group_advanced(img_rgb, img_bgr, gray):
    a = {}
    gs = cap(gray)
    img_rgb_s = cap(img_rgb)

    # ADV1 — prnu_score (standalone, used by ensembleScorer advanced section)
    def _prnu():
        res = []
        for c in range(3):
            ch = img_rgb_s[:,:,c].astype(np.float32)
            bl = cv2.GaussianBlur(ch,(3,3),0)
            res.append((ch-bl).flatten())
        rg = float(np.corrcoef(res[0],res[1])[0,1])
        gb = float(np.corrcoef(res[1],res[2])[0,1])
        return round((rg+gb)/2, 4)
    a['prnu_score'] = try_feat(_prnu)

    # ADV2 — channel_noise_ratio
    def _cnr():
        vs = {}
        for i,n in enumerate(['R','G','B']):
            ch = img_rgb_s[:,:,i].astype(np.float32)
            bl = cv2.GaussianBlur(ch,(3,3),0)
            vs[n] = round(float(np.var(ch-bl)),4)
        gv = vs['G']+1e-9
        vs['RG_ratio'] = round(vs['R']/gv,4)
        vs['BG_ratio'] = round(vs['B']/gv,4)
        return vs
    a['channel_noise_ratio'] = try_feat(_cnr, default={})

    # ADV3 — shot_noise_score
    def _sn():
        dm = gs < 60; bm = gs > 180
        lap = cv2.Laplacian(gs, cv2.CV_64F)
        dn = float(np.std(lap[dm])) if dm.any() else 0.0
        bn = float(np.std(lap[bm])) if bm.any() else 0.0
        return round(dn/(bn+1e-9), 4)
    a['shot_noise_score'] = try_feat(_sn)

    # ADV4 — high_freq_entropy
    def _hfe():
        k = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=np.float32)
        hp = cv2.filter2D(gs.astype(np.float32),-1,k)
        hn = cv2.normalize(hp,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.calcHist([hn],[0],None,[256],[0,256]).flatten()
        hist /= (hist.sum()+1e-9)
        return round(-float(np.sum(hist*np.log2(hist+1e-9))),4)
    a['high_freq_entropy'] = try_feat(_hfe)

    # ADV5 — texture_repetition
    def _tri():
        bs=32; sg=gs.astype(np.float32)
        patches=[sg[y:y+bs,x:x+bs].flatten()
                 for y in range(0,gs.shape[0]-bs,bs) for x in range(0,gs.shape[1]-bs,bs)]
        if len(patches)<2: return 0.0
        dc=0
        for i in range(len(patches)):
            for j in range(i+1,min(i+5,len(patches))):
                if float(np.mean((patches[i]-patches[j])**2)) < 200:
                    dc+=1
        return round(dc/(len(patches)+1e-9),4)
    a['texture_repetition'] = try_feat(_tri)

    # ADV6 — plasticity_smoothness
    def _pls():
        gx=cv2.Sobel(gs,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(gs,cv2.CV_32F,0,1,ksize=3)
        mag=np.sqrt(gx**2+gy**2)
        return round(float(np.sum(mag<5))/(mag.size+1e-9),4)
    a['plasticity_smoothness'] = try_feat(_pls)

    # ADV7 — blur_inconsistency
    def _bi():
        bs=32
        lvs=[float(cv2.Laplacian(gs[y:y+bs,x:x+bs],cv2.CV_64F).var())
             for y in range(0,gs.shape[0]-bs,bs) for x in range(0,gs.shape[1]-bs,bs)]
        return round(float(np.std(lvs)) if lvs else 0.0, 4)
    a['blur_inconsistency'] = try_feat(_bi)

    # ADV8 — color_entropy
    def _ce():
        et=0.0
        for c in range(3):
            hist=cv2.calcHist([img_rgb_s[:,:,c]],[0],None,[256],[0,256]).flatten()
            hist/=(hist.sum()+1e-9)
            et+=-float(np.sum(hist*np.log2(hist+1e-9)))
        return round(et/3,4)
    a['color_entropy'] = try_feat(_ce)

    # ADV9 — overgrading_index
    def _ogi():
        bgr_s = cv2.cvtColor(img_rgb_s, cv2.COLOR_RGB2BGR)
        hsv=cv2.cvtColor(bgr_s,cv2.COLOR_BGR2HSV)
        return round(float(np.sum(hsv[:,:,1]>200))/(hsv.shape[0]*hsv.shape[1]+1e-9),4)
    a['overgrading_index'] = try_feat(_ogi)

    # ADV10 — flat_rgb_ratio
    def _frr():
        ri=img_rgb_s[:,:,0].astype(int)
        gi=img_rgb_s[:,:,1].astype(int)
        bi=img_rgb_s[:,:,2].astype(int)
        fl=(np.abs(ri-gi)<5)&(np.abs(gi-bi)<5)
        return round(float(np.sum(fl))/(fl.size+1e-9),4)
    a['flat_rgb_ratio'] = try_feat(_frr)

    # ADV11 — straight_line_ratio
    def _slr():
        ed=cv2.Canny(gs,50,150)
        ln=cv2.HoughLinesP(ed,1,np.pi/180,threshold=50,minLineLength=30,maxLineGap=10)
        ep=float(np.sum(ed>0))
        if ep<100: return 0.0
        return round((len(ln) if ln is not None else 0)/(ep/100+1e-9),4)
    a['straight_line_ratio'] = try_feat(_slr)

    # ADV12 — symmetry_score
    def _ss():
        left=gs[:,:gs.shape[1]//2].astype(np.float32)
        right=np.fliplr(gs[:,gs.shape[1]//2:]).astype(np.float32)
        mw=min(left.shape[1],right.shape[1])
        diff=np.abs(left[:,:mw]-right[:,:mw])
        return round(max(0.0,min(1.0,1.0-float(np.mean(diff))/128.0)),4)
    a['symmetry_score'] = try_feat(_ss)

    # ADV13 — gan_fingerprint_spikes
    def _gan():
        f=np.fft.fft2(gs.astype(np.float32))
        mag=np.abs(np.fft.fftshift(f))
        h_,w_=mag.shape; mag[h_//2,w_//2]=0
        mm=float(np.mean(mag)); ms=float(np.std(mag))
        return int(np.sum(mag>mm+4*ms))
    a['gan_fingerprint_spikes'] = try_feat(_gan, default=0)

    # ADV14 — spectral_entropy
    def _se():
        mag=np.abs(np.fft.fft2(gs.astype(np.float32))).flatten()
        mag=mag/(mag.sum()+1e-9)
        return round(-float(np.sum(mag*np.log2(mag+1e-9))),4)
    a['spectral_entropy'] = try_feat(_se)

    # ADV15 — dct_block_artifact
    def _dct():
        h_,w_=gs.shape; h_=(h_//8)*8; w_=(w_//8)*8
        g=gs[:h_,:w_].astype(np.float32)
        hd=float(np.mean(np.abs(g[8::8,:]-g[7:-1:8,:]))) if h_>8 else 0.0
        vd=float(np.mean(np.abs(g[:,8::8]-g[:,7:-1:8]))) if w_>8 else 0.0
        return round((hd+vd)/2,4)
    a['dct_block_artifact'] = try_feat(_dct)

    return a


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY FIELDS
# ══════════════════════════════════════════════════════════════════════════════

def legacy_fields(gray):
    try:
        gs = cap(gray)
        bl = cv2.GaussianBlur(gs.astype(np.float32),(5,5),0)
        lap_var = float(cv2.Laplacian(gs, cv2.CV_64F).var())
        bs=16; h,w=gs.shape
        stds=[float(np.std(gs[y:y+bs,x:x+bs]))
              for y in range(0,h-bs,bs) for x in range(0,w-bs,bs)]
        ms=float(np.mean(stds)) if stds else 0.0
        ss=float(np.std(stds))  if stds else 0.0
        return {
            'laplacian_variance': safe(lap_var),
            'local': {'mean':safe(ms),'std':safe(ss),'uniformity':safe(ss/(ms+1e-6))},
        }
    except Exception:
        return {'laplacian_variance':0.0,'local':{'mean':0.0,'std':0.0,'uniformity':0.0}}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — every group fully isolated
# ══════════════════════════════════════════════════════════════════════════════

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

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def run(fn, *args):
        try: return fn(*args)
        except BaseException as ex: return {'_error': str(ex)}

    grp_a = run(group_a_sensor_noise, img_bgr, img_rgb, gray)
    grp_b = run(group_b_texture, img_bgr, gray)
    grp_c = run(group_c_color, img_rgb, img_bgr)
    grp_d = run(group_d_edges, gray, img_bgr)
    grp_e = run(group_e_frequency, gray)
    grp_f = run(group_f_file, image_path, img_rgb)
    grp_g = run(group_g_semantic, img_bgr, gray, img_rgb)
    grp_adv = run(group_advanced, img_rgb, img_bgr, gray)
    leg   = legacy_fields(gray)

    # Extract frequency sub-dicts safely
    fft_d = grp_e.pop('fft', {}) if isinstance(grp_e, dict) else {}
    wav_d = {k: grp_e.get(k, 0.0) for k in
             ['level1_diagonal_energy','level1_horizontal_energy',
              'level1_vertical_energy','wavelet_energy_ratio']} if isinstance(grp_e,dict) else {}
    edg_d = {
        'edge_density':  (grp_d.get('canny_edge_density',0.0) if isinstance(grp_d,dict) else 0.0),
        'mean_gradient': (grp_d.get('mean_gradient',0.0) if isinstance(grp_d,dict) else 0.0),
        'std_gradient':  (grp_d.get('std_gradient',0.0) if isinstance(grp_d,dict) else 0.0),
    }

    result = {
        'noise':     {'laplacian_variance':leg['laplacian_variance'],'local':leg['local']},
        'frequency': {'fft': fft_d, 'wavelet': wav_d},
        'edges':     edg_d,
        'group_a':   grp_a,
        'group_b':   grp_b,
        'group_c':   grp_c,
        'group_d':   grp_d,
        'group_e':   grp_e,
        'group_f':   grp_f,
        'group_g':   grp_g,
        'advanced':  grp_adv,
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
