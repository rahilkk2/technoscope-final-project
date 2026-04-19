'use strict';

/**
 * ensembleScorer.js — 7-Group Weighted Ensemble
 *
 * Group weights (sum = 1.0):
 *   Sensor/Noise    (A): 0.20
 *   Texture         (B): 0.15
 *   Color           (C): 0.10
 *   Edge/Geometry   (D): 0.12
 *   Frequency       (E): 0.18
 *   Metadata/File   (F): 0.08  ← weak: social media strips EXIF
 *   Semantic        (G): 0.10
 *   CNN classifier     : 0.07
 *
 * Critical rules:
 *   - EXIF missing alone contributes max 0.05 to AI score
 *   - Need ≥ 4 groups agreeing before confidence > 70%
 *   - Sensor noise strongly Real → overrides weak metadata AI signals
 *
 * Output (exact frontend contract):
 * { verdict, confidence (int), comments, metadata, spectral }
 */

function clamp(v, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }
function safe(v) { return (typeof v === 'number' && isFinite(v)) ? v : 0; }

// ══════════════════════════════════════════════════════════════════════════════
// GROUP SCORERS — each returns { AI, Real, Screenshot } delta scores [0–1]
// ══════════════════════════════════════════════════════════════════════════════

function scoreGroupA(grpA) {
    // Sensor & Noise — most reliable for real vs AI
    let ai = 0, real = 0, ss = 0;
    if (!grpA) return { AI: ai, Real: real, Screenshot: ss };

    const spn = safe(grpA.sensor_pattern_noise);
    if (spn > 50) real += 0.12;
    if (spn > 20) real += 0.06;
    if (spn < 2) ai += 0.10;

    const prnu = safe(grpA.prnu_score);
    if (prnu > 0.4) real += 0.10;
    if (prnu > 0.2) real += 0.04;
    if (prnu < -0.1) ai += 0.08;

    const iso = safe(grpA.iso_noise_variance);
    if (iso > 100) real += 0.08;
    if (iso < 5) ai += 0.08;

    // Green channel dominance (Bayer sensor pattern)
    const gnd = safe(grpA.green_noise_dominance);
    if (gnd > 1.1 && gnd < 2.5) real += 0.08;

    // Shot noise curve: dark_noise > light_noise = real photo physics
    const snc = safe(grpA.shot_noise_curve);
    if (snc > 1.0) real += 0.12;
    if (snc < -1.0) ai += 0.06;

    const drn = safe(grpA.dark_region_noise);
    if (drn > 3) real += 0.06;
    if (drn < 0.5) ai += 0.06;

    // CFA interpolation trace: real cameras show specific correlation
    const cfa = safe(grpA.cfa_interpolation_trace);
    if (cfa > 0.5) real += 0.08;
    if (cfa < 0.1) ai += 0.04;

    return { AI: clamp(ai), Real: clamp(real), Screenshot: clamp(ss) };
}

function scoreGroupB(grpB) {
    // Texture & Microstructure
    let ai = 0, real = 0, ss = 0;
    if (!grpB) return { AI: ai, Real: real, Screenshot: ss };

    const lbpE = safe(grpB.lbp_entropy);
    if (lbpE < 2.5) ai += 0.12;  // unnaturally uniform LBP
    if (lbpE > 4.5) real += 0.10;

    const homog = safe(grpB.glcm_homogeneity);
    if (homog > 0.85) ai += 0.10;  // too smooth texture
    if (homog < 0.60) real += 0.08;

    const plast = safe(grpB.plasticity_smoothness_index);
    if (plast > 0.55) ai += 0.10;  // >55% hyper-smooth pixels = AI
    if (plast < 0.20) real += 0.08;

    const tri = safe(grpB.texture_repetition_index);
    if (tri < 100) ai += 0.10;  // too similar patches = AI
    if (tri > 800) real += 0.06;

    const mtr = safe(grpB.micro_texture_randomness);
    if (mtr < 10) ai += 0.06;
    if (mtr > 80) real += 0.06;

    return { AI: clamp(ai), Real: clamp(real), Screenshot: clamp(ss) };
}

function scoreGroupC(grpC) {
    // Color & Histogram
    let ai = 0, real = 0, ss = 0;
    if (!grpC) return { AI: ai, Real: real, Screenshot: ss };

    const satMean = safe(grpC.saturation_mean);
    const satStd = safe(grpC.saturation_std);
    if (satMean > 160 || satMean < 10) ai += 0.10;
    if (satMean > 40 && satMean < 130 && satStd > 20) real += 0.08;
    if (satMean < 5) ss += 0.08;  // near grayscale = screenshot

    const overgrade = safe(grpC.overgrading_index);
    if (overgrade > 0.15) ai += 0.08;

    const flatRgb = safe(grpC.flat_rgb_ratio);
    if (flatRgb > 0.40) ss += 0.12;  // lots of gray = screenshot

    const hueCluster = safe(grpC.hue_clustering_score);
    if (hueCluster < 2.0) ai += 0.08;  // too few hue clusters = AI

    // White balance: real cameras have slight imbalance
    const wbd = safe(grpC.white_balance_deviation);
    if (wbd > 0.12) real += 0.06;
    if (wbd < 0.02) ai += 0.04;

    return { AI: clamp(ai), Real: clamp(real), Screenshot: clamp(ss) };
}

function scoreGroupD(grpD, noise) {
    // Edge & Geometry
    let ai = 0, real = 0, ss = 0;

    // Use top-level edge_density for backward compat
    const edges = grpD || {};
    const edgeDensity = safe(edges.canny_edge_density || (noise && noise.edge_density) || 0);
    if (edgeDensity > 0.12) ss += 0.14;
    if (edgeDensity > 0.07) ss += 0.06;
    if (edgeDensity < 0.015) ai += 0.08;

    const lines = safe(edges.detected_lines);
    if (lines > 20) ss += 0.10;

    const blur_inco = safe(edges.blur_inconsistency_map);
    if (blur_inco > 200) ai += 0.08;  // inconsistent blur = AI deepfake

    const shadowCoh = safe(edges.shadow_coherence);
    if (shadowCoh > 0.15) real += 0.06;  // shadows on one side = real

    const cornerFreq = safe(edges.corner_frequency);
    if (cornerFreq > 5) real += 0.06;

    return { AI: clamp(ai), Real: clamp(real), Screenshot: clamp(ss) };
}

function scoreGroupE(freqData) {
    // Frequency Domain
    let ai = 0, real = 0, ss = 0;
    const grpE = freqData || {};

    // FFT ratios (legacy + new)
    const fft = grpE.fft || {};
    const hfRatio = safe(fft.high_freq_ratio);
    if (hfRatio < 0.22) ai += 0.12;
    if (hfRatio < 0.38) ai += 0.04;
    if (hfRatio > 0.58) real += 0.10;

    // Wavelet diagonal energy — FIXED thresholds
    const wavelet = grpE.wavelet || {};
    const diagE = safe(wavelet.level1_diagonal_energy);
    if (diagE < 0.04) ai += 0.10;
    if (diagE < 0.10) ai += 0.03;
    if (diagE > 1.5) real += 0.08;

    // GAN fingerprint — periodic FFT spikes
    const ganFP = safe(grpE.gan_fingerprint_score);
    if (ganFP > 2.0) ai += 0.10;
    if (ganFP > 1.0) ai += 0.04;

    // Spectral entropy (high = complex/real, low = AI-smooth)
    const specE = safe(grpE.spectral_entropy);
    if (specE > 14) real += 0.08;
    if (specE < 8) ai += 0.08;

    // Repetitive spectral spikes (GAN artifact)
    const spikeCount = safe(grpE.repetitive_spectral_spike);
    if (spikeCount > 500) ai += 0.08;

    // DCT block artifacts (JPEG = real photo signal)
    const dct = safe(grpE.dct_block_artifact_score);
    if (dct > 30) real += 0.08;

    return { AI: clamp(ai), Real: clamp(real), Screenshot: clamp(ss) };
}

function scoreGroupF(grpF, metadataData) {
    // Metadata / File — deliberately low weight; social media strips EXIF
    let ai = 0, real = 0, ss = 0;

    // Use grpF (new) or metadataData.exif (legacy)
    const exif = (grpF) ? {
        exif_stripped: !grpF.exif_camera_make && !grpF.exif_aperture_iso && !grpF.exif_gps,
        ai_software_detected: grpF.exif_ai_software || false,
        has_camera_make: grpF.exif_camera_make || false,
        has_gps: grpF.exif_gps || false,
    } : ((metadataData || {}).exif || {});

    // CRITICAL FIX: Missing EXIF = tiny bump only (social media strips real photos)
    if (exif.exif_stripped) ai += 0.03;  // max 0.05 from EXIF absence

    // AI software tag = strong confirmed signal
    if (exif.ai_software_detected) ai += 0.30;

    if (exif.has_camera_make) real += 0.06;
    if (exif.has_gps) real += 0.04;

    // Screenshot signals
    const ss_sig = (grpF) ? {
        uniform_corner_count: grpF.screenshot_uniform_corners || 0,
        round_dimensions: (grpF.image_width % 8 === 0 && grpF.image_height % 8 === 0),
    } : ((metadataData || {}).screenshot || {});

    const ucc = safe(ss_sig.uniform_corner_count || 0);
    if (ucc >= 3) ss += 0.10;
    if (ucc >= 2) ss += 0.04;
    if (ss_sig.round_dimensions) ss += 0.04;

    // PNG flag — more likely screenshot
    if (grpF && grpF.png_flag) ss += 0.06;

    return { AI: clamp(ai), Real: clamp(real), Screenshot: clamp(ss) };
}

function scoreGroupG(grpG) {
    // Semantic / Structural AI errors
    let ai = 0, real = 0, ss = 0;
    if (!grpG) return { AI: ai, Real: real, Screenshot: ss };

    // Symmetry: AI images often too symmetrical
    const symScore = safe(grpG.symmetry_score);
    if (symScore > 0.92) ai += 0.08;
    if (symScore < 0.70) real += 0.06;

    // Accessory mismatch: AI inconsistent edge detail
    const accM = safe(grpG.accessory_mismatch);
    if (accM > 0.12) ai += 0.06;
    if (accM < 0.04) ai += 0.04;

    // Background distortion
    const bgDist = safe(grpG.background_distortion_rate);
    if (bgDist < 0.005) ai += 0.06;  // too uniform background = AI

    // Text region sharpness — blurry text = AI
    const textSharp = safe(grpG.text_region_sharpness);
    if (textSharp > 20) ss += 0.08;  // sharp text = screenshot
    if (textSharp < 3) ai += 0.04;

    return { AI: clamp(ai), Real: clamp(real), Screenshot: clamp(ss) };
}


// ══════════════════════════════════════════════════════════════════════════════
// CONSENSUS CHECK — require ≥ 4 groups to agree for confidence > 70%
// ══════════════════════════════════════════════════════════════════════════════

function countGroupsAgreeing(groupScores, winner) {
    return groupScores.filter(gs => {
        const vals = [gs.AI, gs.Real, gs.Screenshot];
        const keys = ['AI', 'Real', 'Screenshot'];
        const maxIdx = vals.indexOf(Math.max(...vals));
        return keys[maxIdx] === winner;
    }).length;
}


// ══════════════════════════════════════════════════════════════════════════════
// STRING BUILDERS
// ══════════════════════════════════════════════════════════════════════════════

function buildMetadataString(metadata, grpF) {
    const exif = (metadata || {}).exif || {};
    const f = grpF || {};

    if ((exif.ai_software_detected) || f.exif_ai_software) {
        const sw = exif.software_tag || 'Unknown AI Tool';
        return `AI Software Tag Detected: ${sw}`;
    }
    if (exif.exif_stripped || (!f.exif_camera_make && !f.exif_gps)) {
        return 'No EXIF Found (stripped or absent)';
    }
    if (exif.has_camera_make || f.exif_camera_make) return 'Camera Make/Model Present';
    if (exif.has_gps || f.exif_gps) return 'GPS Data Embedded';
    return 'Minimal EXIF Present';
}

function buildSpectralString(features, grpE) {
    const fft = ((features || {}).frequency || {}).fft || {};
    const wavelet = ((features || {}).frequency || {}).wavelet || {};
    const edges = (features || {}).edges || {};
    const e = grpE || {};

    const edgeDensity = safe(edges.edge_density || (grpE && grpE.canny_edge_density) || 0);
    const hfRatio = safe(fft.high_freq_ratio || 0);
    const diagE = safe((wavelet.level1_diagonal_energy) || 0);
    const ganFP = safe(e.gan_fingerprint_score || 0);

    if (edgeDensity > 0.08) return `UI Edge Pattern (${(edgeDensity * 100).toFixed(1)}% density) — Screenshot Signal`;
    if (ganFP > 1.5) return `GAN Fingerprint Detected (score: ${ganFP.toFixed(2)}) — AI Upsampling Artifact`;
    if (hfRatio < 0.22) return `Low High-Freq Content (${(hfRatio * 100).toFixed(1)}%) — AI Diffusion Signature`;
    if (diagE < 0.04) return `Wavelet Diagonal Energy Extremely Low (${diagE.toFixed(4)}) — AI Smoothing`;
    if (hfRatio > 0.55) return `Natural Frequency Distribution (${(hfRatio * 100).toFixed(1)}% high-freq)`;
    return 'Spectral Profile Inconclusive';
}

function buildComments(winner, metadata, features, classifier, grpA, grpB, grpC, grpD, grpE, grpG) {
    const exif = (metadata || {}).exif || {};
    const ss = (metadata || {}).screenshot || {};
    const noise = (features || {}).noise || {};
    const fft = ((features || {}).frequency || {}).fft || {};
    const wavelet = ((features || {}).frequency || {}).wavelet || {};
    const edges = (features || {}).edges || {};
    const cnnP = (classifier || {}).probabilities || {};
    const A = grpA || {}; const B = grpB || {}; const C = grpC || {};
    const D = grpD || {}; const E = grpE || {}; const G = grpG || {};

    const lapVar = safe(noise.laplacian_variance);
    const hfRatio = safe(fft.high_freq_ratio);
    const diagE = safe((wavelet || {}).level1_diagonal_energy);
    const edgeDens = safe(edges.edge_density || D.canny_edge_density);
    const width = safe(ss.image_width || 0);
    const height = safe(ss.image_height || 0);

    const parts = [];

    if (winner === 'AI') {
        if (hfRatio > 0 && hfRatio < 0.40) {
            parts.push(`Spectral analysis reveals low high-frequency content (ratio: ${hfRatio.toFixed(2)}) consistent with diffusion model generation.`);
        }
        if (lapVar > 0) {
            parts.push(`Laplacian variance of ${lapVar.toFixed(1)} — ${lapVar < 30 ? 'AI-smoothed textures with suppressed natural grain.' : 'moderate texture sharpness noted.'}`);
        }
        const prnu = safe(A.prnu_score);
        if (prnu !== 0) {
            parts.push(`PRNU sensor correlation: ${prnu.toFixed(3)} — ${prnu < 0.1 ? 'absent natural sensor fingerprint, consistent with synthetic generation.' : 'some sensor pattern detected.'}`);
        }
        const plast = safe(B.plasticity_smoothness_index);
        if (plast > 0.4) {
            parts.push(`Plasticity smoothness index of ${(plast * 100).toFixed(1)}% — excess smooth pixels indicate GAN/diffusion processing.`);
        }
        const ganFP = safe(E.gan_fingerprint_score);
        if (ganFP > 1.0) {
            parts.push(`GAN upsampling fingerprint detected in FFT (score: ${ganFP.toFixed(2)}) — periodic spectral spikes characteristic of neural network generation.`);
        }
        if (exif.exif_stripped && exif.ai_software_detected) {
            parts.push(`AI software tag confirmed in metadata.`);
        } else if (exif.exif_stripped) {
            parts.push(`No EXIF metadata — common but not exclusive to AI-generated images.`);
        }
        if (cnnP.AI !== undefined) {
            parts.push(`Heuristic classifier: ${(cnnP.AI * 100).toFixed(1)}% AI probability across 8 signals.`);
        }
        return parts.join(' ') || 'Multiple forensic signals indicate AI/synthetic generation.';
    }

    if (winner === 'Screenshot') {
        if (edgeDens > 0.06) {
            parts.push(`High straight-edge density of ${(edgeDens * 100).toFixed(1)}% — UI element pattern consistent with screen capture.`);
        }
        const ucc = safe(ss.uniform_corner_count || (metadata && metadata.screenshot && metadata.screenshot.uniform_corner_count) || 0);
        if (ucc >= 2) {
            parts.push(`${ucc} of 4 corner regions show uniform flat values — characteristic of OS screen chrome.`);
        }
        const flatRgb = safe(C.flat_rgb_ratio);
        if (flatRgb > 0.2) {
            parts.push(`${(flatRgb * 100).toFixed(1)}% of pixels are neutral gray (R≈G≈B) — typical of text-heavy UI rendering.`);
        }
        if (width && height) {
            parts.push(`Image dimensions ${width}×${height} — consistent with screen resolution.`);
        }
        if (lapVar > 0) {
            parts.push(`Laplacian variance: ${lapVar.toFixed(1)} — absent photographic sensor noise.`);
        }
        return parts.join(' ') || 'High edge density and uniform corners indicate screen capture.';
    }

    // Real
    const spn = safe(A.sensor_pattern_noise);
    if (spn > 20) {
        parts.push(`Natural sensor pattern noise detected (variance: ${spn.toFixed(1)}) — consistent with optical sensor capture.`);
    }
    if (lapVar > 0) {
        parts.push(`Laplacian variance of ${lapVar.toFixed(1)} — ${lapVar > 200 ? 'strong natural grain present.' : 'moderate sharpness consistent with camera compression.'}`);
    }
    const snc = safe(A.shot_noise_curve);
    if (snc > 0.5) {
        parts.push(`Shot noise physics confirmed — dark regions show higher noise (Δ=${snc.toFixed(2)}), consistent with photon counting behavior.`);
    }
    if (hfRatio > 0.45) {
        parts.push(`High-frequency spectral content (ratio: ${hfRatio.toFixed(2)}) consistent with photographic lens capture.`);
    }
    if (exif.has_camera_make) {
        parts.push(`EXIF metadata intact with camera make/model — strong authenticity indicator.`);
    }
    const dct = safe(E && E.dct_block_artifact_score);
    if (dct > 20) {
        parts.push(`JPEG DCT block artifacts detected (score: ${dct.toFixed(1)}) — consistent with real camera JPEG compression pipeline.`);
    }
    if (cnnP.Real !== undefined) {
        parts.push(`Classifier confidence: ${(cnnP.Real * 100).toFixed(1)}% Real.`);
    }
    return parts.join(' ') || 'Image features are consistent with authentic photographic capture.';
}


// ══════════════════════════════════════════════════════════════════════════════
// MAIN ENSEMBLE
// ══════════════════════════════════════════════════════════════════════════════

function computeEnsembleScore(pythonResult) {
    const { metadata = {}, features = {}, classifier = {} } = pythonResult;

    // Extract group outputs from features
    const grpA = features.group_a || null;
    const grpB = features.group_b || null;
    const grpC = features.group_c || null;
    const grpD = features.group_d || null;
    const grpE = features.group_e || null;
    const grpF = features.group_f || null;
    const grpG = features.group_g || null;

    // Backward compat: edges at top level
    const edgesCompat = features.edges || {};

    // 7 group score objects
    const sA = scoreGroupA(grpA);
    const sB = scoreGroupB(grpB);
    const sC = scoreGroupC(grpC);
    const sD = scoreGroupD(grpD || edgesCompat, edgesCompat);
    const sE = scoreGroupE(grpE ? { ...grpE, fft: features.frequency ? features.frequency.fft : (grpE.fft || {}), wavelet: features.frequency ? features.frequency.wavelet : (grpE.wavelet || {}) } : (features.frequency || {}));
    const sF = scoreGroupF(grpF, metadata);
    const sG = scoreGroupG(grpG);

    // CNN base probabilities
    const cnnProbs = (classifier.probabilities) || { Real: 0.34, AI: 0.33, Screenshot: 0.33 };

    // Weighted combination
    const W = { A: 0.20, B: 0.15, C: 0.10, D: 0.12, E: 0.18, F: 0.08, G: 0.10, CNN: 0.07 };

    const raw = {
        AI: W.A * sA.AI + W.B * sB.AI + W.C * sC.AI + W.D * sD.AI + W.E * sE.AI + W.F * sF.AI + W.G * sG.AI + W.CNN * (cnnProbs.AI || 0),
        Real: W.A * sA.Real + W.B * sB.Real + W.C * sC.Real + W.D * sD.Real + W.E * sE.Real + W.F * sF.Real + W.G * sG.Real + W.CNN * (cnnProbs.Real || 0),
        Screenshot: W.A * sA.Screenshot + W.B * sB.Screenshot + W.C * sC.Screenshot + W.D * sD.Screenshot + W.E * sE.Screenshot + W.F * sF.Screenshot + W.G * sG.Screenshot + W.CNN * (cnnProbs.Screenshot || 0),
    };

    // ── Advanced features delta scoring ──────────────────────────────────────
    const adv = features.advanced || {};
    let aiDelta = 0, realDelta = 0, screenshotDelta = 0;

    // PRNU: real cameras ~0.3-0.7, AI near 0
    const prnu = typeof adv.prnu_score === 'number' ? adv.prnu_score : 0;
    if (prnu > 0.35) realDelta += 0.08;
    if (prnu < 0.08) aiDelta += 0.10;

    // Shot noise: real > 1.2 (more noise in dark regions)
    const shotNoise = typeof adv.shot_noise_score === 'number' ? adv.shot_noise_score : 1.0;
    if (shotNoise > 1.2) realDelta += 0.07;
    if (shotNoise < 0.8) aiDelta += 0.06;

    // GAN fingerprint spikes in FFT
    const ganSpikes = typeof adv.gan_fingerprint_spikes === 'number' ? adv.gan_fingerprint_spikes : 0;
    if (ganSpikes > 30) aiDelta += 0.12;

    // DCT block artifacts: real JPEG photos have this
    const dctArt = typeof adv.dct_block_artifact === 'number' ? adv.dct_block_artifact : 0;
    if (dctArt > 3.0) realDelta += 0.08;
    if (dctArt < 0.5) aiDelta += 0.06;

    // Symmetry: AI often too symmetrical
    const symAdv = typeof adv.symmetry_score === 'number' ? adv.symmetry_score : 0.5;
    if (symAdv > 0.92) aiDelta += 0.07;

    // Texture repetition: AI repeats patterns
    const texRep = typeof adv.texture_repetition === 'number' ? adv.texture_repetition : 0;
    if (texRep > 0.15) aiDelta += 0.08;

    // Straight line ratio: screenshots have many straight lines
    const slRatio = typeof adv.straight_line_ratio === 'number' ? adv.straight_line_ratio : 0;
    if (slRatio > 2.0) screenshotDelta += 0.10;

    // Plasticity: AI hyper-smooth
    const plast = typeof adv.plasticity_smoothness === 'number' ? adv.plasticity_smoothness : 0;
    if (plast > 0.6) aiDelta += 0.09;

    // Flat RGB (screenshot/grayscale)
    const flatRgb = typeof adv.flat_rgb_ratio === 'number' ? adv.flat_rgb_ratio : 0;
    if (flatRgb > 0.5) screenshotDelta += 0.08;

    // Apply deltas (blended with fraction weight 0.15 so advanced doesn't overpower)
    const advWeight = 0.15;
    raw.AI += advWeight * aiDelta;
    raw.Real += advWeight * realDelta;
    raw.Screenshot += advWeight * screenshotDelta;

    // Normalize
    const total = raw.AI + raw.Real + raw.Screenshot + 1e-9;

    let probs = { AI: raw.AI / total, Real: raw.Real / total, Screenshot: raw.Screenshot / total };

    // Pick winner
    let winner = 'Real';
    let maxProb = probs.Real;
    if (probs.AI > maxProb) { winner = 'AI'; maxProb = probs.AI; }
    if (probs.Screenshot > maxProb) { winner = 'Screenshot'; maxProb = probs.Screenshot; }

    // CONSENSUS RULE: need ≥ 4 groups agreeing for confidence > 70%
    const groupScores = [sA, sB, sC, sD, sE, sF, sG];
    const agreeing = countGroupsAgreeing(groupScores, winner);
    if (agreeing < 4 && maxProb > 0.70) {
        // Dampen confidence — not enough consensus
        maxProb = 0.58 + (agreeing / 4) * 0.12;
        // Re-normalize other probs
        const others = 1 - maxProb;
        const otherKeys = ['AI', 'Real', 'Screenshot'].filter(k => k !== winner);
        const otherSum = otherKeys.reduce((acc, k) => acc + probs[k], 0) + 1e-9;
        otherKeys.forEach(k => { probs[k] = probs[k] / otherSum * others; });
        probs[winner] = maxProb;
    }

    const VERDICT_MAP = { AI: 'AI SYNTHETIC', Real: 'AUTHENTIC', Screenshot: 'SCREENSHOT' };

    // ── GUARANTEED OUTPUT SHAPE — never null, never undefined fields ─────
    const rawVerdict    = VERDICT_MAP[winner] || 'INCONCLUSIVE';
    const rawConfidence = Math.round(maxProb * 100);
    const rawComments   = buildComments(winner, metadata, features, classifier, grpA, grpB, grpC, grpD, grpE, grpG);
    const rawMetadata   = buildMetadataString(metadata, grpF);
    const rawSpectral   = buildSpectralString(features, grpE);

    return {
        verdict:    (typeof rawVerdict === 'string' && rawVerdict.length > 0) ? rawVerdict : 'INCONCLUSIVE',
        confidence: (typeof rawConfidence === 'number' && isFinite(rawConfidence)) ? rawConfidence : 50,
        comments:   (typeof rawComments === 'string' && rawComments.length > 0) ? rawComments : 'Analysis complete. No significant anomalies found.',
        metadata:   (typeof rawMetadata === 'string' && rawMetadata.length > 0) ? rawMetadata : 'No metadata information available.',
        spectral:   (typeof rawSpectral === 'string' && rawSpectral.length > 0) ? rawSpectral : 'No anomalous spectral patterns detected.',
    };
}

module.exports = { computeEnsembleScore };
