'use strict';

/**
 * v1.js — B2B Forensic API routes
 *
 *   POST /v1/analyze        — single image analysis (JSON-only, auth required)
 *   POST /v1/batch_analyze  — batch analysis with webhook callback
 *
 * Both endpoints accept image data as URL or base64 string.
 * Auth is handled by authMiddleware (x-api-key header → keys.json).
 */

const express  = require('express');
const path     = require('path');
const fs       = require('fs');
const crypto   = require('crypto');
const axios    = require('axios');

const { runPythonAnalysis }  = require('../services/pythonBridge');
const { computeEnsembleScore } = require('../utils/ensembleScorer');
const { logAnalysis }        = require('../utils/analysisLogger');
const authMiddleware         = require('../middleware/authMiddleware');

const router    = express.Router();
const uploadDir = path.join(__dirname, '..', 'uploads');

// Ensure upload dir exists
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}


// ══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Persist an image (from URL or base64) to a temp file and return the path.
 */
async function saveImageToTemp(imageInput) {
    const stamp = `${Date.now()}-${Math.round(Math.random() * 1e6)}`;

    // ── URL download ─────────────────────────────────────────────────────────
    if (imageInput.startsWith('http://') || imageInput.startsWith('https://')) {
        const resp = await axios.get(imageInput, {
            responseType: 'arraybuffer',
            timeout: 30000,
            maxContentLength: 100 * 1024 * 1024,   // 100 MB limit
        });

        const ct  = (resp.headers['content-type'] || '').toLowerCase();
        const ext = ct.includes('png') ? '.png'
                  : ct.includes('webp') ? '.webp'
                  : ct.includes('bmp')  ? '.bmp'
                  : '.jpg';

        const filePath = path.join(uploadDir, `api-${stamp}${ext}`);
        fs.writeFileSync(filePath, Buffer.from(resp.data));
        return filePath;
    }

    // ── Base64 decode ────────────────────────────────────────────────────────
    let b64  = imageInput;
    let ext  = '.jpg';

    if (b64.startsWith('data:')) {
        const match = b64.match(/^data:image\/(\w+);base64,/);
        if (match) {
            ext = '.' + match[1].replace('jpeg', 'jpg');
            b64 = b64.replace(/^data:image\/\w+;base64,/, '');
        }
    }

    const buffer   = Buffer.from(b64, 'base64');
    const filePath = path.join(uploadDir, `api-${stamp}${ext}`);
    fs.writeFileSync(filePath, buffer);
    return filePath;
}

/**
 * Run the full analysis pipeline and return the B2B API response shape.
 */
async function analyzeImage(imagePath) {
    // Step 1 — Python feature extraction + classifier
    const pythonResult = await runPythonAnalysis(imagePath);

    // Step 2 — Weighted ensemble scoring
    const ensemble = computeEnsembleScore(pythonResult);

    // Step 3 — Map verdict to short API form
    const VERDICT_MAP = {
        'AI SYNTHETIC': 'AI',
        'AUTHENTIC':    'Real',
        'SCREENSHOT':   'Screenshot',
    };

    return {
        verdict:    VERDICT_MAP[ensemble.verdict] || ensemble.verdict,
        confidence: ensemble.confidence,
        evidence: {
            sensor_noise:  pythonResult.features.group_a  || {},
            texture:       pythonResult.features.group_b  || {},
            color:         pythonResult.features.group_c  || {},
            edge_geometry: pythonResult.features.group_d  || {},
            frequency:     pythonResult.features.group_e  || {},
            file_metadata: pythonResult.features.group_f  || {},
            semantic:      pythonResult.features.group_g  || {},
            advanced:      pythonResult.features.advanced  || {},
        },
        comments:         ensemble.comments,
        metadata_summary: ensemble.metadata,
        spectral_summary: ensemble.spectral,
    };
}


// ══════════════════════════════════════════════════════════════════════════════
// POST /v1/analyze — single image
// ══════════════════════════════════════════════════════════════════════════════

router.post('/analyze', authMiddleware, async (req, res) => {
    const { image } = req.body;

    if (!image || typeof image !== 'string') {
        return res.status(400).json({
            error: 'Missing or invalid "image" field (provide a URL or base64 string)',
        });
    }

    let imagePath = null;
    try {
        imagePath = await saveImageToTemp(image);
        const result = await analyzeImage(imagePath);

        // Log
        logAnalysis({
            image_ref:  image.substring(0, 120),
            verdict:    result.verdict,
            confidence: result.confidence,
            evidence:   result.evidence,
            comments:   result.comments,
            metadata:   result.metadata_summary,
            spectral:   result.spectral_summary,
        });

        return res.json(result);

    } catch (err) {
        console.error('[v1/analyze] Error:', err.message);
        return res.status(500).json({
            error:   'Analysis failed',
            message: err.message,
        });
    } finally {
        if (imagePath) fs.unlink(imagePath, () => {});
    }
});


// ══════════════════════════════════════════════════════════════════════════════
// POST /v1/batch_analyze — batch with webhook
// ══════════════════════════════════════════════════════════════════════════════

router.post('/batch_analyze', authMiddleware, async (req, res) => {
    const { images, webhook } = req.body;

    if (!images || !Array.isArray(images) || images.length === 0) {
        return res.status(400).json({
            error: 'Missing or empty "images" array',
        });
    }

    if (images.length > 20) {
        return res.status(400).json({
            error: 'Batch limited to 20 images per request',
        });
    }

    const jobId = crypto.randomUUID();

    // ── Respond immediately ──────────────────────────────────────────────────
    res.json({
        job_id:      jobId,
        status:      'queued',
        image_count: images.length,
    });

    // ── Background processing ────────────────────────────────────────────────
    setImmediate(async () => {
        console.log(`[batch] Job ${jobId} started — ${images.length} images`);
        const results = [];

        for (let i = 0; i < images.length; i++) {
            const imageInput = images[i];
            let imagePath = null;

            try {
                imagePath = await saveImageToTemp(imageInput);
                const result = await analyzeImage(imagePath);

                results.push({
                    index:      i,
                    image:      imageInput.substring(0, 120),
                    verdict:    result.verdict,
                    confidence: result.confidence,
                    evidence:   result.evidence,
                });

                // Log each
                logAnalysis({
                    image_ref:  imageInput.substring(0, 120),
                    verdict:    result.verdict,
                    confidence: result.confidence,
                    evidence:   result.evidence,
                    comments:   result.comments,
                    metadata:   result.metadata_summary,
                    spectral:   result.spectral_summary,
                });

            } catch (err) {
                console.error(`[batch] Image ${i} failed:`, err.message);
                results.push({
                    index:   i,
                    image:   imageInput.substring(0, 120),
                    error:   'failed to process',
                    message: err.message,
                });
            } finally {
                if (imagePath) fs.unlink(imagePath, () => {});
            }
        }

        // ── POST to webhook ──────────────────────────────────────────────────
        if (webhook && typeof webhook === 'string') {
            try {
                await axios.post(webhook, {
                    job_id:  jobId,
                    status:  'completed',
                    results,
                }, { timeout: 15000 });

                console.log(`[batch] Webhook delivered → ${webhook} (job ${jobId})`);
            } catch (err) {
                console.error(`[batch] Webhook delivery failed for job ${jobId}:`, err.message);
            }
        }

        console.log(`[batch] Job ${jobId} completed — ${results.length} processed`);
    });
});


module.exports = router;
