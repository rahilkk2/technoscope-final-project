'use strict';

// ══════════════════════════════════════════════════════════════════════════════
// ASSUMPTION: Actual project paths:
//   infer.py   → backend/analysis/infer.py
//   models/    → backend/models/
//   Drift sample images:
//     backend/drift_samples/real/  (ground truth: real photos)
//     backend/drift_samples/ai/   (ground truth: AI-generated)
//   Filename convention: <generator>_<year>_<id>.jpg
//   Auth: x-api-key checked against process.env.API_KEY or keys.json
// ══════════════════════════════════════════════════════════════════════════════

const express   = require('express');
const path      = require('path');
const fs        = require('fs');
const { spawn } = require('child_process');

const router = express.Router();

// ── Resolved paths ───────────────────────────────────────────────────────────
const BACKEND_DIR   = path.join(__dirname, '..');
const INFER_SCRIPT  = path.join(BACKEND_DIR, 'analysis', 'infer.py');
const MODELS_DIR    = path.join(BACKEND_DIR, 'models');
const MANIFEST_PATH = path.join(MODELS_DIR, 'manifest.json');
const SAMPLES_DIR   = path.join(BACKEND_DIR, 'drift_samples');
const REAL_DIR      = path.join(SAMPLES_DIR, 'real');
const AI_DIR        = path.join(SAMPLES_DIR, 'ai');

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);

// ── Auth middleware ──────────────────────────────────────────────────────────
function requireApiKey(req, res, next) {
    const provided = req.headers['x-api-key'];
    if (!provided) {
        return res.status(401).json({ error: 'Unauthorized', message: 'Missing x-api-key header' });
    }
    const envKey = process.env.API_KEY;
    if (envKey && provided === envKey) return next();
    try {
        const keysPath = path.join(BACKEND_DIR, 'keys.json');
        if (fs.existsSync(keysPath)) {
            const keys = JSON.parse(fs.readFileSync(keysPath, 'utf-8')).keys || [];
            if (keys.includes(provided)) return next();
        }
    } catch { /* ignore */ }
    return res.status(401).json({ error: 'Unauthorized', message: 'Invalid API key' });
}

// ── Run infer.py on a single image ───────────────────────────────────────────
function inferImage(imagePath) {
    return new Promise((resolve, reject) => {
        // Determine which model weights to use
        let modelPath = path.join(MODELS_DIR, 'classifier.pt');
        try {
            if (fs.existsSync(MANIFEST_PATH)) {
                const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'));
                if (manifest.current_model) {
                    const candidate = path.join(MODELS_DIR, manifest.current_model);
                    if (fs.existsSync(candidate)) {
                        modelPath = candidate;
                    }
                }
            }
        } catch { /* use default */ }

        const args = [INFER_SCRIPT, imagePath, '--pt-path', modelPath];
        const proc = spawn('python', args, {
            cwd: BACKEND_DIR,
            env: process.env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', c => { stdout += c.toString(); });
        proc.stderr.on('data', c => { stderr += c.toString(); });

        proc.on('error', err => reject(err));
        proc.on('close', code => {
            if (code !== 0) {
                return reject(new Error(`infer.py exit ${code}: ${stderr.trim() || stdout.trim()}`));
            }
            // Parse last JSON line
            const lines = stdout.trim().split('\n');
            for (let i = lines.length - 1; i >= 0; i--) {
                const line = lines[i].trim();
                if (line.startsWith('{')) {
                    try {
                        return resolve(JSON.parse(line));
                    } catch (e) {
                        return reject(new Error(`JSON parse error: ${e.message}`));
                    }
                }
            }
            reject(new Error('infer.py produced no JSON output'));
        });
    });
}

// ── List images in a directory ───────────────────────────────────────────────
function listImages(dir) {
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir)
        .filter(f => IMAGE_EXTS.has(path.extname(f).toLowerCase()))
        .map(f => ({
            filename: f,
            fullPath: path.join(dir, f),
        }));
}

// ── Parse filename: <generator>_<year>_<id>.jpg ──────────────────────────────
function parseFilename(filename) {
    const base = path.basename(filename, path.extname(filename));
    const parts = base.split('_');
    let generator = 'unknown';
    let year = 0;

    if (parts.length >= 2) {
        generator = parts[0].toLowerCase();
        const yearCandidate = parseInt(parts[1], 10);
        if (yearCandidate >= 2000 && yearCandidate <= 2099) {
            year = yearCandidate;
        }
    }
    return { generator, year };
}

// ── GET /v1/drift_report ─────────────────────────────────────────────────────
router.get('/drift_report', requireApiKey, async (req, res) => {
    const wantHtml = req.query.format === 'html';

    // ── Ensure sample dirs exist ─────────────────────────────────────────────
    if (!fs.existsSync(REAL_DIR) || !fs.existsSync(AI_DIR)) {
        fs.mkdirSync(REAL_DIR, { recursive: true });
        fs.mkdirSync(AI_DIR, { recursive: true });

        const msg = `Drift sample dirs created but empty. ` +
                    `Place ground-truth images in:\n  ${REAL_DIR}\n  ${AI_DIR}\n` +
                    `Filename convention: <generator>_<year>_<id>.jpg`;

        if (wantHtml) {
            return res.type('html').send(`<p>${msg.replace(/\n/g, '<br>')}</p>`);
        }
        return res.json({ error: msg, accuracy_real: null, fpr_ai: null, by_generator: [] });
    }

    const realImages = listImages(REAL_DIR);
    const aiImages   = listImages(AI_DIR);

    if (realImages.length === 0 && aiImages.length === 0) {
        const msg = 'No images found in drift_samples/real/ or drift_samples/ai/';
        if (wantHtml) return res.type('html').send(`<p>${msg}</p>`);
        return res.json({ error: msg, accuracy_real: null, fpr_ai: null, by_generator: [] });
    }

    // ── Run inference on all images ──────────────────────────────────────────
    let correctReal = 0;
    let totalReal   = realImages.length;
    let falseRealFromAI = 0;
    let totalAI     = aiImages.length;

    // Per-generator tracking for AI set
    const generatorMap = {};  // key: "generator_year" → { total, falseReal }

    // Process real images
    for (const img of realImages) {
        try {
            const result = await inferImage(img.fullPath);
            const predicted = (result.predicted_class || '').toLowerCase();
            if (predicted === 'real') {
                correctReal++;
            }
        } catch (err) {
            console.error(`[drift] Failed on ${img.filename}:`, err.message);
            // Count as missed
        }
    }

    // Process AI images
    for (const img of aiImages) {
        try {
            const result = await inferImage(img.fullPath);
            const predicted = (result.predicted_class || '').toLowerCase();
            const { generator, year } = parseFilename(img.filename);
            const key = `${generator}_${year}`;

            if (!generatorMap[key]) {
                generatorMap[key] = { generator, year, total: 0, falseReal: 0 };
            }
            generatorMap[key].total++;

            // FPR: AI image incorrectly called "real"
            if (predicted === 'real') {
                falseRealFromAI++;
                generatorMap[key].falseReal++;
            }
        } catch (err) {
            console.error(`[drift] Failed on ${img.filename}:`, err.message);
        }
    }

    // ── Compute metrics ──────────────────────────────────────────────────────
    const accuracyReal = totalReal > 0 ? Math.round((correctReal / totalReal) * 10000) / 10000 : null;
    const fprAI        = totalAI > 0   ? Math.round((falseRealFromAI / totalAI) * 10000) / 10000 : null;

    const byGenerator = Object.values(generatorMap)
        .map(g => ({
            generator: g.generator,
            year:      g.year,
            fpr:       g.total > 0 ? Math.round((g.falseReal / g.total) * 10000) / 10000 : 0,
            total:     g.total,
            false_real: g.falseReal,
        }))
        .sort((a, b) => a.generator.localeCompare(b.generator) || a.year - b.year);

    const report = {
        accuracy_real: accuracyReal,
        fpr_ai:        fprAI,
        total_real:    totalReal,
        total_ai:      totalAI,
        by_generator:  byGenerator,
    };

    // ── Return format ────────────────────────────────────────────────────────
    if (wantHtml) {
        let html = `<!DOCTYPE html>
<html><head><title>TechnoScope Drift Report</title></head><body>
<h2>TechnoScope — Model Drift Report</h2>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Real Accuracy</td><td>${accuracyReal !== null ? (accuracyReal * 100).toFixed(2) + '%' : 'N/A'}</td></tr>
<tr><td>AI False Positive Rate</td><td>${fprAI !== null ? (fprAI * 100).toFixed(2) + '%' : 'N/A'}</td></tr>
<tr><td>Total Real Samples</td><td>${totalReal}</td></tr>
<tr><td>Total AI Samples</td><td>${totalAI}</td></tr>
</table>
<h3>By Generator</h3>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Generator</th><th>Year</th><th>FPR</th><th>Total</th><th>False Real</th></tr>`;

        for (const g of byGenerator) {
            html += `<tr>
<td>${escHtml(g.generator)}</td>
<td>${g.year || '—'}</td>
<td>${(g.fpr * 100).toFixed(2)}%</td>
<td>${g.total}</td>
<td>${g.false_real}</td>
</tr>`;
        }

        html += `</table>
<p><em>Generated at ${new Date().toISOString()}</em></p>
</body></html>`;

        return res.type('html').send(html);
    }

    return res.json(report);
});

function escHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

module.exports = router;
