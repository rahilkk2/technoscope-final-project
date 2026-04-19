'use strict';

// ══════════════════════════════════════════════════════════════════════════════
// ASSUMPTION: Actual project paths differ from generic spec:
//   train.py        → PROJECT_ROOT/train.py   (not /backend/python_scripts/)
//   infer.py        → backend/analysis/infer.py
//   models/         → backend/models/
//   Auth: checked against process.env.API_KEY  (set via env or .env)
//        Falls back to keys.json if API_KEY env is not set.
// ══════════════════════════════════════════════════════════════════════════════

const express  = require('express');
const path     = require('path');
const fs       = require('fs');
const os       = require('os');
const crypto   = require('crypto');
const { spawn } = require('child_process');
const axios    = require('axios');
const unzipper = require('unzipper');

const router = express.Router();

// ── Resolved paths ───────────────────────────────────────────────────────────
const BACKEND_DIR   = path.join(__dirname, '..');
const PROJECT_ROOT  = path.join(BACKEND_DIR, '..');
const MODELS_DIR    = path.join(BACKEND_DIR, 'models');
const TRAIN_SCRIPT  = path.join(PROJECT_ROOT, 'train.py');
const MANIFEST_PATH = path.join(MODELS_DIR, 'manifest.json');

// ── Auth middleware (scoped to this file) ─────────────────────────────────────
function requireApiKey(req, res, next) {
    const provided = req.headers['x-api-key'];
    if (!provided) {
        return res.status(401).json({ error: 'Unauthorized', message: 'Missing x-api-key header' });
    }

    // Primary: check env var
    const envKey = process.env.API_KEY;
    if (envKey && provided === envKey) {
        return next();
    }

    // Fallback: check keys.json (existing project auth)
    try {
        const keysPath = path.join(BACKEND_DIR, 'keys.json');
        if (fs.existsSync(keysPath)) {
            const keys = JSON.parse(fs.readFileSync(keysPath, 'utf-8')).keys || [];
            if (keys.includes(provided)) {
                return next();
            }
        }
    } catch { /* ignore parse errors */ }

    return res.status(401).json({ error: 'Unauthorized', message: 'Invalid API key' });
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function sha256File(filePath) {
    return new Promise((resolve, reject) => {
        const hash = crypto.createHash('sha256');
        const stream = fs.createReadStream(filePath);
        stream.on('data', chunk => hash.update(chunk));
        stream.on('end', () => resolve(hash.digest('hex')));
        stream.on('error', reject);
    });
}

function downloadFile(url, destPath) {
    return new Promise(async (resolve, reject) => {
        try {
            const resp = await axios.get(url, {
                responseType: 'stream',
                timeout: 300_000,           // 5 min for large datasets
                maxContentLength: 2 * 1024 * 1024 * 1024,  // 2 GB
            });
            const writer = fs.createWriteStream(destPath);
            resp.data.pipe(writer);
            writer.on('finish', resolve);
            writer.on('error', reject);
        } catch (err) {
            reject(err);
        }
    });
}

function extractZip(zipPath, destDir) {
    return new Promise((resolve, reject) => {
        fs.createReadStream(zipPath)
            .pipe(unzipper.Extract({ path: destDir }))
            .on('close', resolve)
            .on('error', reject);
    });
}

function runTraining(dataPath, outputPath) {
    return new Promise((resolve, reject) => {
        const args = [TRAIN_SCRIPT, '--data', dataPath, '--output', outputPath];
        console.log(`[retrain] Spawning: python ${args.join(' ')}`);

        const proc = spawn('python', args, {
            cwd: PROJECT_ROOT,
            env: process.env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', chunk => { stdout += chunk.toString(); });
        proc.stderr.on('data', chunk => { stderr += chunk.toString(); });

        proc.on('error', err => {
            if (err.code === 'ENOENT') {
                return reject(new Error('Python not found in PATH'));
            }
            reject(err);
        });

        proc.on('close', code => {
            if (code !== 0) {
                const detail = stderr.trim() || stdout.trim() || '(no output)';
                return reject(new Error(`train.py exited with code ${code}: ${detail}`));
            }
            resolve(stdout);
        });
    });
}

function cleanupPaths(...paths) {
    for (const p of paths) {
        try {
            if (!p || !fs.existsSync(p)) continue;
            const stat = fs.statSync(p);
            if (stat.isDirectory()) {
                fs.rmSync(p, { recursive: true, force: true });
            } else {
                fs.unlinkSync(p);
            }
        } catch { /* best-effort cleanup */ }
    }
}

// ── POST /v1/retrain ─────────────────────────────────────────────────────────
router.post('/retrain', requireApiKey, async (req, res) => {
    const { dataset_url } = req.body;

    if (!dataset_url || typeof dataset_url !== 'string') {
        return res.status(400).json({ error: 'Missing or invalid "dataset_url" field' });
    }

    // Temp paths
    const stamp    = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    const tempDir  = path.join(os.tmpdir(), `technoscope-retrain-${stamp}-${Date.now()}`);
    const zipPath  = path.join(tempDir, 'dataset.zip');
    const dataPath = path.join(tempDir, 'data');
    const modelOut = path.join(MODELS_DIR, `classifier_v${stamp}.pt`);

    fs.mkdirSync(tempDir, { recursive: true });
    fs.mkdirSync(dataPath, { recursive: true });
    fs.mkdirSync(MODELS_DIR, { recursive: true });

    const t0 = Date.now();

    try {
        // Step 1 — Download ZIP
        console.log(`[retrain] Downloading dataset from ${dataset_url}`);
        await downloadFile(dataset_url, zipPath);

        if (!fs.existsSync(zipPath) || fs.statSync(zipPath).size === 0) {
            throw new Error('Downloaded file is empty or missing');
        }
        console.log(`[retrain] Downloaded ${(fs.statSync(zipPath).size / 1024 / 1024).toFixed(1)} MB`);

        // Step 2 — Extract ZIP
        console.log(`[retrain] Extracting to ${dataPath}`);
        await extractZip(zipPath, dataPath);

        // Step 3 — Run train.py
        console.log(`[retrain] Starting training → ${modelOut}`);
        await runTraining(dataPath, modelOut);

        if (!fs.existsSync(modelOut)) {
            throw new Error('Training completed but no model file was produced');
        }

        // Step 4 — SHA256
        const hash = await sha256File(modelOut);
        console.log(`[retrain] Model sha256: ${hash}`);

        // Step 5 — Write manifest.json
        const manifest = {
            current_model: path.basename(modelOut),
            trained_on:    new Date().toISOString(),
            dataset_url:   dataset_url,
            sha256:        hash,
        };
        fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2), 'utf-8');
        console.log(`[retrain] Manifest written to ${MANIFEST_PATH}`);

        // Step 6 — Cleanup temp
        cleanupPaths(tempDir);

        const durationMs = Date.now() - t0;
        console.log(`[retrain] Complete in ${durationMs}ms`);

        return res.json({
            success:     true,
            model_file:  manifest.current_model,
            sha256:      hash,
            duration_ms: durationMs,
        });

    } catch (err) {
        console.error('[retrain] Failed:', err.message);

        // Cleanup — do NOT touch the existing model or manifest
        cleanupPaths(tempDir);
        // Remove the partially-written new model if it exists
        if (fs.existsSync(modelOut)) {
            try { fs.unlinkSync(modelOut); } catch { /* ignore */ }
        }

        return res.status(500).json({
            error:   'Retraining failed',
            message: err.message,
        });
    }
});

module.exports = router;
