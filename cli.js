#!/usr/bin/env node
'use strict';

/**
 * cli.js — TechnoScope offline CLI
 *
 * Runs the full forensic pipeline (Python feature extraction + classifier +
 * ensemble scorer) directly via child_process — no HTTP server needed.
 *
 * Usage:
 *   node cli.js --image photo.jpg
 *   node cli.js --image photo.jpg --output result.json
 *   node cli.js --image-url https://example.com/photo.jpg
 *   node cli.js --image-url https://example.com/photo.jpg --output result.json
 *
 * Exit codes:
 *   0 — success (JSON on stdout)
 *   1 — error   (message on stderr)
 */

const fs     = require('fs');
const path   = require('path');
const os     = require('os');
const crypto = require('crypto');

// ── Resolve project paths (cli.js lives at project root) ─────────────────────
const PROJECT_ROOT = __dirname;
const BACKEND_DIR  = path.join(PROJECT_ROOT, 'backend');
const ANALYSIS_DIR = path.join(BACKEND_DIR, 'analysis');

// ── Lazy-load backend modules (same ones the server uses) ────────────────────
const { computeEnsembleScore } = require(path.join(BACKEND_DIR, 'utils', 'ensembleScorer'));

// ── Parse CLI arguments ──────────────────────────────────────────────────────
function parseArgs() {
    const args = process.argv.slice(2);
    const opts = { image: null, imageUrl: null, output: null };

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--image':
                opts.image = args[++i];
                break;
            case '--image-url':
                opts.imageUrl = args[++i];
                break;
            case '--output':
                opts.output = args[++i];
                break;
            case '--help':
            case '-h':
                console.log(`
TechnoScope CLI — Offline Image Forensics

Usage:
  node cli.js --image <local-path>              Analyse a local file
  node cli.js --image-url <url>                 Download + analyse a remote image
  node cli.js --image <path> --output out.json  Write result to file

Exit codes: 0 = success, 1 = error
`);
                process.exit(0);
            default:
                fatal(`Unknown flag: ${args[i]}. Use --help for usage.`);
        }
    }

    if (!opts.image && !opts.imageUrl) {
        fatal('Provide --image <path> or --image-url <url>. Use --help for usage.');
    }
    return opts;
}

function fatal(msg) {
    process.stderr.write(`[ERROR] ${msg}\n`);
    process.exit(1);
}

// ── Run a Python script and return parsed JSON ───────────────────────────────
function runPythonScript(scriptName, imagePath) {
    const { spawnSync } = require('child_process');
    const scriptPath = path.join(ANALYSIS_DIR, scriptName);

    const result = spawnSync('python', [scriptPath, imagePath], {
        cwd: ANALYSIS_DIR,
        encoding: 'utf-8',
        timeout: 120_000,       // 2 min max per script
        maxBuffer: 50 * 1024 * 1024,
    });

    if (result.error) {
        if (result.error.code === 'ENOENT') {
            fatal('Python not found. Make sure "python" is in your PATH.');
        }
        throw result.error;
    }

    if (result.status !== 0) {
        const detail = (result.stdout || '').trim() || (result.stderr || '').trim();
        fatal(`${scriptName} exited with code ${result.status}.\n${detail}`);
    }

    // Parse last JSON line from stdout
    const lines = (result.stdout || '').trim().split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith('{') || line.startsWith('[')) {
            return JSON.parse(line);
        }
    }
    fatal(`${scriptName} produced no JSON output.`);
}

// ── Download image from URL to temp file ─────────────────────────────────────
async function downloadToTemp(url) {
    // Dynamic import of axios (already in backend/node_modules)
    let axios;
    try {
        axios = require(path.join(BACKEND_DIR, 'node_modules', 'axios'));
    } catch {
        // Fallback: try global
        axios = require('axios');
    }

    const resp = await axios.get(url, {
        responseType: 'arraybuffer',
        timeout: 30_000,
        maxContentLength: 20 * 1024 * 1024,
    });

    const ct  = (resp.headers['content-type'] || '').toLowerCase();
    const ext = ct.includes('png') ? '.png' : ct.includes('webp') ? '.webp' : '.jpg';
    const tmp = path.join(os.tmpdir(), `technoscope-${crypto.randomUUID()}${ext}`);
    fs.writeFileSync(tmp, Buffer.from(resp.data));
    return tmp;
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
    const opts = parseArgs();
    let imagePath = opts.image;
    let tempFile  = null;

    // ── Resolve image path ───────────────────────────────────────────────────
    if (opts.imageUrl) {
        process.stderr.write(`[cli] Downloading ${opts.imageUrl} …\n`);
        try {
            imagePath = await downloadToTemp(opts.imageUrl);
            tempFile  = imagePath;
        } catch (err) {
            fatal(`Download failed: ${err.message}`);
        }
    }

    if (!fs.existsSync(imagePath)) {
        fatal(`File not found: ${imagePath}`);
    }

    const absPath = path.resolve(imagePath);

    // ── Run the 3 Python scripts (same as pythonBridge.js) ───────────────────
    process.stderr.write(`[cli] Analysing: ${absPath}\n`);
    const t0 = Date.now();

    let metadata, features, classifier;
    try {
        metadata   = runPythonScript('metadata.py',         absPath);
        features   = runPythonScript('featureExtractor.py', absPath);
        classifier = runPythonScript('classifier.py',       absPath);
    } catch (err) {
        fatal(`Python analysis failed: ${err.message}`);
    }

    // ── Ensemble scoring (same as server) ────────────────────────────────────
    const pythonResult = { metadata, features, classifier };
    const ensemble     = computeEnsembleScore(pythonResult);

    // ── Build output object ──────────────────────────────────────────────────
    const output = {
        file:       path.basename(absPath),
        verdict:    ensemble.verdict,
        confidence: ensemble.confidence,
        comments:   ensemble.comments,
        metadata_summary: ensemble.metadata,
        spectral_summary: ensemble.spectral,
        evidence: {
            sensor_noise:  features.group_a  || {},
            texture:       features.group_b  || {},
            color:         features.group_c  || {},
            edge_geometry: features.group_d  || {},
            frequency:     features.group_e  || {},
            file_metadata: features.group_f  || {},
            semantic:      features.group_g  || {},
            advanced:      features.advanced || {},
        },
        classifier: classifier,
        timing_ms:  Date.now() - t0,
    };

    // ── Output ───────────────────────────────────────────────────────────────
    const json = JSON.stringify(output, null, 2);
    process.stdout.write(json + '\n');

    if (opts.output) {
        fs.writeFileSync(opts.output, json, 'utf-8');
        process.stderr.write(`[cli] Result written to ${opts.output}\n`);
    }

    // ── Cleanup temp file ────────────────────────────────────────────────────
    if (tempFile) {
        try { fs.unlinkSync(tempFile); } catch { /* ignore */ }
    }

    process.stderr.write(`[cli] Done in ${output.timing_ms}ms\n`);
    process.exit(0);
}

main().catch(err => { fatal(err.message); });
