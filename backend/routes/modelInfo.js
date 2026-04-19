'use strict';

// ══════════════════════════════════════════════════════════════════════════════
// GET /v1/model_info — public, no auth
// Reads backend/models/manifest.json and returns it.
// If manifest doesn't exist, returns a sensible default.
// ══════════════════════════════════════════════════════════════════════════════

const express = require('express');
const path    = require('path');
const fs      = require('fs');

const router = express.Router();

const MANIFEST_PATH = path.join(__dirname, '..', 'models', 'manifest.json');

router.get('/model_info', (req, res) => {
    try {
        if (fs.existsSync(MANIFEST_PATH)) {
            const raw      = fs.readFileSync(MANIFEST_PATH, 'utf-8');
            const manifest = JSON.parse(raw);
            return res.json(manifest);
        }
    } catch (err) {
        console.error('[model_info] Failed to read manifest.json:', err.message);
    }

    // Default when manifest doesn't exist yet
    return res.json({
        current_model: 'classifier.pt',
        trained_on:    null,
        sha256:        null,
        dataset_url:   null,
    });
});

module.exports = router;
