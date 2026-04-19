'use strict';

/**
 * authMiddleware.js — API-key gate for /v1/* endpoints
 * Reads valid keys from keys.json on every request (hot-reload friendly).
 * Apply to any route that requires authentication.
 */

const fs   = require('fs');
const path = require('path');

const KEYS_FILE = path.join(__dirname, '..', 'keys.json');

/**
 * Load valid API keys from disk.
 * Re-reads on every call so you can edit keys.json without restarting.
 */
function loadKeys() {
    try {
        const raw    = fs.readFileSync(KEYS_FILE, 'utf-8');
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed.keys) ? parsed.keys : [];
    } catch (e) {
        console.error('[auth] Failed to load keys.json:', e.message);
        return [];
    }
}

/**
 * Express middleware — checks x-api-key header against keys.json.
 */
function authMiddleware(req, res, next) {
    const apiKey = req.headers['x-api-key'];

    if (!apiKey) {
        return res.status(401).json({
            error:   'Unauthorized',
            message: 'Missing x-api-key header',
        });
    }

    // NOTE FOR PRODUCTION: API keys should be in .env or a secrets manager.
    // process.env.API_KEY is checked first, then keys.json for additional keys.
    const validKeys = loadKeys();
    const envKey = process.env.API_KEY || '';
    if (envKey && !validKeys.includes(envKey)) {
        validKeys.push(envKey);
    }

    if (!validKeys.includes(apiKey)) {
        return res.status(401).json({
            error:   'Unauthorized',
            message: 'Invalid API key',
        });
    }

    next();
}

module.exports = authMiddleware;
