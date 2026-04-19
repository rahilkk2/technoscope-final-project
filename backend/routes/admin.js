'use strict';

/**
 * admin.js — Admin dashboard routes
 *
 *   GET  /admin       — serves the admin dashboard HTML
 *   GET  /admin/logs  — returns recent analysis log entries as JSON
 */

const express = require('express');
const path    = require('path');
const { readLogs } = require('../utils/analysisLogger');

const router = express.Router();

// ── GET /admin → dashboard page ──────────────────────────────────────────────
router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'public', 'admin.html'));
});

// ── GET /admin/logs → JSON array of recent analyses ─────────────────────────
router.get('/logs', (req, res) => {
    const logs = readLogs();
    res.json(logs);
});

module.exports = router;
