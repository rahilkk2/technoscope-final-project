'use strict';

/**
 * analysisLogger.js — Append-only analysis log (JSON flat file)
 * Stores the last 500 analysis entries for the admin dashboard.
 * Thread-safe enough for single-process Express.
 */

const fs   = require('fs');
const path = require('path');

const LOG_FILE    = path.join(__dirname, '..', 'logs.json');
const MAX_ENTRIES = 500;

/**
 * Read all log entries from disk.
 * Returns [] on any error.
 */
function readLogs() {
    try {
        if (fs.existsSync(LOG_FILE)) {
            const raw = fs.readFileSync(LOG_FILE, 'utf-8');
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
        }
    } catch (e) {
        console.error('[logger] Failed to read logs.json:', e.message);
    }
    return [];
}

/**
 * Write the full log array to disk.
 */
function writeLogs(logs) {
    try {
        fs.writeFileSync(LOG_FILE, JSON.stringify(logs, null, 2), 'utf-8');
    } catch (e) {
        console.error('[logger] Failed to write logs.json:', e.message);
    }
}

/**
 * Append a single analysis entry.
 * @param {Object} entry - { image_ref, verdict, confidence, evidence, comments?, metadata?, spectral? }
 */
function logAnalysis(entry) {
    const logs = readLogs();

    logs.unshift({
        timestamp:  new Date().toISOString(),
        image_ref:  entry.image_ref  || 'unknown',
        verdict:    entry.verdict    || 'unknown',
        confidence: entry.confidence || 0,
        evidence:   entry.evidence   || {},
        comments:   entry.comments   || '',
        metadata:   entry.metadata   || '',
        spectral:   entry.spectral   || '',
    });

    // Trim to max
    if (logs.length > MAX_ENTRIES) {
        logs.length = MAX_ENTRIES;
    }

    writeLogs(logs);
}

module.exports = { logAnalysis, readLogs };
