'use strict';

const express = require('express');
const path    = require('path');
const fs      = require('fs');

const detectRouter    = require('./routes/detect');
const reverseRouter   = require('./routes/reverse');
const reportRouter    = require('./routes/report');
const casesRouter     = require('./routes/cases');
const searchRouter    = require('./routes/search');
const v1Router        = require('./routes/v1');
const adminRouter     = require('./routes/admin');
const retrainRouter   = require('./routes/retrain');
const driftRouter     = require('./routes/drift');
const modelInfoRouter = require('./routes/modelInfo');

const app  = express();
const PORT = 8000;  // ✅ Fixed: frontend connects to port 8000

// ── Read current model version at startup ────────────────────────────────────
const MANIFEST_PATH = path.join(__dirname, 'models', 'manifest.json');
let currentModel = 'classifier.pt';

try {
    if (fs.existsSync(MANIFEST_PATH)) {
        const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'));
        currentModel = manifest.current_model || currentModel;
        console.log(`[TECHNO SCOPE] Model version: ${currentModel}`);
    }
} catch (err) {
    console.warn('[TECHNO SCOPE] Could not read manifest.json, using default model');
}

// Parse JSON bodies — increased limit for high-res forensic images
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));

// CORS — allow frontend (aranged.html) to communicate from any host
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-api-key');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// ── Model version middleware — attach to all /v1/* responses ─────────────────
app.use('/v1', (req, res, next) => {
    // Re-read manifest on each request so retrain updates are picked up
    try {
        if (fs.existsSync(MANIFEST_PATH)) {
            const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'));
            res.locals.model_version = manifest.current_model || currentModel;
        } else {
            res.locals.model_version = currentModel;
        }
    } catch {
        res.locals.model_version = currentModel;
    }

    // Monkey-patch res.json to inject model_version into every /v1 response
    const originalJson = res.json.bind(res);
    res.json = function(body) {
        if (body && typeof body === 'object' && !Array.isArray(body)) {
            body.model_version = res.locals.model_version;
        }
        return originalJson(body);
    };

    next();
});

// Serve the frontend static files from the project root
app.use(express.static(path.join(__dirname, '..')));
app.use('/public', express.static(path.join(__dirname, 'public')));

// Serve saved case files (PDFs + JSONs) from backend/projects/
app.use('/projects', express.static(path.join(__dirname, 'projects')));

// ── Routes ───────────────────────────────────────────────────────────────────
app.use('/detect', detectRouter);
app.use('/analyze', detectRouter);
app.use('/reverse', reverseRouter);
app.use('/report', reportRouter);             // POST /report → PDF generation
app.use('/v1/case', casesRouter);             // POST /v1/case, GET /v1/case/list
app.use('/search-similar', searchRouter);     // POST /search-similar → reverse image search
app.use('/v1', v1Router);                     // POST /v1/analyze, POST /v1/batch_analyze
app.use('/v1', retrainRouter);                // POST /v1/retrain
app.use('/v1', driftRouter);                  // GET  /v1/drift_report
app.use('/v1', modelInfoRouter);              // GET  /v1/model_info
app.use('/admin', adminRouter);               // GET  /admin, GET /admin/logs

// Root route — open localhost:8000 directly in browser
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'aranged.html'));
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', port: PORT, model: currentModel, timestamp: new Date().toISOString() });
});

// Global error handler
app.use((err, req, res, next) => {
  if (err.type === 'entity.too.large') {
      return res.status(413).json({ error: 'Payload too large', message: 'The request body exceeds the limit (100MB).' });
  }
  console.error('[Server Error]', err);
  res.status(500).json({ error: 'Internal server error', message: err.message });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`[TECHNO SCOPE] Server running at http://localhost:${PORT}`);
  console.log(`[TECHNO SCOPE] POST /analyze          — primary detection endpoint`);
  console.log(`[TECHNO SCOPE] POST /detect           — alias endpoint`);
  console.log(`[TECHNO SCOPE] POST /reverse          — image reverse engineering`);
  console.log(`[TECHNO SCOPE] POST /report           — PDF forensic report`);
  console.log(`[TECHNO SCOPE] POST /v1/case          — save comparison case`);
  console.log(`[TECHNO SCOPE] GET  /v1/case/list     — list saved cases`);
  console.log(`[TECHNO SCOPE] POST /search-similar   — reverse image search`);
  console.log(`[TECHNO SCOPE] POST /v1/analyze       — B2B API (auth required)`);
  console.log(`[TECHNO SCOPE] POST /v1/batch_analyze — batch API (auth + webhook)`);
  console.log(`[TECHNO SCOPE] POST /v1/retrain       — retrain model from dataset ZIP`);
  console.log(`[TECHNO SCOPE] GET  /v1/drift_report  — model drift analysis`);
  console.log(`[TECHNO SCOPE] GET  /v1/model_info    — current model metadata`);
  console.log(`[TECHNO SCOPE] GET  /admin            — admin dashboard`);
});

module.exports = app;
