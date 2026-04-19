'use strict';

const express = require('express');
const path = require('path');
const fs = require('fs');
const router = express.Router();

// ── Projects directory (file-based storage, no DB) ───────────────────────────
const projectsDir = path.join(__dirname, '..', 'projects');
if (!fs.existsSync(projectsDir)) {
    fs.mkdirSync(projectsDir, { recursive: true });
}

/**
 * POST /v1/case
 *
 * Save a two-image comparison case. Stores JSON + generates comparative PDF.
 * Body: { case_id?, original: {...}, altered: {...} }
 */
router.post('/', async (req, res) => {
    try {
        const {
            case_id = `Case_${Date.now()}`,
            original = {},
            altered = {},
        } = req.body;

        // Sanitize case_id for filesystem safety
        const safeName = case_id.replace(/[^a-zA-Z0-9_\-]/g, '_');

        // Save JSON
        const jsonPath = path.join(projectsDir, `${safeName}.json`);
        fs.writeFileSync(jsonPath, JSON.stringify({ case_id: safeName, original, altered, created: new Date().toISOString() }, null, 2));
        console.log(`[cases] Saved case JSON: ${safeName}.json`);

        // Generate comparative PDF
        const timestamp = new Date().toLocaleString('en-US', { dateStyle: 'full', timeStyle: 'medium' });

        function buildSideColumn(label, data) {
            const v = data.verdict || 'N/A';
            const c = data.confidence || 0;
            const vColor = v === 'AUTHENTIC' ? '#34d399' : '#f87171';
            const scores = data.spectral_scores || {};
            const labels = {
                A_sensor_noise: 'A — Sensor', B_texture: 'B — Texture',
                C_color: 'C — Color', D_edge: 'D — Edge',
                E_frequency: 'E — Frequency', F_metadata: 'F — Meta',
                G_semantic: 'G — Semantic',
            };
            let bars = '';
            for (const [k, lbl] of Object.entries(labels)) {
                const val = scores[k] ?? 0;
                const clr = val > 70 ? '#f87171' : val > 50 ? '#fbbf24' : '#34d399';
                bars += `
                    <div style="margin-bottom:8px;">
                        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;">
                            <span>${lbl}</span><span>${val}%</span>
                        </div>
                        <div style="background:#1a1a1a;border-radius:3px;height:10px;">
                            <div style="width:${val}%;height:100%;background:${clr};border-radius:3px;"></div>
                        </div>
                    </div>`;
            }
            return `
                <div style="flex:1;padding:16px;background:#111;border-radius:8px;border:1px solid #222;">
                    <h3 style="color:#888;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">${label}</h3>
                    <div style="text-align:center;margin-bottom:16px;">
                        <div style="font-size:24px;font-weight:700;color:${vColor};">${v}</div>
                        <div style="font-size:14px;color:#aaa;">Confidence: ${c}%</div>
                    </div>
                    <div style="font-size:12px;color:#999;margin-bottom:12px;">${data.comments || ''}</div>
                    ${bars}
                </div>`;
        }

        const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
    *{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0a;color:#e0e0e0;padding:40px;line-height:1.5;}
    .header{text-align:center;margin-bottom:30px;border-bottom:1px solid #222;padding-bottom:20px;}
    .header h1{font-size:24px;color:#fff;letter-spacing:2px;}
    .header p{color:#777;font-size:12px;margin-top:4px;}
    .footer{margin-top:40px;text-align:center;color:#555;font-size:10px;border-top:1px solid #222;padding-top:12px;}
</style></head><body>
    <div class="header">
        <h1>TECHNO SCOPE — Case Comparison Report</h1>
        <p>Case ID: ${safeName} | Generated: ${timestamp}</p>
    </div>
    <div style="display:flex;gap:16px;">
        ${buildSideColumn('Original Image', original)}
        ${buildSideColumn('Altered Image', altered)}
    </div>
    <div class="footer">
        <p>TECHNO SCOPE — AI Image Detection & Forensic Platform</p>
        <p>Comparative analysis. Results are probabilistic.</p>
    </div>
</body></html>`;

        // Render PDF
        const puppeteer = require('puppeteer');
        const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox', '--disable-setuid-sandbox'] });
        const page = await browser.newPage();
        await page.setContent(html, { waitUntil: 'networkidle0' });
        const pdfBuffer = await page.pdf({ format: 'A4', printBackground: true, margin: { top: '15mm', bottom: '15mm', left: '12mm', right: '12mm' } });
        await browser.close();

        const pdfPath = path.join(projectsDir, `${safeName}.pdf`);
        fs.writeFileSync(pdfPath, pdfBuffer);
        console.log(`[cases] Generated case PDF: ${safeName}.pdf`);

        res.json({ ok: true, case_id: safeName, pdf_url: `/projects/${safeName}.pdf` });

    } catch (err) {
        console.error('[cases] Save failed:', err.message);
        res.status(500).json({ error: 'Case save failed', message: err.message });
    }
});

/**
 * GET /v1/case/list
 * Lists all saved case files.
 */
router.get('/list', (req, res) => {
    try {
        const files = fs.readdirSync(projectsDir);
        const cases = {};

        files.forEach(f => {
            const ext = path.extname(f);
            const base = path.basename(f, ext);
            if (!cases[base]) cases[base] = {};
            if (ext === '.json') cases[base].json = f;
            if (ext === '.pdf') cases[base].pdf = f;
        });

        const list = Object.entries(cases).map(([id, files]) => ({
            case_id: id,
            json_url: files.json ? `/projects/${files.json}` : null,
            pdf_url: files.pdf ? `/projects/${files.pdf}` : null,
        }));

        res.json(list);
    } catch (err) {
        res.status(500).json({ error: 'Failed to list cases', message: err.message });
    }
});

module.exports = router;
