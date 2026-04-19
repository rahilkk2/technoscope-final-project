'use strict';

const express = require('express');
const router = express.Router();

/**
 * POST /report
 * 
 * Accepts forensic analysis JSON, renders a branded HTML report,
 * converts it to PDF via Puppeteer, and streams the file back.
 *
 * Body: { verdict, confidence, spectral_scores, metadata, spectral,
 *         comments, heatmap_base64?, image_url? }
 */
router.post('/', async (req, res) => {
    try {
        const {
            verdict = 'N/A',
            confidence = 0,
            spectral_scores = {},
            metadata = '',
            spectral = '',
            comments = '',
            heatmap_base64 = '',
            image_url = '',
            forensic_analysis = '',
        } = req.body;

        const timestamp = new Date().toLocaleString('en-US', {
            dateStyle: 'full', timeStyle: 'medium',
        });

        // Build spectral bars HTML from A-G scores
        const spectralLabels = {
            A_sensor_noise: 'A — Sensor Noise',
            B_texture:      'B — Texture',
            C_color:        'C — Color',
            D_edge:         'D — Edge',
            E_frequency:    'E — Frequency',
            F_metadata:     'F — Metadata',
            G_semantic:     'G — Semantic',
        };

        let spectralBarsHTML = '';
        for (const [key, label] of Object.entries(spectralLabels)) {
            const val = spectral_scores[key] ?? 0;
            const color = val > 70 ? '#f87171' : val > 50 ? '#fbbf24' : '#34d399';
            spectralBarsHTML += `
                <div style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">
                        <span>${label}</span><span>${val}%</span>
                    </div>
                    <div style="background:#1a1a1a;border-radius:4px;height:14px;overflow:hidden;">
                        <div style="width:${val}%;height:100%;background:${color};border-radius:4px;"></div>
                    </div>
                </div>`;
        }

        // Heatmap image section (only if provided)
        const heatmapSection = heatmap_base64
            ? `<div style="margin-top:30px;">
                 <h2 style="font-size:18px;border-bottom:1px solid #333;padding-bottom:8px;">Heatmap Visualization</h2>
                 <img src="data:image/png;base64,${heatmap_base64}" style="max-width:100%;border-radius:8px;margin-top:12px;border:1px solid #333;">
               </div>`
            : '';

        // Uploaded image section
        const imageSection = image_url
            ? `<div style="margin-top:30px;">
                 <h2 style="font-size:18px;border-bottom:1px solid #333;padding-bottom:8px;">Analyzed Image</h2>
                 <img src="${image_url}" style="max-width:100%;border-radius:8px;margin-top:12px;border:1px solid #333;">
               </div>`
            : '';

        const verdictColor = verdict === 'AUTHENTIC' ? '#34d399' : '#f87171';

        // ── Full report HTML (dark-themed, branded) ─────────────────────────
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        @page {
            size: A4;
            margin: 20mm 15mm;
        }
        * { box-sizing:border-box; margin:0; padding:0; }
        body {
            font-family: Arial, Helvetica, sans-serif;
            background: #0a0a0a; color: #e0e0e0;
            padding: 40px; line-height: 1.6;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        .header {
            text-align: center; margin-bottom: 40px;
            border-bottom: 1px solid #222; padding-bottom: 24px;
        }
        .header h1 { font-size: 28px; color: #fff; letter-spacing: 2px; }
        .header p { color: #777; font-size: 13px; margin-top: 6px; }
        .verdict-box {
            text-align: center; margin: 30px 0;
            padding: 24px; border-radius: 12px;
            background: #111; border: 1px solid #222;
        }
        .verdict-label { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .verdict-value { font-size: 36px; font-weight: 700; color: ${verdictColor}; margin: 8px 0; }
        .confidence-value { font-size: 18px; color: #aaa; }
        .section { margin-top: 30px; page-break-inside: avoid; }
        .section h2 { font-size: 18px; color: #fff; border-bottom: 1px solid #333; padding-bottom: 8px; margin-bottom: 16px; }
        .meta-block {
            background: #111; border: 1px solid #222; border-radius: 8px;
            padding: 16px; font-size: 13px; white-space: pre-wrap;
            font-family: 'Courier New', Courier, monospace; color: #aaa;
        }
        .comments-block {
            background: #111; border: 1px solid #222; border-radius: 8px;
            padding: 16px; font-size: 14px; color: #ccc; font-style: italic;
        }
        .footer {
            margin-top: 50px; text-align: center; color: #555;
            font-size: 11px; border-top: 1px solid #222; padding-top: 16px;
        }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Image Sentinel</h1>
        <p>Forensic Image Analysis Report</p>
        <p style="margin-top:4px;">${timestamp}</p>
    </div>

    <div class="verdict-box">
        <div class="verdict-label">Engine Verdict</div>
        <div class="verdict-value">${verdict}</div>
        <div class="confidence-value">Confidence: ${confidence}%</div>
    </div>

    ${comments ? `
    <div class="section">
        <h2>Analysis Comments</h2>
        <div class="comments-block">${comments}</div>
    </div>` : ''}

    <div class="section">
        <h2>7-Group Spectral Breakdown</h2>
        ${spectralBarsHTML}
    </div>

    ${forensic_analysis ? `
    <div class="section">
        <h2>Forensic Analysis</h2>
        <div class="comments-block">${forensic_analysis}</div>
    </div>` : ''}

    <div class="section">
        <h2>Metadata Summary</h2>
        <div class="meta-block">${typeof metadata === 'string' ? metadata : JSON.stringify(metadata, null, 2)}</div>
    </div>

    <div class="section">
        <h2>Spectral Summary</h2>
        <div class="meta-block">${typeof spectral === 'string' ? spectral : JSON.stringify(spectral, null, 2)}</div>
    </div>

    ${heatmapSection}
    ${imageSection}

    <div class="footer">
        <p>Image Sentinel — AI Image Detection &amp; Forensic Platform</p>
        <p>This report was generated automatically. Results are probabilistic, not definitive.</p>
    </div>
</body>
</html>`;

        // ── Render PDF via Puppeteer ────────────────────────────────────────
        const puppeteer = require('puppeteer');
        let browser;
        try {
            browser = await puppeteer.launch({
                headless: 'new',
                args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
                timeout: 60000,
            });
            const page = await browser.newPage();

            // Set viewport for consistent rendering
            await page.setViewport({ width: 1024, height: 1400 });

            // Use domcontentloaded — base64 data URIs don't need network;
            // networkidle0 hangs or times out with large embedded images,
            // producing empty / truncated PDFs.
            await page.setContent(html, {
                waitUntil: 'domcontentloaded',
                timeout: 30000,
            });

            // Small delay to let any inline images decode
            await new Promise(r => setTimeout(r, 500));

            const pdfResult = await page.pdf({
                format: 'A4',
                printBackground: true,
                preferCSSPageSize: false,
                margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' },
            });

            await browser.close();

            // Puppeteer v24+ returns Uint8Array — convert to Buffer
            const pdfBuffer = Buffer.from(pdfResult);

            // Validate PDF — must start with %PDF header
            if (!pdfBuffer || pdfBuffer.length < 100 || pdfBuffer.toString('ascii', 0, 5) !== '%PDF-') {
                throw new Error('Generated PDF is empty or invalid');
            }

            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', 'attachment; filename="forensic_report.pdf"');
            res.setHeader('Content-Length', pdfBuffer.length);
            res.setHeader('Cache-Control', 'no-cache');
            return res.end(pdfBuffer);

        } catch (puppeteerErr) {
            console.error('[report] Puppeteer execution failed:', puppeteerErr);
            if (browser) { try { await browser.close(); } catch(_) {} }
            throw puppeteerErr;
        }

    } catch (err) {
        console.error('[report] PDF generation failed:', err.message);
        res.status(500).json({ error: 'PDF generation failed', message: err.message });
    }
});

module.exports = router;
