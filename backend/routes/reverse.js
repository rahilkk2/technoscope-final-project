'use strict';

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const router = express.Router();

// ── Multer storage (reuses same upload dir as detect.js) ─────────────────────
const uploadDir = path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => {
        const unique = `${Date.now()}-${Math.round(Math.random() * 1e6)}`;
        const ext = path.extname(file.originalname) || '.jpg';
        cb(null, `reverse-${unique}${ext}`);
    },
});

const fileFilter = (req, file, cb) => {
    const allowed = /jpeg|jpg|png|webp|bmp|tiff/i;
    const ext = path.extname(file.originalname);
    if (allowed.test(ext)) {
        cb(null, true);
    } else {
        cb(new Error(`Unsupported file type: ${ext}`), false);
    }
};

const upload = multer({
    storage,
    fileFilter,
    limits: { fileSize: 100 * 1024 * 1024 }, // 100 MB max
});


// ── Run reverseEngineer.py ───────────────────────────────────────────────────

function runReverseAnalysis(imagePath) {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, '..', 'analysis', 'reverseEngineer.py');

        const proc = spawn('python', [scriptPath, imagePath], {
            cwd: path.join(__dirname, '..', 'analysis'),
            env: process.env,
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
        proc.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

        proc.on('error', (err) => {
            if (err.code === 'ENOENT') {
                return reject(new Error(
                    'Python not found. Make sure Python is installed and in your PATH.'
                ));
            }
            reject(new Error(`Failed to start reverseEngineer.py: ${err.message}`));
        });

        proc.on('close', (code) => {
            if (code !== 0) {
                const stdoutTrimmed = stdout.trim();
                const stderrTrimmed = stderr.trim();
                let detail = '';
                if (stdoutTrimmed) detail += `\nstdout: ${stdoutTrimmed}`;
                if (stderrTrimmed) detail += `\nstderr: ${stderrTrimmed}`;
                return reject(new Error(
                    `reverseEngineer.py exited with code ${code}.${detail || ' (no output)'}`
                ));
            }

            // Parse last valid JSON line
            try {
                const lines = stdout.trim().split('\n');
                let parsed = null;
                for (let i = lines.length - 1; i >= 0; i--) {
                    const line = lines[i].trim();
                    if (line.startsWith('{') || line.startsWith('[')) {
                        parsed = JSON.parse(line);
                        break;
                    }
                }
                if (!parsed) throw new Error('No JSON output found in stdout');
                resolve(parsed);
            } catch (parseErr) {
                reject(new Error(
                    `reverseEngineer.py JSON parse error: ${parseErr.message}\nstdout: ${stdout}`
                ));
            }
        });
    });
}


// ── POST /reverse ────────────────────────────────────────────────────────────

router.post('/', (req, res, next) => {
    upload.single('image')(req, res, (err) => {
        if (err instanceof multer.MulterError) {
            if (err.code === 'LIMIT_FILE_SIZE') {
                return res.status(400).json({ error: 'File too large', message: 'Maximum upload size is 100MB.' });
            }
            return res.status(400).json({ error: 'Upload failed', message: err.message });
        } else if (err) {
            return res.status(500).json({ error: 'Server error', message: err.message });
        }
        next();
    });
}, async (req, res) => {
    if (!req.file) {
        return res.status(400).json({
            error: 'No image file provided. Use field name "image".',
        });
    }

    const imagePath = req.file.path;

    try {
        console.log('[reverse.js] Running reverse engineering on:', imagePath);
        const result = await runReverseAnalysis(imagePath);

        // Check for Python-side errors
        if (result.error) {
            return res.status(500).json({
                error: 'Reverse analysis error',
                message: result.error,
            });
        }

        return res.status(200).json(result);
    } catch (err) {
        console.error('[reverse.js] Analysis failed:', err.message);
        return res.status(500).json({
            error: 'Reverse analysis failed',
            message: err.message,
        });
    } finally {
        // Clean up uploaded temp file
        fs.unlink(imagePath, () => { });
    }
});

module.exports = router;
