'use strict';

const { spawn } = require('child_process');
const path = require('path');

const ANALYSIS_DIR = path.join(__dirname, '..', 'analysis');

/**
 * Runs a single Python analysis script and returns parsed JSON output.
 * Includes stdout in error messages so missing-library errors are visible.
 */
function runScript(scriptName, imagePath) {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(ANALYSIS_DIR, scriptName);

        const proc = spawn('python', [scriptPath, imagePath], {
            cwd: ANALYSIS_DIR,
            env: process.env,
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
        proc.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

        proc.on('error', (err) => {
            // 'python' not found in PATH — try python3
            if (err.code === 'ENOENT') {
                return reject(new Error(
                    `Python not found. Make sure Python is installed and in your PATH.\n` +
                    `Try running: python --version`
                ));
            }
            reject(new Error(`Failed to start ${scriptName}: ${err.message}`));
        });

        proc.on('close', (code) => {
            if (code !== 0) {
                // Include stdout so missing-library JSON errors are visible
                const stdoutTrimmed = stdout.trim();
                const stderrTrimmed = stderr.trim();
                let detail = '';
                if (stdoutTrimmed) detail += `\nstdout: ${stdoutTrimmed}`;
                if (stderrTrimmed) detail += `\nstderr: ${stderrTrimmed}`;
                return reject(new Error(
                    `${scriptName} exited with code ${code}.${detail || ' (no output)'}`
                ));
            }

            // Parse last valid JSON line from stdout
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
                    `${scriptName} JSON parse error: ${parseErr.message}\nstdout: ${stdout}`
                ));
            }
        });
    });
}

/**
 * Runs all three Python scripts in parallel and merges results.
 */
async function runPythonAnalysis(imagePath) {
    console.log('[pythonBridge] Running Python analysis on:', imagePath);

    const [metadataResult, featureResult, classifierResult] = await Promise.all([
        runScript('metadata.py', imagePath),
        runScript('featureExtractor.py', imagePath),
        runScript('classifier.py', imagePath),
    ]);

    return {
        metadata: metadataResult,
        features: featureResult,
        classifier: classifierResult,
    };
}

module.exports = { runPythonAnalysis };
