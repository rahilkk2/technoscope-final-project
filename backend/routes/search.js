'use strict';

const express = require('express');
const router = express.Router();

/**
 * Reverse Image Search — Adapter Pattern
 *
 * This module is a pluggable stub. Configure via environment variables:
 *   REVERSE_SEARCH_ENDPOINT  — full URL of the search API
 *   REVERSE_SEARCH_KEY       — API key (sent as X-API-Key header)
 *   REVERSE_SEARCH_PROVIDER  — label shown in the UI (default: "External Search")
 *
 * Expected API contract:
 *   POST <endpoint>  { "url": "https://image.jpg" }
 *   Response:         { "results": [{ "url": "...", "title": "..." }, ...] }
 *
 * Supported adapters (swap by changing REVERSE_SEARCH_PROVIDER):
 *   - "serpapi"  → SerpAPI Google Reverse Image (https://serpapi.com)
 *   - "bing"     → Bing Visual Search API
 *   - "default"  → Generic POST { url } → { results }
 */

// ── Adapter: transforms provider-specific response → { results: [{ url, title }] }
function adaptResponse(provider, rawData) {
    try {
        switch (provider) {
            case 'serpapi':
                // SerpAPI returns: { images_results: [{ link, title, source, ... }] }
                const serpResults = (rawData.images_results || rawData.inline_images || []).slice(0, 15);
                return {
                    results: serpResults.map(r => ({
                        url: r.link || r.original || r.source,
                        title: r.title || r.source || 'Untitled',
                        thumbnail: r.thumbnail || '',
                    })),
                };

            case 'bing':
                // Bing Visual Search returns: { tags[0].actions[].data.value[] }
                const bingResults = [];
                try {
                    const tags = rawData.tags || [];
                    for (const tag of tags) {
                        for (const action of (tag.actions || [])) {
                            if (action.actionType === 'PagesIncluding' && action.data && action.data.value) {
                                for (const page of action.data.value.slice(0, 15)) {
                                    bingResults.push({
                                        url: page.hostPageUrl || page.contentUrl,
                                        title: page.name || 'Untitled',
                                        thumbnail: page.thumbnailUrl || '',
                                    });
                                }
                            }
                        }
                    }
                } catch (_) {}
                return { results: bingResults };

            default:
                // Generic: assume { results: [{ url, title }] } already
                return {
                    results: (rawData.results || []).slice(0, 15).map(r => ({
                        url: r.url || r.link || '#',
                        title: r.title || r.name || 'Untitled',
                        thumbnail: r.thumbnail || '',
                    })),
                };
        }
    } catch (err) {
        console.error('[search] Adapter transform failed:', err.message);
        return { results: [] };
    }
}

/**
 * POST /search-similar
 * Body: { image_url: "https://..." }
 * Returns: { results: [{ url, title, thumbnail? }], provider }
 */
router.post('/', async (req, res) => {
    const { image_url } = req.body;

    if (!image_url) {
        return res.status(400).json({ error: 'image_url is required' });
    }

    const endpoint = process.env.REVERSE_SEARCH_ENDPOINT || '';
    const apiKey = process.env.REVERSE_SEARCH_KEY || '';
    const provider = (process.env.REVERSE_SEARCH_PROVIDER || 'default').toLowerCase();

    // If no API is configured, return clean unavailable response
    if (!endpoint) {
        console.log('[search] No REVERSE_SEARCH_ENDPOINT configured — returning unavailable');
        return res.json({
            status: 'unavailable',
            message: 'Reverse image search is not connected in this build.',
            links: [],
            results: [],
        });
    }

    try {
        // Dynamic import for node-fetch (handles both CJS and ESM)
        let fetchFn;
        try {
            fetchFn = (await import('node-fetch')).default;
        } catch (_) {
            fetchFn = require('node-fetch');
        }

        console.log(`[search] Querying ${provider} at ${endpoint}`);

        const headers = { 'Content-Type': 'application/json' };
        if (apiKey) headers['X-API-Key'] = apiKey;
        if (provider === 'serpapi') headers['Authorization'] = `Bearer ${apiKey}`;

        const response = await fetchFn(endpoint, {
            method: 'POST',
            headers,
            body: JSON.stringify({ url: image_url }),
            timeout: 15000,
        });

        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }

        const rawData = await response.json();
        const adapted = adaptResponse(provider, rawData);

        res.json({ ...adapted, provider });

    } catch (err) {
        console.error('[search] API call failed:', err.message);
        res.json({
            provider,
            error: `Search API call failed: ${err.message}`,
            results: [
                { url: 'https://images.google.com', title: 'Try Google Images manually', thumbnail: '' },
                { url: 'https://yandex.com/images', title: 'Try Yandex Images manually', thumbnail: '' },
            ],
        });
    }
});

module.exports = router;
