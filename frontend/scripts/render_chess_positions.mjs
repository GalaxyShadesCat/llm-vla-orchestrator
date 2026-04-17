/**
 * Render chess benchmark images by loading the frontend snapshot mode and
 * capturing the ChessBoard3D canvas with Playwright.
 */

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright';

function parseArgs(argv) {
  const args = {
    samplesJsonl: '',
    baseUrl: 'http://127.0.0.1:5173',
    timeoutMs: 90000,
    width: 560
  };

  for (let index = 2; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--samples-jsonl') {
      args.samplesJsonl = String(argv[index + 1] || '');
      index += 1;
      continue;
    }
    if (token === '--base-url') {
      args.baseUrl = String(argv[index + 1] || args.baseUrl);
      index += 1;
      continue;
    }
    if (token === '--timeout-ms') {
      args.timeoutMs = Number(argv[index + 1] || args.timeoutMs);
      index += 1;
      continue;
    }
    if (token === '--width') {
      args.width = Number(argv[index + 1] || args.width);
      index += 1;
      continue;
    }
  }

  if (!args.samplesJsonl) {
    throw new Error('--samples-jsonl is required');
  }
  return args;
}

function loadSamples(samplesJsonlPath) {
  const raw = fs.readFileSync(samplesJsonlPath, 'utf8');
  const lines = raw.split('\n').map((line) => line.trim()).filter(Boolean);
  return lines.map((line) => JSON.parse(line));
}

function writePngDataUrlToFile(dataUrl, outputPath) {
  const marker = 'base64,';
  const offset = dataUrl.indexOf(marker);
  if (offset < 0) {
    throw new Error('Canvas output is not a base64 data URL');
  }
  const base64Payload = dataUrl.slice(offset + marker.length);
  const pngBuffer = Buffer.from(base64Payload, 'base64');
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, pngBuffer);
}

async function captureOne(page, sample, args) {
  const fen = encodeURIComponent(String(sample.after_fen || ''));
  const pitch = Number(sample.camera_pitch_deg ?? 65);
  const distance = Number(sample.camera_distance ?? 18);
  const width = Number(args.width || 560);
  const snapshotUrl = `${args.baseUrl}/?snapshot=1&fen=${fen}&pitch=${pitch}&distance=${distance}&width=${width}`;

  await page.goto(snapshotUrl, {
    waitUntil: 'domcontentloaded',
    timeout: args.timeoutMs
  });

  await page.waitForFunction(() => window.__snapshotReady === true, {
    timeout: args.timeoutMs
  });

  let dataUrl = '';
  let lastError = '';
  const minVisualDiversity = 90;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      dataUrl = await page.evaluate(({ expectedWidth, diversityThreshold }) => {
        const canvases = Array.from(document.querySelectorAll('canvas'));
        const canvas = canvases
          .slice()
          .sort((left, right) => (right.width * right.height) - (left.width * left.height))[0] || null;
        if (!canvas) {
          throw new Error('Snapshot canvas not found');
        }
        if (canvas.width < Math.max(500, Number(expectedWidth) - 20)) {
          throw new Error(`Snapshot canvas width too small (${canvas.width})`);
        }
        if (canvas.height < Math.max(500, Number(expectedWidth) - 20)) {
          throw new Error(`Snapshot canvas height too small (${canvas.height})`);
        }
        const snapshot = canvas.toDataURL('image/png');
        if (!snapshot || snapshot.length < 5000) {
          throw new Error('Snapshot data URL too small; frame likely not rendered yet');
        }

        const scratch = document.createElement('canvas');
        scratch.width = canvas.width;
        scratch.height = canvas.height;
        const scratchCtx = scratch.getContext('2d', { willReadFrequently: true });
        if (!scratchCtx) {
          throw new Error('Unable to create scratch canvas for validation');
        }
        scratchCtx.drawImage(canvas, 0, 0);
        const imageData = scratchCtx.getImageData(0, 0, scratch.width, scratch.height).data;
        const sampled = new Set();
        const stepX = Math.max(1, Math.floor(scratch.width / 32));
        const stepY = Math.max(1, Math.floor(scratch.height / 32));
        for (let y = 0; y < scratch.height; y += stepY) {
          for (let x = 0; x < scratch.width; x += stepX) {
            const index = (y * scratch.width + x) * 4;
            sampled.add(`${imageData[index]}-${imageData[index + 1]}-${imageData[index + 2]}`);
          }
        }
        if (sampled.size < diversityThreshold) {
          throw new Error(`Snapshot visual diversity too low (${sampled.size})`);
        }
        return snapshot;
      }, { expectedWidth: width, diversityThreshold: minVisualDiversity });
      if (dataUrl) {
        break;
      }
    } catch (error) {
      lastError = String(error?.message || error);
      await page.waitForTimeout(300);
    }
  }
  if (!dataUrl) {
    throw new Error(`Snapshot capture failed: ${lastError || 'unknown error'}`);
  }

  const outputPath = String(sample.image_path || '').trim();
  if (!outputPath) {
    throw new Error(`Sample ${sample.sample_id} has no image_path`);
  }
  writePngDataUrlToFile(dataUrl, outputPath);
}

async function main() {
  const args = parseArgs(process.argv);
  const samples = loadSamples(args.samplesJsonl);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: Number(args.width), height: Number(args.width) }
  });
  const page = await context.newPage();

  try {
    for (const sample of samples) {
      await captureOne(page, sample, args);
      process.stdout.write(`captured ${sample.sample_id}\n`);
    }
  } finally {
    await page.close();
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${String(error?.stack || error)}\n`);
  process.exit(1);
});
