import { NextRequest, NextResponse } from 'next/server';
import { createWriteStream, existsSync, unlinkSync } from 'node:fs';
import { mkdir, rename } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import { getModelsRoot } from '@/server/settings';

const MODEL_EXTENSIONS = new Set(['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf']);

// Sanitize (don't reject) so real-world model names — spaces, parentheses,
// "Realistic Vision V5 (fp16).safetensors" — upload fine. path.basename strips
// any directory component (no traversal), disallowed chars collapse to '_', and
// we only hard-fail when there's no recognized model extension to anchor on.
function sanitizeFilename(raw: string): string | null {
  const base = path.basename(raw);
  if (!base || base === '.' || base === '..') return null;
  const ext = path.extname(base).toLowerCase();
  if (!MODEL_EXTENSIONS.has(ext)) return null;
  let stem = base
    .slice(0, base.length - ext.length)
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/^[._-]+|[._-]+$/g, '');
  if (!stem) stem = 'model';
  return `${stem}${ext}`.slice(0, 255);
}

// Streaming, CHUNKED upload. The client splits the file so each request body
// stays under the edge proxy's request cap (RunPod fronts pods with Cloudflare,
// which rejects a single multi-GB body with a 400/413 before it ever reaches
// this handler). Chunks carry X-Chunk-Index / X-Total-Chunks / X-Chunk-Offset;
// we write each at its byte offset into a `<name>.part` file (so a retried chunk
// is idempotent), then rename into place on the final chunk. A request with no
// chunk headers is treated as a single whole-file chunk (back-compatible).
export async function POST(request: NextRequest) {
  const rawName = request.headers.get('x-filename');
  if (!rawName) {
    return NextResponse.json({ error: 'X-Filename header is required' }, { status: 400 });
  }
  const filename = sanitizeFilename(rawName);
  if (!filename) {
    return NextResponse.json(
      { error: `Unsupported model file "${rawName}". Use a .safetensors/.ckpt/.pt/.pth/.bin/.gguf file.` },
      { status: 400 },
    );
  }
  if (!request.body) {
    return NextResponse.json({ error: 'Request body is empty' }, { status: 400 });
  }

  const totalChunks = Math.max(1, parseInt(request.headers.get('x-total-chunks') ?? '1', 10) || 1);
  const chunkIndex = Math.max(0, parseInt(request.headers.get('x-chunk-index') ?? '0', 10) || 0);
  const offset = Math.max(0, parseInt(request.headers.get('x-chunk-offset') ?? '0', 10) || 0);
  if (chunkIndex >= totalChunks) {
    return NextResponse.json({ error: 'X-Chunk-Index out of range' }, { status: 400 });
  }

  const modelsDir = await getModelsRoot();
  await mkdir(modelsDir, { recursive: true });
  const target = path.join(modelsDir, filename);
  const partPath = `${target}.part`;

  if (chunkIndex === 0 && existsSync(target)) {
    return NextResponse.json({ error: 'A model with that filename already exists' }, { status: 409 });
  }
  if (chunkIndex > 0 && !existsSync(partPath)) {
    return NextResponse.json(
      { error: 'No in-progress upload for this file (chunk arrived before chunk 0)' },
      { status: 409 },
    );
  }

  try {
    const stream = Readable.fromWeb(request.body as any);
    // chunk 0 truncates/creates ('w'); later chunks write at their offset ('r+').
    const ws = chunkIndex === 0
      ? createWriteStream(partPath, { flags: 'w' })
      : createWriteStream(partPath, { flags: 'r+', start: offset });
    await pipeline(stream, ws);

    if (chunkIndex === totalChunks - 1) {
      if (existsSync(target)) {
        try { unlinkSync(partPath); } catch { /* ignore */ }
        return NextResponse.json({ error: 'A model with that filename already exists' }, { status: 409 });
      }
      await rename(partPath, target);
      return NextResponse.json({ ok: true, complete: true, filename, path: target });
    }
    return NextResponse.json({ ok: true, complete: false, filename, chunkIndex, totalChunks });
  } catch (e: any) {
    // On any failure, drop the partial so a stale .part doesn't block a retry
    // or masquerade as a model. The client restarts from chunk 0.
    try {
      if (existsSync(partPath)) unlinkSync(partPath);
    } catch {
      // ignore
    }
    return NextResponse.json({ error: e?.message ?? 'Upload failed' }, { status: 500 });
  }
}
