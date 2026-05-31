import { NextRequest, NextResponse } from 'next/server';
import { createWriteStream, existsSync, unlinkSync, statSync } from 'node:fs';
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

function resolvePaths(modelsDir: string, filename: string) {
  const target = path.join(modelsDir, filename);
  return { target, partPath: `${target}.part` };
}

// GET /api/models/upload?filename=foo.safetensors
// Reports how many bytes are already on disk for an in-progress upload so the
// client can resume from there after a disconnect or reload instead of
// restarting a multi-GB transfer from zero.
export async function GET(request: NextRequest) {
  const rawName = request.nextUrl.searchParams.get('filename');
  if (!rawName) {
    return NextResponse.json({ error: 'filename query param is required' }, { status: 400 });
  }
  const filename = sanitizeFilename(rawName);
  if (!filename) {
    return NextResponse.json({ error: `Unsupported model file "${rawName}".` }, { status: 400 });
  }
  const modelsDir = await getModelsRoot();
  const { target, partPath } = resolvePaths(modelsDir, filename);
  if (existsSync(target)) {
    return NextResponse.json({ filename, exists: true, complete: true, uploaded: 0 });
  }
  let uploaded = 0;
  try {
    if (existsSync(partPath)) uploaded = statSync(partPath).size;
  } catch {
    uploaded = 0;
  }
  return NextResponse.json({ filename, exists: false, complete: false, uploaded });
}

// Streaming, CHUNKED, RESUMABLE upload. Each chunk is its own request so the
// body stays under the RunPod/Cloudflare edge body cap (a single multi-GB body —
// and even a 50 MiB one — is rejected with a 400 at the edge before it reaches
// this handler). The client picks the chunk size adaptively, so we can't assume
// a fixed count: instead each chunk carries X-Chunk-Offset (its byte position)
// and X-File-Size (the total), and we finalize once the .part reaches that size.
//
// Writing at the byte offset makes a re-sent chunk idempotent, and we deliberately
// keep the .part on a chunk error so the client resumes from the server-reported
// offset rather than restarting. A request with neither header is treated as a
// single whole-file upload (back-compatible).
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

  const offset = Math.max(0, parseInt(request.headers.get('x-chunk-offset') ?? '0', 10) || 0);
  const fileSizeHeader = request.headers.get('x-file-size');
  // No X-File-Size => single-shot whole-file upload (finalize immediately).
  const fileSize = fileSizeHeader != null ? Math.max(0, parseInt(fileSizeHeader, 10) || 0) : null;

  const modelsDir = await getModelsRoot();
  await mkdir(modelsDir, { recursive: true });
  const { target, partPath } = resolvePaths(modelsDir, filename);

  if (existsSync(target)) {
    return NextResponse.json({ error: 'A model with that filename already exists' }, { status: 409 });
  }
  // A continued/resumed chunk (offset > 0) needs an in-progress .part to append
  // to, and the offset must not skip past what we already have (no holes). If
  // the client is out of sync, tell it where we actually are so it can resync.
  if (offset > 0) {
    const have = existsSync(partPath) ? statSync(partPath).size : 0;
    if (have < offset) {
      return NextResponse.json(
        { error: 'Chunk offset is ahead of the partial file; resync required', code: 'OFFSET_GAP', uploaded: have },
        { status: 409 },
      );
    }
  }

  try {
    const stream = Readable.fromWeb(request.body as any);
    // offset 0 truncates/creates ('w'); a continued chunk writes at its offset ('r+').
    const ws = offset === 0
      ? createWriteStream(partPath, { flags: 'w' })
      : createWriteStream(partPath, { flags: 'r+', start: offset });
    await pipeline(stream, ws);

    const uploaded = statSync(partPath).size;
    const complete = fileSize == null ? true : uploaded >= fileSize;

    if (complete) {
      if (existsSync(target)) {
        try { unlinkSync(partPath); } catch { /* ignore */ }
        return NextResponse.json({ error: 'A model with that filename already exists' }, { status: 409 });
      }
      await rename(partPath, target);
      return NextResponse.json({ ok: true, complete: true, filename, uploaded, path: target });
    }
    return NextResponse.json({ ok: true, complete: false, filename, uploaded });
  } catch (e: any) {
    // Keep the .part: the client resumes from the server-reported offset and only
    // re-sends the failed tail, so one bad chunk doesn't cost a multi-GB restart.
    return NextResponse.json({ error: e?.message ?? 'Upload failed' }, { status: 500 });
  }
}
